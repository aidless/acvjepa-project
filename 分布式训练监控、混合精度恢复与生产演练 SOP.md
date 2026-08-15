# 分布式训练监控、混合精度恢复与生产演练 SOP

**作者：Manus AI**  
**日期：2026-08-15**  
**适用范围：** AC-VJEPA 的 2:1 异构微批、NCCL/DDP、多节点 `torchrun` Rendezvous 重建、全局数据游标账本及受控 RDMA/rail 演练。  
**非适用范围：** 本文不授予网络、交换机、RDMA、GPU reset、机器人或生产发布操作权限。

## 1. 总体设计与边界

该系统将运行时观测拆为四条相互可关联、但权限分离的证据线：训练进程导出低基数指标；游标账本保存高维数据/计划/检查点血缘；日志与 NCCL RAS/flight recorder 保存故障细节；对象存储保存 checkpoint 与全状态 fingerprint。Prometheus/Grafana 的职责是聚合、可视化和告警；它只能建议**冻结新的 UpdatePlan、标记作业 `SUSPECT`、保留工件并通知责任人**，不能直接执行网络扰动、集群重启、cursor 推进或模型发布。

Prometheus 官方建议指标使用稳定前缀、单一量/单位和基础单位；每个独特 label 组合都会产生新时间序列，因此不得把 work ID、轨迹、hash、checkpoint URI、错误原文或人员标识放入标签。[1] 这些高维证据必须保存在账本/日志，并通过受控 experiment ID 或 incident ID 关联。

```text
AC-VJEPA rank workers
  ├─ 2:1 UpdatePlan / NCCL / AMP runtime events
  ├─ ElasticCursorLedger: committed cursor + attempt + checkpoint lineage
  ├─ PrecisionCheckpoint: model / EMA / AdamW / scaler or FP8 metadata / RNG
  └─ RailChaosGuard: approval / isolation / rollback receipt
           │
           ▼
Training metrics facade  ──►  metrics endpoint  ──► Prometheus
                                                     │
                                    audit links ◄────┼────► Grafana panels / alert rules
                                                     │
                    detailed evidence ◄─────────────┴────► ledger, object store, RAS, logs
```

| 层 | 允许写入 | 禁止写入 | 保留周期建议 |
|---|---|---|---|
| Prometheus 指标 | 低基数健康、时延、计数、gauge、阶段 | work/window ID、全 hash、原始数据、错误正文、授权人 | 用于在线告警的短/中期时序 |
| Grafana 面板与告警 | PromQL、阈值、runbook 链接、通知策略 | 网络/训练执行命令、审批密钥 | 版本化 dashboard/rules |
| cursor/检查点账本 | 完整 manifest/plan/checkpoint digest、attempt 状态 | 原始传感器内容（仅引用） | 至少覆盖回滚与审计窗口 |
| 对象存储/RAS/trace | checkpoint、全 state fingerprint、flight recorder、环境快照 | 自动提升模型到生产的授权 | 与事故/合规策略一致 |

## 2. Prometheus 指标合同

### 2.1 导出拓扑

生产中建议由每个训练 worker 在本地写运行状态，由同节点受控 exporter 采集；rank 0 或一个专门 aggregator 汇总 job 级 counter/gauge，避免每个 rank 对同一 job 级计数重复累加。仅在需要排查 2:1 槽位时导出有限的 `rank_slot` 标签；其域必须受 `nproc_per_node × 最大节点数` 上限约束。所有 exporter target 必须含 `cluster`、`job`、`environment` 三个稳定标签。

`distributed_training_observability.py` 是可运行的指标 facade。它采用 `prometheus-client`，但不负责打开 HTTP 服务或改动部署环境；生产封装应由受控的 metrics endpoint 提供 scrape 接入。指标名称均为 `acvjepa_training_*`，单位使用 seconds/bytes/total 等规范形式。[1]

| 指标族 | 类型 | 标签 | 用途 |
|---|---|---|---|
| `acvjepa_training_plan_digest_consensus` | Gauge | cluster, job, environment | 当前计划摘要是否所有 rank 一致；0 必须阻止 backward |
| `acvjepa_training_update_global_samples` | Gauge | 公共标签 | 当前 armed plan 的全局有效样本数；2:1 双 rank/batch=2 时为 6 |
| `acvjepa_training_plan_micro_batches` | Gauge | + `rank_slot` | 显示 rank 0=2、rank 1=1 等受限槽位计划 |
| `acvjepa_training_update_attempts_total` | Counter | + outcome | `committed`、`aborted`、`rejected`；不含高维原因 |
| `acvjepa_training_allreduce_duration_seconds` | Histogram | + `rail_class` | 最终同步 AllReduce 的尾延迟；具体 rail identity 留在账本/RAS |
| `acvjepa_training_cursor_next_offset` | Gauge | 公共标签 | 已确认全局游标的下一未提交 window offset |
| `acvjepa_training_cursor_reservations` | Gauge | 公共标签 | `PREPARED` reservation 数；通常为 0 或 1 |
| `acvjepa_training_checkpoint_commits_total`、`...age_seconds` | Counter/Gauge | 公共标签 | 原子 checkpoint 提交率与可容忍失工作窗口 |
| `acvjepa_training_rendezvous_rebuilds_total`、`...restart_count`、`...recovery_duration_seconds` | Counter/Gauge/Histogram | + bounded cause | 弹性重建频率、当前 restart 与恢复 p95 |
| `acvjepa_training_state_alignment_verified` | Gauge | + component | `model`、`ema`、`optimizer`、`rng` 的 checkpoint load 后精确验证 |
| `acvjepa_training_grad_scaler_*` | Gauge/Counter | 公共标签或 precision mode | FP16 scaler 状态、overflow；BF16 通常 disabled |
| `acvjepa_training_fp8_metadata_verified` | Gauge | 公共标签 | FP8 scaling/AMAX metadata 和 engine contract 是否准确恢复 |
| `acvjepa_training_chaos_*` | Gauge/Counter | phase/rejection reason | 演练阶段和守卫拒绝；experiment identity 进入审计账本而非 label |

### 2.2 从账本到指标的映射

任何 `ElasticCursorLedger` 事务变化都应按如下顺序更新观测：先写 durable ledger，成功后更新 exporter 的 gauge/counter；若指标发送失败，不能回滚 ledger 事务。指标是观测副本，不是控制真相。恢复时由 ledger 重建 gauge，而不是从 Prometheus 反推 cursor。

```python
# 只在成功 ledger 事务后调用；指标失败不得影响 checkpoint commit。
cursor = ledger.commit_update(...)
metrics.record_cursor(
    next_offset=cursor.next_offset,
    prepared_reservations=0,
    checkpoint_age_seconds=0.0,
    committed=True,
)

# rebuild 后：只报告，不自动执行网络或发布动作。
metrics.record_rebuild(
    cause="node_failure",
    restart_count=current_restart_count,
    recovery_seconds=elapsed_seconds,
)
```

### 2.3 Recording rules 与告警策略

交付文件 `monitoring/prometheus_rules.yml` 包含 4 条 recording rules 与 9 条告警规则。Recording rules 预聚合常用的 update rate、AllReduce p95、recovery p95 和 abort ratio，避免 dashboard/告警重复运行昂贵 PromQL。Grafana 文档也建议对高成本查询预聚合，并对短暂后端错误使用 `Keep Last State`、合理 pending period 和 `absent()`/`absent_over_time()` 检测指标消失。[2]

| 告警 | 条件摘要 | 级别 | 自动允许动作 | 人工 runbook 重点 |
|---|---|---|---|---|
| `ACVJEPAUpdatePlanConsensusLost` | plan consensus=0 持续 1m | Critical | 冻结新计划 | 比对 topology/work manifest/JSON digest；旧计划不可重试 |
| `ACVJEPAPreparedCursorStuck` | reservation>0 且 10m 无 checkpoint commit | Warning | 标记 SUSPECT | 判断 update 是未提交、卡住还是 exporter 滞后 |
| `ACVJEPACommittedCheckpointStale` | checkpoint age 超阈值 | Warning | 通知 | 校准阈值为批准的失工作预算，而不是复制示例 900s |
| `ACVJEPARendezvousRebuildStorm` | 15m 重建超过预算 | Critical | 禁止自动重启升级 | 保留 RAS/trace/健康工件，升级基础设施团队 |
| `ACVJEPAExactStateRestoreFailed` | model/EMA/optimizer/RNG 任一验证为 0 | Critical | 阻止恢复 update | hash、precision backend、optimizer 实现、checkpoint 读回 |
| `ACVJEPAFP8MetadataRestoreFailed` | FP8 job 的 metadata verify=0 | Critical | 阻止恢复 update | 版本、`_extra_state`、AMAX/scale recipe、backend key 映射 |
| `ACVJEPAAllReduceTailLatencyHigh` | 有流量时 AllReduce p95 超基线 | Warning | 观察/冻结可选 | topology、rail、RAS、NIC/GPU 健康；不可只加 timeout |
| `ACVJEPAUpdateAbortRatioHigh` | 5m abort ratio 高 | Warning | 不推进 cursor | 数据合同、rank skew、watchdog、loader/plan 对齐 |
| `ACVJEPAMetricsTargetMissing` | metrics 消失 | Warning | 通知 | 区分 job 完成、exporter 故障、scrape 故障与 worker 崩溃 |

告警的**故障动作**应写成 alert label/annotation 与人工流程，而不是 webhook 直接执行命令。对于 Grafana-managed rules，查询错误或超时应采用 `Keep Last State`，否则短暂的 Prometheus 网络错误会形成虚假事故；但频繁 query error 本身仍应被调查。[2]

### 2.4 Grafana 面板

交付 `monitoring/grafana_acvjepa_elastic_dashboard.json`，可导入到带 Prometheus datasource 的 Grafana 实例。面板按三个下拉变量过滤 `cluster`、`job`、`environment`，包含：

| 面板区域 | 关键问题 | 判读方式 |
|---|---|---|
| 顶部状态卡 | plan 是否一致？cursor 是否存在预留？checkpoint 是否新鲜？恢复 state 是否完整？ | 任一红色状态先阻止下一个 update，再看详细证据 |
| 2:1 计划 | 是否仍为 `2/1`？是否有意外缩减/扩大？ | 计划变化要与 telemetry/topology epoch/overflow 相关联 |
| NCCL 尾延迟 | final AllReduce p95 是否上升？ | 有进度的短暂差异不等同通信故障；结合 RAS 与 plan guard |
| Update outcome | committed/aborted/rejected 比率怎样？ | abort 上升需回到 ledger attempt 证据而非只看 counter |
| Elastic recovery | restart count、recovery p95 是否满足预算？ | rebuild storm 应停止重复重试并升级 |
| Precision safety | GradScaler scale/overflow、FP8 metadata 是否通过？ | BF16 的 GradScaler 不应被误报为失败；FP8 metadata=0 必须阻断恢复 |
| Chaos guard | 演练是否已 rollback？请求为何被拒绝？ | 任何 guard rejection 都优先证明安全，而不是“绕过后再试” |

## 3. BF16/FP16/FP8 的弹性恢复数值合同

### 3.1 “绝对数值对齐”的精确定义

在此系统中，**绝对对齐**是一个 checkpoint-boundary 合同：在加载同一个 `COMMITTED` checkpoint 后、下一次 forward/随机采样/collective 前，当前进程组上的模型、EMA、AdamW、precision state 和 RNG 与写入 checkpoint 的 bytes/state-dict 完全一致。`mixed_precision_elastic_recovery.py` 使用类型感知 SHA-256 fingerprint 验证该合同。

它**不**表示以下情况仍可逐 bit 连续：中断前的 in-flight update；world-size 改变后的下一次梯度归约；GPU/驱动/内核/TF32/通信压缩变化后的下一次计算；或不同 FP8 engine/version 的 metadata 解释。此时正确主张是“从同一确认 checkpoint 无损加载”，而不是“恢复了一个永未提交的计算”。

| 比较时刻 | 相同 world size / backend | world size 改变 | 是否允许提交 |
|---|---|---|---|
| 刚 load `COMMITTED` checkpoint | model/EMA/optimizer/scaler/FP8/RNG fingerprint 必须完全相同 | 同样必须完全相同 | 仅在所有 required component=true 后允许新 plan |
| 重建后第一次 forward 前 | precision contract 必须相同 | 同左 | 允许进入 shadow/update 准备 |
| 下一次更新后 | 可做固定输入参考容差比较 | 不承诺逐 bit；以新规模 baseline、cursor、loss/health 合同判断 | 需通过 plan/数据/安全门控 |
| 故障时 in-flight update | 不可拼接 | 不可拼接 | 必须 `UNCOMMITTED` 并重读 cursor range |

### 3.2 必须同时 checkpoint 的状态

| 状态组 | BF16 | FP16 | FP8 | 原因 |
|---|---|---|---|---|
| 学生模型、EMA、buffer | 必须 | 必须 | 必须 | 预测与 EMA target 的唯一事实来源 |
| AdamW `state` 与 `param_groups` | 必须；通常是 FP32 moment | 必须 | 必须；常仍为 FP32 master/moment | 包括 `exp_avg`、`exp_avg_sq`、`step`、lr、betas、weight decay、参数组顺序 |
| scheduler、global step、clip config | 必须 | 必须 | 必须 | 防止恢复后训练率/裁剪语义突变 |
| CPU/CUDA RNG（每 rank）与 global sample seed | 必须 | 必须 | 必须 | 固定规模 replay/数据顺序；变规模必须显式重新派生 seed |
| autocast dtype/backend policy | `bf16` 配置 | `fp16` 配置 | FP8 recipe/format/engine 版本 | 与 state 一起验证，不能默认为相容 |
| `GradScaler.state_dict()` | 通常不存在/disabled | **必须** | 只有所选 backend 明确使用时才存 | FP16 dynamic scale 是数值状态，不是可丢弃配置 |
| FP8 scale/AMAX history | 不适用 | 不适用 | **必须** | FP8 delayed scaling 的状态影响下一迭代量化 |

PyTorch AMP 文档说明 fp16 梯度缩放用于减轻 underflow，`autocast` 与 `GradScaler` 可分别使用；其 BF16 示例只使用 autocast，且 GradScaler 的 scale 不保证始终大于 1。[3] 因此本系统中“BF16 job 未导出 GradScaler scale”是正常状态，而“FP16 job 的 enabled GradScaler state 缺失”是恢复阻断。

Transformer Engine 文档指出 FP8 scaling factors 与 AMAX histories 存在 checkpoint 的 `._extra_state`，且不同版本的 key 位置可能变化。[4] 因而 FP8 checkpoint 必须绑定 engine/package 版本、FP8 recipe/format、所需 `_extra_state` 键集合和 metadata fingerprint；缺失、默认初始化或 key 迁移未映射都不能视为精确恢复。

### 3.3 FP16 overflow 的特别提交规则

`GradScaler.step()` 可能因非有限梯度跳过 optimizer step，但 `GradScaler.update()` 仍会改变 scale。若简单把该事件当作“未发生”并丢弃 scaler 新状态，Rendezvous 后就无法保持 precision state 对齐；若又推进数据 cursor，则会把没有 optimizer 更新的 window 错记为已训练。

推荐扩展游标账本的 commit 类型：

```text
UPDATE_COMMIT:
  optimizer_step = true
  EMA/scheduler 已更新
  next_offset 前移
  committed_step 前移

PRECISION_STATE_COMMIT:
  optimizer_step = false（例如 FP16 overflow skip）
  GradScaler/FP8 scale 发生合法变化
  next_offset 不前移
  committed_step 不前移；precision_event_seq 前移
  下一 attempt 重读同一 global work range
```

这样能同时满足两点：scale backoff 被持久化；`work-0..2` 未被虚假消耗。对 FP16，正确顺序为：`unscale_` → clip → `scaler.step(optimizer)` → `scaler.update()` → 判断是否实际 step/overflow → 更新 EMA/scheduler（仅实际 step 时）→ 写相应 checkpoint/ledger commit。不要在一个 attempt 内多次无条件调用 `unscale_` 或 `update()`。

### 3.4 固定规模与变规模恢复流程

```text
恢复前：冻结新 UpdatePlan，确认上一条 COMMITTED checkpoint 与 cursor
   ↓
加载模型 / EMA / AdamW / scheduler / RNG / precision state
   ↓
同版本 precision contract、checkpoint hash、state fingerprints 全部为真？
   ├─ 否：BLOCK，保留证据，人工升级
   └─ 是：记录 state_alignment_verified=1
        ↓
world size 是否变化？
   ├─ 否：可运行固定输入的全状态参考测试
   └─ 是：清除旧 rank-local RNG/sampler 假设；从 global cursor 新分片
        ↓
采集新 topology + work manifest；新 UpdatePlan digest 共识
   ↓
shadow update → 正常训练（仍需现有发布门控）
```

`mixed_precision_elastic_recovery.py` 在 CPU 上验证三种合同：BF16（无 enabled scaler）、FP16（GradScaler state）和 FP8（显式 scale/AMAX mapping）。它递归 fingerprint model、EMA、AdamW 的所有 tensor/标量状态、scaler、FP8 metadata 与 CPU RNG；smoke test 已通过。实际 CUDA 部署必须增加 per-rank CUDA RNG、scheduler、FSDP/ZeRO shard 以及具体 FP8 引擎 state 的 load/save 适配，并在目标 CUDA/driver/backend 版本上复测。

## 4. 生产上线前验收 Checklist

以下 checklist 应作为**签字式门槛**而非“尽量完成”的建议。所有 `否` 或 `未知` 都意味着不进入生产 rollout，而是保持隔离环境或 shadow 状态。

### A. 架构与数据合同

| 检查项 | 通过标准 | 证据 | Owner |
|---|---|---|---|
| 全局 manifest | `dataset_commit`、排序、provenance、digest 已冻结 | manifest artifact + hash | 数据 owner |
| Cursor 原子性 | `PREPARED` 不推进 offset；仅 `COMMITTED` 前移 | `elastic_data_cursor_ledger.py` 回归 + ledger 审阅 | 训练 owner |
| 2:1 语义 | rank 0=2、rank 1=1，`global_samples=6`，loss scale 正确 | full-state / reference test | 训练 owner |
| Plan binding | topology/work manifest digest 都验证；旧 epoch 拒绝 | topology smoke + negative tests | 训练 owner |
| Checkpoint | model/EMA/optimizer/scheduler/RNG/precision state/ledger hash 都存在 | checkpoint manifest/readback | 平台 owner |

### B. NCCL、弹性与基础设施

| 检查项 | 通过标准 | 证据 | Owner |
|---|---|---|---|
| 拓扑基线 | GPU/NIC/NUMA/rail inventory 可信且版本化 | topology manifest + RAS/NCCL diagnosis | 基础设施 owner |
| 通信基线 | 单/多节点 AllReduce p50/p95/p99 与批准基线匹配 | benchmark + Grafana snapshot | 基础设施 owner |
| watchdog/trace | timeout dump、trace buffer、RAS 工件可读且有容量控制 | 预生产 dry run | 平台 owner |
| Rendezvous | 固定与弹性 worker-group 重建经 L2 演练 | restart evidence bundle | 平台 owner |
| 故障预算 | checkpoint cadence、max restarts、检测/恢复窗口已批准 | experiment manifest | 服务 owner |

### C. 混合精度与状态恢复

| 检查项 | BF16 | FP16 | FP8 |
|---|---|---|---|
| precision contract 固定 | autocast dtype/backend | + enabled scaler 参数 | + engine/version/recipe/format |
| state fingerprint | model/EMA/AdamW/RNG exact | + GradScaler exact | + scale/AMAX/`_extra_state` exact |
| fixed-world restore | 通过 | 通过，包含 overflow 路径 | 通过，包含 metadata key 负例 |
| world-change resume | 新 cursor/plan、shadow 指标通过 | 同左，scale 连续 | 同左，engine 版本严格相同 |
| non-finite 处理 | 明确 NaN policy | `PRECISION_STATE_COMMIT` 语义验证 | AMAX/scale 更新语义验证 |

### D. 观测、告警与人类处置

| 检查项 | 通过标准 |
|---|---|
| 指标基数 | 无 work ID/hash/raw error/个人标识 label；series budget 经审阅 |
| Dashboard | 2:1、cursor、checkpoint、NCCL、Rendezvous、precision、chaos 面板可用 |
| 告警 | alert rule 在预生产触发、去重、路由、静默和恢复通知均演练 |
| Runbook | 每条 critical alert 有 owner、stop condition、证据清单和人工恢复步骤 |
| 自动化边界 | 告警不会直接执行网络操作、生产发布或 cursor 变更 |

### E. 发布门控

| 检查项 | 通过标准 |
|---|---|
| Shadow | 候选模型在影子模式无未解释退化；RCA 工件完整 |
| Canary | 现有 Canary gate 满足安全/性能阈值且有回滚 checkpoint |
| HITL | 隔离难例、patch、审核责任和数据血缘完整 |
| 机器人安全 | 训练/监控/网络演练路径与 `SafetyKernel`、LOCAL_HOLD、LLM supervision 独立 |
| 签字 | 训练、平台/基础设施、安全/发布各自批准；不以单一人员替代 |

## 5. 故障演练 SOP

### 5.1 角色与通信

每轮需至少三类责任：**演练负责人**定义假设/范围；**训练负责人**确认 cursor/checkpoint/模型状态；**基础设施负责人**确认专用资源和受限 executor。对于 L3+ 或任何生产候选环境，另需发布/安全值班确认。没有相应 owner 的演练不能执行。

| 角色 | 允许决策 | 不允许决策 |
|---|---|---|
| 训练负责人 | freeze plan、验证 checkpoint/precision/cursor、停止训练 | 变更网络或绕过 guard |
| 基础设施负责人 | 在专用资源执行已批准 fault profile、确认 rollback | 修改训练账本或发布模型 |
| 演练负责人 | 推进/终止演练、组织证据包 | 单方面降低安全阈值 |
| 发布/安全值班 | 判断是否进入 shadow/canary/rollback | 以监控“绿色”替代训练/基础设施证据 |

### 5.2 每轮标准步骤

**步骤 0：登记。** 创建 experiment manifest，固定演练 ID、环境、批准的 L0—L5 层级、目标 group/rail、fault profile、TTL、重启预算、最新 committed checkpoint hash、dataset/manifest digest、topology epoch、停止条件与两位审批签名。不要在未完成登记后开始。

**步骤 1：Preflight。** 确认节点池是 dedicated、non-production、non-shared-fabric、无机器人控制；检查 GPU ECC/Xid、温度、NIC/RDMA/交换机健康、对象存储可写、Prometheus scrape、Grafana 面板、NCCL RAS/trace 和当前 checkpoint readback。运行 L0 baseline：2:1 full-state、cursor smoke、precision restore、metrics exporter 均必须通过。

**步骤 2：Arm。** 将训练置于已知的 update boundary；写入安全 marker，记录 current cursor/plan/topology/work digest；开启临时 trace。`RailChaosGuard` 先执行 `DRY_RUN`，必须验证所有 guard；被拒绝意味着安全成功，不能绕过。

**步骤 3：Inject。** 对 L1/L2 使用受控逻辑 failpoint 或 worker 退出。对 L3/L4，仅由基础设施负责人通过受限 executor 在批准 TTL 内执行预注册 profile。训练代码不得包含网络命令。每次仅注入一个变量；若时间、范围或错误预算超限立即停止。

**步骤 4：Observe。** 观察 plan consensus、update outcome、cursor reservation、checkpoint age、NCCL p95、watchdog、RAS、restart count、precision overflow、state alignment、GPU/NIC 健康。关键问题是：update 是否 `UNCOMMITTED`？old plan 是否失效？证据是否可读？不要在卡顿时仅提高 timeout。

**步骤 5：Rollback。** 不论观测成功、失败或脚本异常，guard 的 `finally` 都请求 executor rollback，且必须获得 `ROLLED_BACK` receipt。回滚不确认、NCCL heartbeat 无法收敛、GPU/NIC 健康变红、trace 不可写或非目标资源受到影响时，转人工事件处置，禁止自动重试。

**步骤 6：Recover。** `torchrun` 形成新 worker group 后，从最新 `COMMITTED` checkpoint load；验证 precision fingerprints；使旧 reservation `ABORTED`；重新采集 topology、重新构建 work manifest/plan；确认旧 epoch/digest 不会被接受。固定 world-size 做全状态参考；变化 world-size 做 cursor/plan/precision/影子 update 合同验证。

**步骤 7：Close。** 归档 experiment manifest、approval、fault/rollback receipt、ledger 前后记录、checkpoint hash、metrics/Grafana snapshot、RAS/trace、GPU/NIC health、测试结论、未解决项和签字。恢复训练不等于进入生产发布；按现有 shadow/canary/HITL gate 单独评审。

### 5.3 强制停止条件

出现任一条件立即停止并升级：生产/共享 fabric 识别失败；目标超出 allowlist；第二个活跃故障；GPU Xid/ECC/热告警；rollback receipt 缺失；checkpoint hash 或全状态 fingerprint 不匹配；cursor 被错误推进；old plan 在新 epoch 被接受；metrics/log/trace 无法保留；或者人员安全/机器人控制路径受影响。

## 6. 交付文件与验证

| 文件 | 用途 | 已验证内容 |
|---|---|---|
| `distributed_training_observability.py` | 低基数 Prometheus metrics facade | Python smoke test 通过；导出 2:1/cursor/rebuild/precision/chaos 指标 |
| `monitoring/prometheus_rules.yml` | recording rules 与 9 条告警 | YAML 结构人工审阅；需在目标 Prometheus 用 `promtool check rules` 再验收 |
| `monitoring/grafana_acvjepa_elastic_dashboard.json` | 可导入 Grafana dashboard | JSON 解析通过；需在实际 datasource 中检查 PromQL、权限和阈值 |
| `mixed_precision_elastic_recovery.py` | BF16/FP16/FP8 committed-state exact restore contract | CPU smoke test 通过，含 AdamW/GradScaler/FP8 metadata |
| `监控混合精度恢复与生产演练_安全边界.md` | 监控/恢复/演练权限边界 | 文档交付 |

## 参考资料

[1] [Prometheus：Metric and Label Naming](https://prometheus.io/docs/practices/naming/)。

[2] [Grafana：Prometheus Alerting](https://grafana.com/docs/grafana/latest/datasources/prometheus/alerting/)。

[3] [PyTorch 2.13：Automatic Mixed Precision (`torch.amp`)](https://docs.pytorch.org/docs/2.13/amp.html)。

[4] [NVIDIA Transformer Engine：FP8 Checkpoint Compatibility FAQ](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/faq.html)。

[5] [PyTorch 2.13：torchrun（Elastic Launch）](https://docs.pytorch.org/docs/2.13/elastic/run.html)。

[6] [NVIDIA NCCL：Communicator、错误处理与 Fault Tolerance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)。
