# Failpoint 实时观测与 CI/CD 混沌演练方案

**作者：Manus AI**  
**日期：2026-08-15**  
**系统范围：** 2:1 异构微批 AC-VJEPA、全局 cursor、Rendezvous 重建、三层 verified checkpoint cache 与五类离线 failpoint。  
**安全范围：** CI 仅运行逻辑 failpoint、cache 模拟、指标/配置校验和工件归档。它不执行真实 NCCL、节点抢占、RDMA/rail 分区、对象存储、KV、模型发布或部署动作。

## 1. 监控目标：实时量化“可信恢复”而非仅记录异常

恢复的核心 SLO 应是 `RTO_trusted`：从 fault trigger 开始，到新 worker group 已完成 committed checkpoint 读取、precision/optimizer/RNG 校验、cursor 回放、topology/plan digest 共识以及 shadow warm-up 后，允许创建下一份 `UpdatePlan` 的时间。进程重新出现或 retry counter 增加不是完成恢复的充分证据。

Prometheus 的 histogram 可在服务端按时间窗口聚合并计算分位数，因此适合按 fault class 汇总恢复时延；相比之下 summary 的预计算分位数不可跨 worker 聚合。[1] 训练/缓存作为 offline processing，应该分别追踪每一阶段输入、进行中、输出和耗时；cache 至少要追踪 query、hit、错误和前端服务时延。[2]

| 观测问题 | 低基数指标 | 主要用途 |
|---|---|---|
| 哪类 fault 发生、是否结束？ | `acvjepa_training_failpoint_triggers_total{fault_class,outcome}`、`..._failpoint_active` | rate、故障频率、active recovery |
| 恢复慢在哪个阶段？ | `acvjepa_training_failpoint_stage_duration_seconds{fault_class,phase}` | p50/p95、阶段瓶颈分解 |
| cache 是否真的减轻 durable 压力？ | `..._checkpoint_cache_fetches_total`、`..._bytes_total`、`..._fetch_duration_seconds` | tier/component/outcome hit/fallback/失败 |
| cache 是否污染恢复？ | `outcome="integrity_failed"` rate | 立即 freeze、回退 durable、取证 |
| plan/cursor/state 是否仍安全？ | 既有 plan digest、cursor reservation、full-state alignment、precision state 指标 | 恢复完成的 hard gate |

所有指标仅使用以下稳定标签：`cluster`、`job`、`environment`、`fault_class`、`phase`、`cache_tier`、`component_class`、`outcome`。**禁止**以 attempt ID、work ID、checkpoint hash、GPU UUID、rail ID、错误正文或用户/审批人作为 Prometheus label。Prometheus 文档警告每个 labelset 都会增加资源成本，且潜在高基数维度应该转移到日志/工件系统。[2]

## 2. 五类 Failpoint 的分段计时合同

### 2.1 通用阶段模型

真实训练控制器应在同一个协调器进程或同一 recovery trace 内用 `time.monotonic()` 写下本地 marker；不使用跨节点 wall-clock 相减。跨节点只传递受限的 trace ID 到 ledger/audit log，用于关联，而不是进入 metric label。

```text
T0 trigger
 ├─ trigger_to_detect
T1 detect
 ├─ detect_to_freeze
T2 new-plan freeze
 ├─ freeze_to_rendezvous
T3 new group / new topology epoch
 ├─ rendezvous_to_cursor_replay
T4 committed cursor and work window replayed
 ├─ cursor_replay_to_checkpoint_load
T5 verified checkpoint bytes available
 ├─ checkpoint_load_to_state_verify
T6 model/EMA/AdamW/scaler/FP8/RNG exact
 ├─ state_verify_to_training_ready
T7 new plan digest consensus + shadow warm-up complete
```

端到端 span 为 `trigger_to_training_ready = T7 - T0`。每阶段都可以独立 `observe()`；最终 span 用于跨 fault class 的 p95/RTO 预算。`record_failpoint()` 已对白名单 `fault_class` 与 `phase` 验证，避免将动态字符串写成无界 label。

```python
start = time.monotonic()
metrics.record_failpoint(
    fault_class="node_loss_final_allreduce",
    phase="trigger_to_detect",
    duration_seconds=time.monotonic() - start,
    active=True,
)
# 执行仅限已批准的恢复控制流程；任何未确认 update 都不 commit。
metrics.record_failpoint(
    fault_class="node_loss_final_allreduce",
    phase="trigger_to_training_ready",
    duration_seconds=time.monotonic() - start,
    active=False,
    outcome="passed",
)
```

| Fault class | 触发开始 | 最重要阶段 | TrainingReady 前的额外 hard gate |
|---|---|---|---|
| `node_loss_final_allreduce` | final synchronized backward/watchdog marker | `freeze_to_rendezvous`、`rendezvous_to_cursor_replay` | 旧 2:1 attempt 已 abort；`{w0,w1,w2}` 重读；新 checkpoint 才可推进 cursor |
| `network_partition_after_prepare` | `PREPARED` 后 data/control-plane 失败 marker | `detect_to_freeze`、`freeze_to_rendezvous` | reservation abort；旧 plan 不能重试/commit；alert 只 freeze/取证/通知 |
| `stale_plan_after_rendezvous` | new epoch 已形成而旧 attempt 请求 commit | `rendezvous_to_cursor_replay` | stale identity/attempt 被 ledger fencing；cursor 不动 |
| `plan_topology_mismatch` | plan/topology digest 不一致 | `trigger_to_detect` | prepare 阶段 fail closed；不得创建第二 reservation 或进入 backward |
| `cache_corruption_during_restore` | cache length/hash/precision contract 失败 | `checkpoint_load_to_state_verify` | evict cache；durable fallback；pointer revision 不变；full state exact |

`run_failpoint_observability_drill.py` 在 CI 中为五类 fault 各执行一次 end-to-end logical span，并生成 `artifacts/failpoint_metrics.prom`。这证明指标 schema、label contract 与离线恢复断言能够共同运行；真实集群需要在上述每个 recovery boundary 调用相同的 `record_failpoint()`，才能得到真实 p95。

## 3. 缓存命中率、字节命中率与完整性追踪

### 3.1 原始指标

| 指标 | 标签 | 语义 |
|---|---|---|
| `acvjepa_training_checkpoint_cache_fetches_total` | `cache_tier=node_local|rdma|durable`、`component_class`、`outcome` | 一次 verified fetch 的计数 |
| `acvjepa_training_checkpoint_cache_bytes_total` | 同上 | verified/attempted byte 量，支持字节加权分析 |
| `acvjepa_training_checkpoint_cache_fetch_duration_seconds` | 同上 | 每 tier/component/outcome 的 load 分布 |

`outcome` 仅允许 `hit`、`miss`、`fallback`、`negative_hit`、`integrity_failed`、`rejected`。cache 的 `hit` 只有在 content hash、字节数、precision contract 和 committed manifest 验证成功后才可计数；`integrity_failed` 不可被当作 miss 处理。

### 3.2 核心 PromQL

```promql
# 按 cache tier 与 component 计算 5 分钟 request hit ratio。
job:acvjepa_training_checkpoint_cache_hit_ratio:5m

# 直接写法：
sum by (cluster, job, environment, cache_tier, component_class) (
  rate(acvjepa_training_checkpoint_cache_fetches_total{outcome="hit"}[5m])
)
/
clamp_min(
  sum by (cluster, job, environment, cache_tier, component_class) (
    rate(acvjepa_training_checkpoint_cache_fetches_total{outcome=~"hit|miss|fallback"}[5m])
  ),
  0.001
)

# 按 byte 而非 request 计算命中率，更适合大 optimizer shard：
sum by (cluster, job, environment, cache_tier, component_class) (
  rate(acvjepa_training_checkpoint_cache_bytes_total{outcome="hit"}[5m])
)
/
clamp_min(
  sum by (cluster, job, environment, cache_tier, component_class) (
    rate(acvjepa_training_checkpoint_cache_bytes_total{outcome=~"hit|miss|fallback"}[5m])
  ),
  1
)

# 任意 tier/component 的完整性失败速率：
sum by (cluster, job, environment, cache_tier, component_class) (
  rate(acvjepa_training_checkpoint_cache_fetches_total{outcome="integrity_failed"}[5m])
)

# RDMA optimizer shard fetch p95：
histogram_quantile(
  0.95,
  sum by (le, cluster, job, environment) (
    rate(acvjepa_training_checkpoint_cache_fetch_duration_seconds_bucket{
      cache_tier="rdma", component_class="optimizer", outcome="hit"
    }[5m])
  )
)
```

request hit ratio 反映调度有效性；byte hit ratio 反映是否真正减轻 durable I/O。两者都要看：大量小 metadata hit 可能掩盖大 optimizer shard 仍持续 fallback 的事实。

## 4. Grafana 与告警

现有仪表盘扩展至 **14 个 panel**。新增的第 13 个 panel 按 `fault_class` 展示 `trigger_to_training_ready` p95 及通过次数；第 14 个 panel 按 `cache_tier/component_class` 展示 verified cache hit ratio 和 integrity failure。相关 recording rules 已写入 `monitoring/prometheus_rules.yml`：

```promql
# 30 分钟、按 failpoint 类别的 end-to-end p95。
histogram_quantile(
  0.95,
  sum by (le, cluster, job, environment, fault_class) (
    rate(acvjepa_training_failpoint_stage_duration_seconds_bucket{
      phase="trigger_to_training_ready"
    }[30m])
  )
)
```

新增两条仅观测/冻结型告警：

| 告警 | 条件 | 允许响应 | 禁止响应 |
|---|---|---|---|
| `ACVJEPAFailpointRecoveryTailHigh` | fault recovery p95 超过暂定 300s（阈值需用集群 baseline 校准） | freeze new plan、复核 RTO 预算、取证/升级 | 自动调大 retry、改变节点/网络/rail |
| `ACVJEPACacheIntegrityFailure` | verified cache integrity failure rate > 0 | freeze、隔离 suspect cache tier、durable restore 验证 | 接受错误 shard、覆盖 pointer、推进 cursor |

## 5. 自动化混沌演练执行器

命令：

```bash
cd /home/ubuntu/lecun_analysis
python3 run_failpoint_observability_drill.py \
  --report artifacts/failpoint_drill_report.json \
  --metrics artifacts/failpoint_metrics.prom
```

执行器串行运行五类 logical scenario，每一类都单独创建 experiment ID 和临时资源；输出经过脱敏的报告与 Prometheus exposition。报告包含 fault class、成功状态、单调时钟 end-to-end duration 和 assertion 数；不包含 work ID、checkpoint hash、trace 原文或真实基础设施信息。

当前本地 smoke test 结果为：5 个 failpoint 完成、cache cold-key 演练产生 1 次 durable fallback 和 8 次 node-local hit，并生成 `failpoint_metrics.prom`。这些数字来自离线模拟器，不应被作为 production cache 容量或 RTO 基准。

## 6. CI/CD 流水线方案

### 6.1 流水线分层

| 层级 | 触发条件 | 执行内容 | 通过条件 | 产物 |
|---|---|---|---|---|
| PR Contract | PR 修改 Python/monitoring/workflow | compile、五类独立 unittest、cache load-shedding、aggregate chaos、observability drill、Prometheus/Grafana 校验 | 任一断言/配置失败即失败 | text logs、JSON report、Prometheus exposition |
| Main Branch Regression | 合并到 `main` | 与 PR 相同的离线回归 | 无 schema/semantic drift | 同上，作为基线 |
| 受保护的集群演练 | 人工审批的隔离环境任务 | 另行由平台 executor 发起真实 rendezvous/RDMA/NCCL 演练 | 真实 RTO、state/cursor/plan gate、rollback evidence | 集群 trace/RAS/ledger/metric snapshot |

**PR 与 Main 流水线不能持有任何集群、网络、KV、对象存储、发布或部署凭据。** 真正的基础设施演练必须在独立受保护环境中审批运行，其结果可被引用但不应通过 PR 中不可信代码自动解析为部署动作。

### 6.2 已交付 GitHub Actions 配置

`.github/workflows/failpoint-chaos-ci.yml` 采用以下安全属性：

1. 仅 `pull_request`、`push main` 和手动 `workflow_dispatch`；不使用 `pull_request_target`。
2. `permissions: contents: read`，没有 secrets、packages、deployments、id-token 或 write 权限。
3. 通过固定完整 SHA 使用 checkout/setup/upload action；安全准则建议以完整 commit SHA 固定 action，使用最小权限 token，并谨慎对待不可信 PR/工件。[3]
4. 超时为 10 分钟，`concurrency` 取消同 ref 的旧任务，避免重复 CI 雪崩。
5. 所有产物写入 `artifacts/`，使用唯一 run/run-attempt 名称上传；artifact 未找到只 warning，默认不含隐藏文件，保留 14 天。`upload-artifact` 支持 artifact digest、唯一命名和保留期，且工件本身仍应当作不可信输入处理。[4]

工作流命令顺序：

```text
py_compile
  → five-failpoint unittest
  → offline cache load-shedding
  → aggregate chaos framework
  → failpoint observability drill
  → Prometheus/Grafana config validator
  → upload sanitized artifacts (always)
```

`validate_failpoint_ci_config.py` 还会静态校验：workflows 只有 `contents: read`、action 均固定 SHA、无 `pull_request_target`、无 secrets、无 `kubectl/iptables/tc/ssh/RDMA/torchrun/NCCL` 命令、且 artifact step 满足目录与 retention contract。

## 7. 生产接线与验收清单

生产训练器启动时，将 `DistributedTrainingMetrics.registry` 注册到受认证的 `/metrics` scrape endpoint；注册本身不改变训练状态。建议从当前已有的 exporter wiring 开始，而不是让 CI 进程临时启动 endpoint。

| 验收项 | 必须满足 |
|---|---|
| 低基数 | metric labels 不包含 attempt/work/hash/GPU UUID/错误正文；以 audit artifact 承载详细关联 |
| 指标完整性 | 五类 fault 至少各有 trigger、end-to-end recovery span、terminal outcome；cache 各 tier/component 有 hit/fallback/integrity 计数 |
| 时钟正确性 | 采用同进程单调 span；跨节点只关联 trace ID，不跨 wall clock 计算 duration |
| 仪表盘 | p95、通过率、active recovery、cache hit/byte hit、integrity fail、cursor/plan/state gate 同屏可查 |
| 告警边界 | 告警仅产生 freeze/取证/通知意图；无基础设施 mutation credential |
| CI 安全 | 离线 workflow 仅 contents read；真实演练单独审批、独立身份/runner、无 PR 权限继承 |
| 真实集群验收 | 用独立真实 NCCL/RDMA/KV 容量演练校准 300s 暂定阈值、histogram bucket 与 token/wave 上限 |

## 参考资料

[1] [Prometheus：Histograms and Summaries](https://prometheus.io/docs/practices/histograms/)。

[2] [Prometheus：Instrumentation](https://prometheus.io/docs/practices/instrumentation/)。

[3] [GitHub Docs：Secure Use Reference for GitHub Actions](https://docs.github.com/en/actions/reference/security/secure-use)。

[4] [GitHub Actions：upload-artifact](https://github.com/actions/upload-artifact)。
