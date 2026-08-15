# 大规模抢占下的快速 Rendezvous 重建与安全告警联动实战手册

**作者：Manus AI**  
**日期：2026-08-15**  
**范围：** 2:1 异构微批 AC-VJEPA 的多节点 DDP/NCCL 训练、节点抢占、RDMA/rail 网络分区、`torchrun` Rendezvous 重建、cursor/checkpoint 恢复与 Prometheus/Grafana 告警联动。  
**核心边界：** 本手册中的自动化只允许冻结新 UpdatePlan、标记作业为 `SUSPECT`、捕获现有证据和通知人工；不允许直接改变网络、节点、Rendezvous、数据 cursor 或模型发布状态。

## 1. 先纠正一个目标：缩短“可信恢复”，不是缩短进程再次启动

大规模抢占或网络分区后，最快启动的进程不一定是最快恢复的训练。`torchrun` 在 worker、agent、节点故障或成员变化后会停止现存 worker 并形成拥有新 `RANK`/`WORLD_SIZE` 的 worker group；官方明确指出 `RANK` 不稳定，成员变化时不能保留旧 rank 假设。[1] 因此本方案优化的恢复目标是：

> 从故障检测到**新 worker group 读取同一条 COMMITTED checkpoint、通过 precision/cursor/topology/plan 验证、完成影子 warm-up 并允许创建新 UpdatePlan**的时间。

这一口径避免用“进程 alive”掩盖下列风险：旧 communicator 仍被使用、PREPARED cursor 被错误提交、AdamW/GradScaler/FP8 AMAX 不完整、旧 topology digest 被接受，或重建后的 rank-local 数据位置被误当成全局恢复位置。

| 阶段 | 指标符号 | 优化目标 | 绝不能牺牲的验证 |
|---|---:|---|---|
| 检测与冻结 | `T_detect + T_freeze` | 快速发现、一次冻结、去重 | 不因瞬时噪声无限创建告警/重启 |
| 工作组收敛 | `T_rdzv` | 受控 admission、避免 thundering herd | 新 `run_id/restart/world size` 可见、旧 group 已失效 |
| checkpoint 读取 | `T_load` | 局部缓存、并发限流、按 hash 去重 | 只读 COMMITTED artifact，哈希正确 |
| 数值状态验证 | `T_state` | 并行 hash / shard read / metadata 检查 | model、EMA、AdamW、RNG、scaler/FP8 metadata 全通过 |
| 拓扑与计划 | `T_topology + T_plan` | manifest 缓存、成本分桶、一次广播 | 新 topology/work manifest/UpdatePlan digest 共识 |
| 影子 warm-up | `T_shadow` | 小窗口、只读/不提交 | 禁止将 warm-up 当作 optimizer/cursor commit |

定义端到端恢复时间：

```text
RTO_trusted = T_detect + T_freeze + T_rdzv + max(T_load, T_topology) + T_state + T_plan + T_shadow
```

其中 `max(T_load, T_topology)` 应并行执行；但 `T_state`、`T_plan` 与任何 optimizer update 必须串行守住 commit 边界。

## 2. 快速重建的关键路径优化

### 2.1 先冻结，再收敛，再恢复

故障第一反应不是增加 `max_restarts` 或延长 NCCL timeout，而是在稳定的 job key 上写入幂等 `FROZEN_FOR_RECOVERY` 状态。该状态拒绝新的 `UpdatePlan`，但允许收集 trace/RAS、等待未完成进程退出和读 checkpoint。冻结键为 `(cluster, job, environment)`，而不是不稳定的 rank 或 node ID。

随后进入新 `torchrun` group；旧 group 的所有 in-flight gradient、DDP bucket、未提交 optimizer/EMA、GradScaler、FP8 scale/AMAX 和 cursor reservation 都视作无效。PyTorch 文档建议将 `load_checkpoint → initialize → train` 作为弹性脚本结构，并强调 surviving workers 在失败/成员变化后也会终止；checkpoint 频率应与可接受失工作量匹配。[1]

### 2.2 把大规模重启从“齐步”变成“分波”

大规模抢占最常见的次生事故是所有幸存/替换节点同时对 Rendezvous、对象存储、metadata 服务和 Prometheus endpoint 发起请求。`rapid_recovery_alert_drill.py` 的 `RecoveryWavePlanner` 将 node admission 切分为受限 wave：

```text
backoff(restart) = min(max_backoff, base_backoff × 2^restart_count)
wave(node)       = sorted_node_index // max_nodes_per_wave
join_after(node) = wave × backoff + deterministic_jitter(run_id, restart, node)
load_slot(node)  = sorted_node_index mod checkpoint_load_concurrency
```

**确定性抖动**由 `(run_id, restart_count, node_id)` 派生，既能打散连接风暴，又能在故障复盘中重现 admission 顺序。wave 边界不是执行许可：节点只有在通过 checkpoint hash、precision fingerprint、global cursor、topology epoch 与新 plan digest 后才能成为可用 worker。

| 参数 | 初始建议 | 依据与调节方式 |
|---|---:|---|
| `max_nodes_per_wave` | 3–4 | 从 Rendezvous/对象存储 p95、每节点 GPU 数及许可 recovery RTO 反推；不是硬编码常量 |
| `base_backoff_seconds` | 1–2s | 应高于瞬时 TCP/NCCL error 抖动，但低于批准的恢复预算 |
| `max_backoff_seconds` | 30–60s | 防止反复失败产生高速重启风暴 |
| `jitter_ratio` | 10–20% | 打散波内请求；必须确定性可审计 |
| `checkpoint_load_concurrency` | 小于等于 wave 容量 | 从存储读吞吐、单 checkpoint 大小和网络容量压测得出 |

脚本中 8 个节点、wave=3、`restart_count=2` 的示例产生 3 个波次和 2 个 checkpoint load slot。它只生成计划和证据，不启动进程或触碰外部网络。

### 2.3 把 checkpoint 拆成“可并行读、原子认定”

为压缩 `T_load + T_state`，建议 checkpoint manifest 将下列对象按完整性单元列出：学生模型、EMA、AdamW state、scheduler、precision state、RNG、cursor ledger pointer、数据/拓扑/plan fingerprints。每个对象有内容 hash，顶层 manifest 再有整体 hash。节点可并行预取不同对象或从受控本地只读缓存命中；但**只有顶层 manifest 与所有必要对象都验证通过时**才可形成 `COMMITTED` 恢复候选。

对于 2:1，不能只缓存 rank 0 的局部 state；完整复制的 DDP/AdamW replica 都要加载相同 committed state。若使用 optimizer sharding，则 checkpoint manifest 必须包含 shard ownership/重建规则，不能套用本代码的“每 rank 全量 AdamW”假设。

### 2.4 为恢复建立两道 ready gate

`ProcessReady` 表示新进程已加入并能读取静态配置；`TrainingReady` 表示该进程可进入下一次计划。两者必须分开。

```text
ProcessReady
  ├─ 新 TORCHELASTIC_RUN_ID / RESTART_COUNT / WORLD_SIZE 已读取
  ├─ 新 NCCL ProcessGroup 已创建，旧 group 不可使用
  └─ checkpoint bytes 已取得

TrainingReady
  ├─ checkpoint / model / EMA / AdamW / RNG / AMP/FP8 state fingerprint=exact
  ├─ latest COMMITTED cursor 已加载；所有旧 PREPARED attempt 已 ABORTED
  ├─ new topology manifest 和 work manifest digest 已验证
  ├─ new UpdatePlan 已广播、所有 rank digest consensus=1
  └─ shadow warm-up 已完成，未产生 optimizer/cursor commit
```

任何 `TrainingReady` 条件为假都保持 freeze。这样可缩短进程/IO 的并行阶段，而不会让不安全的 worker 提前进行同步 backward。

### 2.5 快速数据恢复：只从全局 cursor 重分片

Rendezvous 形成新 group 后，调用 `ElasticCursorLedger.recover_after_rendezvous()`：读取最后 `COMMITTED` `next_offset=o`，将所有未提交 reservation 标为 `ABORTED`，按新 group 的微批 capacity peek `manifest[o : o + ΣK'_r]`，然后用新 topology planner 分配 window。旧 2:1 `rank0=[w0,w1], rank1=[w2]` 可重排为新 `rank0=[w0], rank1=[w1,w2]`；允许变的是位置，不允许变的是连续全局 range、provenance 和 cursor 起点。

为了缩短 `T_plan`，可缓存 immutable work manifest 和每个 cost bucket 的摘要；但必须每次绑定当前 `dataset_commit`、work manifest digest、topology epoch、elastic identity。不能把缓存的 rank assignment 当作新 group 的计划。

### 2.6 以预算而非固定 timeout 控制恢复

下面是推荐的 recovery SLO 分解，数字必须通过目标集群基线压测后替换，不是通用阈值。

| 指标 | 公式/来源 | 预警意图 |
|---|---|---|
| `T_freeze` | Alert firing → ledger freeze audit event | 避免同一故障重复 arm plan |
| `T_rdzv` | new restart observation → all ProcessReady | 是否出现 admission/Rendezvous 风暴 |
| `T_load` | checkpoint read start → verified manifest | 对象存储/本地缓存/并发控制是否饱和 |
| `T_state` | load complete → all required state exact | 是否出现 GradScaler/FP8/optimizer 恢复缺失 |
| `T_plan` | topology ready → digest consensus=1 | 通信控制面或 manifest 代价过高 |
| `RTO_trusted` | 以上关键路径和影子 warm-up | 真实恢复 SLO，而非仅进程 restart |

将 alert 用 `for`/`keep_firing_for` 或 Alertmanager grouping 做适度去抖与限频；Prometheus 定义的 `for` 让条件持续一段时间才 firing，`keep_firing_for` 可减少间歇恢复导致的抖动。[2] 对于已知节点批量抢占这一类明确事件，可用协调器/故障信号提前 freeze，但仍不能绕过数据和状态验证。

## 3. Prometheus 与 Grafana 核心查询

交付脚本 `rapid_recovery_alert_drill.py` 内置 `PROMQL` 查询目录；下表列出最重要的面板语句。所有查询沿用 `$cluster`、`$job`、`$environment` 变量，并且只依赖低基数 metric。

| 面板/告警意图 | PromQL |
|---|---|
| 新计划摘要共识 | `min(acvjepa_training_plan_digest_consensus{cluster="$cluster",job="$job",environment="$environment"})` |
| 2:1 槽位 | `acvjepa_training_plan_micro_batches{cluster="$cluster",job="$job",environment="$environment"}` |
| AllReduce p95 | `job:acvjepa_training_allreduce_p95_seconds:5m{cluster="$cluster",job="$job",environment="$environment"}` |
| cursor 卡住 | `acvjepa_training_cursor_reservations{cluster="$cluster",job="$job",environment="$environment"} > 0 and on(cluster,job,environment) increase(acvjepa_training_checkpoint_commits_total[10m]) == 0` |
| 重建风暴 | `sum by(cluster,job,environment) (increase(acvjepa_training_rendezvous_rebuilds_total[15m]))` |
| recovery p95 | `job:acvjepa_training_recovery_p95_seconds:30m{cluster="$cluster",job="$job",environment="$environment"}` |
| model/EMA/AdamW/RNG 对齐 | `min(acvjepa_training_state_alignment_verified{cluster="$cluster",job="$job",environment="$environment",component=~"model|ema|optimizer|rng"})` |
| FP8 metadata | `acvjepa_training_fp8_metadata_verified{cluster="$cluster",job="$job",environment="$environment"}` |
| abort ratio | `job:acvjepa_training_update_abort_ratio:5m{cluster="$cluster",job="$job",environment="$environment"}` |

频繁使用的 p95、recovery、abort ratio 应采用 recording rule 预计算；Prometheus 官方说明 recording rules 用于预计算频繁或昂贵查询，减少 dashboard 重复计算。[3]

## 4. 安全告警联动：从 Alertmanager 到受限控制面

Prometheus alert rules 决定“哪些当前状态应触发告警”，而 Alertmanager 负责分组、限频、静默和 receiver 路由。[2] [4] 本系统不把 webhook 当作集群控制通道，而是使用如下最小化模型：

```text
Prometheus rule
  → Alertmanager grouping / dedupe / routing
  → authenticated webhook adapter
  → SafeAlertControlPlane
       ├─ freeze_new_plans       (durable control flag)
       ├─ mark_suspect           (durable job status)
       ├─ capture_existing_evidence request
       └─ notify_owner / escalation intent
  → human runbook + existing RailChaosGuard / cursor / release gates
```

`rapid_recovery_alert_drill.py` 解析 Alertmanager 风格的 payload，但它没有 HTTP server、外部连接或 mutation credential。该解析器只接受 `alertname`、`cluster`、`job`、`environment`、`severity`、`status`、`fingerprint`。未在 allowlist 的 alert、非 allowlisted environment、缺失字段或重复 fingerprint 均 fail closed/幂等处理。

| Alert | 允许的联动 | 禁止的联动 |
|---|---|---|
| `ACVJEPARendezvousRebuildStorm` | freeze、SUSPECT、capture、notify | 自动追加 `max_restarts`、重启节点、改变 RDMA/rail |
| `ACVJEPAExactStateRestoreFailed` | freeze、SUSPECT、capture、notify | 使用部分 optimizer/scaler state 重试 |
| `ACVJEPAFP8MetadataRestoreFailed` | freeze、SUSPECT、capture、notify | 默认初始化 AMAX 后继续训练 |
| `ACVJEPAUpdatePlanConsensusLost` | freeze、capture、notify | 沿用旧 UpdatePlan 或单 rank optimizer step |
| `ACVJEPAAllReduceTailLatencyHigh` | SUSPECT、capture、notify | 自动故障注入、盲目扩大 NCCL timeout |

生产 webhook adapter 应在服务器端追加 mTLS/工作负载身份校验、请求大小/频率限制、job/environment 静态 allowlist、fingerprint idempotency 的 durable store、审计 append-only log 和无出站网络/集群管理凭据的最小权限容器。脚本中的 `SafeAlertControlPlane` 只证明逻辑，不替代这些部署控制。

## 5. 演练脚本与运行结果

```bash
cd /home/ubuntu/lecun_analysis
python3 rapid_recovery_alert_drill.py
```

该命令执行的全是本地、无网络的模拟：发出 2:1 plan/cursor/rebuild Prometheus 指标，生成 8 节点的分波 admission，解析一个 Alertmanager 风格 `ACVJEPARendezvousRebuildStorm` payload，创建 freeze/SUSPECT/capture/notify 四条审计意图，再用相同 fingerprint 重投递验证幂等性。预期关键结果如下：

| 字段 | 预期 |
|---|---|
| `smoke_test` | `passed` |
| `new_plans_frozen` | `true` |
| `audit_events` | `4` |
| `metrics_contains_plan` | `true` |
| `admissions` | 8 个节点、最多 3 个 wave、load slot 在预设范围内 |

这不是网络故障注入器、集群调度器或 Alertmanager server。它是上线前验证“控制面不会因告警而越权”的实战 harness。

## 6. 部署选择与取舍

| 方案 | 组成 | 优点 | 局限与适用场景 |
|---|---|---|---|
| 轻量预生产演练 | 训练 worker metrics endpoint、单 Prometheus/Grafana、受限 dry-run control plane | 部署快，适合验收 2:1/ledger/alert 语义 | 不适合高可用或大量历史指标；需人工运行演练 |
| 生产级观测控制面 | 高可用 Prometheus/远端时序存储、Grafana、Alertmanager、独立审计/控制服务 | 能承受大规模重建、告警去重、长期审计与多团队路由 | 需要身份、存储、SLO、容量、灾备和变更管理；控制服务必须最小权限 |

两种方案都应保留“告警只冻结和升级”的安全边界。生产级观测不等于自动故障修复权限。

## 7. 验收标准

上线前需至少满足以下可度量标准；阈值由集群容量基线与业务 RTO 决定。

1. 在固定规模和变规模两类演练中，所有旧 `PREPARED` attempt 均变为 `ABORTED`，cursor 仅从最后 `COMMITTED` offset 读取，且不出现 silent skip。
2. 从新 group load 结束到 `state_alignment_verified` 全组件为 1 的时间有 p50/p95 基线；FP16 的 GradScaler 和 FP8 的 metadata 均有负例阻断测试。
3. 大规模模拟抢占时，Rendezvous 与 checkpoint request 峰值不超过批准容量，且 `RTO_trusted` p95 满足恢复预算；增加 worker 数不会造成重建风暴。
4. 对同一 Alertmanager fingerprint 重投递不产生重复 freeze/通知；未知 alert、生产外目标和缺失 labels 被拒绝。
5. `ACVJEPAExactStateRestoreFailed`、`ACVJEPAFP8MetadataRestoreFailed`、`ACVJEPAUpdatePlanConsensusLost` 的告警均能冻结新计划，但没有任何测试证明其可以重启节点、修改网络、推进 cursor 或发布模型。
6. 真实 Alertmanager、Prometheus/Grafana、Rendezvous、checkpoint store 和 NCCL/RAS 集成必须在隔离集群完成一次受控演练；本地 smoke test 不能替代该验收。

## 参考资料

[1] [PyTorch 2.13：torchrun（Elastic Launch）](https://docs.pytorch.org/docs/2.13/elastic/run.html)。

[2] [Prometheus：Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)。

[3] [Prometheus：Recording Rules](https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/)。

[4] [Prometheus Alertmanager：Configuration](https://prometheus.io/docs/alerting/latest/configuration/)。
