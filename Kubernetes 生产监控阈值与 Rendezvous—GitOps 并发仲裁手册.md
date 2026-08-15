# Kubernetes 生产监控阈值与 Rendezvous—GitOps 并发仲裁手册

**作者：Manus AI**  
**日期：2026-08-15**  
**范围：** AC-VJEPA 2:1 异构微批训练的 verified cache、五类 failpoint、Rendezvous 恢复、RecoveryDeploymentEpoch 和 GitOps 并发控制。

> **关键结论：** 不应以一个“全局正确的 500 ms / 95% hit ratio”常数管理所有集群。生产阈值必须以拓扑、checkpoint 大小、有效 world size、存储 tier、恢复频率和批准的 lost-work/RTO budget 分层校准。交付中的 `0.85` hit ratio、`5 s` durable fallback p95、`900 s` frozen/pending age 等是**保守的初始保护栏**，不是事实性性能承诺。它们的默认动作仅为冻结、取证与人工升级；不得直接改 pointer、cursor、网络、Pod、节点或发布状态。

## 1. 监控目标与指标分层

### 1.1 三层信号

| 层级 | 问题 | 代表指标 | 告警目的 | 允许自动动作 |
|---|---|---|---|---|
| **正确性阻断** | 是否可能读取/提交/部署错误状态？ | plan digest、state alignment、FP8 metadata、cache integrity、inputs binding、Git revision match、fence rejection | 立即阻止不可信恢复/部署 | freeze 新计划/新 apply、取证、通知 |
| **容量与尾延迟** | 恢复会不会形成 cache/object-store/NCCL 雪崩？ | cache hit ratio、durable fallback p95、AllReduce p95、recovery p95、rebuild rate、admission queue | 提前降载而非等故障扩大 | freeze 新 recovery admission、标记 suspect、人工调预算 |
| **流程与人工响应** | 仲裁是否卡死或 desired change 积压？ | state age、GitOps pending age、deferred ratio、chaos guard rejection | 发现长期 freeze、审批/流程瓶颈 | 工单/值班通知、工件保留 |

Prometheus 规则中的 `for` 可让条件持续一段时间才 firing，`keep_firing_for` 可减少短暂恢复造成的抖动；Alertmanager 才负责通知聚合、限频与静默。[1] 本方案将“正确性阻断”与“性能退化”分开：前者可以 `for: 0m` 或很短时间，后者使用 5–10 分钟窗口，并在低流量/idle 条件下不告警。

### 1.2 低基数指标合同

新指标均沿用 `(cluster, job, environment)`，只添加枚举型 `state`、`actor`、`reason`、`component_class`、`cache_tier`。Git commit SHA、checkpoint hash、cursor ID、work ID、fencing token 和异常正文全部保留在仲裁账本/审计日志中，**不得**写入 Prometheus label。

| 指标 | 含义 | 关键观察 |
|---|---|---|
| `acvjepa_training_recovery_deployment_state{state}` | 当前 RecoveryDeploymentEpoch 的 one-hot 状态 | 是否 `FROZEN`、是否从 `RECOVERY_READY` 转为 `DEPLOYMENT_ARMED` |
| `..._generation` | 单调 generation，不暴露 token | 是否发生新恢复 epoch |
| `..._inputs_valid` | checkpoint/cursor/precision/topology/plan/Git bind 全通过才为 1 | 任何 0 都阻断 |
| `..._git_revision_match` | 恢复绑定与当前 desired revision 是否一致 | ready/armed 时为 0 表示 drift |
| `..._fence_rejections_total{actor,reason}` | 拒绝陈旧 writer | 非零是并发边界证据，不是可忽略重试 |
| `..._gitops_sync_attempts_total{outcome}` | applied/deferred/rejected/failed | 观察被恢复门控延迟的变更 |
| `..._gitops_pending_age_seconds` | desired change 等待仲裁时间 | 长期积压需人工决定 |

## 2. 生产阈值：从初始保护栏到校准 SLO

### 2.1 建议的初始阈值

下表对应已写入 `monitoring/prometheus_rules.yml` 的第一版设置。上线前必须按实际恢复样本复核；在样本不足时宁可保持 warning/freeze，而不要为降低告警噪音而放宽正确性门。

| 信号 | 初始表达/阈值 | 持续时间 | 级别 | 解释与行动 |
|---|---:|---:|---|---|
| Cache integrity failure | `rate(integrity_failed) > 0` | 立即 | Critical | cache tier 不可信；freeze，新恢复只能走 verified durable fallback |
| Cache stampede risk | hit ratio `< 0.85` 且 verified fetch `>0.1/s` | 10 min | Warning | 先检查 planned cold start，再限 admission/single-flight/TTL，而非关闭 hash 验证 |
| Durable fallback p95 | `>5 s` | 10 min | Warning | 初始值；按 component/checkpoint size/storage baseline 校准；调查 load slots、对象存储、NUMA/rail |
| AllReduce p95 | `>0.5 s` 且 update rate `>0.01/s` | 5 min | Warning | 关联 NCCL/RAS、topology/rail、straggler；不要先调大 timeout |
| Rendezvous rebuild storm | 15 min 内 `>2` | 1 min | Critical | freeze 自动 restart/admission，保留 NCCL/PyTorch/RAS 与 lease 证据 |
| Failpoint recovery p95 | `>300 s` | 5 min | Warning | 检查阶段分解：load、cursor replay、state verify、warm-up |
| Inputs invalid | `inputs_valid == 0` | 1 min | Critical | checkpoint/cursor/precision/topology/plan/Git 任一绑定无效，阻断恢复和 deploy |
| Git revision drift | ready/armed 且 `git_revision_match == 0` | 1 min | Critical | freeze GitOps apply；创建新 epoch，而非对旧 epoch 换 revision |
| Fencing rejection | 5 min rate `>0` | 立即 | Critical | 表示有旧 worker/旧 GitOps writer 试图写入，取证后仲裁 |
| FROZEN age / pending age | `>900 s` | 10 min | Warning | 超出初始人工响应预算；必须决定 resume、supersede 或 rollback |

这些表达均被配置为具有明确 `action` label 和 runbook；action 指向“冻结、取证、升级”，不绑定具有基础设施写权限的 webhook。

### 2.2 校准方法

对每一个 `{cluster class, checkpoint-size bucket, world-size bucket, precision mode, storage tier}` 组合，收集至少一次完整的正常恢复链路：`trigger → detect → freeze → rendezvous → cursor replay → checkpoint load → state verify → TrainingReady`。如果没有足够历史样本，使用隔离预生产的批准演练补足，而不是从生产故障中学习。

建议用 28 天滚动窗口保存成功恢复样本，并对每个阶段计算 p50/p95/p99。第一个生产版本可用下列政策：将 warning 设为 `max(绝对安全下限, baseline_p95 × 1.5)`，将 critical 设为 `max(绝对硬上限, baseline_p99 × 1.5)`，再以批准的 trusted-RTO 总预算约束所有阶段之和。这个比例是工程起点，不是统计保证；容量明显不同的 rail、对象存储或 checkpoint shard 分组必须分别建基线。

对于 hit ratio，必须同时加**最小读流量条件**；训练暂停或已完成时，零读/零命中不代表 cache 退化。对于 cache cold start，应以事先标记的 recovery wave 抑制或 route warning，而不应全局 silence integrity failure。对于 FROZEN 状态，不要简单将 900 秒延长来消除告警，因为它可能代表 lease/epoch/Git drift 未被处理。

## 3. 核心 PromQL 与 Grafana 看板

### 3.1 生产排障查询

```promql
# Cache verified request hit ratio；低流量时避免单独解释
job:acvjepa_training_checkpoint_cache_hit_ratio:5m{
  cluster="$cluster", job="$job", environment="$environment"
}

# durable fallback 的 component p95
job:acvjepa_training_checkpoint_durable_fallback_p95_seconds:5m{
  cluster="$cluster", job="$job", environment="$environment"
}

# 任意 stale writer 的近 5 分钟速率
sum by (actor, reason) (
  job:acvjepa_training_recovery_deployment_fence_rejections:rate5m{
    cluster="$cluster", job="$job", environment="$environment"
  }
)

# ready/armed 阶段的 Git desired revision drift
(acvjepa_training_recovery_git_revision_match{
  cluster="$cluster", job="$job", environment="$environment"
} == 0)
and on (cluster, job, environment)
(acvjepa_training_recovery_deployment_state{
  cluster="$cluster", job="$job", environment="$environment",
  state=~"RECOVERY_READY|DEPLOYMENT_ARMED"
} == 1)

# 五类 failpoint 的端到端 trusted recovery p95
job:acvjepa_training_failpoint_recovery_p95_seconds:30m{
  cluster="$cluster", job="$job", environment="$environment"
}
```

### 3.2 Grafana 信息架构

仪表盘已由 14 个面板扩展为 **19 个面板**。原有 Plan/Cursor/Checkpoint/Precision/NCCL/Failpoint/Cache 面板保留；新增五个面板分别显示：

| 新面板 | 用途 | 值班员首先要问的问题 |
|---|---|---|
| RecoveryDeploymentEpoch State and Generation | 观察状态、generation、state age | 是否有一个明确的新 epoch，还是多个恢复控制者在争夺？ |
| Recovery and GitOps Input Binding Gate | `inputs_valid` 与 Git revision match | 是否所有恢复输入仍绑定同一 checkpoint/cursor/Git revision？ |
| Fencing Rejections and GitOps Sync Outcomes | stale writer 与 applied/deferred/rejected | 是否有晚到写者或被门控的 Git change？ |
| Cache Stampede Protection: Verified Hit Ratio | node-local/RDMA tier ratio | 这是计划 cold start 还是异常 fallback 放大？ |
| Cache Stampede Protection: Durable Fallback p95 | durable tail 与 integrity events | 存储路径是否进入长尾并压迫 recovery budget？ |

在跨 region/rail 大集群中，必须添加受控的 `cluster class`/`storage tier` dashboard split 或单独 dashboard，而不是给 Prometheus metric 再加高基数 node、work ID 或 checkpoint hash label。

## 4. Rendezvous 与 GitOps 并发：为何会出现极端不一致

Rendezvous 与 GitOps 是两个独立的控制闭环。Rendezvous 试图把训练恢复到某一已确认 checkpoint/cursor；GitOps 试图把 Git desired state 同步到 cluster live state。若二者都认为自己可写，会出现以下极端竞态：

| 竞态 | 错误后果 | 不可接受的“修复” |
|---|---|---|
| 老 worker 在新 group 已建立后完成 AllReduce/restore callback | 旧 optimizer/plan 对新 topology 提交 | “old worker 看起来成功，所以继续 step” |
| Git revision 在 `RECOVERY_READY` 与 deploy 之间变化 | 新代码/新配置绑定旧 checkpoint 或旧 topology | 将新 revision 直接 apply 到旧 recovery epoch |
| GitOps self-heal 看到临时 recovery freeze 视为 drift | 自动覆盖 interlock，启动不可信 workload | 允许 self-heal 无条件修改 training resources |
| controller leader 切换/Lease 暂时分区 | 两个 controller 都以为是 leader | 仅用 pod liveness 或 wall-clock 判断 owner |
| checkpoint 已上传但 cursor ledger 未 COMMITTED | deployment 指向不可恢复/未提交状态 | cache hit、object 存在或 Git commit 存在即视为可用 |
| Alert 与 Git commit 同时到达 | alert automation freeze 后 GitOps 又 apply | 将告警动作与 apply 视为独立幂等操作 |

Kubernetes Lease 可用于确保组件单一 leader；官方控制面也用 Lease 进行 leader election。[2] 但 Lease 只能提供“谁暂时主动”的 liveness 协调，不能使 checkpoint pointer、cursor ledger、precision metadata、topology plan 和 Git desired revision 天然成为一个原子事务。因此，Lease 是**必要但不充分**的部件。

Argo CD 自动同步会在 Git desired manifests 与 cluster live state 不一致时同步，且 CI 可仅提交 Git 而无需直接调用 Argo CD API。[3] 这降低了 CI 的集群权限，但也意味着 recovery interlock 必须成为 GitOps desired state 与 admission policy 的一部分，否则 auto-sync/self-heal 会绕过临时 freeze。

## 5. RecoveryDeploymentEpoch 仲裁协议

### 5.1 单一真相记录

交付的 `recovery_deployment_arbiter.py` 是 in-memory 参考状态机。生产应将同样的数据模型写入一个线性一致存储，使用 CAS/`resourceVersion` 或等价事务。每条记录含：

```text
generation                    # 单调递增；等同 fenced epoch，不暴露为 metric label
state                         # IDLE | RECOVERING | RECOVERY_READY | DEPLOYMENT_ARMED | FROZEN
lease_holder + lease_expiry   # 只用于当前 recovery writer liveness
checkpoint_hash               # 已 COMMITTED checkpoint
cursor_commit_id              # 对应 durable cursor ledger commit
precision_contract_hash       # BF16/FP16/FP8/optimizer/scaler/AMAX 合同
work_manifest_digest          # 数据血缘
(topology_epoch, plan_digest) # 新 process-group 和动态 UpdatePlan 绑定
git_revision                  # 选定 desired state
inputs_digest                 # 上列绑定字段 canonical hash
reason                        # 冻结原因，完整文本留审计日志
```

每个 transition 都携带 `expected_generation`。任一新 generation 出现后，旧 writer 虽然可能仍有 network response、checkpoint bytes 或 Git artifact，也只能被 fence；它不能 mark ready、arm deployment 或回填 cache。该模型比“重试一次”更重要：重试在分区中会制造第二写者，fencing 才能使旧写者失去写权限。

### 5.2 合法状态转换

```text
IDLE/FROZEN/expired lease
      │ recovery controller CAS(expected_generation)
      ▼
RECOVERING ── all state/cursor/checkpoint/precision/topology/Git bindings verified ──► RECOVERY_READY
      │                                                                         │
      │ Git revision drift / invalid input / lease ambiguity                    │ GitOps CAS(expected_generation + inputs_digest)
      ▼                                                                         ▼
   FROZEN ◄──────────────────────────────────────────────────────── DEPLOYMENT_ARMED
      │                                                               │
      └── create a *new* generation after human/ledger reconciliation ─┘
```

`RECOVERY_READY → DEPLOYMENT_ARMED` 必须同时满足：(a) state 未过期，(b) `inputs_valid=1`，(c) `git_revision_match=1`，(d) GitOps 传入的完整 binding digest 与仲裁记录相等，(e) expected generation 一致。GitOps 在没有这些条件时只能记录 `deferred`，不能 apply。

### 5.3 原子性边界：不要声称跨系统两阶段提交

对象存储、cursor SQLite/数据库、Kubernetes API 和 Git repository 通常不存在一个全局 ACID transaction。正确做法不是虚构“全局原子”，而是建立顺序化的 canonical commit：

1. 所有 checkpoint shards 先写为 immutable content-addressed bytes；逐 shard 验证。
2. durable checkpoint manifest、验证摘要和 cursor advance 在 ledger 的单一 commit 中确认；未 COMMITTED 的对象永远不能成为恢复输入。
3. recovery controller 从**这一条** committed record 构造 `RecoveryInputs`；其 canonical digest 写入 RecoveryDeploymentEpoch。
4. GitOps/admission 只接受包含同一 generation + inputs digest 的 manifest/artifact。
5. 若 Git revision、topology、precision 或 cursor 变化，freeze 当前 epoch，创建新 generation；不在原 generation 上修改字段。

这给训练数据提供 at-least-once work-window replay 语义，并给控制面提供“每个 generation 至多一个 accepted writer”的语义。它不是端到端 exactly-once 训练承诺，也不应被描述为此。

## 6. 极端边界的处置矩阵

| 事件 | 检测 | 仲裁结果 | 恢复条件 |
|---|---|---|---|
| 节点抢占导致 torchrun rebuild | rebuild counter、Lease 续约、Rendezvous state | 当前 `PREPARED` abort；cursor 不前进；新 generation 取得 lease | 新 topology/plan、同一 committed checkpoint/cursor、all-state verify |
| 老 rank 晚到 | expected generation 不匹配 | `stale_generation` fence rejection；Critical alert | 不重试旧 callback；用当前 generation 重新计划 |
| Git commit 在 ready 时抵达 | Git revision match=0 | state=`FROZEN`；GitOps sync=`deferred` | 对新 Git revision 重新生成 RecoveryInputs + new generation |
| GitOps auto-sync/self-heal | admission/pre-sync gate 发现非 armed generation | deny/defer，不应用 live change | manifest 绑定 current `DEPLOYMENT_ARMED` generation 与 inputs digest |
| lease 过期但旧 leader仍运行 | lease deadline + generation | 新 leader 可 CAS 接管；旧 leader publish 被 fence | current leader 验证 committed inputs 后重新 ready |
| checkpoint cache 损坏 + Git revision drift | integrity failure + Git drift alerts | cache tier suspect；epoch frozen；不改变 pointer/cursor | verified durable fallback、选择 desired Git revision、new generation |
| monitor 丢失 | `absent_over_time`/scrape target | 不自动解除 freeze | 从 ledger/GitOps 状态核对作业是否完成、停止或 exporter 故障 |

## 7. 上线前验收

1. 在每种 cluster/storage/checkpoint 组合上完成 baseline 恢复，配置阈值 ownership、评审人和有效期。
2. 验证 18 条 alerts、10 条 recording rules 和 19 个 dashboard 面板均引用低基数 exported metrics。
3. 运行 `recovery_deployment_arbiter.py`：Git drift 后 generation=2，旧 recovery/GitOps writers 均以 `stale_generation` 被 fence。
4. 在隔离 namespace 演练“Rendezvous rebuild + Git desired revision update”交错场景；验收 cursor/pointer 不前进、old generation 不能 arm、new generation 才能 apply。
5. 检查 Argo CD/等价 GitOps 配置：training application 的同步必须受 `DEPLOYMENT_ARMED` admission/pre-sync gate 约束；不要让 self-heal 覆盖 `FROZEN` interlock。
6. 验证 Alertmanager 路由只调用 freeze/evidence/notification control plane；其身份没有 Kubernetes 写、网络、node 或 pointer 直接权限。
7. 对实际生产启用前，平台、训练、安全和发布责任人共同签字，明确 rollback owner、RTO/RPO、告警值班与 GitOps override 规则。

## 参考资料

[1] [Prometheus: Alerting Rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/)。

[2] [Kubernetes: Leases](https://kubernetes.io/docs/concepts/architecture/leases/)。

[3] [Argo CD: Automated Sync Policy](https://argo-cd.readthedocs.io/en/stable/user-guide/auto_sync/)。
