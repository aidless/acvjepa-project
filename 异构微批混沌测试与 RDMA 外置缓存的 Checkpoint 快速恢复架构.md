# 异构微批混沌测试与 RDMA 外置缓存的 Checkpoint 快速恢复架构

**作者：Manus AI**  
**日期：2026-08-15**  
**适用范围：** AC-VJEPA 的 2:1 异构微批同步 DDP、`torchrun` 弹性重建、全局数据 cursor、BF16/FP16/FP8 committed-state 恢复。  
**不适用范围：** 本文不提供生产网络中断、节点终止、RDMA 设备控制、交换机变更或机器人控制命令。

## 1. 结论：把“恢复快”与“恢复正确”拆开设计

大规模节点抢占和网络分区中，恢复瓶颈通常分属三条路径：Rendezvous 控制面、checkpoint 数据面和状态验证/计划控制面。RDMA 外置缓存或分布式 KV 只能压缩第二条路径的一部分；如果它们被错误地当成提交事实来源，反而会放大 cursor、optimizer 和精度状态不一致风险。

> **唯一的恢复事实来源仍是 durable ledger 指向的 COMMITTED checkpoint manifest/hash。** 分布式 KV 仅保存很小的线性化 commit pointer、lease 和 shard availability 元数据；对象存储/并行文件系统保存 durable shard；节点本地 NVMe/RAM 和 RDMA 外置内存池只做可验证的只读缓存。

PyTorch Distributed Checkpoint（DCP）支持多 rank 并行 save/load，以及在不同保存/加载拓扑之间进行 load-time resharding；每个 rank 可只读取满足其 local shard 所需的数据。[1] 这正适合与内容寻址的并行 cache reader 组合，但前提是 checkpoint manifest、precision contract 和完整性验证不被缓存层绕过。

| 层级 | 典型介质/服务 | 保存什么 | 一致性要求 | 是否可独立恢复 |
|---|---|---|---|---|
| L0：提交控制面 | etcd 或同级强一致 KV、cursor ledger | commit pointer、manifest digest、step、lease、fencing epoch | 严格串行化/CAS | 是，指向下一层 |
| L1：耐久数据面 | 对象存储/并行文件系统 | checkpoint shard、完整 manifest、签名/哈希 | 内容耐久、不可变版本 | 是，恢复最终回退源 |
| L2：节点本地缓存 | NVMe、page cache、pinned CPU RAM | 热 shard | 可丢失、按 hash 验证 | 否 |
| L3：RDMA 外置缓存 | 专用 RDMA memory pool、内存服务器或支持 RDMA 的缓存服务 | 热 shard 的只读副本 | 可丢失、TTL/租约/内容 hash | 否 |
| L4：GPU/CPU staging | pinned RAM、GPU buffer、DCP staging | 当前 load/save 的临时缓冲 | 进程生命周期内 | 否 |

etcd 对 KV API 提供 durability 和 strict serializability，适合小元数据的 compare-and-swap；线性化也有共识开销，因此不应该承载 GB/TB 级张量。[4] GPUDirect RDMA 提供 GPU 与 NIC/存储等第三方 PCIe peer 的直接数据交换能力，但其行为受 root complex、PCIe/NUMA、IOMMU、BAR 空间、memory pin/unpin 和 driver callback 等约束。[3]

## 2. 针对 2:1 异构微批的自动化混沌测试框架

### 2.1 为什么普通 DDP chaos 测试不够

常规“kill 一个 rank 后能否重启”的测试不能证明 2:1 语义正确：rank 0 在一次 update 内可能已完成第一个 `no_sync()` 微批，而 rank 1 在最终同步 backward 前失效。若只检查进程重新出现，就可能掩盖了局部梯度残留、cursor 提前前移或已读窗口被错误跳过。

框架的测试预言机必须同时覆盖以下不变量。

| 不变量 | 2:1 的具体含义 | 失败后预期 |
|---|---|---|
| 全局数据连续性 | 本 update 的连续范围为 `{w0,w1,w2}`；旧布局可为 `[w0,w1] / [w2]` | 未提交时 cursor 仍为 `0`，重建后完整重读该集合 |
| 同步更新边界 | rank 0 第一个微批只局部累积；所有 rank 最后微批才 collective | final AllReduce 前/中故障不允许 optimizer/EMA/cursor commit |
| 弹性 fencing | `RANK`、`WORLD_SIZE`、topology epoch 可变 | 旧 attempt/old UpdatePlan 无法在新 identity commit |
| checkpoint 完整性 | 只读 COMMITTED checkpoint | model/EMA/AdamW/precision/RNG state 全部验证 |
| cache 非权威性 | 缓存是加速副本，不是 checkpoint 事实来源 | 损坏/miss/过期回退 durable；不使用不匹配缓存 |
| 告警安全性 | rebuild/恢复异常会触发控制面 | 仅 freeze/SUSPECT/取证/通知，不能执行网络/调度操作 |

### 2.2 框架结构

交付的 `heterogeneous_microbatch_chaos_framework.py` 是一个**纯逻辑、离线、默认 dry-run** 的测试 harness。它复用 `ElasticCursorLedger`、`VerifiedReadThroughCache`、Prometheus 指标 facade 和 `SafeAlertControlPlane`，但不包含 kill、`tc`、`iptables`、RDMA、SSH、调度器或云 API。

```text
ChaosScenario (seed, fault, dry_run, isolated environment)
  → static guard
  → one logical failpoint
       ├─ ElasticCursorLedger PREPARED/ABORTED/COMMITTED
       ├─ new ElasticIdentity / topology epoch
       ├─ 2:1 → 1:2 rank-local reassignment
       ├─ verified checkpoint cache
       └─ safe alert control plane
  → invariants / assertions / Prometheus observation
  → result artifact
```

每次只运行一个 fault，使用确定 experiment ID/seed，并按 `finally`/临时目录隔离所有 SQLite、checkpoint 和缓存状态。这让 CI 可以重复地验证语义；真实 L3/L4 故障演练仍由基础设施团队用受控 executor 单独执行。

### 2.3 已实现的 fault 矩阵

| Scenario | 逻辑注入点 | 核心断言 | 保护的生产风险 |
|---|---|---|---|
| `node_loss_final_allreduce` | rank 1 在 final synchronized backward 失效 | PREPARED abort；cursor 不动；`{w0,w1,w2}` 重读；新 group 可从 2:1 重分配到 1:2；新 checkpoint 后才 offset=3 | 把 rank 0 `no_sync` 局部进度错误视作已提交 |
| `network_partition_after_prepare` | reservation 之后、commit 之前逻辑分区 | reservation abort；cursor=0；rebuild storm 告警冻结新 plan | 分区中盲目重试/推进 cursor |
| `stale_plan_after_rendezvous` | 新 epoch 形成后尝试提交旧 attempt | old attempt 被 fencing；stale commit 拒绝 | 旧 rank/world/topology plan 穿透新 group |
| `plan_topology_mismatch` | manifest 正确但 plan topology digest 错误 | `prepare_next_update` 拒绝，未产生第二 reservation | topology 变化后错误复用工作分桶 |
| `cache_corruption_during_restore` | cache 条目 payload 被逻辑篡改 | hash mismatch、删除缓存条目、从 durable source 重新加载 | 损坏/陈旧 RDMA 缓存导致 optimizer 状态静默偏移 |

运行：

```bash
cd /home/ubuntu/lecun_analysis
python3 heterogeneous_microbatch_chaos_framework.py
```

当前 smoke test 已完成五类 scenario、17 条断言。它证明的是 ledger/cache/control-contract 的本地语义，不是 NCCL、RDMA 或 scheduler 真实故障已经被压测。

### 2.4 从逻辑 failpoint 到真实集群演练

生产化应把 `ChaosScenario` 映射到抽象的 `FaultExecutor`，而不是将故障命令写进训练器：

```text
LogicalChaosExecutor          → CI/local test，当前实现
ApprovedInfrastructureExecutor → 专用隔离集群，由平台团队托管
```

真实 executor 的每次调用必须再次独立验证 experiment approval、专用 resource pool、目标 rail/node allowlist、TTL、同一时刻仅一个 fault、GPU/NIC health、近期 checkpoint、rollback receipt 和审计 append-only log。训练代码只能观察并验证 `ABORTED → new group → verified restore`，不能拥有基础设施 mutation credential。

## 3. RDMA 外置缓存与分布式 KV 加速 checkpoint 恢复

### 3.1 一个容易犯的错误：把 KV 当 tensor store

对于 checkpoint，KV 与 cache 的职责必须不同。将 AdamW state、FP8 AMAX history 或模型 shard 直接塞入 Raft KV 会将大量二进制流量和高频变更推入共识路径，降低而非提高 RTO；更严重的是把“cache 已写”误认为“checkpoint 已提交”。正确模式为：

```text
(1) Durable shard writer writes immutable content-addressed objects
(2) Hash/size/schema/precision metadata form immutable CheckpointManifest
(3) L0 KV / cursor ledger CAS publishes only:
      run_key -> {revision, checkpoint_hash, manifest_digest, committed_step}
(4) Caches may prewarm shard bytes after (3), never before it becomes recoverable
(5) Recovering workers linearly read pointer -> verify manifest -> parallel fetch shards
```

KV 的 transaction revision 提供 fencing：任何过期 writer 以旧 revision publish pointer 都被拒绝。Watch 可用于提示 cache prewarm，但 etcd Watch 不保证线性化；恢复前必须以 KV 的线性化 read/CAS 或 ledger 读确认当前 pointer/revision。[4]

### 3.2 内容寻址和 cache key

`verified_checkpoint_cache.py` 使用以下 cache key 思想：

```text
cache namespace = namespace : precision_contract_hash : checkpoint_hash
shard key       = component : shard_id : sha256
```

其中 `precision_contract_hash` 绑定 BF16/FP16/FP8 mode、AMP/FP8 backend version、optimizer schema、必要 recipe；`component` 至少区分 `model`、`ema`、`optimizer`、`scaler`、`fp8_metadata`、`rng`。因此，即使字节碰巧相同，不同作业、不同 checkpoint、不同 precision backend 也不会复用同一缓存命名空间。

| 读取路径 | 行为 | 正确性要求 |
|---|---|---|
| cache hit | 取内容、核对 length + SHA-256 | 成功才进入 destination buffer |
| cache miss | 从 durable store 拉取 | 再验证，再 warming cache |
| cache corruption | 删除 entry、计数、从 durable 拉取 | 不重试坏 payload，不推进 cursor |
| cache expiry | 删除 entry、回退 durable | TTL 仅影响性能，不影响恢复事实 |
| KV pointer changed | 丢弃旧 manifest/read plan，重新查询 | 禁止混用多个 checkpoint 的 shard |
| KV quorum/linearizable read 失败 | 保持 freeze 或回退到 durable ledger 可验证副本 | 不以 stale watch 结果恢复 |

原型 smoke test 已验证首读 durable、第二次命中 cache、逻辑损坏后发现 hash 不符并 evict、再从 durable fallback，以及旧 revision CAS 发布被拒绝。

### 3.3 并行加载与 DCP 的结合

DCP 支持多 rank 并行 load，且为本地 shard 计划所需的最少读操作。[1] 可实现一个自定义 StorageReader，使 DCP 的每个 `ReadItem` 先走 verified cache，再走 durable store。读取层的并发由 `(worker, node, rack/rail, object store)` 四级令牌桶控制：

```text
DCP ReadItem
  → resolve current committed manifest from L0 KV / ledger
  → compute content-addressed descriptor
  → acquire per-node + per-rail + global load tokens
  → L2 node-local cache
  → L3 RDMA external cache
  → L1 durable store fallback
  → verify bytes/hash → destination tensor
  → release token, report source/latency/corruption metrics
```

DCP 的 `async_save` 可用 CPU/pinned memory stage 来降低 checkpoint 保存对训练关键路径的阻塞；但官方也提醒 pinned/CPU staging 带来内存压力，通常应限制同时进行的异步 checkpoint 数量。[2] 因此大规模恢复中不应既让所有 node 同时 `async_save`，又让所有恢复 node 同时拉取全部 checkpoint；要统一由 token budget 管理 save、prewarm 和 load。

### 3.4 RDMA 何时有效，何时得不偿失

GPUDirect RDMA 可减少数据在 GPU、CPU、NIC/存储设备间的拷贝/路径长度，但它不是无条件加速：GPU 与 peer 的 PCIe root complex、NUMA 路径、NIC、IOMMU、BAR 大小、驱动/firmware 都会改变或限制性能与可用性。[3] 针住 GPU memory 的 pin/unpin 本身可达毫秒量级；NVIDIA 文档因此建议 lazy unpin 或 registration cache，同时要处理失效 callback、BAR 容量和资源清理。[3]

| 方案 | 优点 | 主要风险 | 首选场景 |
|---|---|---|---|
| Node-local NVMe/page cache | 简单、故障域小、易验证 | 冷 node miss；本地盘容量 | 首先实施的 baseline |
| Pinned CPU RAM staging | 可加速 GPU↔CPU copy，DCP 有相关支持 | page-locked 内存稀缺，压力持续 | checkpoint save/load 热路径 |
| RDMA 外置 CPU memory cache | 跨节点共享热 shard，可能缓解对象存储 thundering herd | NIC/NUMA/网络拥塞、缓存一致性/多租户安全 | checkpoint 大、重启节点多、缓存命中高 |
| GPUDirect RDMA 到 GPU buffer | 可能减少 CPU bounce | 硬件/驱动/注册/失效复杂，BAR 资源受限 | 已验证 GPU-NIC 亲和且 copy 占主导的专用集群 |
| 强一致 KV | pointer/CAS/fencing 正确性强 | 共识延迟/吞吐有限，不适合大 payload | manifest、lease、commit pointer、cache index |

建议路线是先用 DCP 并行分片加载 + node-local cache + manifest hash 验证形成基线；若分析表明 `T_load` 主要来自 durable storage hot-spot 且 cache hit 可以维持较高，才在专用 topology 上引入 RDMA L3。若 `T_state`、Rendezvous 或 NCCL re-init 占主导，RDMA cache 的投入不会显著降低 `RTO_trusted`。

### 3.5 面向抢占风暴的 cache warming 和配额

cache warming 必须发生在 checkpoint 成为 COMMITTED 之后，并且不要把所有 shard 广播到所有节点。更适合的方法是按 topology/role 和预测 shard ownership 分层预热：

1. 提交阶段：完成 durable checkpoint，CAS 发布 manifest pointer；此时才发出低优先级 cache warm hint。
2. 抢占检测：冻结 new UpdatePlan，Rendezvous 分波 admission；每 wave 读取当前 pointer。
3. 重建节点：优先预拉 model/optimizer 中本 rank/DCP local read plan 所需 shard，而非全量。
4. 熔断：缓存 hit rate 下降、corruption、KV quorum failure、rail 延迟或对象存储 429 时降低并发，回退 durable、延后下一 wave。
5. 验证：所有必要 component 的 fingerprint 通过，再允许 `TrainingReady` 和新 UpdatePlan。

建议新增的低基数指标如下，实际可扩展 `distributed_training_observability.py`：

| 指标 | 类型 | 标签 | 目的 |
|---|---|---|---|
| `acvjepa_training_checkpoint_cache_fetches_total` | Counter | source=`node_local|rdma|durable`, outcome | hit/miss/fallback 观测 |
| `acvjepa_training_checkpoint_cache_load_duration_seconds` | Histogram | source、component class | 识别 `T_load` 真实来源 |
| `acvjepa_training_checkpoint_cache_integrity_failures_total` | Counter | bounded reason | cache corruption/contract mismatch 预警 |
| `acvjepa_training_checkpoint_load_inflight` | Gauge | tier | 防止恢复风暴超配额 |
| `acvjepa_training_checkpoint_pointer_revision` | Gauge | job | 已确认 manifest revision，非 shard ID |
| `acvjepa_training_checkpoint_kv_read_duration_seconds` | Histogram | consistency=`linearizable|watch_hint` | 防止把 stale watch 用作恢复事实 |

## 4. 端到端自动化演练流程

```text
L0 baseline: 2:1 DDP + full-state/cursor/cache tests
  ↓
L1 logical failpoint: node loss / partition / stale plan / topology mismatch
  ↓
L2 cache fault: miss / TTL expiry / corruption / precision contract mismatch
  ↓
L3 scale simulation: N nodes × wave admission × token budget
  ↓
L4 isolated infra drill: approved node/rail event, no production traffic
  ↓
L5 admission: RTO_trusted, state exactness, cursor continuity, safe alert actions all pass
```

每轮至少收集：scenario/seed、start/end time、cursor before/after、reservation/attempt states、checkpoint/manifest/pointer hashes、topology epoch、DCP read plan、cache source/latency/integrity、Rendezvous restart、NCCL/RAS、precision fingerprint、Prometheus/Grafana snapshot、alert fingerprint/actions 和 rollback receipt。任何以下情况都意味着失败：cursor 前移但 update 未 commit；任一旧 plan 在新 epoch 成功；cache payload hash 不符仍被使用；precision state 缺失；缓存/KV 失败后继续用 stale pointer；或告警执行越权基础设施动作。

## 5. 交付与验证状态

| 文件 | 作用 | 本地验证 |
|---|---|---|
| `heterogeneous_microbatch_chaos_framework.py` | 五类逻辑 chaos scenario 与 2:1 recovery assertions | smoke test 通过：5 scenarios、17 assertions |
| `verified_checkpoint_cache.py` | 强一致 pointer、内容寻址缓存、损坏/过期回退、并行 slot 分配 | smoke test 通过：cache hit、corrupt fallback、CAS fence |
| `异构微批混沌测试与RDMA外置缓存_Checkpoint恢复架构.md` | 本架构与演练手册 | 文档交付 |
| `异构微批混沌注入与RDMA缓存_安全边界.md` | 操作、权限与缓存事实来源边界 | 文档交付 |

## 参考资料

[1] [PyTorch 2.13：Distributed Checkpoint](https://docs.pytorch.org/docs/2.13/distributed.checkpoint.html)。

[2] [PyTorch：Asynchronous Saving with Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)。

[3] [NVIDIA CUDA：GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)。

[4] [etcd：API Guarantees](https://etcd.io/docs/v3.5/learning/api_guarantees/)。

[5] [PyTorch：torchrun（Elastic Launch）](https://docs.pytorch.org/docs/2.13/elastic/run.html)。
