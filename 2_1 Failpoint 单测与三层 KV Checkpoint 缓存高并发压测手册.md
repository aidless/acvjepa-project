# 2:1 Failpoint 单测与三层 KV Checkpoint 缓存高并发压测手册

**作者：Manus AI**  
**日期：2026-08-15**  
**范围：** 离线 2:1 异构微批恢复语义测试，以及将强一致 KV、durable shard、verified cache 推向隔离生产化压测的测试计划。  
**安全声明：** 所有交付脚本只操作内存、临时文件和 SQLite；不触碰真实 `torchrun`、NCCL、RDMA、网络、节点、对象存储或 KV 集群。

## 1. 交付结构

本轮将原先由 `heterogeneous_microbatch_chaos_framework.py` 聚合执行的五类 failpoint，拆分为独立 `unittest` 测试。每个测试实例化新的 framework 和 experiment ID，避免 AlertControlPlane 的 freeze/idempotency 状态跨测试泄漏。额外的 production/非 dry-run 拒绝测试证明框架不会被误用于真实基础设施。

| 文件 | 作用 | 执行方式 |
|---|---|---|
| `test_heterogeneous_microbatch_failpoints.py` | 五个独立 failpoint 单测 + 安全拒绝测试 | `python3 -m unittest -v test_heterogeneous_microbatch_failpoints.py` |
| `heterogeneous_microbatch_chaos_framework.py` | 被测恢复语义与 logical fault handler | `python3 heterogeneous_microbatch_chaos_framework.py` |
| `checkpoint_cache_load_shedding_simulator.py` | 高并发/防雪崩离散事件模拟 | `python3 checkpoint_cache_load_shedding_simulator.py` |
| `verified_checkpoint_cache.py` | 内容寻址、CAS pointer、verified cache 原型 | `python3 verified_checkpoint_cache.py` |

本地结果：独立单测 **6/6 通过**；五类 chaos 框架 smoke test **5 scenarios、17 assertions 通过**；cache load shedding smoke test 通过。

## 2. 五类 failpoint 的测试代码与断言

### 2.1 Failpoint 1：最终 AllReduce 的节点失效

**故障模型。** 初始 `UpdatePlan` 为 2:1：rank 0 消费 `[w0,w1]`，rank 1 消费 `[w2]`。rank 0 第一微批处于 `no_sync()`，rank 0 的第二微批与 rank 1 的唯一微批共同构成最终同步 backward。故障发生在最终 AllReduce，任何局部梯度都不是提交事实。

测试方法 `test_node_loss_at_final_allreduce_replays_exact_global_range` 调用：

```python
result = self._run(ChaosFault.NODE_LOSS_FINAL_ALLREDUCE)
self.assertEqual(
    result.assertions,
    (
        "uncommitted_cursor_unchanged",
        "same_global_range_replayed",
        "rank_layout_reassigned",
        "new_checkpoint_committed",
    ),
)
```

框架内部的关键断言是：初始 reservation 为 `[start=0,end=3)`；`abort_uncommitted()` 后 cursor 仍为 0；新的 elastic identity 下，`rank0=[w0]`、`rank1=[w1,w2]` 必须再次准备同一个全局集合；仅新 checkpoint commit 成功后 `next_offset=3`、`committed_step=1`。

| 断言 | 检测的错误实现 |
|---|---|
| `uncommitted_cursor_unchanged` | 将 rank 0 的两个局部微批错误视为已消费 |
| `same_global_range_replayed` | 重建后跳过/重复部分 work window |
| `rank_layout_reassigned` | 把旧 global rank/local sampler 位置当成稳定身份 |
| `new_checkpoint_committed` | 在 final collective 失败后仍提交 optimizer/EMA/cursor |

### 2.2 Failpoint 2：`PREPARED` 后的逻辑网络分区

**故障模型。** 账本已写入 `PREPARED` reservation，但 checkpoint/commit 尚未发生。控制面或数据面分区可能造成 watchdog、rendezvous 或对象存储不可用；这不是重试旧 plan 或推进 cursor 的理由。

```python
result = self._run(ChaosFault.NETWORK_PARTITION_AFTER_PREPARE)
self.assertIn("reservation_aborted", result.assertions)
self.assertIn("cursor_not_advanced", result.assertions)
self.assertIn("new_plans_frozen", result.assertions)
self.assertIn("no_network_operation_executed", result.assertions)
```

测试先调用 `ledger.abort_uncommitted()`，再触发模拟 `ACVJEPARendezvousRebuildStorm`。`SafeAlertControlPlane` 只记录 `freeze_new_plans`、`mark_suspect`、`capture_existing_evidence`、`notify_owner`；测试明确证明它不执行网络操作。

| 断言 | 预防的失效模式 |
|---|---|
| `reservation_aborted` | 孤儿 `PREPARED` 阻塞后续恢复或被意外 commit |
| `cursor_not_advanced` | 分区时把无更新的样本标记为训练完成 |
| `new_plans_frozen` | 故障仍在发生时继续创建 UpdatePlan |
| `no_network_operation_executed` | 监控/测试脚本越权控制 network/rail |

### 2.3 Failpoint 3：Rendezvous 后提交旧计划

**故障模型。** 新 worker group 已形成，topology epoch/restart count 已变，但旧 attempt 持有曾经有效的 checkpoint bytes。即使 bytes 完整，也不允许旧 plan 以旧 identity commit。

```python
result = self._run(ChaosFault.STALE_PLAN_AFTER_RENDEZVOUS)
self.assertEqual(
    result.assertions,
    ("old_attempt_fenced", "stale_commit_rejected", "cursor_not_advanced"),
)
self.assertNotIn("new_checkpoint_committed", result.assertions)
```

被测逻辑先调用 `recover_after_rendezvous()`，它将所有 `PREPARED` attempt 标记为 `ABORTED`。随后故意调用 `commit_update(old_attempt, old_identity)`，预期捕获 `CursorContractError`。这验证 fencing 发生在 ledger commit path，而不是依赖外部“不要这样做”的约定。

### 2.4 Failpoint 4：拓扑摘要不匹配

**故障模型。** 数据 manifest 和 work IDs 看似合法，但 `topology_epoch`/`topology_digest` 来自错误/陈旧 process group。此类错误应在 prepare 阶段失败，不能等到 NCCL collective 或 optimizer step 才暴露。

```python
result = self._run(ChaosFault.PLAN_TOPOLOGY_MISMATCH)
self.assertEqual(
    result.assertions,
    ("topology_digest_mismatch_rejected", "no_second_reservation_created"),
)
```

框架构造 `bad_plan`，其 ID 形式正确、`world_size=2`，但 epoch/digest 与当前 `ElasticIdentity` 不同。`prepare_next_update()` 必须抛出 `CursorContractError`。第二个断言尤为重要：不只要报错，还要保证数据库没有遗留第二条 reservation。

### 2.5 Failpoint 5：恢复期间 optimizer cache 损坏

**故障模型。** 为恢复加速而缓存的 `optimizer/rank-shard-0` payload 被篡改。重点不在“缓存报错”，而在“损坏的 AdamW `exp_avg/exp_avg_sq` 绝不进入恢复状态”。

```python
result = self._run(ChaosFault.CACHE_CORRUPTION_DURING_RESTORE)
self.assertEqual(
    result.assertions,
    (
        "cache_corruption_detected",
        "entry_evicted",
        "durable_fallback_verified",
        "commit_pointer_unchanged",
    ),
)
```

`VerifiedReadThroughCache` 对每次 cache hit 做 `byte_count + SHA-256` 验证。测试先以 durable source 填热 cache，再使用 test-only helper 替换为 `b"bad"`，下次 fetch 必须：发现 hash mismatch、删除 entry、从 durable store 拉取正确 payload、再次验证，并保持 KV 的 committed pointer revision 不变。

### 2.6 Failpoint 框架的安全拒绝

虽然不属于五类业务 failpoint，`test_framework_rejects_non_isolated_or_non_dry_run_scenario` 是必需的 guard rail：

```python
with self.assertRaises(PermissionError):
    framework.run(
        ChaosScenario(
            fault=ChaosFault.NODE_LOSS_FINAL_ALLREDUCE,
            experiment_id="must-reject",
            seed=0,
            environment="production",
            dry_run=False,
        )
    )
```

这避免有人把 CI 语义测试误解为有权对 production 进行节点/网络故障注入。

## 3. 三层 KV Checkpoint 缓存的高并发压测模型

### 3.1 被压测的业务流程

压测对象不是“能否把 GB 级 tensor 写进 KV”，而是以下三层恢复路径在 mass preemption 下的协调能力：

```text
L0: strongly-consistent KV / ledger
      └─ CAS: run_key -> committed pointer(revision, checkpoint hash, manifest digest)
L1: durable object / parallel storage
      └─ immutable content-addressed model/EMA/optimizer/precision/RNG shards
L2/L3: node-local + RDMA external verified cache
      └─ hot immutable shard copies; TTL, capacity and hash verification
```

恢复 worker 必须先做一次**线性化** L0 pointer 读取，再将该 pointer 绑定到 immutable manifest；cache warm hints、watch notification 或 Redis client-side invalidation 可用于性能优化，但不能决定“当前 checkpoint 是什么”。etcd 对 KV API 提供严格串行化，但线性化读要经 quorum，开销大于可返回陈旧数据的 serializable read。[1] 因此应让线性化读只承载小 pointer，而非每个 shard。

### 3.2 压测维度与阶段

| 阶段 | 读/写比例与负载 | 变量 | 主要验收 |
|---|---|---|---|
| P0 基线 | 单 node、热/冷 read | shard 大小、cache tier | 量化 L1/L2/L3 latency 与 hit rate |
| P1 并发递增 | N worker 同时读取同 manifest | N、connection、per-node token | p50/p95/p99、queue、token block、durable QPS |
| P2 冷启动风暴 | 所有 worker 请求同一个冷 optimizer shard | single-flight on/off、wave size | durable fetch 必须接近 distinct shard 数，而非 worker 数 |
| P3 多 key 热点 | Zipf/热点 manifest + 多 shard | cache size、eviction、TTL jitter | hit rate、热点倾斜、cache memory、invalidations |
| P4 指针写风暴 | 多个 writer 竞争 CAS publish | writer count、lease、retry budget | 一个 revision 仅一 winner；冲突重试有上限 |
| P5 退化/故障 | cache miss、corrupt、TTL mass expiry、KV slow/quorum unavailable | failure ratio、recovery latency | 熔断、降并发、durable fallback、cursor/commit 不变 |
| P6 恢复全链路 | admission waves + DCP read plan | preemption size、rail/topology | `RTO_trusted` p95、state exactness、cursor continuity |

etcd 官方建议新环境首先运行实际 benchmark；其延迟与吞吐会受并发、网络 RTT、磁盘 `fdatasync`、Raft、compaction 等环境因素影响。[1] 所以 P0/P1 的阈值必须用目标集群测得，不能照搬本地 CPU 的数字。

### 3.3 压测流量与合成数据

使用合成的 content-addressed shard，而非真实模型或传感器数据。建议设三档 payload：小 metadata（1–16 KiB）、中等 optimizer shard（1–64 MiB）、大模型 shard（256 MiB–数 GiB），并以固定随机种子生成。每个 read request 带：`run_key`、revision、manifest digest、component class、shard ID、tier preference 和 worker/node/rail bucket；Prometheus 标签只保留 tier/component/outcome，不携带 shard ID/hash。

真实 KV 与对象存储压测必须在专用 namespace、独立 quota、固定最大 QPS/bytes、清晰清理策略和人工 stop switch 下进行。etcd 空间 quota 触发后会进入维护/限制模式；history compaction 和 defragmentation 也会影响延迟，故需把 `NOSPACE`、backend size、fsync、leader changes 纳入观测。[2]

## 4. 防雪崩控制面：完整策略

### 4.1 Single-flight：同 key 一次落盘读取

mass preemption 时，数百 worker 可能同时请求同一个 checkpoint shard。对 key：

```text
(namespace, precision_contract_hash, checkpoint_hash, component, shard_id, shard_sha256)
```

只允许一个 leader 向 L1 durable store 请求；其余请求加入同一 future，leader 完成后所有 follower 从 verified payload 分发或命中 cache。若 leader 失败，followers 收到同一错误/退避 token，而不是并发重试。

`checkpoint_cache_load_shedding_simulator.py` 的 32 节点 cold-key storm 断言：

```python
report = simulator.cold_key_storm(
    key="optimizer:step-42:shard-0",
    nodes=[f"node-{index}" for index in range(32)],
)
assert report.durable_fetches == 1
assert report.coalesced_waiters == 31
assert report.cache_hits_after_fill == 32
```

这不代表真实 RDMA 性能，仅证明调度逻辑不会把 32 个同 key request 放大为 32 次 durable read。

### 4.2 分层 token bucket 与 admission wave

配置四个同时生效的额度：global durable QPS/bytes、per-rack/rail、per-node、per-checkpoint/manifest。优先级为：commit/recovery pointer > model/optimizer 必需 shard > EMA/precision state > optional prewarm。令牌不足时排队带 deadline；超过 deadline 触发 wave 延后而不是 retry storm。

```text
Acquire(global durable) ∧ Acquire(rail) ∧ Acquire(node) ∧ Acquire(manifest)
  → verified fetch
否则
  → queue / join single-flight / next wave after jitter
```

`RecoveryWavePlanner` 的分波重入与 cache tier token 必须共享同一恢复预算；不能让 wave 限制了 Rendezvous，却让 cache prewarm 在后台同时淹没对象存储。

### 4.3 TTL jitter、stale-while-verify 与负缓存

大量缓存以同一 TTL 过期会产生“过期雪崩”。将 immutable checkpoint shard 的 TTL 通过 key/seed 加确定性抖动分散；checkpoint hash 不变时可以采用 `stale-while-verify`，但必须先验证 pointer 仍指向同一 content-addressed checkpoint，且不能对 precision contract 已变的条目提供 stale 数据。

不存在的 shard、manifest key 或访问拒绝可放入短 TTL negative cache，减少重复 L1/KV 探测。negative cache 只能缓存明确的、可验证失败，不能吞掉超时/权限/完整性异常。模拟器中 10 个相同 missing request 只发起 3 次 durable probe、7 次 negative hit：

```python
negative = simulator.negative_cache_storm(
    missing_key="missing:fp8-meta", requests=10
)
assert negative == {"requests": 10, "durable_probes": 3, "negative_hits": 7}
```

Redis client-side cache 文档指出缓存失效和失效连接丢失存在陈旧风险；失效通道断开时应 flush local cache，且应对每项设置最大 TTL 与内存限制。[3] 对 checkpoint restore 而言，这意味着失效通知只用作 cache hygiene；线性化 pointer/hash 验证仍是硬门槛。

### 4.4 自适应并发、熔断和指数退避

采集每个 tier 的 p50/p95、timeout、integrity error、queue length、KV leader latency、L1 429/5xx、rail utilization。若 `p95 > target`，用保守乘法下降（例如令牌上限减半，底线为 1）；若连续健康窗口低于目标的 70%，缓慢线性增加。这与 Envoy 的基于 minRTT/样本时延调整 outstanding request 上限的思路一致。[4]

完整性失败、pointer/manifest 不匹配、KV quorum 丢失、或 cache corruption 连续超过预算时，打开 circuit：停止 RDMA/L3 cache admission，freeze new UpdatePlan，回退 L1 或等待人工健康检查。熔断器关闭只能由成功 health check 与审计事件触发，不能仅因 timeout 消失自动打开。

模拟器断言：

```python
simulator.controller.observe(p95_ms=250, target_ms=100)
assert simulator.controller.durable.limit == 1
simulator.controller.observe(p95_ms=80, target_ms=100, integrity_failures=1)
assert simulator.controller.circuit_open
```

### 4.5 CAS 写风暴和幂等 publish

checkpoint writer 只在所有 shard durable 写入、manifest hash 验证后尝试发布小 pointer：

```text
CAS(run_key, expected_revision=r,
    new = {checkpoint_hash, manifest_digest, committed_step})
```

若冲突，writer 线性化读取当前 pointer，判断自己是否已被另一个 writer supersede；绝不盲目覆盖。模拟器以 16 个并发逻辑 writer 验证 `wins=1`、`cas_conflicts=15`、`revision=1`。这一约束确保预热/缓存层不能把旧 checkpoint 重新变成当前恢复锚。

## 5. 生产压测的指标、门槛与停止条件

| 指标 | 建议观测 | 通过/告警原则 |
|---|---|---|
| pointer linearizable read p95/p99 | KV quorum path | 在目标 RTO 预算内；突升时停止扩大 load |
| pointer CAS conflict rate | writer competition | 冲突可预期，但必须有一个 winner、重试受预算限制 |
| cache hit ratio（按 tier） | L2/L3/durable | 先看 byte-weighted hit；低 hit 时禁止盲目增加 cache 容量 |
| distinct-shard durable amplification | durable reads / distinct cold shards | single-flight 后应接近 1；显著大于 1 说明击穿 |
| durable/L3 load p95 | 每 tier、每 rail | 超 target 降 token，非简单增加 timeout |
| queue wait / blocked token | admission health | 长队列应减小 wave 或提高经过压测的容量，不应无限排队 |
| integrity failure | hash/contract/byte mismatch | 任意非零开启高优先级调查；连续触发 circuit |
| KV NOSPACE/compaction/leader change | etcd health | 停止 publish/prewarm；保留恢复证据 |
| `RTO_trusted` p50/p95 | 从故障 marker 到 TrainingReady | 必须包含 full state/cursor/plan gate，不只测进程 restart |

**强制停止条件**包括：生产/共享 namespace 被识别；总带宽、QPS、cache memory、p99 或 KV backend size 超预算；出现 NOSPACE、持续 leader churn、cache integrity error、pointer revision skew、cursor 前移异常、state fingerprint mismatch、目标以外流量、或人工 stop 触发。

## 6. 上线压测 SOP

1. 在测试池创建唯一 namespace、合成 checkpoint 集、固定随机 seed 和明确字节/QPS/节点上限；确认 durable store、KV、cache tier 和监控隔离。
2. 运行 P0/P1，确定各 tier baseline；为 token、wave、TTL jitter、negative TTL、circuit 阈值写入 versioned experiment manifest。
3. 运行 P2/P3，验证 single-flight/TTL jitter/negative cache；比较 durable amplification 与 cache hit。
4. 运行 P4，验证 pointer CAS/lease/revision/fencing；每次冲突都保留 audit record。
5. 运行 P5，逐个引入 cache corruption、KV slow、L1 timeout、TTL mass expiry，验证 fail closed/fallback；不得同时注入多种故障。
6. 仅在上述证据完整后运行 P6 mass-preemption admission wave；将 cache load、Rendezvous、DCP read plan、cursor replay、precision exactness 合成 `RTO_trusted`。
7. 归档 Prometheus/Grafana snapshot、KV health、cache stats、DCP/ledger artifacts、审批和 stop logs；由训练、平台、数据/安全 owner 共同决定是否提升 wave/并发上限。

## 参考资料

[1] [etcd：Performance](https://etcd.io/docs/v3.5/op-guide/performance/)。

[2] [etcd：Maintenance](https://etcd.io/docs/v3.5/op-guide/maintenance/)。

[3] [Redis：Client-side Caching Reference](https://redis.io/docs/latest/develop/reference/client-side-caching/)。

[4] [Envoy：Adaptive Concurrency](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/adaptive_concurrency_filter)。

[5] [PyTorch：Distributed Checkpoint](https://docs.pytorch.org/docs/2.13/distributed.checkpoint.html)。
