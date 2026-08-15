# 2:1 异构微批的游标回滚对齐与 RDMA/rail 安全演练

**作者：Manus AI**  
**日期：2026-08-15**  
**范围：AC-VJEPA 同步 DDP 的离线训练恢复与隔离集群可靠性演练。**

## 1. 设计结论

在 2:1 异构微批中，rank 0 处理两个本地窗口、rank 1 处理一个窗口，但**数据游标绝不能按 rank 本地位置保存**。节点宕机后 `torchrun` 会停止/重建 worker group，新的 `RANK`、`WORLD_SIZE` 及 rank-to-node 映射都可能改变；因此旧 rank 0 的“已读两批”和旧 rank 1 的“已读一批”没有恢复身份意义。[1] 唯一可靠的 cursor 是一条与 rank 无关的、不可变全局 work-window manifest 上的**已确认全局 offset**。

> **提交规则：** 只有“同步 DDP update 完成、EMA 更新、完整 checkpoint 写入并哈希验证、checkpoint 原子发布、commit ledger 写入完成”五者都成功时，`next_offset` 才能前移。任一步失败（包括最后 AllReduce 中节点宕机）都使本轮保持 `UNCOMMITTED`，新 worker group 从上一个 `COMMITTED` offset 重新取得同一全局 window range；新拓扑可以重新给 rank 分配该范围，但不能跳过、重复分配或静默丢弃其中的 window。

这种语义是**按 work-window 的 at-least-once 重读**，而不是 PyTorch DataLoader 自动提供的 exactly-once 语义。它选择可审计和收敛正确性，而非试图在进程失效后拼接部分 rank 的 optimizer/EMA/梯度状态。

| 需要稳定的对象 | 可随弹性重建变化的对象 | 设计处理 |
|---|---|---|
| `dataset_commit`、有序 `work_id`、window provenance hash、manifest digest | `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、节点/rail、local sampler offset | 前者进入 durable cursor ledger；后者只绑定单次 attempt/plan |
| 上一个 `COMMITTED` checkpoint hash、model/EMA/optimizer/scaler 状态 | in-flight 梯度、DDP bucket、未发布 optimizer step、临时 CUDA/NCCL 状态 | 后者全部丢弃并从 checkpoint 重建 |
| `next_offset`、commit ID、数据血缘 | 2:1 中哪个 rank 拿两个窗口 | 新 group 基于全局 offset 与新计划重新分桶 |
| 当前 global work range 的完整 ID 集合 | rank-local work 顺序 | 集合必须等于连续全局 slice；顺序可因拓扑重排 |

## 2. 全局游标数据模型

### 2.1 Work window 不是 rank-local batch

将数据集提交预处理为一个有序清单：

```text
GlobalWorkManifest(dataset_commit, ordered_windows)

ordered_windows = [
  (offset=0, work_id=w0, provenance_hash=h0, cost=c0),
  (offset=1, work_id=w1, provenance_hash=h1, cost=c1),
  (offset=2, work_id=w2, provenance_hash=h2, cost=c2),
  ...
]
```

`work_id` 必须是稳定的窗口身份，例如 `dataset_commit / trajectory / window_start / augmentation_seed`；`provenance_hash` 必须绑定原始 RGB-D/点云、动作、接触、本体和生成配置的版本。清单按确定性规则排序并计算 SHA-256。它是游标语义的锚；仅仅对计划中的 ID 集合哈希不足以发现同 ID 被赋予不同 cost、节点偏好或数据血缘的情况。

### 2.2 三类 ledger 记录

`elastic_data_cursor_ledger.py` 用 SQLite 实现了可运行参考账本，核心记录如下。

| 记录 | 关键字段 | 写入时机 | 语义 |
|---|---|---|---|
| `CommittedCursor` | `commit_id`、`committed_step`、`next_offset`、checkpoint hashes、dataset/manifest digest | 原子 checkpoint 验证后 | 唯一有效恢复点 |
| `UpdateReservation` | parent commit、`[start_offset,end_offset)`、完整 plan、elastic identity、全局 work IDs | 本 update 开始前 | 预留但不推进 cursor |
| `ABORTED/COMMITTED` attempt | 原因/时间/checkpoint | 出错或 commit 后 | 解释重读、拒绝重复提交、保留审计链 |

`CursorBoundPlan` 是 `TopologyAwareUpdatePlan` 的数据面投影。它包含 plan version、topology epoch/digest、work manifest digest、world size 以及每个 rank 的 `work_item_ids`。游标验证全局 range 的 **ID 集合**，而训练器使用每 rank 的 ID **顺序**构造 iterator。

## 3. 2:1 的 prepare/commit/abort 对齐算法

### 3.1 准备阶段

设最近确认 cursor 为 `C=(step=s, next_offset=o, commit_id=p)`；当前拓扑感知 planner 给出 `K0=2`、`K1=1`，因此本 update 需要 `M=3` 个 work window。训练根节点在安全 update 边界执行：

```text
1. 从 immutable manifest 读取 R = windows[o : o + M]。
2. 采集当前 rendezvous identity：run_id、restart_count、world_size、topology epoch/digest。
3. 生成新 TopologyAwareUpdatePlan，要求其所有 rank_work_ids 的集合恰好等于 R 的 ID 集合。
4. 在 SQLite BEGIN IMMEDIATE 中写 PREPARED reservation：
   parent_commit_id=p, start=o, end=o+M, plan digest, elastic identity, R IDs。
5. 不改变 CommittedCursor.next_offset。
6. 广播并验证 UpdatePlan；rank 0/1 分别按计划 ID 顺序加载 2/1 个微批。
```

这里有一个重要的“循环”问题：需要知道计划微批总数 `M` 才能取候选 window，但拓扑 planner 的 assignment 又需要 window cost。正确实现中应先以 telemetry/拓扑计算每 rank 槽位数 `K_r`（不依赖具体 window），然后从全局 cursor peek `ΣK_r` 个 window，再用这些 window 的 cost/locality 进行 assignment。当前 `TopologyAwarePlanner` 已把 slot count 的核心计算写在 plan 中；生产实现宜将其提炼成无副作用的 `preview_slot_counts()`，以避免用不完整清单“猜”计划。

### 3.2 2:1 正常提交

例如 `o=0`，前三个 window 是 `[w0,w1,w2]`，新拓扑分配为：

```text
rank 0 -> [w0, w1]  # 第一个 no_sync，第二个同步 backward
rank 1 -> [w2]      # 同步 backward
```

在 `acvjepa_dynamic_update()` 中，各 local loss 以 `valid_samples × world_size / global_samples` 缩放；因此对于 batch size 2、world size 2、全局 6 样本，三项均以 `2 × 2 / 6 = 2/3` 缩放。最终 DDP rank 平均正好得到六样本全局平均梯度。只有所有 rank 完成最后同步 backward、梯度裁剪、optimizer step、EMA 更新和原子 checkpoint 后才发生：

```text
CommittedCursor.next_offset := 3
CommittedCursor.committed_step := s + 1
CommittedCursor.commit_id := hash(parent, attempt, checkpoint, end_offset, elastic_identity)
```

### 3.3 节点宕机发生在最终 AllReduce 时

假设 rank 1 在最后同步 backward 中宕机。rank 0 可能已完成第一个 `no_sync` 的本地梯度，甚至已启动第二个 backward；但它没有得到一个可提交的全局 update。此时算法绝不检查“rank 0 到底做到了哪里”来推进游标：

```text
PREPARED(o=0, end=3) → node loss / NCCL error
  → mark attempt ABORTED(reason=node_lost_during_final_allreduce)
  → CommittedCursor 仍是 (step=s, next_offset=0, commit_id=p)
  → 所有 in-memory grad/bucket/optimizer/EMA 丢弃
```

这避免了最危险的伪恢复：把 rank 0 的两个窗口认为“已训练”、而把 rank 1 的窗口认为“未训练”。那样既破坏全局梯度语义，也会在重建后导致 silent data skip。

### 3.4 Rendezvous 重建后的重新对齐

当 `torchrun` 重建 worker group 后，它可能让相同物理节点拿到不同 global rank，也可能改变 world size；官方文档明确要求不把 `RANK` 视为稳定标识。[1] 恢复算法为：

```text
1. 从 durable ledger 读取唯一最新 COMMITTED cursor C；验证 checkpoint 文件 SHA-256。
2. 将所有 PREPARED reservation 标记 ABORTED（幂等）；C.next_offset 不变。
3. 初始化新的 process group；读取新的 run/restart/rank/world-size 环境。
4. 重新采集可信 GPU/NIC/rail topology，产生新的 topology epoch/digest。
5. 依据新 telemetry 预览 K'_r，peek manifest[o : o + ΣK'_r]。
6. 为这个精确的全局 range 创建新 UpdatePlan；全体 work ID 必须刚好等于连续 range。
7. 以新 rank-local 分配执行 update；只有成功 checkpoint 后推进 cursor。
```

若重建后仍是 2:1，可能出现新的 rank 分配：

```text
旧 group: rank0=[w0,w1], rank1=[w2], attempt 未提交
新 group: rank0=[w0],    rank1=[w1,w2], 新 attempt
```

两个计划的 rank-local布局不同，但它们都消费相同全局连续范围 `{w0,w1,w2}`。这正是游标与拓扑解耦的价值。若 world size 变化导致 `ΣK'_r` 不同，下一次成功 update 可以消费不同长度的连续前缀；但它仍必须从 `o=0` 开始，不能从旧 rank-local offset 接续。

## 4. 代码交付与已验证路径

### 4.1 `elastic_data_cursor_ledger.py`

该模块实现 `bootstrap → prepare_next_update → abort_uncommitted / recover_after_rendezvous → commit_update`。`commit_update()` 会检查：attempt 仍是 `PREPARED`、parent commit 和 cursor 没移动、当前 elastic identity 与 prepare 时一致、plan/topology/work manifest 与当前组一致；对 `file://` checkpoint 还会重新计算 SHA-256。任一检查失败都会 rollback SQL transaction。

烟雾测试演示了如下精确过程。

| 事件 | 计划/身份 | cursor 结果 |
|---|---|---|
| genesis | 初始 epoch、2 ranks | `next_offset=0` |
| attempt-0 | 2:1：`rank0=[w0,w1]`，`rank1=[w2]` | reservation 为 `[0,3)`；不前移 |
| rank 宕机 | final AllReduce 未完成 | attempt 变 `ABORTED`；仍为 `0` |
| rendezvous 重建 | restart count=1、新 topology epoch | 从同一 commit/offset 恢复 |
| attempt-1 | 新 1:2 分配：`rank0=[w0]`，`rank1=[w1,w2]` | 成功 checkpoint 后前移为 `3` |
| attempt-2 | 2:1：`[w3,w4]` 与 `[w5]` | 成功后前移为 `6` |

运行：

```bash
cd /home/ubuntu/lecun_analysis
python3 elastic_data_cursor_ledger.py
```

预期输出中包含：`replayed_uncommitted_work=["work-0","work-1","work-2"]`、`committed_next_offset=6` 与 `committed_step=2`。这验证未确认 `{w0,w1,w2}` 被完整重读，且游标仅在两个成功 checkpoint 后推进。

### 4.2 变 world-size 的额外约束

world-size 改变时，不能期望“用新规模得到与旧规模未完成 update 完全相同的参数”。正确合同应是：从相同已确认 checkpoint 出发、使用相同全局 manifest、无旧计划/旧 identity 复用、每次新 update 的 global sample 加权正确、所有 commit 有血缘。若需要严格的样本处理一次性语义，还必须给每个 window 加入外部事务性消费 ledger；该能力不应由 `DistributedSampler` 或 rank-local iterator 隐式承担。

## 5. RDMA/rail 自动化演练：安全架构

### 5.1 为什么训练脚本不能直接注入网络故障

真实 RDMA/rail 故障可能影响共享 fabric、其它作业、rendezvous、存储或设备健康；把网络管理权限塞给训练作业是错误的权限边界。NCCL 对通信错误建议 abort communicator 并重建；PyTorch 还提供 watchdog、trace 和 desync 调试，但这些不构成网络操作授权。[2] [3]

因此交付的 `rdma_rail_chaos_guard.py` 是**请求守卫**而不是网络破坏工具。它没有 shell、SSH、路由、firewall、RDMA、交换机或云控制命令；它只有：

```text
训练/演练编排器
  → RailChaosGuard（本地防御层）
     → TrustedRailExecutor（基础设施自有、服务器端再验证）
        → 仅专用测试 rail 的预注册故障 profile
```

真实 executor 必须由基础设施团队单独部署，并再次独立验证 mTLS/workload identity、实验签名、测试资源标签、白名单、TTL、幂等回滚、审计日志和无生产路由。客户端守卫可以被绕过，不能成为唯一安全控制。

### 5.2 多重 fail-closed 拦截器

| 拦截层 | `RailChaosGuard` 的检查 | 失败结果 |
|---|---|---|
| 默认模式 | `DRY_RUN`；`EXECUTE` 还要求本机执行 interlock phrase | 只生成验证/演练计划，不调用 executor |
| 范围 | 静态 target-group allowlist；rail 必须在 isolation inventory 的 allowed list | 拒绝任意主机、任意 NIC 或共享 rail |
| 环境 | 必须是 `isolated-preproduction`、专用测试池、非 production、非共享 fabric、无机器人控制 | 生产或未知环境 fail closed |
| 双人授权 | 一名 training owner 与一名 infrastructure owner；不同 principal；HMAC 绑定 experiment digest 和有效期 | 单人、过期、篡改或角色错误拒绝 |
| 状态 | 最近 checkpoint 已 commit；无活跃故障；GPU/NIC 绿色；trace sink 可写 | 防止叠加故障或缺少恢复证据 |
| 训练绑定 | 当前 topology epoch 与 work manifest digest 必须等于被批准的值 | 防止在错误 worker group 上执行 |
| 有界执行 | TTL 和 max restarts 均受 policy 上限 | 防止无限故障/重试 |
| 回滚 | `try/finally` 总是调用 executor rollback，并要求 receipt 变为 `ROLLED_BACK` | 回滚不确认则升级人工，禁止再试 |

### 5.3 自动化事件流程

```text
ARM
  → validate checkpoint + topology/work digest + health + isolation + approvals
  → DRY_RUN 生成 request digest、观察项和回滚计划
  → 人工确认后，EXECUTE 请求交给受限 executor
  → executor 返回有 TTL 的 injection receipt
  → 观察：NCCL RAS / watchdog / agent / cursor ledger / system health
  → finally: request rollback
  → 验证 ROLLED_BACK receipt
  → 等 rendezvous 新组形成
  → 执行 cursor/restart/full-state/shadow 验收
  → CLOSE 或 ESCALATE
```

演练过程中，`after_step_before_commit` 或 final AllReduce 故障都应验证：本轮 reservation 未提交、旧 `next_offset` 未推进、恢复组产生新 topology epoch、旧 plan 被拒绝，以及数据重新从同一全局 offset 分配。

### 5.4 本地模拟测试

模块的 `RecordingExecutor` 仅记录 `mock_inject` 与 `mock_rollback` 事件。它用于验证守卫流程，不触发任何网络 I/O：

```bash
cd /home/ubuntu/lecun_analysis
python3 rdma_rail_chaos_guard.py
```

烟雾测试确认：干运行可通过双人批准；生产标志会被拒绝；不管观察回调成功或异常，`finally` 路径都确认 rollback。任何真实 rail 扰动都必须由受限 executor 在服务器端二次执行同样的策略验证。

## 6. L0—L5 的数据面演练对齐表

| 层级 | 数据游标预期 | 网络/训练预期 | 最低通过证据 |
|---|---|---|---|
| L0 基线 | cursor 未变，计划 digest 一致 | 无注入，2:1 通过全状态对照 | baseline checkpoint、plan、RAS/trace 可读 |
| L1 逻辑 failpoint | `PREPARED → ABORTED`，offset 不前移 | 不进入或不完成 optimizer commit | 失败 attempt、无新 commit、旧计划拒绝 |
| L2 worker 退出 | in-flight range 重读 | agent 重启新 group | 新 restart/epoch、同一 range、checkpoint hash |
| L3 RDMA/rail 分区 | 不提交受影响 update | async/watchdog 触发、旧 communicator 退出 | fault receipt、trace/RAS、rollback receipt、恢复 commit |
| L4 rendezvous 分区 | 保持上一个 commit | 旧 group 停止，endpoint 恢复后重建 | 新 `RANK/WORLD_SIZE`、fresh manifests/plan |
| L5 硬件告警/多域 | 不尝试自动 cursor 推进 | 人工隔离/升级，不自动重试 | GPU/NIC 健康、incident、恢复授权 |

## 7. 运行与发布前门槛

真实集群试验前，至少需要两份独立批准：训练团队确认 checkpoint 与数据回放合同，基础设施团队确认专用节点、rail、rendezvous 和回滚范围。先运行 L0/L1；只有其证据完整，才逐层进入 L2/L3。L3 不是高频自动化任务，不能使用无人工批准的定时触发。

恢复成功后，还应执行：checkpoint/ledger hash 复核、new plan digest 共识、data cursor 连续性检查、固定规模全状态等价测试或变规模 shadow 指标、HITL/发布门控审阅。恢复训练不等于自动发布候选模型。

## 参考资料

[1] [PyTorch 2.13：torchrun（Elastic Launch）](https://docs.pytorch.org/docs/2.13/elastic/run.html)。

[2] [NVIDIA NCCL：Creating a Communicator、错误处理与 Fault Tolerance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)。

[3] [PyTorch 2.13：ProcessGroupNCCL Environment Variables](https://docs.pytorch.org/docs/2.13/torch_nccl_environment_variables.html)。

[4] [NVIDIA NCCL：RAS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html)。
