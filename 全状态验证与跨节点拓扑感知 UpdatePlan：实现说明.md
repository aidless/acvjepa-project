# 全状态验证与跨节点拓扑感知 UpdatePlan：实现说明

**作者：Manus AI**  
**日期：2026-08-15**  
**状态：已在双进程 CPU/Gloo 语义环境验证；真实 GPU/NCCL 多节点验收仍需在隔离集群进行。**

## 1. 目标与正确性边界

本次增强补齐了两个此前没有被充分验证的层面。其一，2:1 异构微批更新后不再只比较一个卷积权重，而是逐项比较**完整 AC-VJEPA `state_dict`、EMA 目标编码器、全部 buffer 和完整 AdamW `optimizer.state_dict`**。其二，补充一个跨节点拓扑感知的控制平面：它在安全 update 边界采集每个 rank 的 GPU/NIC/NUMA/rail 事实，生成拓扑 epoch，结合数据—通信 p95 与成本分桶计划本轮异构微批和 work assignment，再将不可变计划广播给所有 rank。

> **关键边界：** 这套实现通过动态微批和数据分桶改善同步 DDP 的有效样本吞吐；它不会让快 rank 独立执行 optimizer step，也不会在 active collective 中变更成员、调整 NIC、修改路由或控制硬件。节点离开、rendezvous 重启、world-size/epoch 不匹配都会使当前计划失效，必须从已确认 checkpoint 建立新进程组后再计划。[1]

| 交付物 | 覆盖的正确性问题 | 明确不覆盖 |
|---|---|---|
| `test_dynamic_nccl_full_state_equivalence.py` | 2:1 更新后的跨 rank 模型/EMA/optimizer 一致性；与同六样本单进程参考的数值等价 | CUDA/NCCL、NVLink、IB/RoCE、真实多节点尾延迟或硬件故障 |
| `topology_aware_update_plan.py` | 跨节点拓扑清单、epoch、work manifest digest、成本分桶、2:1 计划广播和 legacy DDP 适配 | 自动配置网络、强制 GPU/NIC 绑定、弹性组重建、外部 inventory 真实性 |
| `全状态一致性与拓扑计划_安全边界.md` | 防止验证/计划逻辑被误用为生产集群控制或物理系统授权 | 基础设施访问控制、KMS、集群调度和机器人安全内核 |

## 2. 增强的 2:1 全状态等价测试

### 2.1 测试输入与计划仍然是 2:1

测试以两个 Gloo rank 启动。rank 0 的报告吞吐为 `40 samples/s`，rank 1 为 `17 samples/s`；目标 update 时间是 120 ms，每个微批固定 2 个样本。现有 planner 因而得出：rank 0 处理 2 个微批（4 样本），rank 1 处理 1 个微批（2 样本），全局有效样本数 `N=6`。每个 rank 的数据由不同种子 `900 + rank` 生成，因此两个 rank 不会因为看到相同输入而“假一致”。

```text
rank 0: batch A (2) ─ no_sync ─┐
                                ├─ batch B (2) ─ DDP AllReduce ─ AdamW ─ EMA
rank 1:                         └─ batch C (2) ─ DDP AllReduce ─ AdamW ─ EMA

全局：N = 2 + 2 + 2 = 6；每个局部均值 loss 以 2 × (W/N) = 2/3 缩放。
```

DDP 的默认梯度归约对 rank 做平均。对于 world size `W=2`，所有局部微批的缩放为 `local_mean_loss × valid_samples × W/N`；最终归约后恰好为六个样本的全局平均梯度。NCCL/Gloo collective 必须由所有 rank 按相同次序参与；该测试让非最终微批使用 `no_sync()`，以保证 rank 1 不会缺失与 rank 0 第一微批对应的 collective。[2]

### 2.2 逐项比较的对象

测试用 `to_cpu_tree()` 深拷贝每个 rank 的快照，然后仅在受信任的本地 CI/Gloo 测试中使用 `all_gather_object` 收集它们。生产 UpdatePlan 控制面不使用 object/pickle collective；其使用有长度上限的 canonical JSON 字节张量和摘要校验。

`assert_tree_close()` 递归检查以下每一层：字典键集合、list/tuple 长度、张量 shape、dtype、浮点张量的 `allclose`、整型/bool 张量的精确相等，以及非张量标量值相等。由此覆盖的状态如下。

| 快照路径 | 包含内容 | 跨 rank 要求 | 单进程参考要求 |
|---|---|---|---|
| `model_state` | 学生编码器、动作 tokenizer、GRU、预测头、EMA `target_encoder`、未来 buffer | `atol=rtol=1e-6` | `atol=rtol=2e-5` |
| `optimizer_state.state` | 每个参数的 AdamW `step`、`exp_avg`、`exp_avg_sq` 等状态 | `1e-6` | `2e-5` |
| `optimizer_state.param_groups` | 学习率、betas、weight decay、参数组结构及其它标量配置 | 精确或对应容差 | 精确或对应容差 |

跨 rank 使用更严格的容差，因为所有 rank 经过同一次 DDP 归约和相同 optimizer step，理论上应是同一副本。串行参考允许很小的浮点归约树差异：单进程按三个微批相加，而 DDP 是先对 rank 0 的两微批累积、再做跨 rank 归约。这个误差预算不能用于掩盖跨 rank 参数分叉，也不能直接复制到混合精度、通信压缩或多 update 的生产验收中。

### 2.3 单进程参考路径避免了一个常见的“测试预言机”缺陷

更强的测试不只验证“两个分布式副本彼此相等”，还把相同的三个 batch（A、B、C）在 rank 0 以单进程方式重放：

```python
for batch, count in [(A, 2), (B, 2), (C, 2)]:
    losses = action_conditioned_jepa_loss(...)
    global_mean_loss += losses.total * (count / 6)

global_mean_loss.backward()
torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
optimizer.step()
module.update_ema_target()
```

参考路径与分布式路径必须在 loss 权重、AdamW、梯度裁剪和 EMA 更新上完全对齐。实现过程中曾发现参考路径遗漏 `clip_grad_norm_`；这会使 AdamW 一阶矩表现出明显差异，即使 DDP 本身正确。修复后，测试在 `2e-5` 参考容差下通过。这是一个重要经验：**若参考实现遗漏任一优化器前处理或后处理，状态差异首先意味着测试预言机不可信，而不是 DDP 有 bug。**

当前运行结果为：跨 rank 比对了 **118 个张量条目**，与单进程参考也比对了 **118 个张量条目**；全局样本为 6，rank 0/1 本地样本分别为 4/2，测试通过。该数量来自当前轻量 AC-VJEPA 和 AdamW 状态结构；模型架构或 optimizer 改变后应由递归遍历自动更新，而不是写死计数。

### 2.4 运行方法与通过条件

```bash
cd /home/ubuntu/lecun_analysis
torchrun --standalone --nproc_per_node=2 test_dynamic_nccl_full_state_equivalence.py
```

通过只说明以下逻辑已被验证：`UpdatePlan` 的 2:1 预算被执行；实际样本数为 4+2；rank 副本的完整训练状态一致；并且这一 update 与相同样本集合的单进程参考在限定容差内一致。它不说明 NCCL 或真实网络正常；多 GPU 集群须将同一测试迁移到 NCCL，保留 deterministic input artifact，并单独校准 GPU、TF32、AMP、融合算子和通信压缩下的容差预算。

## 3. 跨节点 GPU 拓扑感知控制平面

### 3.1 从可信节点事实构造 `TopologyManifest`

每个 rank 在**完整 optimizer/checkpoint 边界**提交一个 `LocalTopologyRecord`。记录仅描述已被基础设施或特权只读探针确认的事实：`node_id`、`local_rank`、GPU 标识、NUMA 节点、NIC/HCA、rail、GPU-to-NIC distance、可观测的 collective floor 和 inventory epoch。训练进程不根据该记录修改网卡或系统设置。

```python
LocalTopologyRecord(
    rank=global_rank,
    node_id=trusted_node_id,
    local_rank=LOCAL_RANK,
    gpu_id=trusted_gpu_uuid,
    numa_node=trusted_numa,
    nic_id=trusted_hca_port,
    rail_id=trusted_rail,
    gpu_nic_distance=trusted_distance_class,
    expected_collective_floor_ms=measured_or_inventory_floor,
    inventory_epoch=inventory_generation,
)
```

所有 rank 通过长度受限的 JSON 字节 tensor `all_gather` 上送其记录。rank 0 按 global rank 排序，构造 `TopologyManifest`，并计算 `topology_epoch`：该哈希绑定了完整 rank 记录、当前 world size、`TORCHELASTIC_RUN_ID` 和 `TORCHELASTIC_RESTART_COUNT`。之后 rank 0 广播 manifest，所有 rank 独立重新计算 epoch 和 SHA-256 digest。这样可阻止以下危险复用：上轮进程组的计划在节点替换、rank 重映射、inventory 变化或弹性重启后继续被使用。

PyTorch 文档明确说明，`torchrun` 发生故障或成员变化时会停止现有 worker 并形成带有新 `RANK`/`WORLD_SIZE` 的 worker group，且 rank 并不稳定；因而 topology epoch 必须由当前运行的成员集合重建。[1]

### 3.2 为什么不直接让训练进程自动“探测并调优”网络

NCCL 会进行拓扑探测，并可导出拓扑 XML；其故障排查将 GPU-to-GPU、GPU-to-NIC、IOMMU/ACS、拓扑及网络问题作为不同诊断领域。[3] 但自动把临时探测结果写成 `NCCL_SOCKET_IFNAME`、HCA 选择或网络规则会将训练进程升级成基础设施控制面，也可能误伤共享集群。因此本实现仅**读取和承诺**可信 inventory/probe 输出；实际 NIC/rail 策略由基础设施团队在压测、变更控制和节点配置层完成。

| 字段 | 作用 | 不可替代的外部责任 |
|---|---|---|
| `gpu_nic_distance` | 对远端数据分配施加软惩罚、记录拓扑亲和 | GPU/NIC 真实映射的采集和校验 |
| `expected_collective_floor_ms` | 为本 update 的通信 guard 提供下限 | 使用真实 NCCL p95/p99 校准，不是硬件承诺 |
| `rail_id` | 避免把全部成本敏感 work 同时压向同一网络 rail | rail 映射、交换机隔离和流量工程 |
| `inventory_epoch` | 发现硬件/固件/映射变更 | inventory 服务的认证、可用性和变更审计 |
| elastic run/restart 字段 | 禁止重启后复用旧计划 | rendezvous、checkpoint 与 job scheduler 的正确配置 |

## 4. 拓扑感知异构 `UpdatePlan` 的计算逻辑

### 4.1 通信 guard 进入微批预算

此前的基础 planner 主要以 `samples_per_second × target_update_ms` 推断微批数，虽然 telemetry 中包含 p95 字段，但其未进入公式。新 `TopologyAwarePlanner` 将实际通信/数据尾延迟作为 guard：

```text
network_guard_r = max(p95_allreduce_ms_r, expected_collective_floor_ms_r)
compute_budget_r = target_update_ms - p95_data_ms_r - network_guard_r - safety_jitter_ms
K_r = clamp(round(samples_per_second_r × compute_budget_r / 1000 / batch_size_r), K_min, K_max)
```

其中 `expected_collective_floor_ms` 是拓扑路径下的下限，实际 `p95_allreduce_ms` 是近期作业观测。取 max 防止在网络一时安静时过度分配，也防止 inventory 过于乐观。若 `compute_budget_r <= 0`，planner 直接拒绝计划；此时应该调查数据 I/O、网络尾延迟、GPU/NIC 路径或减少目标 update，而非强制至少一个微批后让 rank 在 collective 中超时。

在双 rank 模拟中：rank 0 的 network guard 为 10 ms、compute budget 为 93 ms，因而得到 2 个微批；rank 1 的拓扑 floor 为 20 ms、compute budget 为 83 ms，得到 1 个微批。这保留全局 6 样本与 `loss_sum_scale = W/N = 1/3` 的 DDP 数学语义。

### 4.2 成本分桶与拓扑亲和 assignment

`WorkItem` 对应一个已经完成预处理、可由数据集 commit 验证的微批窗口。它包含不可变 `work_id`、估计 `cost_units`、可选 `preferred_nodes` 和 `provenance_hash`。root planner 首先为每个 rank 创建 `K_r` 个固定槽位；再以 cost 从高到低的顺序进行贪心分配，选择预计完成时间最小的可用 rank：

```text
estimated_finish(r, item)
  = 1000 × (assigned_cost_r + item.cost_units) / samples_per_second_r
  + remote_work_penalty(item.preferred_nodes, node_r)
  + gpu_nic_distance_r × distance_penalty_ms
```

算法把高成本微批优先放置到较快、较本地的 rank，并让每 rank 内的高成本工作先执行，尽量让“最后一个会触发同步 backward 的微批”成为较小的尾部任务。它是一个透明、确定性、可测试的启发式，不承诺全局最优调度；在真实集群中应先对比固定分片、轮转分片、该启发式和任何更复杂的求解器，观察 step p99、数据等待、global samples/s 与收敛。

### 4.3 防止 work 清单分歧

仅绑定 `work_id` 不够：不同 rank 可能对同一 ID 使用不同 cost、偏好节点或数据血缘。新的 `work_manifest_digest` 对按 `work_id` 排序后的完整 `WorkItem` canonical JSON 求 SHA-256，并被写入 `TopologyAwareUpdatePlan`。计划验证时会重新计算该 digest；任意 ID、cost、locality 或 provenance 变化都会拒绝计划。

```text
所有 rank：验证 manifest digest、work manifest digest、world size、epoch、rank 顺序
rank 0：构建并广播不可变 topology-aware plan
所有 rank：验证广播字节 SHA-256 一致
所有 rank：按本 rank 的 work_item_ids 构造确定性 iterator
所有 rank：仅最后一局部微批打开 DDP 同步 backward
```

NCCL Broadcast 将 root 缓冲区复制到所有 rank，而 AllReduce 需要所有参与者匹配集体调用；因此计划广播和摘要校验发生在任何 backward 之前，工作分配只能影响输入顺序/成本，不能改变一轮内 collective 次序。[2]

## 5. 与现有 AC-VJEPA 训练循环的集成

拓扑计划保留更丰富的 manifest、rail 和 work assignment；现有 `acvjepa_dynamic_update()` 只消费通用的 `UpdatePlan`（微批数、样本数、loss scale）。`as_legacy_ddp_plan()` 是显式适配器：它保留同一 `global_samples`、`micro_batches` 和 `loss_sum_scale`，而调用者保留 topology plan 作为审计记录，并严格按 `work_item_ids` 建立本地数据迭代器。

```python
# 只在完整 optimizer/checkpoint 边界调用。
manifest, topo_plan = topology_aware_next_plan(
    local_topology=trusted_local_topology(),
    local_telemetry=measure_rank_telemetry(),
    local_micro_batch_size=config.per_rank_batch_size,
    work_items=verified_global_work_manifest,
    planner=planner_if_rank0_else_none,
    device=device,
)

# `work_item_ids` 是本 rank 本次 update 的唯一合法输入顺序。
my = topo_plan.ranks[dist.get_rank()]
micro_batches = iter(load_verified_batches(my.work_item_ids, dataset_commit))
legacy_plan = as_legacy_ddp_plan(topo_plan)

metrics = acvjepa_dynamic_update(
    ddp=ddp,
    optimizer=optimizer,
    scaler=scaler,
    plan=legacy_plan,
    micro_batches=micro_batches,
    device=device,
    config=DynamicStepConfig(amp=config.amp, gradient_clip_norm=config.clip_grad_norm),
)
record_plan_and_metrics(manifest, topo_plan, metrics, checkpoint_lineage)
```

集成代码还应在 `record_plan_and_metrics` 中记录 topology epoch、manifest/work digests、数据集 commit、checkpoint hash、`TORCHELASTIC_RESTART_COUNT`、每 rank work IDs、local/global effective samples、p95 data/AllReduce 和 ETA。若 loader 无法提供计划中的 item、实际 batch size 不符、manifest digest 不同、rank/epoch 变化或 NCCL 发生异步错误，`acvjepa_dynamic_update()` 不应进入/完成 optimizer step；调用方应终止本轮、保留诊断，并从最近确认 checkpoint 重建。

## 6. 已运行验证与下一步

| 验证 | 命令 | 通过信号 | 限制 |
|---|---|---|---|
| 完整状态 2:1 等价 | `torchrun --standalone --nproc_per_node=2 test_dynamic_nccl_full_state_equivalence.py` | 118 个跨 rank 张量条目严格一致；118 个参考条目在 `2e-5` 内；4+2=6 样本 | Gloo/CPU，单 update，未覆盖 AMP/NCCL/通信压缩 |
| 拓扑计划/分桶广播 | `torchrun --standalone --nproc_per_node=2 topology_aware_update_plan.py --smoke-test` | topology epoch/digest 通过；work digest 绑定；2:1，rank 0 获两个 node-a 软体窗口，rank 1 获一个 node-b 窗口 | node/GPU/NIC 是模拟可信事实，未探测真实硬件 |

下一步应把两类测试带入隔离的实际 GPU 集群：一轮使用稳定 world size 的 NCCL；一轮在 checkpoint 边界做受控 worker/node 重启；一轮更换 topology inventory epoch 或篡改 work manifest，并确认计划在 backward 前 fail closed。对 AMP、TF32、PowerSGD/fp16 通信 hook，应建立单独的参考误差预算，不能直接沿用 CPU/Gloo 的 `2e-5`。

## 参考资料

[1] [PyTorch 2.13：torchrun（Elastic Launch）](https://docs.pytorch.org/docs/2.13/elastic/run.html)。

[2] [NVIDIA NCCL：Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

[3] [NVIDIA NCCL：Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)。

[4] [NVIDIA NCCL：Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。
