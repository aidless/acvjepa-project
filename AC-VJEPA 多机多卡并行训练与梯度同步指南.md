# AC-VJEPA 多机多卡并行训练与梯度同步指南

## 1. 先选对并行策略

对于本方案的轻量级 AC-VJEPA（例如冻结或轻度适配 80M 级 V-JEPA 骨干，再训练小预测器/动作头），首选 **PyTorch DistributedDataParallel（DDP）**。每张 GPU 保留完整模型副本，反向传播时 DDP 对梯度执行 all-reduce；实现简单、调试可控，且在模型状态能够放入单卡显存时通常比过早采用分片方案更合适。

| 模型/训练形态 | 首选策略 | 原因 |
|---|---|---|
| 冻结 80M 骨干，仅训预测器/动作头 | DDP + 混合精度 + 梯度累积 | 模型和优化器状态易放入单卡；通信开销低。 |
| 80M 骨干末层适配或部分解冻 | DDP + 激活检查点 + `no_sync()` 梯度累积 | 仍可用复制式训练，先减少同步频率和激活占用。 |
| 300M–1B 骨干大面积解冻 | FSDP / HSDP 或张量并行评估 | 参数、梯度和优化器状态可能超过单卡显存。 |
| 多节点、数据读取成为瓶颈 | DDP/FSDP + 本地数据分片或高吞吐对象存储缓存 | 防止 GPU 等待远端小文件读取。 |

不要因为“多机多卡”就自动选择模型并行。先测单卡峰值显存和每步通信/计算比例：若模型可装入单卡、每步计算足够重，DDP 的数据并行常是最清晰可靠的起点。

## 2. 训练数据与动作语义必须跨 rank 一致

多 GPU 训练能同步数值梯度，却不能自动修复数据语义不一致。每台机器必须使用相同的：动作块 schema、坐标系、尺度/单位、相机预处理、相机标定版本、本体状态字段、事件标签词典和数据清洗规则。特别是 `executed_actions` 必须是最终实际执行、经安全限幅后的 `ActionBlock`，而不是 LLM 计划或安全门裁剪前的候选动作。

训练脚本通过 `DistributedSampler` 对 episode/window 级样本无重叠分片，并在每轮调用 `sampler.set_epoch(epoch)` 使各 rank 以一致但不同的随机顺序读取数据。训练、验证和测试仍要按环境布局、对象实例和任务组合隔离，不能只按随机帧切分。

## 3. 有效批量与梯度同步

总有效批量为：

> `per_rank_batch_size × world_size × gradient_accumulation`

脚本仅在每个累积边界同步梯度；中间微批通过 `DDP.no_sync()` 避免重复 all-reduce。这样可以在不提高单卡显存峰值的情况下提高有效批量，尤其适合视频窗口较长、单卡 batch 很小的场景。

```python
is_update_step = (batch_index + 1) % gradient_accumulation == 0
sync_context = ddp.no_sync() if isinstance(ddp, DDP) and not is_update_step else nullcontext()

with sync_context:
    prediction = ddp(context_video, context_proprio, executed_actions)
    targets = ddp.module.target_latents(future_video, future_proprio)
    loss = action_conditioned_jepa_loss(prediction, targets, future_events).total
    scaler.scale(loss / gradient_accumulation).backward()

if is_update_step:
    scaler.unscale_(optimizer)
    clip_grad_norm_(ddp.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()
```

关键点是必须调用 `ddp(...)` 的标准 `forward`，而不是绕过 DDP 调用 `ddp.module.predict(...)`；后者会使梯度钩子和 all-reduce 失效。

## 4. EMA 目标编码器如何保持一致

AC-VJEPA 的目标编码器不参与反向传播。每个 rank 在 DDP 同步后的学生权重上执行相同 EMA 更新，理论上会保持一致；但混合精度、执行顺序和恢复检查点都可能造成微小漂移。因此训练脚本每隔固定优化步从 rank 0 对 `target_encoder` 的参数与 buffer 广播一次。

```python
module.update_ema_target()            # 每个 rank 在同步后的学生权重上更新
if global_step % ema_broadcast_interval == 0:
    broadcast_module(module.target_encoder, src=0)
```

这比每一步广播更省通信，也避免不同 rank 长期使用略有不同的目标表征。若使用 FSDP，EMA 同步应按照其全参数/分片状态字典规范实现，不能直接假定普通 DDP 的参数广播语义。

## 5. 多机网络与数据吞吐

| 层面 | 推荐做法 | 需要监测的信号 |
|---|---|---|
| GPU 通信 | 使用 NCCL；确保节点间高速互连、正确网卡选择和驱动一致；启用异步错误处理。 | all-reduce 时长、NCCL 超时/重试、rank 失联。 |
| 节点内通信 | 尽量使用高带宽 GPU 互连；固定 GPU 到 CPU/网卡拓扑。 | GPU 利用率不均、PCIe/NVLink 瓶颈。 |
| 数据读取 | 先将使用的 shard 缓存到每台节点本地 NVMe；按 episode shard 分片，而非大量远端小文件。 | GPU 空闲、DataLoader 等待、文件系统 I/O 延迟。 |
| 视频预处理 | 在 CPU worker 或离线缓存中完成解码；固定输入形状和常用裁剪。 | CPU 利用率、解码队列、host-to-device 拷贝等待。 |
| 日志/检查点 | 仅 rank 0 写入；临时文件写完后原子替换；其它 rank 在必要处 barrier。 | 文件损坏、保存时长、对象存储一致性。 |

建议在每个节点的启动环境中显式配置与集群网络、GPU 拓扑相匹配的 NCCL 设置，并先运行带宽/集合通信测试。具体网卡名、RDMA 开关和 NCCL 环境变量取决于硬件与集群运维规范，不应从通用示例硬编码到训练代码。

## 6. 启动与恢复

单节点四卡示例：

```bash
torchrun --standalone --nproc_per_node=4 train_ac_vjepa_ddp.py \
  --manifest /data/ac_vjepa/manifest.jsonl \
  --output /checkpoints/ac_vjepa_run \
  --per-rank-batch-size 4 \
  --gradient-accumulation 4 \
  --num-workers 8
```

多节点示例应在每个节点上启动一次，并由集群调度器提供 `NODE_RANK`、主节点地址和端口：

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  train_ac_vjepa_ddp.py \
  --manifest /data/ac_vjepa/manifest.jsonl \
  --output /checkpoints/ac_vjepa_run
```

检查点必须包含模型、优化器、GradScaler、epoch/step、训练配置、动作 schema 版本、相机/本体预处理版本和 world size。恢复到不同 world size 时，应重新审视学习率、有效批量、学习率调度和数据采样可重复性，而不是假定断点续训等价。

## 7. 从 DDP 扩展到 FSDP/HSDP 的触发条件

当完整模型、梯度、优化器状态和激活无法安全地装入单卡，或者大部分 300M/1B 视频编码器需要解冻时，再考虑 FSDP/HSDP。该切换会增加调试复杂度、状态字典管理与目标编码器同步难度，因此应先通过以下步骤降低显存：缩短视频窗口、降低分辨率、冻结早层、启用 bfloat16、激活检查点、减少不必要的密集 token 和使用梯度累积。

若这些办法仍不能满足峰值显存，则使用 FSDP 对可训练块分片，并为 EMA 目标编码器明确设计“full state dict 生成—rank 0 更新/广播或每 rank 本地更新—一致性校验”的流程。不要把普通 DDP checkpoint 直接迁移到 FSDP 而不验证参数重组与 optimizer state。

## 8. 性能与正确性验收

训练吞吐提升不应牺牲动作世界模型的语义一致性。每次扩容前，至少验证：

1. 所有 rank 的 `ActionBlock` 归一化统计和 schema 哈希一致；
2. 同一固定 mini-batch 在单卡与多卡上的损失/梯度统计在容差内一致；
3. EMA 目标编码器定期参数校验一致；
4. 每 rank 读取的样本 ID 无意外重叠或遗漏；
5. all-reduce、DataLoader、解码、GPU 前向和 checkpoint 的 p50/p95/p99 都被记录；
6. 节点失败、NCCL 异常和 checkpoint 中断不会悄然生成部分有效的模型文件；
7. 多卡训练所得模型仍通过离线故障注入、仿真闭环和影子实机验收。

分布式训练的目标不是最大化 GPU 利用率本身，而是在保证动作—观察对齐、目标编码器一致和可复现安全评测的前提下，以可控成本扩大数据与模型规模。
