# NCCL 动态负载均衡、梯度累积与 Human-in-the-Loop 难例校正蓝图

## 1. 设计结论

在同步 DDP 中，所谓“动态负载均衡”不能理解为让快卡随意多更新、慢卡少更新。NCCL AllReduce 是紧密同步的集合通信；某个 rank 缺席、跳步或在不同参数版本上进入同一 collective，会导致挂死或语义错误。[1] 因而正确做法是：在**一个完整 optimizer update 之间**允许每个 rank 处理不同数量的局部微批；所有 rank 在预先广播的计划下完成局部累积，然后共同执行一次同步反向传播、AllReduce 和 optimizer step。

人类在环也不应被设计成“人替模型开机器人”。它是隔离难例的证据修复和数据治理机制：审核原始视频/点云/动作/接触/不确定性/根因报告，修正事件或数据语义，提出受批准范围约束的物理先验/仿真覆盖，双重复核后形成可追溯的数据 patch。该 patch 只能进入 SimJob 编译器或 curated dataset manifest，不能修改在线安全阈值或绕过影子/灰度发布。

## 2. NCCL 多机动态训练

NCCL 为多 GPU、多节点提供拓扑感知的集合通信，支持 AllReduce、ReduceScatter、AllGather 等原语，并可在 PCIe、NVLink、InfiniBand 或 IP sockets 上运行。[1] 但 NCCL 处理的是通信，不是异构数据/计算调度器。因此负载均衡必须在训练器中把不同长度视频、不同点云数量、数据解码等待和不同 GPU 吞吐转化为**下一轮更新的局部工作预算**。

### 2.1 两时标控制

| 时标 | 组件 | 决策 | 不能做的事 |
|---|---|---|---|
| 每个微批 | rank 本地 loader/预取器 | 长度分桶、视频帧/token 预算、非阻塞传输、下一个样本选择 | 改变 DDP world size 或自行跳过最终同步。 |
| 每个 update | rank 0 计划器 + `broadcast_object_list` | 广播下一 update 的每 rank 微批数、micro-batch 大小、全局有效样本数 | 在 AllReduce 中途改变计划。 |
| 每个 checkpoint 边界 | 弹性作业控制器 | 评估健康 rank、持久化状态、重启/重建 process group | 允许成员变更继承未完成 collective。 |
| 每个 epoch/评测窗口 | 训练治理 | 评估吞吐、收敛、校准、安全集表现 | 因吞吐提升而自动放宽质量或安全门。 |

### 2.2 以样本成本而非文件数分桶

对每条训练窗口预先估计 `cost = alpha × 视频帧数 + beta × 有效点数 + gamma × 解码字节数`。使用长度/成本分桶后，rank 优先领取总成本接近目标的批组，而非随机读任意文件。这样减少慢节点因柔性物体点云密度、视频分辨率或远端缓存 miss 形成的 straggler。

每个 rank 每隔一个稳定窗口上报：有效 samples/s、p95 数据等待、p95 前向/反向、p95 AllReduce、显存余量、温度/降频状态和数据错误率。rank 0 不应根据一次尖峰调整，而应使用窗口中位数/分位数和最小变化阈值，生成版本化 `UpdatePlan`。计划必须被广播、记录到 run manifest，并在所有 rank 确认后才用于下一 update。

### 2.3 可变局部累积与正确归一化

设参与 update 的 world size 为 \(W\)，rank \(r\) 在本次 update 处理 \(n_r\) 个有效样本，全球有效样本数为 \(N=\sum_r n_r\)。若 PyTorch DDP 默认在 AllReduce 后对 rank 梯度取平均，而每个本地微批损失先对该微批样本求平均，则每个微批应在反向前乘以：

\[
\mathrm{scale}=\frac{W \times n_{\text{micro}}}{N}.
\]

这样每个 rank 累积的是 \(W/N\) 乘以本地样本梯度和；DDP 的 rank 平均后恰好得到全局样本平均梯度。若使用 token/有效点掩码，\(n_{\text{micro}}\) 应替换为实际参与 loss 的有效权重，而不是固定 batch 大小。所有 rank 都必须在自己的最后一个微批离开 `no_sync()`，共同执行**一次且仅一次**同步 backward。PyTorch 文档明确说明，`no_sync()` 内的梯度会累积，并在随后退出上下文的第一次 forward-backward 时同步。[2]

`adaptive_ddp_accumulation.py` 将 rank 吞吐转换为本地 `micro_batches`，同时输出 `global_samples` 和 `loss_sum_scale`。示例中快 rank 处理 32 个样本、慢 rank 处理 20 个样本，但两者使用相同的 `loss_sum_scale = W/N`，仍在一次共同更新中学习。

### 2.4 网络与 NCCL 策略

优先级应为：先通过 profile 判断瓶颈是否真在 AllReduce；然后使用长度分桶、预取、梯度累积减少同步次数、`gradient_as_bucket_view` 和通信/反向重叠；最后才启用 FP16/BF16 或 PowerSGD 通信压缩。不同节点的 GPU/网络能力不一致时，限制每 rank 的局部预算上限，让最慢 rank 的 p95 update time 保持在可接受范围，比试图“追平每张卡的利用率”更可靠。

成员变更或持续网络故障不应在 process group 存活时硬恢复。应在最近一次原子 checkpoint 后停止当前组、启动新的 rendezvous/world size，并重新计算 batch、学习率调度和 `UpdatePlan`。PyTorch 的弹性启动教程说明，故障后应从应用快照重新初始化；快照须包含持续训练需要的状态。[3]

## 3. 人类在环的隔离难例校正

### 3.1 审核队列

隔离 case 应包含不可变证据包：episode commit、视频/点云摘要、实际动作、接触/力、本体状态、传感器质量、V-JEPA 预测与不确定性、影子 RCA、当前 DR 配置、simulator/asset 版本、哈希与风险等级。系统按“高不确定性持续、预测残差、潜在碰撞/保持漏失、数据漂移、软体物理缺口、覆盖稀缺度、样本多样性”排序；但硬件/传感器故障直接转到安全事件而非训练数据审核。

| 审核动作 | 输出 | 下游用途 | 必要复核 |
|---|---|---|---|
| 纠正接触、滑移、形变事件边界 | 版本化事件标签 | `future_events`、校准压力集 | 高风险/软体样本双审。 |
| 确认物体类别、材料/几何状态 | 受限物理 prior 提案 | SimJob 的分层重加权 | 超出批准范围必须单独审批。 |
| 标记传感器/标定故障 | 数据处置/系统工单 | 排除训练，修复数据契约 | 可由安全负责人终审。 |
| 选择低风险人类示教或视频片段 | curated evidence | 后续模拟反事实、replay | 来源/许可审核。 |
| 拒绝无效/隐私不合规样本 | `REJECTED` | 永不进入训练 | 审计保留。 |

### 3.2 双重审核与 patch

`hitl_quarantine_review.py` 实现最小审核账本：第一位审核者提交 `CorrectionPatch` 后，软体/接触 case 进入 `NEEDS_SECOND_REVIEW`；第二位独立审核者确认或拒绝。仅 `APPROVED_FOR_DATA` case 能导出包含两次审核、证据哈希、版本和下游限制的 patch。其 `forbidden_downstream` 明确包含 `robot_control`、`safety_threshold_change` 和 `direct_production_deploy`。

```text
quarantined case
  → 自动聚合证据 + 去重 + 优先级排序
  → reviewer A: 标注/处置/物理先验/仿真焦点
  → reviewer B: 独立确认或拒绝
  → approved correction patch
  → DR/SimJob 编译器（仅批准范围内）或 curated data manifest
  → verified dataset commit
  → incremental training + frozen replay + offline/sim/edge evaluation
  → shadow only → existing canary gate
```

高效来自“让人修正最有信息量的证据”，而非让人逐帧标注一切。审核界面应提供同步的 RGB-D/点云视图、动作/接触时间轴、模型预测与真实观测差、候选物理分层、同类历史 case 和一键拒绝/升级；同时保持所有修改为结构化 patch，避免自由文本直接进入训练参数。

## 4. 运行准则

1. 为每个 update 写入 `UpdatePlan`、world size、有效全局样本数、每 rank 累积步数、通信压缩配置和 p50/p95/p99 指标。
2. 只在 checkpoint 边界接受成员变化；恢复后重建 sampler、预取器、NCCL process group 和全局 batch 解释。
3. 将吞吐/通信优化与模型质量看作双目标：新计划必须通过冻结回归集、软体压力集和不确定性校准集，而不只是 steps/s 变快。
4. 任何审核 patch 均需保留原始 evidence hash，且只能生成新数据/新训练运行，不能修改历史数据集提交。
5. 训练后仍沿用离线、仿真、Jetson、影子和灰度门控；HITL 通过不等于获得实机控制权限。

## 参考资料

[1]: [NVIDIA, *Overview of NCCL*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)

[2]: [PyTorch, *DistributedDataParallel*](https://docs.pytorch.org/docs/2.13/generated/torch.nn.parallel.DistributedDataParallel.html)

[3]: [PyTorch, *Fault-tolerant Distributed Training with torchrun*](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
