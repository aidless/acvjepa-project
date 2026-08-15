# NCCL 真实集群压测与 Gloo 异构微批验证实施手册

**作者：Manus AI**  
**日期：2026-08-15**  
**适用范围：AC-VJEPA 的同步 DDP/NCCL 训练；仅限隔离 GPU 集群和离线模型工件。**

## 1. 结论与前置边界

真实多 GPU 集群的压测不应被理解为“运行一次 `all_reduce_perf` 并记录一个 GB/s”。对动态 `UpdatePlan` 训练而言，需要同时验证四件事：物理路径是否符合预期拓扑；集体通信的带宽与尾延迟是否稳定；应用中的计算—数据—通信重叠是否有效；发生进程、节点或网络故障后，训练是否从**最后一个已确认 checkpoint**以正确的成员集合恢复。任一项缺失都可能形成“微基准很好、训练仍频繁卡死或梯度语义错误”的假阳性。

> **安全边界：** 本手册建议在专用或预生产 GPU 队列、无真实机器人控制链路、可丢弃数据副本和已批准变更窗口中进行压测。节点重启、GPU reset、网络隔离/限速、驱动重载等破坏性操作必须由基础设施值班人员执行；本手册不提供绕过权限或影响共享生产作业的指令。每一轮故障实验开始前都必须有可验证的 checkpoint。

NCCL 集体操作要求所有 rank 以一致的集合调用完整 collective；不满足时会发生未定义行为，包括挂起、崩溃或数据损坏。[1] 因而压测的第一成功标准不是峰值带宽，而是：**任意故障均能在错误 optimizer step 之前被检测、记录并收敛到可恢复状态。**

| 层级 | 应回答的问题 | 关键工件 | 不应由该层得出的结论 |
|---|---|---|---|
| 硬件/拓扑 | GPU、PCIe、NVLink/NVSwitch、NIC、NUMA 和网络 rail 怎样连接？ | GPU/NIC/NUMA 清单、拓扑 XML、PCIe/NVLink 状态 | 不能仅据拓扑推断实际训练 p99 |
| NCCL 微基准 | 不同消息大小、rank 数、节点数的 all-reduce/p2p 表现与方差如何？ | 每 rank 延迟、`algbw`、`busbw`、错误日志 | 不能推断真实模型的计算通信重叠 |
| 应用压测 | UpdatePlan 的 2:1、3:1 等异构微批对样本吞吐、等待时间、梯度正确性有何影响？ | plan、有效样本数、step p50/p95/p99、参数/EMA 哈希 | 不能替代故障恢复验证 |
| 故障恢复 | watchdog、进程组重建、弹性成员变化及 checkpoint 血缘是否正确？ | 根因日志、restart count、rendezvous ID、checkpoint hash | 不能等价于机器人安全或上线批准 |

## 2. 实验前：建立“可比较”的基线

### 2.1 固定软件和物理清单

在所有节点上记录 GPU 型号、显存、驱动、CUDA、NCCL、PyTorch、OFED/网卡固件、Linux kernel、容器镜像 digest、`torchrun` 参数与训练代码 commit。记录每个 GPU 到 CPU NUMA、NIC 和其他 GPU 的距离，以及每个 NIC 所属的网络 rail。不要只凭设备编号推断距离；设备枚举顺序并不等于 PCIe/NVLink 或 GPU-to-NIC 亲和关系。

NCCL 可以探测拓扑，也提供 `NCCL_TOPO_DUMP_FILE` 以导出 XML 拓扑；其故障排查文档将 GPU-to-GPU、GPU-to-NIC、IOMMU/ACS、拓扑和多节点 NVLink 明确列为独立诊断方向。[2] 这些导出的拓扑和诊断日志可能包含主机/网络细节，应存放在受控的实验工件库中。

建议在每个节点仅以**读取方式**采集以下信息，并将时间戳与作业 ID 关联：

| 范畴 | 建议采集 | 判读目的 |
|---|---|---|
| GPU 运行状态 | GPU 型号、功耗/温度、ECC/Xid、时钟、利用率、MIG 模式 | 识别热降频、硬件错误和隔离配置差异 |
| GPU 间路径 | GPU 拓扑矩阵、NVLink/NVSwitch 状态、PCIe 代际/宽度 | 区分 P2P、PCIe、NVLink/NVSwitch 受限路径 |
| CPU/NUMA | GPU/NIC 与 CPU socket/NUMA 对应关系、CPU 绑核 | 排除跨 NUMA 进程或 dataloader/NCCL helper 争用 |
| 网络路径 | NIC/HCA、端口速率、IB/RoCE GID/PKey、rail—switch 映射、端口错误计数 | 验证 GPU-to-NIC 亲和和 fabric 健康 |
| 软件配置 | NCCL/PyTorch 版本、NCCL 环境、容器/驱动 digest | 防止把版本漂移误判为拓扑性能波动 |

### 2.2 不要先“调参”

第一轮必须使用尽可能接近默认且可审计的设置，只显式指定经基础设施确认的通信接口或 HCA。NCCL 文档明确区分系统配置参数与调试参数，并警告调试参数不应长期留在生产脚本中，因为可能造成性能下降、崩溃或挂起。[3]

如果集群采用多个 NIC/rail，`NCCL_CROSS_NIC` 的选择取决于实际 fabric：rail-optimized 网络通常更希望同一 ring/tree 使用同一 NIC，而所有 NIC 汇入同一交换结构时可能适合允许跨 NIC。不要把任一固定取值视为通用优化；应在已确认的网络设计上做单变量 A/B 实验。[3]

## 3. 分阶段 NCCL 带宽与拓扑压测方案

### 3.1 阶段 A：节点内路径确认

先在单节点、1→2→4→全部 GPU 的规模上测试 P2P 和 all-reduce。消息大小需覆盖控制面的小张量、模型梯度 bucket 常见大小以及接近最大 bucket 的大张量；每个点先 warm-up，再进行足以报告中位数、p95、p99 和失败次数的重复采样。不要只取平均值，因为动态 `UpdatePlan` 的 straggler 主要由尾延迟决定。

输出应按“GPU 集合 × 消息大小 × 集体操作 × 算法/协议自动选择 × rank”分组，至少包含 wall-clock、每 rank 时长、`algbw`、`busbw`、GPU 温度/功率和 NCCL 初始化日志。`busbw` 用于比较集体通信折算后的链路使用情况，`algbw` 用于观察应用视角数据率；两者都需在同样的 rank 数、消息大小与 collectives 下横向对比，不能跨不同规模直接比较。

### 3.2 阶段 B：双节点与多节点网络基线

固定每节点 GPU 数后，逐步扩展节点数，并保持一个作业只使用同构的 `LOCAL_WORLD_SIZE`。PyTorch 的 `torchrun` 文档要求每个 GPU 进程独占一张 GPU；它也说明弹性运行目前假定每个节点的本地 worker 数同构。[4] 因此，不要用“某节点 8 GPU、另一节点 4 GPU”的方式测试本方案的异构**微批**；这里的异构发生在每个 rank 的 `micro_batches`，而不是每节点进程数。

建议压测矩阵如下。数值阈值不在文档中预设，因为它们取决于 GPU、网络、消息大小、算法和集群负载；项目应以经过至少三轮复现的**健康基线区间**作为回归门槛。

| 因子 | 最小取值集 | 观测重点 | 一次只改变的变量 |
|---|---|---|---|
| 节点数 | 1、2、目标规模 | 扩展效率、跨节点 p99、错误率 | 节点数 |
| 每节点 rank 数 | 1、半节点、全节点 | GPU-to-NIC/NUMA 亲和、ring 构成 | rank 映射 |
| 消息大小 | 小控制张量、中等梯度 bucket、大 bucket | 延迟拐点、带宽饱和、尾延迟 | 张量字节数 |
| collective | all-reduce、all-gather、reduce-scatter、broadcast | 与 DDP/控制面行为的对应 | 操作类型 |
| 通信路径 | 自动选择；经确认的 NIC/HCA 或 rail 配置 | NIC 选择与跨 rail 冲突 | 单个路径配置 |
| 负载 | 空闲；受控 CPU/I/O 负载；训练共存 | helper 线程/NUMA/数据加载竞争 | 一种负载条件 |
| 训练策略 | 固定累积；2:1；3:1；计划关闭 | 样本吞吐、最终同步等待、收敛正确性 | `UpdatePlan` 配置 |

对于 TCP socket 路径，NCCL 提供 `NCCL_SOCKET_NTHREADS` 和 `NCCL_NSOCKS_PERTHREAD`；提高它们可能增加性能，但会增加 CPU 使用，而且两者乘积有上限。对于 InfiniBand/RoCE，网络超时和重试设置会改变“错误多快浮现”，不应把超时增大当作修复。NCCL 文档给出了 `NCCL_IB_TIMEOUT` 与 `NCCL_IB_RETRY_CNT` 的乘积关系；过高或无限等待会将实际网络错误伪装为长期卡死。[3]

### 3.3 阶段 C：应用级动态 UpdatePlan 压测

微基准通过后，才使用实际 AC-VJEPA 模型、真实数据形状、AMP、目标 EMA 和通信 hook 配置进行训练压测。每个 update 必须写入：`plan_version`、每 rank `micro_batches`、每 rank 有效样本数、global samples、data/forward/backward/AllReduce/optimizer 时长、显存峰值、NCCL collective p50/p95/p99、最终同步等待时间、梯度范数、loss、EMA/学生参数哈希、数据集 commit 与 checkpoint ID。

动态计划的基线对比不应只问“吞吐变快了吗”，还要问：快 rank 的额外微批是否填满了慢 rank 在最终 AllReduce 前的等待窗口；p99 是否更糟；global sample 的增加是否改变了学习率/调度语义；模型更新是否仍和单进程参考一致。建议对同一固定输入窗口运行：固定 1:1 累积、计划 2:1、计划 3:1、禁用计划但保持相同有效样本数四组；每组至少重复三次，随机种子、数据顺序、模型初值、gradient bucket 配置均固定。

应特别检查 `DynamicAccumulationPlanner` 的当前实现边界：它记录 `p95_step_ms`、`p95_data_ms` 和 `p95_allreduce_ms`，但本代码的计划公式实际上只使用 `samples_per_second × target_update_ms`，再除以每微批样本数并取整。因此当前集成测试证明了**计划广播和数学缩放**，但没有证明 p95 延迟已进入分配决策。将 p95、显存余量、成本分桶和网络健康引入 planner 前，应单独做单变量回归测试，避免把“字段存在”误当作“控制器已使用”。

### 3.4 阶段 D：可观测性与诊断配置

在受控实验中开启足够的 PyTorch/NCCL 诊断，但不要把调试配置固化到常规训练。PyTorch 的 ProcessGroupNCCL 变量可启用每 collective timing、flight recorder、timeout dump、desync debug 与 watchdog monitoring；`TORCH_NCCL_DUMP_ON_TIMEOUT` 需要与非零 `TORCH_NCCL_TRACE_BUFFER_SIZE` 配合。[5]

推荐为**故障实验作业**保存如下最小诊断集合：

| 目的 | 诊断信号 | 成功判据 |
|---|---|---|
| collective 时间线 | 每 collective timing、trace buffer、rank 时间戳 | 可定位 p99 来自 data、compute 还是某个 collective |
| desync 定位 | desync debug、计划 digest、rank 的 `plan_version` | 明确第一个未按序进入 collective 的 rank |
| watchdog 处置 | heartbeat、timeout dump、异步错误策略 | 超时后有完整工件且 job 终止/重启，而不是无限占用 GPU |
| 拓扑验证 | NCCL 初始化/拓扑日志、GPU-NIC 映射 | 通信路径与预期 rail/接口一致 |
| 物理健康 | GPU ECC/Xid、温度、功率、NIC 计数器 | 排除硬件/网络错误被误判为训练代码问题 |

## 4. 受控故障恢复实验

### 4.1 恢复语义：停止所有 worker，而不是部分继续

`torchrun` 的固定规模容错和弹性模式都不是“幸存 rank 继续训练”。官方文档说明：worker 失败时，会停止并重启所有 worker；节点离开或加入时，也会停止现有 worker、建立新的 worker group，并用新的 `RANK` 与 `WORLD_SIZE` 启动。[4] 因此前序 `UpdatePlan` 在成员变化后绝不可复用。新的进程组必须从最后一个原子 checkpoint 载入，并在新成员集合上重新收集 telemetry 和广播新计划。

这直接带来两个实现要求。其一，checkpoint 必须绑定模型、优化器、scaler、EMA、global step、数据集 commit、action/preprocess schema、原计划版本和 world-size。其二，更新必须具有提交边界：只有在 optimizer、EMA 与 checkpoint 记录均完成后，才可将 global step 标记为成功。故障发生在 update 中部时，应重放或丢弃该 update，而不是试图把一半 rank 的梯度/优化器状态与另一半拼接。

### 4.2 安全的故障注入阶梯

按照破坏性从低到高的顺序运行。每轮只注入一种故障，并在开始前确认最近 checkpoint、资源隔离、日志目标与人工终止条件。

| 阶梯 | 注入方式（需授权） | 验证点 | 不应做的事 |
|---|---|---|---|
| 0：正常恢复 | 主动中断训练入口，随后重新启动同一 checkpoint | step、模型/EMA hash、数据血缘是否连续 | 不要跳过 checkpoint 完整性验证 |
| 1：受控 worker 退出 | 在测试脚本内、指定 update 边界让单个 worker 自行退出 | 所有 worker 停止；`max_restarts` 内重建；无重复/跳过提交 | 不要在共享节点向任意 PID 发送信号 |
| 2：collective 迟滞 | 在仅测试作业的指定 rank、指定 collective 前插入有界 sleep | watchdog/dump 能定位阻塞和 rank；作业按策略终止/重启 | 不要用无限 sleep 制造不可回收 GPU 占用 |
| 3：rendezvous 可用性 | 在隔离环境演练 rendezvous 服务短暂不可用或 endpoint 配置错误 | 错误可诊断；不会连接到错误作业 | 不要影响其他作业共用的 rendezvous 服务 |
| 4：节点/网络失联 | 由基础设施在测试 VLAN/节点池模拟节点离开或 rail 中断 | group 重建、rank/world-size 变化、重新计划、恢复窗口 | 不要在生产交换机、共享 NIC 或真实机器人网络施加规则 |
| 5：硬件告警路径 | 通过现有监控或厂商批准的演练机制模拟/重放告警 | 训练停止、工件保留、交由硬件值班 | 不要随意 GPU reset、卸载驱动或反复忽略 Xid/ECC |

实验前设置一个**故障预算**，例如每个作业最多一次注入、最多一次重启验证、超时后人工确认，防止多重故障叠加使根因不可判读。恢复成功必须同时满足：启动使用预期 checkpoint hash；新的 `WORLD_SIZE` 与 `RANK` 没有被代码硬编码假定；新的 `UpdatePlan.world_size` 与实际 process group 一致；重启前未确认 update 未被计为已完成；重启后的评估和影子门控仍独立执行。

## 5. 潜在风险与缓解

| 风险 | 典型误判或后果 | 预防与停止规则 |
|---|---|---|
| 拓扑误配 | GPU 使用跨 NUMA/NIC 路径，微基准和训练出现随机长尾 | 先导出拓扑、映射 GPU-NIC、逐节点确认绑核与 rank 映射 |
| 用平均带宽掩盖尾延迟 | 平均 GB/s 高，但 rank 在最终同步长时间等待 | 以 per-rank p95/p99、straggler wait 和 step p99 作为门控 |
| 持久化调试变量 | 调试配置本身引入性能下降、崩溃或 hang | 仅故障实验启用；在实验报告中逐项记录和清理 |
| 无限超时/重试 | 真正 fabric 故障变成长期占用 GPU 的“慢训练” | 为故障实验设置可恢复 watchdog、超时、工件转储与人工上限 |
| 以 worker 单独继续为恢复 | 梯度、optimizer 或 EMA 状态分叉 | 任一 worker 失败时让整个 group 回到最后确认 checkpoint |
| rank 稳定性假设 | 弹性重启后把旧 rank 的数据、计划或设备映射错误复用 | 每次初始化读取环境；绝不硬编码 `RANK`/`WORLD_SIZE` [4] |
| 基准和应用数据形状脱节 | microbenchmark 正常，真实 gradient bucket/AMP 下失败 | 用同一 bucket、batch、AMP、comm hook 和数据读取策略压测 |
| 误把 Gloo 结果当 NCCL 验证 | CPU 测试无法覆盖 CUDA stream、NVLink、IB/RoCE 和 watchdog 行为 | Gloo 仅用作语义回归；GPU NCCL 需独立验收 |
| 故障注入影响他人 | 共享网络/节点/服务的意外中断 | 专用队列、测试 VLAN、变更窗口和人工执行权限 |

## 6. 双进程 Gloo 测试的 2:1 异构微批逻辑

文件 `test_dynamic_nccl_acvjepa_integration.py` 是一个**语义集成测试**，而不是性能基准。它用两个 CPU 进程和 Gloo 后端验证控制面、局部累积、最终同步和副本一致性。生产路径在 CUDA 可用时选择 NCCL；因此该测试无法验证 GPU 拓扑、CUDA/NCCL stream、RDMA/IB/RoCE、NVLink 或 watchdog 行为。

### 6.1 如何得到 2:1 计划

测试在第 60—69 行构造两条确定性 telemetry：rank 0 为 `40 samples/s`，rank 1 为 `17 samples/s`；目标 update 时间为 120 ms，每微批均为 2 个样本。现有 planner 的容量公式是：

```text
capacity_r = samples_per_second_r × target_update_ms / 1000
micro_batches_r = round(capacity_r / samples_per_micro_batch_r)
```

因而：

| rank | 输入吞吐 | 120 ms 容量 | 除以每微批 2 样本 | 取整并裁剪后 `K_r` | 本地有效样本 `n_r` |
|---|---:|---:|---:|---:|---:|
| 0 | 40 samples/s | 4.80 | 2.40 | 2 | 4 |
| 1 | 17 samples/s | 2.04 | 1.02 | 1 | 2 |

因此 `UpdatePlan` 是 **rank 0: rank 1 = 2:1 微批**，不是 2:1 GPU 数、节点数或 batch size。全局有效样本是 `N = 4 + 2 = 6`。`next_update_plan()` 在每个 rank 上先 `all_gather` 数值 telemetry 与本地 batch size，再只让 rank 0 调用 planner，最后通过 `broadcast_update_plan()` 广播同一 JSON 计划，并对计划 SHA-256 进行 `all_gather` 共识校验。任何 world-size、schema、样本数或 digest 不一致都会在 backward 前停止。

需要指出，测试设置了相同的 p95 字段，但当前 planner 尚未使用这些字段计算 `K_r`；本例的 2:1 来自吞吐和 120 ms 目标，不来自 p95 延迟。

### 6.2 为什么 rank 1 可以先进入最终 backward 而不破坏同步

在 `acvjepa_dynamic_update()` 中，第 282—322 行按 `mine.micro_batches` 循环。rank 0 的 `K_0=2`：第一微批在 `ddp.no_sync()` 中反向，因此只累积本地梯度；第二微批不使用 `no_sync()`，触发 DDP 的梯度同步。rank 1 的 `K_1=1`：它的第一微批就是最后一微批，直接进入同步 backward。若 rank 1 更早到达 collective，它会等待；当 rank 0 完成其第二微批并进入相同 DDP collective 后，AllReduce 才完成。

```text
rank 0: micro 1 [no_sync, local grad] ── micro 2 [DDP backward → AllReduce] ── step
rank 1:                               micro 1 [DDP backward → AllReduce] ── step
```

这是同步 DDP 的正确形态：快 rank 用额外计算尽可能填充慢 rank 的等待窗口，但**所有 rank 仍只进行一次同步梯度归约和一次共同的 optimizer step**。若 rank 0 的第一微批也同步，那么 rank 1 并没有对应的同序 collective，会触发 desynchronization；若 rank 1 在 rank 0 尚未同步时自行 `optimizer.step()`，则会导致参数分叉。现有代码两种情况都避免了。

### 6.3 梯度为何仍是 6 个样本的全局均值

两个 rank 的每微批 batch 都是 2，world size `W=2`，global samples `N=6`。计划为每个 rank 设定：

```text
loss_sum_scale = W / N = 2 / 6 = 1 / 3
weighted_loss = local_mean_loss × 2 × 1/3
```

每个本地 `losses.total` 是微批上的均值。乘以 2 把均值转成该微批的梯度和，乘以 `W/N` 补偿 DDP 默认对 rank 梯度做的 `1/W` 平均。rank 0 的两个微批贡献 4 个样本，rank 1 的一个微批贡献 2 个样本；最终 DDP 平均后正好得到六个样本的平均梯度，而非“两个 rank 的平均”。

在第 291—296 行，代码还强制实际 batch 的 `context_video.shape[0]` 必须等于计划 batch size；第 324—325 行检查每 rank 实际消费样本数是否等于 `mine.local_samples`；第 338—343 行让所有 rank 对 `local_seen` 求和并与 `plan.global_samples` 比较。由此，测试输出中的 `global_samples: 6.0` 不是仅由 planner 打印的预测值，而是经历了实际数据流消费和一次 `all_reduce(SUM)` 后的守卫结果。

### 6.4 模型副本一致性验证究竟覆盖什么

第 47 行使用相同的 `torch.manual_seed(12345)`，确保两个进程构造出相同的 AC-VJEPA 初始参数；DDP 包装也会建立同步副本。另一方面，第 31—41 行按 `900 + rank` 初始化各自的数据生成器，因此两 rank 看到的训练数据不同。这一点很重要：若两边使用完全相同的数据，即使梯度同步失效，某些简单测试也可能偶然更新到相近参数；使用不同随机批次更能检验 DDP 的梯度归约。

在 update 后，第 87 行选择学生视觉编码器第一个卷积层的权重张量，所有 rank 用 `dist.all_gather` 收集它，然后第 90 行逐个用 `torch.allclose(..., atol=1e-6)` 比较。通过意味着在不同本地数据、2:1 微批和一次同步 step 后，这个被检查参数在所有 rank 上仍一致。测试还以 `assert int(metrics["global_samples"]) == 6` 复核计划输出。

| 已被证明 | 尚未被证明 | 建议增强 |
|---|---|---|
| 两 rank 可完成同序 collective，且无 hang | NCCL/CUDA/IB/RoCE 行为 | 在真实 GPU 多节点集群复跑同一逻辑 |
| 2:1 计划实际消耗 rank 0=4、rank 1=2 个样本 | 与单进程“6 样本一次全局 batch”的逐参数数值等价 | 固定全部输入，构造单进程参考更新，逐参数比较 |
| 被检查的一个卷积权重在两个 rank 间一致 | 所有学生参数、EMA target 参数、buffers、optimizer state 都一致 | 对完整 `state_dict`、EMA 和 optimizer state 逐张量哈希/比较 |
| 输入在 rank 间不同，避免完全相同梯度的弱测试 | 学习率调度、AMP scaler、通信压缩 hook 的长期正确性 | 加入多 update、AMP、hook、checkpoint/restore 回归 |
| 实际 global samples 与计划相符 | p95 telemetry 影响计划质量 | 将 p95/显存/AllReduce 预测纳入 planner 并建立单元测试 |

更强的回归测试应在捕获同一批次张量后，分别执行：单进程 6 样本参考更新；双进程 2:1 更新；再对每一个学生参数、目标 EMA 参数、optimizer state、global step 和数据哈希逐项比较。在浮点非结合性可接受的容差范围内，两条路径应一致；如果采用 AMP 或通信压缩，则应使用单独的、经过阈值标注的误差预算，而不应复用全精度的 `1e-6` 断言。

## 7. 推荐的验收门槛

压测项目不应预先承诺某一固定 GB/s 或毫秒数，因为这些值与 GPU、互联、网络、消息大小和模型 bucket 强耦合。建议采用“基线区间 + 正确性不可退化”的双门槛：性能门槛由同集群健康基线的中位数/p95/p99 相对区间定义；正确性门槛则是绝对的——计划共识、有效样本数、所有 rank 参数/EMA 一致性、checkpoint 血缘和重启语义必须全部通过。

通过后的常态运行应关闭高开销调试，仅保留必要健康指标。任何拓扑变化、驱动/NCCL/PyTorch 升级、HCA 固件变更、网络 rail 调整、模型 bucket 结构变化或通信 hook 变更，都应重新进入阶段 A—C 的最小回归矩阵，并将结果绑定到模型与基础设施版本。

## 参考资料

[1] [NVIDIA NCCL：Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

[2] [NVIDIA NCCL：Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)。

[3] [NVIDIA NCCL：Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。

[4] [PyTorch 2.13：torchrun（Elastic Launch）](https://docs.pytorch.org/docs/2.13/elastic/run.html)。

[5] [PyTorch 2.13：ProcessGroupNCCL Environment Variables](https://docs.pytorch.org/docs/2.13/torch_nccl_environment_variables.html)。
