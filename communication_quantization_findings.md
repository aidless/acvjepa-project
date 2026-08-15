# 通信优化与边缘量化：公开技术事实

## PyTorch DDP 通信钩子

PyTorch 官方 DDP Communication Hooks 文档说明，通信钩子可覆盖 DDP 默认梯度 all-reduce 通信行为。官方文档包含默认 hooks、PowerSGD hooks、量化/低精度 hooks 以及用于状态管理的接口。工程上这意味着现有 `DDP` 实例可在构造后通过 `register_comm_hook` 逐步试验 fp16/bf16 压缩或 PowerSGD，而不需要改写模型前向与训练损失。

来源：https://docs.pytorch.org/docs/2.13/ddp_comm_hooks.html （访问于 2026-08-15）

## ONNX Runtime TensorRT Execution Provider

ONNX Runtime 官方文档说明 TensorRT EP 可在 NVIDIA GPU 上加速 ONNX 图，并建议同时注册 CUDA EP 作为 TensorRT 不支持节点的回退。文档列出 `trt_fp16_enable`、`trt_int8_enable`、workspace、engine cache 和 timing cache 等配置；TensorRT 引擎/时序缓存可以明显缩短 session 创建时间。文档也提示 Jetson 的包与 JetPack/CUDA/TensorRT 版本需要匹配。

来源：https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html （访问于 2026-08-15）

## Jetson Orin 电源与热约束

NVIDIA Jetson Linux 开发者指南说明，Jetson Orin 软件可见的电源、热与电气管理会共同约束时钟、功耗门控、处理器电源状态和可用 CPU 核等运行旋钮；负载调频与热管理可能相互覆盖。因此，边缘推理的 p99 延迟不能只靠单次引擎 benchmark 推断，必须在目标功耗模式、风扇/散热和并发负载下测量。

来源：https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html （访问于 2026-08-15）

## 具身主动学习

Li 与 Silver（2023）提出具身主动学习范式：智能体通过在线交互选择生成信息量高专家查询的动作，并使用神经网络集成的熵评估候选查询的信息量。该工作表明，不确定性可用于选择高价值交互/查询，而不是随机扩大数据量；但它不构成在真实机器人上无监督、自由探索的安全许可。

来源：https://proceedings.mlr.press/v232/li23a.html （访问于 2026-08-15）

## Isaac Lab 的 Sim-to-Real 支撑能力

Isaac Lab 官方文档将其定位为统一、模块化的机器人学习框架，支持 Isaac Sim 的写实场景/RTX 渲染，也支持 Newton 物理后端；文档列出多种环境、传感器、快速/准确物理、向量化渲染和用于提升鲁棒性与适应性的域随机化。其框架可作为将真实难例的场景、物体、传感器和动力学参数化后批量生成合成轨迹的候选基础设施，但软体接触参数仍必须通过真实记录与实验校准。

来源：https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/index.html （访问于 2026-08-15）

## 弹性分布式恢复

PyTorch 的 `torchrun` 官方容错教程说明：进程/节点发生故障时，弹性启动器可以尝试重启全部进程；应用需要定期保存并在入口加载快照，快照可包含模型、优化器、已完成 epoch 等状态。该机制可减少训练中断损失，但不替代数据生成任务的应用级 lease、产物哈希、幂等 commit 和 dataset manifest 一致性设计。

来源：https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html （访问于 2026-08-15）

## DDP 梯度累积与 NCCL

PyTorch DistributedDataParallel 文档说明，在 `no_sync()` 上下文内反向传播的梯度会累积；退出上下文后的下一次前向/反向传播才进行同步。因此所有 rank 必须采用协调一致的更新边界，避免一部分 rank 进入 collective、另一部分 rank 仍在局部累积。

NVIDIA NCCL 官方概览说明，NCCL 提供拓扑感知的跨 GPU/跨节点集合通信（包括 AllReduce），集合通信的通信处理器之间存在紧密同步；NCCL 可利用 PCIe、NVLink、InfiniBand 或 IP sockets 等互连。负载调整应发生在更新/通信轮次之间，不能在单次集合通信中让某些 rank 静默跳过。

来源：https://docs.pytorch.org/docs/2.13/generated/torch.nn.parallel.DistributedDataParallel.html；https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html （访问于 2026-08-15）
