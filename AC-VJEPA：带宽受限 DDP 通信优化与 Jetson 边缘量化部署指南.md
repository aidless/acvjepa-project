# AC-VJEPA：带宽受限 DDP 通信优化与 Jetson 边缘量化部署指南

## 结论先行

对于现有轻量级 AC-VJEPA，通信优化应遵循**先消除无效同步，再尝试低精度压缩，最后才使用低秩/有损近似**的顺序。最先启用的通常不是“激进压缩”，而是固定输入形状、梯度累积的 `no_sync()`、数据本地化、合理 bucket、计算—通信重叠和 NCCL/网络拓扑调优。若确认作业确实受节点间 all-reduce 带宽限制，再用官方 DDP 通信钩子试验 FP16/BF16 或 PowerSGD，并针对同一验证/闭环安全集做回归比较。[1]

Jetson 部署也应按**FP32 参考 → FP16 静态 TensorRT 引擎 → 选择性 INT8 → 仅在验证充分后全图 INT8**逐级推进。不要将训练、EMA 目标编码、LLM、MPC、几何安全、厂商控制器或硬件急停一起导出为 TensorRT 图；边缘引擎只运行动作条件预测子图，安全与实时控制保持独立。

## 1. DDP 通信为什么会成为瓶颈

DDP 在反向传播中把每个 rank 的梯度通过 all-reduce 聚合。对于视频模型，若每卡 batch 很小、模型前向较短、节点间网络较慢，通信时间可能大于计算时间。首先应从 profiler/训练日志确认以下事实：GPU 是否在等待 all-reduce、NCCL 时间是否占 step 的主要比例、DataLoader 是否也在让 GPU 空转、以及跨节点是否比节点内慢得多。

| 症状 | 更可能的根因 | 首选处理 |
|---|---|---|
| GPU 利用率低，NCCL all-reduce 占比高 | 节点间带宽/拓扑不足，bucket 过碎，过于频繁同步 | 增加梯度累积、调整 bucket、重叠通信与反向、检查 NCCL 网卡/拓扑。 |
| GPU 利用率低但 DataLoader 等待高 | 视频解码、远端小文件、CPU worker 不足 | 节点本地 NVMe 缓存、按 shard 读取、离线特征/解码缓存。 |
| 训练损失不稳定，启用压缩后恶化 | 压缩过强、低秩 rank 不足、量化误差、学习率未重调 | 回退全精度基线；再以 FP16/BF16、保留 error feedback 的 PowerSGD 渐进试验。 |
| rank 失联或 step 偶发极慢 | 网络抖动、NCCL/驱动/网卡配置、负载不均 | 做集合通信基准与故障日志，排除数据加载不均和节点健康问题。 |

## 2. 通信优化策略：从无损到有损的启用阶梯

### 2.1 第一层：不改变梯度数学含义

1. **梯度累积与 `DDP.no_sync()`。** 现有训练脚本已实现：只有在累积边界触发 all-reduce，中间微批不通信。有效批量增加为 `per_rank_batch × world_size × accumulation`，但需相应检查学习率和训练稳定性。
2. **计算—通信重叠。** DDP 默认按 bucket 在反向传播中发起通信；保持静态图、避免动态未使用参数、固定输入形状可以帮助稳定 bucket 和重叠。
3. **数据本地化。** 视频训练常因 I/O 而非网络梯度受限。每个节点预缓存其 episode shards 到 NVMe，避免多 rank 从共享文件系统读取大量小文件。
4. **网络/拓扑调优。** 确认 NCCL 选择了正确高速网卡与 GPU 拓扑；从集群运维规范确定 RDMA、网卡绑定和异步错误处理设置。不要把某个通用环境变量硬编码到所有集群。

### 2.2 第二层：官方低精度通信钩子

PyTorch DDP 通信钩子可替换默认 all-reduce 策略；官方文档包含 FP16、BF16 和 PowerSGD 等 hooks。[1] 本交付的 `ac_vjepa_comm_hooks.py` 已封装三种可选模式，并在 `train_ac_vjepa_ddp.py` 中通过命令行启用。

| 模式 | 适用场景 | 优点 | 风险与验证要求 |
|---|---|---|---|
| `none` | 建立基线、调试一致性 | 精确、最容易归因 | 网络带宽可能限制扩展。 |
| `fp16` | 参数规模中等、网络带宽受限、训练对半精度鲁棒 | 通信数据量约减半，最容易启用 | 梯度小值/不确定性头可能更敏感；与全精度损失和校准对照。 |
| `bf16` | 硬件支持 BF16，且希望更大动态范围 | 低精度通信、较稳的指数范围 | 需要兼容硬件；仍是近似通信。 |
| `powersgd` | 大型、近似低秩梯度矩阵且通信显著受限 | 高压缩潜力；可启用 error feedback/warm start | 需选择 rank、预热步数和最小压缩率；对小模型或小 bucket 可能得不偿失。 |

示例：先在固定验证集上试验 FP16 通信：

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  train_ac_vjepa_ddp.py \
  --manifest /data/ac_vjepa/manifest.jsonl \
  --output /checkpoints/fp16_comm \
  --comm-hook fp16
```

PowerSGD 应放在后面试验，并保留 error feedback：

```bash
... train_ac_vjepa_ddp.py \
  --comm-hook powersgd \
  --power-sgd-rank 1 \
  --power-sgd-start-iter 500
```

由于 PowerSGD 的收益依赖梯度矩阵结构和 bucket 大小，不存在不经测量就普适的 rank。应以“每步 p99 时长、通信占比、收敛速度、事件预测/不确定性校准、闭环规划安全指标”而不是吞吐单指标做选择。

### 2.3 第三层：更激进的策略应谨慎

Top-k 稀疏化、符号/1-bit 压缩、局部 SGD、多步异步参数更新和过期梯度在研究中常见，但它们会改变同步 SGD 的语义，并需要残差/error-feedback、参数服务器一致性或特定收敛假设。对于要学习动作后果和不确定性、且最终服务于安全受限机器人规划的 AC-VJEPA，不建议在没有稳定全精度基线、固定离线回放集和仿真闭环安全评测之前启用这些策略。

若网络极其受限，更可靠的第一步通常是**减少每个训练步需要同步的次数与尺寸**：冻结大骨干、只训练动作预测器/适配器、缩短视频窗口、增大本地累积步，或将模型升级限制在可离线批处理的阶段；不要急于让异步梯度改变学习动态。

## 3. 面向 Jetson AGX Orin 的推理子图划分

### 3.1 应部署与不应部署的组件

| 组件 | 是否导出 ONNX/TensorRT | 原因 |
|---|---|---|
| 视觉状态编码器（部署版） | 是，先 FP16，后评估 INT8 | 是预测子图的主要计算来源。 |
| 动作 tokenizer、GRU/时序预测器、潜在/事件/方差头 | 是，优先静态 ONNX | 与候选 ActionBlock 一起构成需要低延迟的滚动预测。 |
| EMA 目标编码器与训练损失 | 否 | 仅训练期需要。 |
| LLM、任务编排、数据库 | 否 | 不在机器人控制关键路径。 |
| MPC 候选生成/代价组合 | 通常保留 CPU/GPU 原生代码 | 可能频繁变动且需可审计；可在后续单独优化。 |
| 确定性几何、保护区、人/宠物、权限、TTL | **绝不依赖 TensorRT** | 安全规则必须在模型失效、量化误差或超时下仍工作。 |
| 厂商伺服、限幅、急停 | **绝不导出** | 必须在独立控制/安全域运行。 |

`export_ac_vjepa_onnx.py` 已将部署图明确限定为：`context_video + context_proprio + action_blocks → future_latents + log_variance + event_logits`。它使用静态 batch、静态历史窗口、静态动作时域和静态图像尺寸，避免 Jetson 在运行时因动态 shape 触发额外 engine build 或 p99 抖动。

### 3.2 量化阶梯

| 阶段 | 精度设计 | 用途 | 放行标准 |
|---|---|---|---|
| Q0 | PyTorch FP32 参考 | 建立数值与闭环基准 | 输出、事件、校准、p99 都可复现。 |
| Q1 | TensorRT/ORT **FP16** | 第一版边缘部署 | 每个输出与 FP32 在容差内；无错误回退增多；真实边缘端 p99 达标。 |
| Q2 | **混合精度**：视觉编码/归一化/方差头保持 FP16，预测器/动作投影优先试 INT8 | 在不破坏不确定性语义的前提下降低主要计算成本 | 事件 F1、潜在误差、不确定性校准和安全回退率不劣化超过预先设定阈值。 |
| Q3 | 选择性或全图 INT8 | 只有 Q2 仍不满足预算时尝试 | 用目标 Jetson 数据校准；通过所有故障注入、影子模式和 p99 验收。 |

不要一开始就全图 INT8。`log_variance`、LayerNorm、softmax/归一化和风险阈值通常比普通视觉特征更容易因低精度改变校准行为。首个高价值目标不是最大吞吐，而是**在较低延迟下保留“高不确定性会触发保持”的正确排序**。

## 4. 校准数据设计：三输入、目标设备、分层采样

TensorRT PTQ 会用目标域样本执行模型、观察 FP32 激活来建立 INT8 映射；因此校准器必须提供全部模型输入，而不仅是图像。[3]

| 输入 | 校准集必须覆盖 | 不能忽略的因素 |
|---|---|---|
| `context_video` | 目标相机、固定预处理、日/夜光照、反光、遮挡、常见厨房/桌面布局 | 相机曝光、裁剪、颜色空间、帧序和真实模糊。 |
| `context_proprio` | 双臂静止/运动、夹爪开合、门接触、负载变化 | 单位、时间同步、延迟补偿、极限但安全的姿态。 |
| `action_blocks` | 已注册技能的实际限幅动作：接近、抓取、持门、搬运、放置、撤退 | 必须使用最终执行/允许的动作范围，不用自由文本计划。 |
| 输出验证 | 正常、困难但安全的观测与候选 | 分别评估潜在误差、事件、方差/不确定性及回退决策。 |

校准集应包含代表性的困难场景，但不应被罕见损坏数据完全主导，否则动态范围被极端激活拉宽，普通任务精度可能下降。故障注入集应作为**量化后的安全回归测试集**，而不是简单地混入全部校准样本。

## 5. ONNX Runtime TensorRT EP 的实机模式

ONNX Runtime 官方建议将 TensorRT EP 置于高优先级，并同时注册 CUDA EP 以承接 TensorRT 不支持的子图；文档还提供 FP16/INT8、engine cache、timing cache 和 workspace 选项。[2]

```python
providers = [
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_int8_enable": False,  # 仅在已完成目标域校准/验证后开启
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/var/cache/ac_vjepa/trt",
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": "/var/cache/ac_vjepa/trt",
    }),
    ("CUDAExecutionProvider", {"device_id": 0}),
]
```

`edge_ac_vjepa_ort.py` 实现了这一模式，并在返回模型结果前检查状态年龄、deadline 和非有限输出。任何 `deadline_miss`、`invalid_output` 或 `runtime_error` 都必须被外部安全协调器转换为 `LOCAL_HOLD`，之后才可能进入 LLM 托管；ORT/TensorRT 本身不拥有机器人控制权。

引擎缓存必须与 ONNX 文件哈希、静态输入 shape、JetPack/CUDA/TensorRT/ORT 版本、GPU 架构和量化校准版本联锁。软件/驱动升级后应重建并重新验证 engine，而不是复用旧缓存。

## 6. 端到端实时部署流程

1. 在训练服务器上冻结部署配置，导出 FP32 静态 ONNX，并保存 schema/预处理/校准元数据。
2. 在目标 Jetson、目标 JetPack/TensorRT/ORT 版本上构建 FP16 engine；预热并缓存，避免首次控制周期才建引擎。
3. 以完全相同的记录传感器窗口比较 PyTorch FP32、ORT CUDA、ORT TensorRT FP16 的输出与事件/不确定性决策。
4. 若 FP16 仍不能满足预算，构建包含真实三输入的 PTQ INT8 校准器；先仅量化动作预测器/投影层，再评估是否扩展。
5. 对每个精度版本运行噪声、掉帧、NaN、过期状态、高不确定性和 deadline 故障注入。模型速度提升不得降低 `LOCAL_HOLD` 的触发正确性。
6. 先在影子模式采集目标 Jetson 的 p50/p95/p99；再以低速、短 TTL、低风险原子技能受限自主运行。
7. 低层伺服与硬件安全环始终独立；边缘推理只替换最新未过期的短期规划窗口。

## 7. 验收指标

| 维度 | 必测指标 | 不能接受的现象 |
|---|---|---|
| 通信 | step p99、all-reduce 占比、压缩后收敛、rank 错误/重试 | 仅吞吐提高但事件预测、校准或闭环安全明显变差。 |
| 数值 | FP32/FP16/INT8 的潜在误差、事件 F1、输出 NaN | 风险头或方差头在低精度下失真却未被发现。 |
| 安全 | 高不确定性/超时触发率、漏触发、误触发、保持时延 | 为了达帧率而继续使用过期或低置信模型输出。 |
| 实时 | Jetson 上端到端 p50/p95/p99、engine warm-up、热状态功耗下抖动 | 只报告平均单次模型推理时间，不测相机/排队/安全门。 |
| 可运维性 | engine cache 可重建、模型与 schema 版本可追溯、回滚可用 | JetPack 或模型版本变更后仍盲目加载旧 engine。 |

## 参考资料

[1]: [PyTorch, *DDP Communication Hooks*](https://docs.pytorch.org/docs/2.13/ddp_comm_hooks.html)

[2]: [ONNX Runtime, *TensorRT Execution Provider*](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

[3]: [PyTorch TensorRT, *Post Training Quantization*](https://docs.pytorch.org/TensorRT/user_guide/ptq.html)
