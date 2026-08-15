# 影子退化 RCA 与软体点云—视频对合成：实施说明

## 1. 影子模式性能退化的自动根因分析

影子模式中发现“候选性能变差”后，CI/CD 不应仅输出一个失败布尔值。应将同一 `correlation_id` 下的基线、候选、传感器/标定、模型/引擎、延迟、事后状态残差和安全事件组成可查询证据包；再依据分层规则产出**可能原因、证据、保守建议和 CI/CD 动作**。输出是可审计假设，不是证明因果，更不允许自动调低安全阈值或直接重新发布。

`shadow_degradation_rca.py` 提供一个离线、确定性规则实现。它按以下优先级识别问题：

| 观测模式 | 推测原因 | 自动建议 | CI/CD 动作 |
|---|---|---|---|
| 候选延迟相对基线回归，或热保护激活 | Edge runtime、队列、engine、功耗或热稳定性问题 | 分解 p99，核验引擎、shape、缓存与热稳态 | 暂停扩大，创建边缘性能工单。 |
| 非有限/无效输出 | ONNX I/O、TensorRT/INT8、校准、版本兼容问题 | 隔离 engine，做 FP32/FP16/INT8 同窗口数值比对 | 隔离候选并重建/验证引擎。 |
| 时间偏差或图像质量显著下降 | 传感器同步、标定或预处理漂移 | 校验时钟、相机标定、预处理版本 | 停止将该问题误标为新物理。 |
| 柔性物体残差显著高于其他对象 | 接触/形变物理缺口或相应数据覆盖不足 | 审核难例、批准物理先验、生成有界反事实、adapter 微调 | 创建软体数据与 Sim-to-Real 工单。 |
| 高残差窗口不对应更高不确定性，或风险发生却未请求保持 | 不确定性校准失真 | 增加 error–uncertainty 排序/校准损失，重跑压力集 | 阻断灰度扩大。 |
| 候选整体残差变差但无明显系统/物理分层 | checkpoint、动作归一化、训练清单或容量回归 | 对比 manifest、优化器/EMA、动作 schema，做消融 | 创建训练回归工单。 |

影子部署通常让新模型处理真实镜像输入但不参与实际输出；其价值包括验证真实服务链路和延迟，而不干扰生产行为。[1] 在机器人中，影子模型的任何结论仍只能进入发布控制面，不能写入轨迹队列或机器人驱动器。

## 2. 诊断接入 CI/CD

```text
shadow telemetry + 事后观测
  → correlation-id join
  → 覆盖/完整性检查
  → rca.analyze(records)
  → RCAReport JSON + Markdown/Issue
  → 安全策略：pause / rollback / keep shadow / schedule data task
  → 人工审核或预定义发布策略决定后续训练与测试
```

运行示例：

```bash
python3 shadow_degradation_rca.py > rca_report.json
```

在真实流水线中，应将报告绑定候选模型、基线模型、数据 manifest、标定、edge engine、JetPack/TensorRT/ORT 版本与时间区间。若可用证据少于版本化样本门槛，模块会返回 `insufficient_evidence`，只建议保持影子模式扩充覆盖。

## 3. 点云—视频对数据契约

面向 AC-VJEPA 的每个合成 episode 不只应保存图像。训练需要跨模态、跨时间对齐的状态与动作信息：

| 字段 | 形状/单位 | 说明 |
|---|---|---|
| `rgb_video` | `[T,H,W,3]`，`uint8` | 同步 RGB 帧序列。 |
| `depth_video_m` | `[T,H,W]`，米 | 深度与 RGB 同时间戳/标定。 |
| `point_cloud_xyz` | `[T,N,3]`，机器人基坐标系 | 由深度和相机内外参反投影；`N` 固定。 |
| `point_cloud_rgb` | `[T,N,3]` | 与点一一对应的颜色。 |
| `point_mask` | `[T,N]` | 指示有效点，避免把 zero padding 当成物体。 |
| `executed_actions` | `[T,A]` | 实际执行而非原始请求动作。 |
| `proprio` / `contacts` | `[T,P]` / `[T,C]` | 本体与接触状态。 |
| `timestamps_ns` | `[T]` | 单调时间轴，验证对齐。 |
| `metadata.json` | JSON | job、seed、物理、相机、版本、质量报告、产物哈希。 |

## 4. 数据生成流程

```text
已审核 CandidateEpisode + 已批准 SoftObjectPrior
  → sim2real_hard_example_compiler.py 生成可复放 SimJob
  → IsaacLab/RoboCasa 适配器：场景 reset、动作回放、有界扰动、物理/视觉/传感器随机化
  → 记录 RGB-D、实际动作、接触、本体、时间戳、相机标定
  → backproject_depth() 生成 robot-frame 固定大小点云
  → schema / 时间 / 深度 / 接触 / 数值质量门控
  → 输出 episode.npz + metadata.json + manifest
  → train / validation / stress 分区与版本化训练
```

`sim2real_pointcloud_video_pipeline.py` 已实现后半段：数据契约、RGB-D 反投影、固定点数掩码、质量门控、NPZ/JSON 产物和 JSONL manifest。它也提供 `SyntheticDeformableBackend` 用于**契约测试**，并明确标注其并不具备物理有效性。真实数据生成应在固定版本的 Isaac Lab/Isaac Sim 或 RoboCasa 等适配器中完成，返回同步 RGB-D、接触和实际动作。Isaac Lab 的官方文档说明其提供模块化任务、传感器、物理/渲染后端与域随机化能力，适合作为此适配层的候选设施。[2]

运行契约演示：

```bash
python3 sim2real_pointcloud_video_pipeline.py \
  --demo --output ./soft_pair_contract_demo --max-points 1024
```

对真实仿真清单，应先用 `sim2real_hard_example_compiler.py` 生成经审核的 `SimJob` JSONL，再在 `IsaacLabAdapter` 中实现 `rollout()`。适配器必须把实际的仿真器版本、场景/资产版本、相机内外参、物理参数、动作 trace、接触与随机 seed 返回给写入器；不能只输出无来源的视频。

## 5. 质量与 Sim-to-Real 门控

合成 episode 的 `accepted=true` 只表示满足基本数据契约，不代表能够代替实机数据。训练前仍需执行：真实/仿真特征分层、接触序列与事件对齐、物体/相机/场景 split 隔离、软物体参数覆盖审查和独立压力集评测。合成数据应围绕真实难例补充材质、形变、光照和动作扰动，而不是压倒真实触觉、力和接触证据。

最终的候选模型仍要经过 replay、真实软体保留集、传感器故障注入、仿真闭环、Jetson 性能、影子模式和受限 canary。持续的高不确定性说明数据/模型仍不充分，正确操作是保持保守策略并补充经审核示教，而不是通过扩大合成规模直接解除安全保持。

## 参考资料

[1]: [JFrog ML, *Shadow deployment vs. canary release of machine learning models*](https://www.qwak.com/post/shadow-deployment-vs-canary-release-of-machine-learning-models)

[2]: [NVIDIA, *Isaac Lab Documentation*](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/index.html)

[3]: [Li & Silver, *Embodied Active Learning of Relational State Abstractions for Bilevel Planning*, CoLLAs 2023](https://proceedings.mlr.press/v232/li23a.html)
