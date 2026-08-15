# 正在把 JEPA 用于 AI Agent 的开源项目与研究团队

**检索截至：2026 年 8 月 14 日。** 本清单严格区分三种情况：第一类是**JEPA 直接进入行动、规划或控制闭环**的开源项目；第二类是**世界模型与物理规划研究平台**；第三类是有价值的**架构/组件库或尚未确认稳定代码的研究线索**。它不把所有冠以“world model”的项目都算作 JEPA，也不把纯视觉表征项目误称为 Agent。

## 结论先行

目前确实已经出现了一批将 JEPA 用于 Agent 的项目，但生态仍处于**早期研究向可复现原型过渡**的阶段。最成熟、最直接的开源起点是 Meta FAIR 的 **V-JEPA 2-AC**：它明确把 V-JEPA 2 后训练为潜在的**动作条件世界模型**，并用于图像目标条件的机器人操控规划。[1] 其余多数项目要么在仿真基准中验证策略/规划，要么是极新的学术代码，尚不能等同于通用、可靠、可商业部署的智能体平台。

> 当前 JEPA Agent 生态的主战场是**具身智能**——机器人操控、视觉导航、自动驾驶与物理规划；不是浏览器、办公流或纯文本工具 Agent。对于后者，JEPA 还没有形成成熟的主流技术栈。

## 第一梯队：已将 JEPA 接入行动或规划闭环的开源项目

| 项目与团队 | JEPA 如何进入 Agent 闭环 | 已公开的任务与证据 | 开源成熟度与限制 |
|---|---|---|---|
| **V-JEPA 2 / V-JEPA 2-AC — Meta FAIR** | 先以自监督视频特征预测训练 V-JEPA 2；再用少量机器人轨迹后训练为动作条件的潜在世界模型 V-JEPA 2-AC，以视觉目标为条件规划动作。 | 官方 PyTorch 仓库同时发布 V-JEPA 2、V-JEPA 2-AC 与 V-JEPA 2.1；项目说明其可部署到 Franka 机械臂，在新环境完成 reaching、grasping、pick-and-place 等任务。[1] [2] | **最值得优先评估。** 官方代码、模型与机器人规划设置均已公开，但任务范围仍是基础操控，不能直接视作通用机器人 Agent。 |
| **VLA-JEPA — 研究团队 / `ginwind`** | 将 Qwen3-VL-2B 的视觉语言行动策略与 V-JEPA 2 编码器结合；通过 JEPA 风格的潜在世界模型学习状态转移，增强 VLA 的动作预测。 | 公开训练代码、权重链接、LIBERO、LIBERO-Plus 和 SimplerEnv 评估；支持在 LeRobot 格式机器人数据与人类视频上继续训练。[3] [4] | **最直接的“LLM/VLA + JEPA”融合原型。** 代码可运行，但属于论文实现，系统依赖和数据准备较重。 |
| **ACT-JEPA — Vujinovic & Kovacevic / `act-jepa`** | 把模仿学习的可执行动作生成与 JEPA 的未来潜在状态预测共同训练；使用学到的表征支持动作重建和 rollout 评估。 | 仓库提供 Push-T、MetaWorld、ManiSkill 的训练配置、数据接口、评估与策略 rollout 代码。[5] [6] | **适合复现与理解“策略表示 + 世界模型”的最小闭环。** 仓库较新、提交历史短，工程成熟度较低。 |
| **Demo-JEPA — `banban3forever`** | 用联合嵌入预测把源机器人视觉示范映射为目标本体兼容的未来潜在轨迹，服务于一次性跨本体模仿。 | 开源仓库含训练、评估、数据采集脚本，集成 RLBench 与 Diffusion Policy 依赖。[7] [8] | **研究型跨本体模仿原型。** 适合研究 demo-to-action，离真实生产机器人系统仍有距离。 |
| **Auto-JEPA — `NoctYang`** | 在轨迹潜在空间预测连续的未来驾驶意图，再以轨迹记忆检索、场景评分与可行驶区域过滤作决策。 | 公开代码与 NAVSIM 评估入口；仓库明确写明仍在从研究工作区迁移、整理为可复现发布。[9] [10] | **自动驾驶方向的早期代码。** 有清晰 Agent 语义，但目前不宜视为完整可复现实车栈。 |

## 第二梯队：用于物理规划与控制研究的 JEPA 世界模型平台

| 项目与团队 | 实际用途 | 为什么值得关注 | 主要边界 |
|---|---|---|---|
| **JEPA-WMs — Meta AI Research / FAIR** | 提供 JEPA 物理世界模型、数据、权重与规划评估；环境包括 DROID、RoboCasa、MetaWorld、Push-T、PointMaze、Wall 及 Franka 轨迹。仓库包含世界模型训练、在线规划评估、仿真规划和可视化结构。[11] [12] | 它是目前较完整的**JEPA 世界模型研究工作台**，可以系统比较 JEPA-WM、DINO-WM 与固定 V-JEPA 2-AC 基线。 | 开源许可证为 **CC-BY-NC 4.0**；文档主轴是研究训练与评估，而非端到端商用 Agent 框架。 |
| **C-JEPA — Galilai Group 等** | 用对象级遮挡替代补丁级遮挡，让模型从对象间关系中预测状态；代码报告在 Agent control tasks 上以远少于补丁世界模型的潜在特征进行规划。[13] [14] | 对“对象、接触、关系与反事实”的表示非常有启发，适合研究什么时候世界模型真的需要因果结构。 | 主要是受控控制与规划实验；不是通用 VLA 或实体机器人部署栈。 |
| **EB-JEPA — Meta AI Research / FAIR** | 一个能量式 JEPA 的开源库，包含图像、视频、动作条件视频 JEPA 示例，以及 JEPA 模型上的规划示例。[15] [16] | 若开发者要自己设计动作条件预测器、能量函数或规划代价，它是很好的底层实验库。 | **它不是开箱即用 Agent。** 需要自行补齐状态接口、控制器、执行、验证与安全层。 |

## 已出现的前沿研究团队：强相关，但不宜误称为稳定开源项目

| 研究方向 | 采用的结构 | 目前公开状态 |
|---|---|---|
| **JEPA-WAM — Lin 等** | 在预训练 V-JEPA 表示空间中，通过共享预测器将潜在状态转移与连续动作生成耦合；论文报告 LIBERO-Plus、RoboTwin 2.0 和真实双臂操控实验。[17] | 2026 年 8 月 10 日刚发布的预印本；论文给出项目主页线索，但本次核验未确认稳定的独立开源仓库。应列为**重点跟踪对象**，不应与成熟代码库等量齐观。 |
| **PiJEPA — Chahe & Zhou** | 先以 Octo 通才策略给出语言条件行动先验，再以 JEPA 世界模型做 MPPI 潜在空间规划；论文报告真实世界视觉导航实验。[18] | 明确展示了“语言模型/策略 + JEPA 世界模型 + MPC”的 Agent 组合方式；本次未确认官方公开代码。 |
| **AMI Labs — Yann LeCun 团队** | 公开定位是发展动作条件世界模型和先进机器智能，而不是单纯聊天模型。 | 它是最受关注的产业研究团队之一，但截至本次核验，不应把它列为可用的开源 Agent 项目；公开产品、代码与独立技术评测仍有限。 |

## 对开发者的实际选择建议

如果目标是快速验证“JEPA 是否能改善视觉 Agent”，应从 **V-JEPA 2-AC** 开始，因为它提供了最直接的动作条件建模和机器人规划路径。若目标是把大模型的任务理解能力和世界模型结合，**VLA-JEPA** 是最贴近这一架构的开源参照：语言/视觉行动策略承担任务语义与动作输出，JEPA 模块承担潜在状态预测。

如果目标是做研究而不是交付机器人产品，**JEPA-WMs** 和 **EB-JEPA** 更合适。前者适于进行环境、数据规模、预测深度与规划器的系统消融；后者更适合从底层搭建动作条件 JEPA 与代价/能量式规划。若研究对象是关系推理或操作中接触关系，**C-JEPA** 的对象级表征值得优先阅读。

对多数软件 Agent（网页操作、数据分析、文档流转）而言，直接引入 JEPA 的性价比目前并不高。更实用的第一步仍是显式状态机、工具后验证、结构化记忆、约束求解和失败恢复。JEPA 的价值会在 Agent 必须理解连续视觉环境、预测行动后果、承受部分可观测性和长时程物理交互时明显上升。

## 判断一个项目是否“真的在做 JEPA Agent”的快速检查表

| 问题 | 通过标准 |
|---|---|
| 是否明确采用 JEPA / joint-embedding / latent feature prediction？ | 论文或仓库描述模型在嵌入空间预测未来状态，而不是只使用普通视觉编码器。 |
| 行动是否进入预测模型？ | 动作是状态转移的条件，或至少世界模型实际被规划器用于选择动作。 |
| 是否有闭环评测？ | 提供 robot rollout、导航/驾驶闭环、MPC 规划或可执行 policy，而非仅做分类、VQA 或表征线性探针。 |
| 是否有真实可复现材料？ | 至少有代码、配置、权重或评测说明；若仅论文或展示视频，应标为研究线索。 |
| 是否区分仿真和实机？ | 仿真成功不能自动外推为真实世界部署；需核查传感器、控制、校准、故障恢复与安全策略。 |

## 参考资料

[1]: [Meta FAIR, V-JEPA 2 官方开源仓库](https://github.com/facebookresearch/vjepa2)

[2]: [Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985](https://arxiv.org/abs/2506.09985)

[3]: [VLA-JEPA 官方开源仓库](https://github.com/ginwind/VLA-JEPA)

[4]: [Sun et al., *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model*, arXiv:2602.10098](https://arxiv.org/abs/2602.10098)

[5]: [ACT-JEPA 官方开源仓库](https://github.com/act-jepa/act-jepa)

[6]: [Vujinovic & Kovacevic, *ACT-JEPA*, arXiv:2501.14622](https://arxiv.org/abs/2501.14622)

[7]: [Demo-JEPA 官方开源仓库](https://github.com/banban3forever/Demo-JEPA)

[8]: [*Demo-JEPA: Joint-Embedding Predictive Architecture for One-shot Cross-Embodiment Imitation*, arXiv:2605.20811](https://arxiv.org/abs/2605.20811)

[9]: [Auto-JEPA 官方开源仓库](https://github.com/NoctYang/Auto-JEPA)

[10]: [Yang et al., *Auto-JEPA: A Latent World Model of Continuous Intent for End-to-End Autonomous Driving*, arXiv:2607.29031](https://arxiv.org/abs/2607.29031)

[11]: [Meta FAIR, JEPA-WMs 官方开源仓库](https://github.com/facebookresearch/jepa-wms)

[12]: [Terver et al., *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?*, arXiv:2512.24497](https://arxiv.org/abs/2512.24497)

[13]: [C-JEPA 官方开源仓库](https://github.com/galilai-group/cjepa)

[14]: [Nam et al., *Causal-JEPA: Learning World Models through Object-Level Latent Interventions*, arXiv:2602.11389](https://arxiv.org/abs/2602.11389)

[15]: [Meta FAIR, EB-JEPA 官方开源库](https://github.com/facebookresearch/eb_jepa)

[16]: [Terver et al., *A lightweight library for energy-based joint-embedding predictive architectures*, arXiv:2602.03604](https://arxiv.org/abs/2602.03604)

[17]: [Lin et al., *JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling*, arXiv:2608.09381](https://arxiv.org/abs/2608.09381)

[18]: [Chahe & Zhou, *Policy-Guided World Model Planning for Language-Conditioned Visual Navigation*, arXiv:2603.25981](https://arxiv.org/abs/2603.25981)
