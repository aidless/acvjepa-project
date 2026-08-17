# JEPA 应用于 AI Agent：项目核验记录

## 纳入分级标准

| 级别 | 定义 |
|---|---|
| A：直接 JEPA + 行动/规划 | 明确使用 JEPA 或 V-JEPA 表征，并将动作条件预测、规划或控制作为实验/代码的一部分。 |
| B：JEPA 世界模型研究平台 | 明确使用 JEPA 进行物理环境预测与规划研究，提供代码/权重，但主要是研究基准或模拟/机器人环境，尚非通用可部署 Agent。 |
| C：JEPA 使能层 | 提供 JEPA 训练、表征或组件代码，却未实现 Agent 行动闭环。 |
| 排除 | 仅使用“world model”营销词、没有 JEPA 证据，或只提供概念/演示而无可核验实现。 |

## 已核验的官方开源线索

| 项目/团队 | 初步分级 | 已确认事实 | 暂定判断 |
|---|---|---|---|
| Meta FAIR / `facebookresearch/vjepa2` | A | 官方仓库公开 V-JEPA 2、V-JEPA 2-AC、V-JEPA 2.1 的 PyTorch 代码；README 称 V-JEPA 2-AC 是从 V-JEPA 2 后训练得到的潜在动作条件世界模型，使用少量机器人轨迹数据，可处理无环境特定数据收集/任务特定训练/校准的机器人操控任务。 | 当前最直接、最成熟的“JEPA → 实际机器人规划”开源起点。 |
| Meta FAIR / `facebookresearch/jepa-wms` | B | 仓库公开代码、数据和权重，对应论文 *What drives success in physical planning with Joint-Embedding Predictive World Models?*；目录含数据、训练、评估和规划相关实验结构。 | 看起来是 JEPA 物理规划研究/基准平台，需继续从 README/论文确认具体环境与可部署范围。 |

## 核验来源

1. https://github.com/facebookresearch/vjepa2
2. https://github.com/facebookresearch/jepa-wms
3. https://arxiv.org/abs/2506.09985

| Demo-JEPA / `banban3forever/Demo-JEPA` | A（研究原型） | 公开仓库含训练、评估、数据采集脚本与 RLBench、Diffusion Policy 子模块；项目定位为一次性跨本体模仿。 | 明确把视觉示范转为目标本体可执行的未来潜在轨迹/策略，面向机器人模仿；开源成熟度仍明显低于 Meta 的主仓库。 |
| ACT-JEPA / `act-jepa/act-jepa` | A（研究原型） | 官方实现公开；README 称模型同时用 IL 学习可执行动作、用 JEPA 预测未来潜在状态；提供 Push-T、MetaWorld、ManiSkill 环境训练、rollout evaluation 与实验数据说明。 | 这是直接把 JEPA 接入策略表示与控制评估的可运行研究代码，但仓库刚发布、仅有很少提交，需将复现风险标为高。 |

## 新增来源

4. https://github.com/banban3forever/Demo-JEPA
5. https://github.com/act-jepa/act-jepa
6. https://arxiv.org/abs/2605.20811
7. https://arxiv.org/abs/2501.14622
| VLA-JEPA / `ginwind/VLA-JEPA` | A（直接 JEPA + VLA） | 开源仓库明确以“视觉语言动作模型 + 潜在世界模型”为定位；依赖 Qwen3-VL-2B 与 V-JEPA 2 encoder；提供训练、LIBERO、LIBERO-Plus、SimplerEnv 评估及自定义 LeRobot 数据训练说明。 | 当前最直接的“LLM/VLA + JEPA”融合开源路线；包括部署目录和 Hugging Face 权重链接，但项目仍处于研究代码成熟度。 |
| JEPA-WAM / Lin et al. | A（最新研究前沿，代码状态待验证） | 2026-08-10 arXiv 论文：在预训练 V-JEPA 表示空间构建潜在世界动作模型，以共享预测器耦合潜在转移预测和连续动作生成；报告 LIBERO-Plus、RoboTwin 2.0 与真实双臂操作实验。 | 技术上高度相关、且直接面向 VLA policy，但发表时间极新；论文摘要提供项目主页线索，未在本次核验中确认稳定独立代码仓库，因此不应与已可复现仓库等量齐观。 |

## 新增来源

8. https://github.com/ginwind/VLA-JEPA/
9. https://arxiv.org/abs/2602.10098
10. https://arxiv.org/abs/2608.09381

### JEPA-WMs 深入核验

- README 明确写明其为 Meta AI Research / FAIR 的 PyTorch 实现，提供 JEPA-WM、DINO-WM 与 V-JEPA-2-AC 基线的预训练模型；支持 DROID、RoboCasa、MetaWorld、Push-T、PointMaze、Wall 等环境。
- 项目数据集包含 RoboCasa 厨房操作、Franka 轨迹、42 个 MetaWorld 任务及导航/操控环境轨迹。
- 训练脚本会按频率自动启动 planning evaluations；仓库含 `simu_env_planning`、`online_plan_evals`、`plan_common` 与世界模型结构目录，说明它是完整的物理规划研究代码，而非仅提供视觉表征权重。
- 重要限制：项目许可证为 **CC-BY-NC 4.0**，并且公开文档聚焦仿真环境和研究评估；不应把它描述为已可商业部署的通用 Agent 框架。

来源： https://github.com/facebookresearch/jepa-wms （README，访问于 2026-08-14）；论文 https://arxiv.org/abs/2512.24497
| C-JEPA / `galilai-group/cjepa` | B（直接 JEPA 世界模型 + 控制研究） | 开源项目提出对象级掩码的 Causal-JEPA；README 表述其在 Agent control tasks 上以显著更少的潜在特征实现可比规划，并提供代码、数据说明和检查点。 | 适合研究“对象交互与反事实表示如何支持规划”；证据指向受控研究环境，不宜宣传为通用机器人平台。 |
| EB-JEPA / `facebookresearch/eb_jepa` | C（Agent 使能库） | Meta FAIR 的 Apache-2.0 开源库，涵盖 Image JEPA、Video JEPA、Action-Conditioned Video JEPA 及 JEPA-based planning 示例。 | 对开发者很有价值，但它是架构/示例库而非开箱即用 Agent；可用来构建自己的动作条件预测与规划模块。 |

新增来源：
11. https://github.com/galilai-group/cjepa
12. https://arxiv.org/abs/2602.11389
13. https://github.com/facebookresearch/eb_jepa
14. https://arxiv.org/abs/2602.03604
| Auto-JEPA / `NoctYang/Auto-JEPA` | A（自动驾驶研究原型） | 项目以联合嵌入预测学习连续未来驾驶意图；将轨迹潜在检索、场景条件候选评分与可行驶区域过滤接入端到端驾驶决策。 | 代码已公开但仓库明确仍在整理可复现发布，当前不宜作为生产级项目推荐。 |
| PiJEPA / Chahe & Zhou | B（研究团队，未核验开源代码） | 论文采用 Octo-based 通才策略形成动作先验，并以 JEPA 世界模型在相同冻结视觉表征空间预测未来潜在状态，用 MPPI 做语言条件视觉导航；报告真实导航实验。 | 是很清楚的“策略 + JEPA 世界模型 + MPC”范式，但本次未确认代码仓库，故按研究线索而非开源项目列出。 |

新增来源：
15. https://github.com/NoctYang/Auto-JEPA
16. https://arxiv.org/abs/2607.29031
17. https://arxiv.org/abs/2603.25981

## WAM 生态学习批次补录（2026-08-15，详见 `WAM生态学习报告_2026-08-15.md`）

> ⚠️ **JEPA-WAM 同名论文冲突（重要，引用必须带 arXiv 号）**：
> - **arXiv:2608.09381**（Lin et al.，本清单原「JEPA-WAM / Lin et al.」条目）——V-JEPA 空间共享预测器，LIBERO-Plus 79.2%，π0.5 实例化 86.3%；
> - **arXiv:2608.10780v2**（Motus WAM + Stage-JEPA）——冻结 V-JEPA2 预测「下一任务阶段」潜在目标，RoboTwin 2.0 90.25%，成功步数 -5.97%；**无公开仓库/权重**；「零样本部署」声称未见于摘要（待核对全文）。

| 项目/团队 | 初步分级 | 已确认事实 | 暂定判断 |
|---|---|---|---|
| INTACT-JEPA / `zju3dv/INTACT-JEPA` | A（免搜索世界模型控制器） | 浙大/清华 AIR/InSpatio/RoboParty Lab；arXiv:2607.26056；MIT；151★；72 个受控 checkpoint + REPRODUCIBILITY.md 契约；1 epoch Direct 95.33%；四接口受控对照（Direct/Pure-CEM/Actor-CEM/Guarded-A）。 | 「意图→动作律同构、零搜索读出」——对本项目 H-T2/H-R1 与 M3 评测设计最直接；发布纪律（72 checkpoint）是预注册样板。 |
| LeWorldModel / `lucas-maes/le-wm` | B（端到端 JEPA 研究平台） | arXiv:2603.19312；MIT；4308★；~15M 参数、两损失（投影头正则防坍塌）、单卡从头端到端；stable-worldmodel CEM/MPC 栈。 | 「从头端到端轻量」= 冻结骨干路线的 A/B 对照臂；CEM 栈可作 M3 A 组基线参考实现。 |
| W2-VLA / `yyyyu120/W2-VLA` | A（研究原型，过新） | arXiv:2608.05369；21★；LICENSE NOASSERTION（待核对）；冻结 V-JEPA2.1 ViT-L/384 腕部编码器 + 任务条件未来腕部潜变量 + DiT 头；W2-CoT 离线进度标注。 | 仅作 H-R1（接口 tokens 注入）与金字塔第 4 层标注管线参考；成熟度低，不宜引用为可复现基线。 |
| worldmodels_ros2 / `rsasaki0109/worldmodels_ros2` | B（集成层） | 9★；Apache-2.0；从 ROS 2 运行现有世界模型（运行时/适配器/基准/可视化）；图像目标检索动力学规划 + 自校准 surprise 监控。 | 吸收模式：无训练检索动力学基线（M3 A 组参考）、self-calibrated surprise 阈值（§4.2 自动化标注第一道闸）、Honest scope 声明文化。 |

**待核验项（经云端 GitHub/arXiv 实抓核验，2026-08-16）**：
- ✅ **W2-VLA LICENSE=MIT**（StarVLA Team，GitHub 原始文件实核）——原「NOASSERTION（待核对）」解除；
- ✅ **JEPA-WAM 10780v2 摘要实核：无「zero-shot」声称**——摘要只提「frozen V-JEPA2 encoder + Stage-JEPA 预测下一阶段的潜变量目标，RoboTwin 2.0 50 任务 90.25%、成功 rollout 步数 −5.97%」；
- ⚠️ **VLA-JEPA 仓库根无 LICENSE 文件**（GitHub API license=null；README 上的 Apache-2.0 徽章与 ECCV 2026 接收状态只能一手确认）；
- ⚠️ **vla-eval 榜单口径**：GitHub API 确认 README 含 leaderboard 部分；「README 2456 模型 vs 论文 657 结果」为学习批次当日摘录，榜单页复核待后续。
