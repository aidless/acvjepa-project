# AC-VJEPA: From "Is LLM a dead end?" to a runnable action-conditioned world-model stack

围绕 Yann LeCun 对「大语言模型是死胡同」这一批评展开的**研究 + 工程**双阶段项目：先把路线之争转化为可证伪命题，再把它落成一套可运行的 **动作条件 V-JEPA（AC-VJEPA）** 工程体系——训练、分布式、混沌工程、监控、数据装配与评测预注册俱全。

- **许可证**：MIT（见 `LICENSE`）。官方 V-JEPA 2 权重不随仓库分发，需自行从 Hugging Face 下载并遵守 Meta 条款（见下文「权重」）。
- **验证基线**：59/59 全绿，`.\verify_all.ps1` 一键可重跑（见 `VERIFY_RESULTS.md`）。
- **语言**：工程手册与研究报告以中文为主，代码/docstring 英文。

## 项目是什么

1. **阶段一 · 研究与评估（2026-08-14）**：事实核验「LeCun 批评 LLM / JEPA 路线 / AMI Labs」叙事，产出审慎评估报告、11 页 PPT 与多份深度报告。核心结论：LeCun 的问题诊断（纯语言缩放不足以解决具身自主性）成立，但「LLM 是死胡同」的绝对化结论超出证据；更可能是 LLM × JEPA 混合智能体。评估报告由 Manus AI 生成（见 `LICENSE` 第三方说明）。
2. **阶段二 · AC-VJEPA 工程实现（2026-08-15 起）**：可运行的 Python 参考实现 + 混沌工程/弹性恢复/监控/CI 体系：动作条件 V-JEPA 训练核心、DDP 多机训练、2:1 异构微批、UpdatePlan 拓扑感知、混合精度弹性恢复、数据游标账本、三层 KV checkpoint 缓存、防篡改 HITL 账本、Prometheus/Grafana 监控、Rendezvous—GitOps 并发仲裁、本地/K8s 混沌演练与 CI 门禁；以及官方 V-JEPA 2.1 权重适配、M2 数据装配（RoboCasa 适配器 / B 层视频切窗 / 端到端装配）、P1/P2 训练入口与 M3 评测预注册。

## 快速上手

```powershell
# 0) 依赖
python -m pip install -r requirements.txt

# 1) 全部离线验证（幂等；结果写入 verify_artifacts/ 与 VERIFY_RESULTS.md）
.\verify_all.ps1

# 2) 单个模块冒烟
python ac_vjepa_core.py                                   # 训练步 + EMA + 安全回退状态机
python heterogeneous_microbatch_chaos_framework.py        # 5 类 2:1 故障注入
python checkpoint_cache_load_shedding_simulator.py        # 高并发缓存防雪崩
python validate_monitoring_config.py                      # 18 alerts / 10 rules / 19 panels

# 3) 单元测试（避开损坏的 deepeval 插件，见 VERIFY_RESULTS 备注）
python -m unittest -v test_heterogeneous_microbatch_failpoints.py

# 4) Gloo CPU 双进程语义回归（Windows 需 USE_LIBUV=0；内存不足时 WinError 1455）
$env:USE_LIBUV='0'; $env:CUDA_VISIBLE_DEVICES='-1'
python make_demo_ddp_data.py --root verify_artifacts/demo_ddp_data
python scripts/manual_gloo_runner.py test_dynamic_nccl_full_state_equivalence.py
```

## 权重（A 层，官方 V-JEPA 2.1 ViT-B 80M）

仓库不包含预训练权重。按 `DATA_MANIFEST.md` 的登记：

- 来源：Hugging Face `davevanveen/vjepa2.1-vitb-fpc64-384`（Meta 官方 facebookresearch/vjepa2 的 HF 转换版，ViT-B/16，384px）；
- 下载后放到 `weights/vjepa2.1-vitb-fpc64-384/model.safetensors`（已在 `.gitignore`），SHA-256 见 `DATA_MANIFEST.md`；
- 用法：`--init-from vjepa2hf:<path>[:frozen|finetune]`（训练入口）、`vjepa_backbone_smoke.py --safetensors <path>`（加载验证）；
- **许可证**：权重遵循 Meta 原条款，与仓库 MIT 协议独立。

## 目录结构

- `.py` 模块位于根目录（模块间 `from ac_vjepa_core import ...` 交叉导入，不可拆散）；
- 手册/报告为根目录 `.md`（每个工程主题都是「手册 + 安全边界 + 资料核验摘要」三件套）；
- `monitoring/`：Prometheus 规则 + Grafana dashboard；`docker/`：Compose 用 Dockerfile/healthcheck/provisioning；
- `k8s/chaos-lab/`：隔离混沌 Job 模板（immutable digest manifest）；`.github/workflows/`：CI workflows；
- `scripts/`：离线契约、Gloo 双进程 runner、CUDA 冒烟、合成帧生成；
- `weights/`（不入库）：官方权重本地存放；
- 验证产物写入 `verify_artifacts/`（不入库）。

## 关键文档索引

| 文档 | 内容 |
|---|---|
| `PROJECT_PLAN.md` | **主计划**：从路线之争到可运行实验体系（M0–M5 里程碑与门禁） |
| `杨立昆、JEPA 与 AMI Labs：技术路线与产业叙事的审慎评估.md` | 阶段一主报告（结论先行 + 事实分级 + 参考资料） |
| `杨立昆、世界模型与 AMI Labs.pptx` / `slide_content.md` | 11 页演示文稿 |
| `JEPA 世界模型训练的最大数据与算力瓶颈.md` | 数据/算力瓶颈分析 |
| `轻量级 V-JEPA：突破数据与算力限制的技术路线与实验室实施方案.md` | P0–P4 五阶段实验路线 |
| `LLM × JEPA 世界模型混合 Agent：可实施架构蓝图.md` | 混合架构（含 .d2/.mmd 源图） |
| `AC-VJEPA 核心模块使用说明.md` / `AC-VJEPA 多机多卡并行训练与梯度同步指南.md` | 训练使用与 DDP 指南 |
| `NCCL 弹性恢复与动态 UpdatePlan 混沌工程演练手册.md` | 弹性恢复 + 混沌演练 |
| `分布式训练监控、混合精度恢复与生产演练 SOP.md` | 监控 SOP |
| `Kubernetes 生产监控阈值与 Rendezvous—GitOps 并发仲裁手册.md` | 生产仲裁 |
| `DATA_MANIFEST.md` | 四层数据金字塔清单（A 官方权重 / B 本地视频 / C RoboCasa / D 真实轨迹） |
| `M3_MPC_EVALUATION_DESIGN.md` | **M3 评测预注册**：P3 MPC 对比的基线/指标/统计/判定阈值 |
| `CLUSTER_VALIDATION_RUNBOOK.md` | B1–B9 集群验证执行清单（前置/命令/验收） |
| `决策记录.md` / `STATUS.md` / `BACKLOG.md` / `VERIFY_RESULTS.md` | 决策与未决影响 / 状态快照 / 待办分级 / 验证结果 |

## 关键代码入口

| 文件 | 用途 |
|---|---|
| `ac_vjepa_core.py` | 动作条件 V-JEPA 训练核心（EMA、损失、单飞推理、安全回退状态机） |
| `train_ac_vjepa_ddp.py` | 动作条件 DDP 训练（`--init-from vjepa2hf:` 支持官方权重） |
| `train_p1_domain_adapt.py` | P1 无动作域适配训练（冻结骨干 + 无动作/事件损失） |
| `vjepa_backbone.py` + `vjepa_backbone_smoke.py` | 官方 V-JEPA 2 权重适配器（frozen/last_k/lora/finetune）+ 冒烟 |
| `robocasa_adapter.py` / `video_to_windows.py` / `assemble_m2_dataset.py` | M2 数据装配（RoboCasa 采集 / B 层切窗 / 端到端装配） |

## 安全边界与贡献

- 本仓库是**离线研究与参考实现**：不含真实网络扰动、机器人控制、云端操作命令（执行器属受控基础设施职责）；合成数据仅用于链路验证。
- 详见 `SECURITY.md` 与 `CONTRIBUTING.md`；贡献前先读 `决策记录.md` 与 `PROJECT_PLAN.md`。

## 致谢

- 研究评估部分由 Manus AI 于 2026-08-14 生成（经权利人授权以 MIT 公开）；
- 预训练权重与原始模型归 Meta AI（facebookresearch/vjepa2，arXiv:2506.09985）。
