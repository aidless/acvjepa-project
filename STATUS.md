# STATUS — 项目当前状态快照

> 快照时间：2026-08-15（接管日 + M1）。验证结果明细以 `VERIFY_RESULTS.md` 为准；里程碑见 `PROJECT_PLAN.md`。

## P1 域适配训练脚本（2026-08-15）— ✅ 完成（J 组 4 项 PASS）

- `train_p1_domain_adapt.py`：无动作 JEPA 域适配训练——复用 `train_ac_vjepa_ddp.py` 的 DDP/EMA/checkpoint 基建；冻结官方 V-JEPA 2.1 骨干（`--init-from vjepa2hf:<path>:frozen`）；损失只含 latent_nll + cosine + calibration，**显式排除动作与事件**（`event_term: 0.0`）。
- 单进程 CPU 26 步冒烟通过（loss 正常演变）；`p1-last.pt`（434 MB）落盘。
- 待办：真实 M2 训练仍需 ≥24GB GPU + 真实 B/C 层数据。

## M2 数据装配工具链（2026-08-15）— ✅ 完成（I 组 4 项 PASS）

- `robocasa_adapter.py`：RoboCasa 采集适配器（`SimulatorAdapter` 契约；`--simulator robocasa` 接口就绪；本机 portable synthetic 契约冒烟；真实采集需 sim 环境）。
- `video_to_windows.py`：B 层无标签视频 → 域适配窗口（零动作/事件 + `domain_adaptation_only` 标记；`WindowEpisodeDataset` 可直读）。
- `assemble_m2_dataset.py`：端到端装配——C 层 episode→commit→windows + B 层窗口 + split 隔离校验（按 clip/job）+ DATA_MANIFEST 登记。
- `scripts/make_synthetic_clips.py`：合成 B 层帧（仅冒烟用）。
- 待办：真实 RoboCasa/Isaac 环境装配（`--simulator robocasa`）与 B 层实拍视频采集。

## M1 实验基座（2026-08-15）— ✅ 完成（PROJECT_PLAN 里程碑）

- `vjepa_backbone.py`：官方 V-JEPA 2 ViT-B/16 权重适配器（键重映射：`encoder.`/`backbone.` 前缀剥离、qkv 合并/拆分、`patch_embed` 别名；微调模式 frozen/last_k/lora/finetune；LoRA 低秩适配；安装时同步替换 student+EMA frame_encoder）。
- `vjepa_backbone_smoke.py`：4 类检查全 PASS（forward 契约 / 模式 / 150 键 strict 重映射 / 安装+训练步+EMA）。
- `train_ac_vjepa_ddp.py`：`--init-from vjepa2:<path>[:mode]` 扩展（含 `--init-lora-rank` / `--init-unfreeze-last-k`）。
- `DATA_MANIFEST.md`：四层数据金字塔登记清单（A 官方权重 / B 本地视频 / C RoboCasa / D 真实轨迹）。
- 待办：官方权重下载（数百 MB，网络/许可前置）→ M2 域适配需 A/B/C 层数据 + ≥24GB GPU。

## 阶段一：研究与评估（2026-08-14）— ✅ 完成

- 事实核验（公司/人物/技术主张）已完成并分级：`company_findings.md`、`jepa_findings.md`、`route_findings.md`、`architecture_comparison_findings.md`、`jepa_agent_project_findings.md`。
- 主评估报告完成（结论先行、区分事实/技术解释/个人评估、7 条一手来源）。
- 11 页 PPT（`杨立昆、世界模型与 AMI Labs.pptx`）与讲稿（`slide_content.md`）完成。
- 深度报告 5 份（数据算力瓶颈、轻量 V-JEPA 实验路线、架构本质区别、开源 JEPA-Agent 清单、jepa-wms 仓库笔记）。

## 阶段二：AC-VJEPA 工程（2026-08-15）— ✅ 代码与文档完成；验证基线全绿（59/59）

- 训练核心：`ac_vjepa_core.py`（CPU 冒烟已跑通：训练步/EMA/NORMAL/LOCAL_HOLD→LLM_SUPERVISION）、`train_ac_vjepa_ddp.py`（DDP/torchrun/Gloo）、`dynamic_nccl_update_plan_train.py`、`topology_aware_update_plan.py`。
- 弹性恢复与一致性：混合精度恢复、数据游标账本、Rendezvous—GitOps 仲裁、快速重建告警、RDMA/rail 守卫。
- 缓存与防雪崩：verified cache、线程安全门、载荷脱落模拟、损坏回退演示。
- 安全与治理：多模态防篡改 HITL 账本（Ed25519 哈希链）、隔离检疫、LLM 安全托管状态机。
- 观测与 CI：Prometheus 18 alerts / 10 recording rules、Grafana 19 panels、failpoint CI、K8s 混沌契约、本地 Compose 演练、离线契约脚本。
- 约 30 本手册，全部带「安全边界」与「资料核验摘要」。

## 本机验证（2026-08-15 接管基线）

- 全部 148 文件已物化为工作副本，无非法文件名；全部模块 import 通过（`healthcheck.py` 为网络探针除外）。
- 依赖补装：`prometheus-client`（原交付缺失）。
- **验证基线：59/59 全绿**（2026-08-15 最终）：
  - 5 个单元测试文件（21 用例）全过；
  - 22 个独立冒烟全过（含 mixed_precision 复跑稳定）；
  - 6 个配置校验全过（还原目录结构后）；
  - **Gloo 双进程语义回归 4/4 全过**（`scripts/manual_gloo_runner.py` 手工双进程，绕开 torchrun 页文件限制）：train/topology/full_state(118+118)/integration；
  - Docker compose 全流程实测：build→up 3 容器 Healthy→chaos-ci profile 容器内跑离线契约→down；
  - **H 组 HF 真实权重训练冒烟 3 项**（`--init-from vjepa2hf:` → demo 数据 + 训练 + checkpoint 落盘）；
  - **I 组 M2 数据装配 4 项**（RoboCasa 契约 / B 层切窗 / 端到端装配）；
  - **J 组 P1 域适配（CPU）4 项**（合成帧 → 384px 切窗 → 域适配训练 → checkpoint）；
  - **K 组 P1 CUDA 4 项**（真实权重冻结骨干 GPU 训练，峰值 510.8MB）；
  - **L 组 P2 动作条件 CUDA 3 项**（GPU 训练峰值 567.2MB，checkpoint 落盘）。
- 已知环境障碍：pytest 全局 deepeval 插件损坏 → 统一用 `python -m unittest`；torch 2.5.1 Windows 无 libuv → `USE_LIBUV=0`；torchrun elastic agent 在本机页文件限制下不可用 → 用 manual_gloo_runner。

## 接管期修复（均已记录于决策记录，逻辑未变）

- `file://` URI 解析跨平台修复（3 个文件）；SQLite/npz 句柄与 TemporaryDirectory 清理（Windows 文件锁）；hitl 演示幂等化；prometheus v3.5.0 flag 兼容。
- **还原被 zip 压平的目录结构**：`monitoring/`、`docker/`（含 grafana provisioning）、`k8s/chaos-lab/`、`.github/workflows/`、`scripts/` —— validate_* 与 compose 按原始布局断言，还原后全部 PASS。
- **容器内契约脚本**：chaos-ci 镜像排除 `.github/workflows/`，`run_offline_chaos_contract.sh` 对 CI workflow 校验在文件缺失时显式 SKIP（仓库场景保持严格）。
- **新增工具**：`scripts/manual_gloo_runner.py`（手工双进程 Gloo 回归）、`requirements.txt`、`CLUSTER_VALIDATION_RUNBOOK.md`（B1–B9 执行清单）、`C9_AMI_JEPA_更新核验_2026-08-15.md`。

## 研究推演与开源治理（2026-08-15 追加）

- **开源化**（MIT → GitHub `aidless/acvjepa-project`）：LICENSE（含第三方声明）、CONTRIBUTING/SECURITY、README 开源版；权重 438.9MB 不入库（.gitignore）。
- **猜想假设集** `HYPOTHESES.md`：五层 11 条可证伪假说（H-R1/R2、H-T1–T5、H-D1、H-P1/P2、H-M1）+ 预注册流程；状态全部 `proposed`。
- **宏大蓝图** `BLUEPRINT.md` v2：十年推演总纲（纯思辨、不执行实验）——判定网络、A/B/C 分支树（含 B4）、X/Y/Z 终局、时间信号（2026Q4–2036）、决策门 G1–G5；经审阅修订（A1–A6/B1–B7 修复 + C 级建议，commit `ca96612`）。
- **C9 检查清单扩展**：三梯队 JEPA 项目活跃度/终止率统计项（2026-11-15 轮生效）。
- **A18/A19 裁定与预注册**（2026-08-15）：D 层拆 D-mini（随 P2）/D-full（G1 分支门控），三方张力化解（决策记录）；H-T2/H-T4 预注册登记完成（HYPOTHESES 状态 → preregistered），执行待解除约束。
- **C9 第二轮核验（提前轮）**：AMI 谢赛宁加入（待一手确认）、JEPA-WMs 权重上 HF、VLA-JEPA 进 LeRobot v0.6.0、INTACT-JEPA 等新开源项目；H-R2/H-P1 首读数方向支持（定性）；原交付四条结论维持。
- **PROJECT_PLAN/轻量方案接口审阅**：成功标准 42/42→59/59 修复；L4 D 层行按 A18 更新；M3 v1.1 指针。
- **第六轮猜想落盘（BLUEPRINT v2.1 §7）**：F 免搜索支线（X1/X2）、Y' 分域收敛、B4×H-P2 表征优先汇合、信号概率更新表；仓库全量内容公开（权重与可再生的 verify_artifacts 除外）。
- **项目与 AGI 关系自审（BLUEPRINT v2.2 §7.5）**：反推演分层结论（直接能力❌/路线裁决✅/方法论✅/认知✅/社会学⚠️）+ 自证预言消毒；定位=裁决节点（地图/罗盘/航标），非 AGI 引擎。
- **AGI 总题猜想（BLUEPRINT v2.3 §7.6）**：定义三分（AGI-1/2/3）+ 三路径 + 2026–2040 时间线 + 表征层会合猜想；三瓶颈=本项目 H-T1/H-T3/H-R1 切片。
- **AGI 机制层深推（BLUEPRINT v2.4 §7.7）**：同构涌现机制、AGI-2 临界形态（长任务链）、反事实密度/校准阈值量化猜想、W 路线、接口错误占比=AGI-2 距离度量。
- **AGI-2 可达性地图与概率账本（BLUEPRINT v2.5 §7.8）**：四象限地图（演示/陷阱/可信但窄/AGI-2 区）+ 终局权重账本（Y 25% 等，C9 轮调权）+ 人类角色=裁决与校准。
- 治理闭环：决策记录追加蓝图落盘与 v2 修订条目；`origin/master` 同步至 `ca96612`。

## 后续审计（2026-08-15 完成）

- **PPTX 一致性**：12 页 ↔ slide_content.md 12 区块对齐，关键事实全过（`verify_artifacts/ppt_audit_report.json`）。
- **C9 动态核验**：ICML 2026 冯雁特邀演讲、AMI 拒绝 AGI 标签聚焦机器人世界模型、SBVA €30M 等后续投资、中国 JEPA 具身热度；原交付结论（LLM 死胡同=路线判断、AMI=科学下注）全部维持。

## 硬约束与 BLOCKED 项（本机无法执行；执行清单已产出）

| 项 | 原因 |
|---|---|
| 真实 NCCL 集群压测（带宽/拓扑/RDMA） | 无多机集群；单卡 6GB 笔记本 → `CLUSTER_VALIDATION_RUNBOOK.md` B1 |
| RDMA/rail/网络分区演练 | 需专用 VLAN/节点池 + 基础设施授权 → RUNBOOK B2 |
| 生产监控阈值校准（0.85/5s/900s 等） | 需真实 cluster/job 基线 → RUNBOOK B3 |
| FP8 metadata / CUDA RNG / FSDP/ZeRO 扩展测试 | 需目标 GPU 集群 → RUNBOOK B4 |
| Gloo 验证升级为 GPU 全状态对比 | 需 GPU 多卡 → RUNBOOK B5 |
| K8s 隔离 Job 实际 apply | 需独立集群 + admission policy + GitOps 身份 → RUNBOOK B6 |
| torchrun 故障恢复真实作业演练 | 需真实调度器 + 共享存储 → RUNBOOK B7 |
| 分波恢复参数校准（wave/concurrency） | 需 rendezvous/对象存储/网络压测 → RUNBOOK B8 |
| checkpoint 三层缓存目标硬件验证 | 需目标 KV/对象存储/RDMA → RUNBOOK B9 |
| GitHub workflow 安装与 action SHA 审核 | 需仓库维护者 |

完整分级待办见 `BACKLOG.md`。
