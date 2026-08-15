# STATUS — 项目当前状态快照

> 快照时间：2026-08-15（接管日 + M1）。验证结果明细以 `VERIFY_RESULTS.md` 为准；里程碑见 `PROJECT_PLAN.md`。

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

## 阶段二：AC-VJEPA 工程（2026-08-15）— ✅ 代码与文档完成；验证基线全绿（42/42）

- 训练核心：`ac_vjepa_core.py`（CPU 冒烟已跑通：训练步/EMA/NORMAL/LOCAL_HOLD→LLM_SUPERVISION）、`train_ac_vjepa_ddp.py`（DDP/torchrun/Gloo）、`dynamic_nccl_update_plan_train.py`、`topology_aware_update_plan.py`。
- 弹性恢复与一致性：混合精度恢复、数据游标账本、Rendezvous—GitOps 仲裁、快速重建告警、RDMA/rail 守卫。
- 缓存与防雪崩：verified cache、线程安全门、载荷脱落模拟、损坏回退演示。
- 安全与治理：多模态防篡改 HITL 账本（Ed25519 哈希链）、隔离检疫、LLM 安全托管状态机。
- 观测与 CI：Prometheus 18 alerts / 10 recording rules、Grafana 19 panels、failpoint CI、K8s 混沌契约、本地 Compose 演练、离线契约脚本。
- 约 30 本手册，全部带「安全边界」与「资料核验摘要」。

## 本机验证（2026-08-15 接管基线）

- 全部 148 文件已物化为工作副本，无非法文件名；全部模块 import 通过（`healthcheck.py` 为网络探针除外）。
- 依赖补装：`prometheus-client`（原交付缺失）。
- **验证基线：38/40 PASS**（详见 `VERIFY_RESULTS.md`）：
  - 5 个单元测试文件（21 用例）全过；
  - 22 个独立冒烟全过（含 mixed_precision 复跑稳定）；
  - 6 个配置校验全过（还原目录结构后）；
  - **Gloo 双进程语义回归 4/4 全过**（`scripts/manual_gloo_runner.py` 手工双进程，绕开 torchrun 页文件限制）：train/topology/full_state(118+118)/integration；
  - Docker compose 全流程实测：build→up 3 容器 Healthy→chaos-ci profile 容器内跑离线契约→down；
  - **验证基线：42/42 全绿**（2026-08-15 最终）。
- 已知环境障碍：pytest 全局 deepeval 插件损坏 → 统一用 `python -m unittest`；torch 2.5.1 Windows 无 libuv → `USE_LIBUV=0`；torchrun elastic agent 在本机页文件限制下不可用 → 用 manual_gloo_runner。

## 接管期修复（均已记录于决策记录，逻辑未变）

- `file://` URI 解析跨平台修复（3 个文件）；SQLite/npz 句柄与 TemporaryDirectory 清理（Windows 文件锁）；hitl 演示幂等化；prometheus v3.5.0 flag 兼容。
- **还原被 zip 压平的目录结构**：`monitoring/`、`docker/`（含 grafana provisioning）、`k8s/chaos-lab/`、`.github/workflows/`、`scripts/` —— validate_* 与 compose 按原始布局断言，还原后全部 PASS。
- **容器内契约脚本**：chaos-ci 镜像排除 `.github/workflows/`，`run_offline_chaos_contract.sh` 对 CI workflow 校验在文件缺失时显式 SKIP（仓库场景保持严格）。
- **新增工具**：`scripts/manual_gloo_runner.py`（手工双进程 Gloo 回归）、`requirements.txt`、`CLUSTER_VALIDATION_RUNBOOK.md`（B1–B9 执行清单）、`C9_AMI_JEPA_更新核验_2026-08-15.md`。

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
