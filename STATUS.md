# STATUS — 项目当前状态快照

> 快照时间：2026-08-15（接管日）。验证结果明细以 `VERIFY_RESULTS.md` 为准。

## 阶段一：研究与评估（2026-08-14）— ✅ 完成

- 事实核验（公司/人物/技术主张）已完成并分级：`company_findings.md`、`jepa_findings.md`、`route_findings.md`、`architecture_comparison_findings.md`、`jepa_agent_project_findings.md`。
- 主评估报告完成（结论先行、区分事实/技术解释/个人评估、7 条一手来源）。
- 11 页 PPT（`杨立昆、世界模型与 AMI Labs.pptx`）与讲稿（`slide_content.md`）完成。
- 深度报告 5 份（数据算力瓶颈、轻量 V-JEPA 实验路线、架构本质区别、开源 JEPA-Agent 清单、jepa-wms 仓库笔记）。

## 阶段二：AC-VJEPA 工程（2026-08-15）— ✅ 代码与文档完成；验证基线已建立

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
  - 21 个独立冒烟全过（`mixed_precision_elastic_recovery` 一次偶发崩溃、单独复跑 PASS）；
  - 6 个配置校验全过（还原目录结构后）；
  - Docker compose 本地演练实测 3 容器 Healthy（修复 prometheus flag 后）；
  - 4 项 torchrun Gloo 双进程回归因本机内存不足（WinError 1455）FAIL → BACKLOG A10 blocked(env)。
- 已知环境障碍：pytest 全局 deepeval 插件损坏 → 统一用 `python -m unittest`；torch 2.5.1 Windows 无 libuv → `USE_LIBUV=0`。

## 接管期修复（均已记录于决策记录，逻辑未变）

- `file://` URI 解析跨平台修复（3 个文件）；SQLite/npz 句柄与 TemporaryDirectory 清理（Windows 文件锁）；hitl 演示幂等化；prometheus v3.5.0 flag 兼容。
- **还原被 zip 压平的目录结构**：`monitoring/`、`docker/`（含 grafana provisioning）、`k8s/chaos-lab/`、`.github/workflows/`、`scripts/` —— validate_* 与 compose 按原始布局断言，还原后全部 PASS。

## 硬约束与 BLOCKED 项（本机无法执行）

| 项 | 原因 |
|---|---|
| 真实 NCCL 集群压测（带宽/拓扑/RDMA） | 无多机集群；单卡 6GB 笔记本 |
| RDMA/rail/网络分区演练 | 需专用 VLAN/节点池 + 基础设施授权 |
| 生产监控阈值校准（0.85/5s/900s 等） | 需真实 cluster/job 基线 |
| FP8 metadata / CUDA RNG / FSDP/ZeRO 扩展测试 | 需目标 GPU 集群 |
| K8s 隔离 Job / GitOps 实际部署 | 无集群与部署凭据 |
| GitHub workflow 安装与 action SHA 审核 | 需仓库维护者 |

完整分级待办见 `BACKLOG.md`。
