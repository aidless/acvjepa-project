# AC-VJEPA / 「杨立昆为何批评大语言模型为"死胡同"？」项目工作副本

> 接管日期：2026-08-15 ｜ 来源交付包：`E:\杨立昆为何批评大语言模型为"死胡同"？.zip`（只读原件，勿改动）

## 项目是什么

一个**研究 + 工程**双阶段任务，围绕 Yann LeCun 对大语言模型的批评展开：

1. **阶段一 · 研究与评估（2026-08-14）**：事实核验「LeCun 批评 LLM / JEPA 路线 / AMI Labs」叙事，产出审慎评估报告、11 页 PPT 与多份深度研究报告。核心结论：LeCun 的问题诊断（纯语言缩放不足以解决具身自主性）成立，但「LLM 是死胡同」的绝对化结论超出证据；更可能是 LLM × JEPA 混合智能体。
2. **阶段二 · AC-VJEPA 工程实现（2026-08-15）**：把蓝图落成可运行的 Python 参考实现 + 混沌工程/弹性恢复/监控/CI 体系：动作条件 V-JEPA 训练核心、DDP/NCCL 多机训练、2:1 异构微批、UpdatePlan 拓扑感知、混合精度弹性恢复、数据游标账本、三层 KV checkpoint 缓存、防篡改 HITL 账本、Prometheus/Grafana 监控、Rendezvous—GitOps 并发仲裁、本地/K8s 混沌演练与 CI 门禁。

## 目录结构（已还原原始布局）

- `.py` 模块位于根目录（模块间 `from ac_vjepa_core import ...` 交叉导入，不可拆散）；
- 手册/报告为根目录 `.md`（每个工程主题都是「手册 + 安全边界 + 资料核验摘要」三件套）；
- `monitoring/`：Prometheus 规则 + Grafana dashboard（validate 与 compose 按此断言）；
- `docker/`：Compose 用 Dockerfile、healthcheck、prometheus.local.yml、grafana provisioning；
- `k8s/chaos-lab/`：隔离混沌 Job 模板（immutable digest manifest）；
- `.github/workflows/`：failpoint-chaos-ci.yml、kubernetes-chaos-contract.yml；
- `scripts/`：`run_offline_chaos_contract.sh`（离线契约，compose chaos-ci 与 CI 共用）；
- `决策记录.md`：关键决策，含依据/来源/未决影响（**接管工作的第一入口**）；
- `_extras/SKILL.md`：原 zip 误带的无关 imagegen 技能文件，仅归档不参与项目；
- 文件名兼容：原 zip 中 `2:1 …`、`…CI/CD…`、`…RDMA/rail…` 的 `:` 与 `/` 是 Windows 非法字符，已改写为 `2_1`、`CI-CD`、`RDMA_rail`；
- 验证产物写入 `verify_artifacts/`（不入库）。

## 环境与依赖

| 项 | 本机实测 |
|---|---|
| Python | 3.12.7 |
| PyTorch | 2.5.1+cu121（CUDA 可用，RTX 3060 Laptop 6GB 单卡） |
| 第三方 | numpy、yaml、cryptography、onnxruntime、prometheus-client（接管时补装） |
| Docker | 29.6.2（CLI；daemon 状态见 VERIFY_RESULTS） |
| pytest | 8.3.4（**注意**：全局 deepeval 插件损坏，统一改用 `python -m unittest`） |

**硬约束**：单卡 6GB 笔记本 GPU → 真实 NCCL 集群压测、RDMA/rail 演练、多机阈值校准在本机不可执行，只能跑 Gloo CPU 多进程语义回归。相关项在 `BACKLOG.md` 中标记 BLOCKED。

## 快速上手

```powershell
# 1) 全部离线验证（幂等，约几分钟；结果写入 verify_artifacts/ 与 VERIFY_RESULTS.md）
.\verify_all.ps1

# 2) 单个模块冒烟
python ac_vjepa_core.py                                   # 训练步 + EMA + 安全回退状态机
python heterogeneous_microbatch_chaos_framework.py        # 5 类 2:1 故障注入
python checkpoint_cache_load_shedding_simulator.py        # 高并发缓存防雪崩
python validate_monitoring_config.py                      # 18 alerts / 10 rules / 19 panels

# 3) 单元测试（本机避开损坏的 deepeval 插件）
python -m unittest -v test_heterogeneous_microbatch_failpoints.py

# 4) Gloo CPU 多进程回归（需要内存充裕环境；本机内存不足时会 WinError 1455）
$env:USE_LIBUV='0'; $env:CUDA_VISIBLE_DEVICES='-1'
python make_demo_ddp_data.py
torchrun --standalone --nproc_per_node=2 train_ac_vjepa_ddp.py --manifest F:\home\ubuntu\lecun_analysis\demo_ddp_data\manifest.jsonl --output verify_artifacts\ddp_train_out --epochs 1
torchrun --standalone --nproc_per_node=2 test_dynamic_nccl_full_state_equivalence.py
```

## 关键文档索引

| 文档 | 内容 |
|---|---|
| `杨立昆、JEPA 与 AMI Labs：技术路线与产业叙事的审慎评估.md` | 阶段一主报告（结论先行 + 事实分级 + 参考资料） |
| `杨立昆、世界模型与 AMI Labs.pptx` / `slide_content.md` | 11 页演示文稿 |
| `JEPA 世界模型训练的最大数据与算力瓶颈.md` | 数据/算力瓶颈分析 |
| `轻量级 V-JEPA：突破数据与算力限制的技术路线与实验室实施方案.md` | P0–P4 五阶段实验路线 |
| `LLM × JEPA 世界模型混合 Agent：可实施架构蓝图.md` | 混合架构（含 .d2/.mmd 源图） |
| `Action-conditioned V-JEPA：双臂机器人的代码集成、时延与控制部署蓝图.md` | 双臂部署蓝图 |
| `AC-VJEPA 核心模块使用说明.md` | `ac_vjepa_core.py` 使用说明 |
| `AC-VJEPA 多机多卡并行训练与梯度同步指南.md` | DDP 训练指南 |
| `NCCL 弹性恢复与动态 UpdatePlan 混沌工程演练手册.md` | 弹性恢复 + 混沌演练 |
| `分布式训练监控、混合精度恢复与生产演练 SOP.md` | 监控 SOP |
| `Kubernetes 生产监控阈值与 Rendezvous—GitOps 并发仲裁手册.md` | 生产仲裁 |
| `决策记录.md` | 全部决策与未决影响 |
| `PROJECT_PLAN.md` | **主计划**：从路线之争到可运行实验体系（M0–M5 里程碑与门禁） |
| `DATA_MANIFEST.md` | 四层数据金字塔清单（A 官方权重 / B 本地视频 / C RoboCasa / D 真实轨迹） |
| `vjepa_backbone.py` + `vjepa_backbone_smoke.py` | **M1 交付**：官方 V-JEPA 2 权重适配器（frozen/last_k/lora/finetune）+ 冒烟 |
| `STATUS.md` / `BACKLOG.md` / `VERIFY_RESULTS.md` | 当前状态 / 待办分级 / 验证结果 |
| `CLUSTER_VALIDATION_RUNBOOK.md` | B1–B9 集群验证执行清单（前置/命令/验收） |
| `C9_AMI_JEPA_更新核验_2026-08-15.md` | AMI/JEPA 最新动态核验（ICML 2026 等） |

## 验证基线与待办

- 验证结果与逐项证据见 `VERIFY_RESULTS.md`（由 `verify_all.ps1` 重跑可刷新）。**当前基线：42/42 全绿**（单元测试 5、独立冒烟 22、配置校验 6、Gloo 双进程 4、容器/契约 4+1）。
- Gloo 双进程回归在本机使用 `scripts/manual_gloo_runner.py`（torchrun 的 elastic agent 会触发本机页文件限制）。
- 未决事项按「本机可做 / 需真实集群硬件 / 需外部系统审批」三级见 `BACKLOG.md`；B 级项的执行步骤见 `CLUSTER_VALIDATION_RUNBOOK.md`。

## 原始交付包与完整性

- 原件：`E:\杨立昆为何批评大语言模型为"死胡同"？.zip`（SHA 见 `_extras/` 或按需重算）。
- 本副本由接管流程从原件物化：148 个文件 + `verify_all.ps1` + 管理文档。
