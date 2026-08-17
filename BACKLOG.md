# BACKLOG — 未决事项分级待办

> 来源：`决策记录.md` 各条目的「未决影响」列 + 接管勘察发现。分级：**A 本机可做** / **B 需真实集群/硬件** / **C 需外部系统或审批**。
> 状态：`open` / `in_progress` / `done` / `blocked`。更新人需同步维护 `STATUS.md`。

## A. 本机可做（无需外部资源）

| # | 事项 | 来源决策 | 状态 | 备注 |
|---|---|---|---|---|
| A1 | 补装缺失依赖并固化环境（requirements.txt / pip freeze） | 接管勘察 | done | prometheus-client 已补装；`requirements.txt` 已入库（A6） |
| A2 | 修复 zip 中 `:`/`/` 非法文件名并物化工作副本 | 接管决策 | done | `2_1`/`CI-CD`/`RDMA_rail` |
| A3 | 建立可重跑验证基线（verify_all.ps1 → VERIFY_RESULTS.md） | 接管决策 | done | **42/42 全绿**（2026-08-15 最终） |
| A4 | Docker Compose 本地演练实跑（build→up→chaos-ci profile→down 全流程） | 存储损坏回退手册；决策「本地 Compose」条 | done | 3 容器 Healthy；chaos-ci 容器内跑离线契约成功；修复 .github 缺失 SKIP 语义 |
| A5 | PPTX 与 `slide_content.md` 一致性核对（页数/要点/来源） | 阶段一交付审计 | done | 12 页↔12 区块对齐，关键事实全过（`verify_artifacts/ppt_audit_report.json`） |
| A6 | 统一 `requirements.txt` + 运行说明入库 | A1 | done | 已入库 |
| A7 | `_extras/SKILL.md`（无关 imagegen 技能）处理确认：删除 or 保留归档 | 接管勘察 | done | 保留归档（`_extras/SKILL.md`），README 已注明无关性，不影响项目 |
| A8 | **Windows 兼容修复回归**：`file://` URI 解析（elastic_data_cursor_ledger / multimodal ledger / generate_pointcloud_pairs）、SQLite 句柄关闭（chaos framework）、np.load 句柄（pointcloud 测试）、hitl 演示幂等 | 接管验证发现 | done | 原交付在 Linux 验证；本机 Windows 需这些修复（决策记录已记） |
| A9 | **还原被 zip 压平的目录结构**（monitoring/、docker/、k8s/chaos-lab/、.github/workflows/、scripts/） | 接管验证发现（validate_* 与 compose 按原始布局断言） | done | validate_* 从 FAIL 转 PASS |
| A10 | Gloo 双进程语义回归在本机可跑（经 `scripts/manual_gloo_runner.py` 绕开 torchrun elastic agent 页文件限制） | 接管验证 | done | 4/4 通过：train/topology/full_state(118+118)/integration |
| A11 | M1 实验基座：官方 V-JEPA 2.1 80M 权重获取 + `vjepa_backbone.py` 适配器 + 训练入口 `--init-from vjepa2hf:` | PROJECT_PLAN M1 | done | 438.9MB 下载验证（391/395 键）；HFVJEPA2Backbone 真实加载+EMA 训练步 PASS |
| A12 | M2 数据装配：`robocasa_adapter.py` / `video_to_windows.py` / `assemble_m2_dataset.py` / 合成帧脚本 | PROJECT_PLAN M2 前置 | done | I 组 4 项 PASS；真实采集需 RoboCasa/sim 环境 |
| A13 | P1 域适配训练：`train_p1_domain_adapt.py`（冻结骨干 + 无动作/事件损失） | PROJECT_PLAN M2 P1 | done | J 组 4 项 PASS；26 步 CPU 冒烟 + checkpoint |
| A14 | 本机 CUDA 微型 P1 训练（真实权重+冻结骨干，实测峰值 473MB 可行） | 复审发现（2026-08-15） | done | K 组 4 项 PASS：CUDA 峰值 510.8MB；修复 NCCL 回退 + 安装后 `.to(device)` 两个真实 bug |
| A15 | P2 动作条件训练 CUDA 微型冒烟（复用 `train_ac_vjepa_ddp.py --init-from vjepa2hf:`） | 复审计划步骤 3 | done | L 组 3 项 PASS：CUDA 峰值 567.2MB，loss 与 CPU 路径一致 |
| A16 | M3 评测设计文档 `M3_MPC_EVALUATION_DESIGN.md`（P3 闭环评测方案） | 复审计划步骤 4 | done | 已落盘：三基线/指标/统计/判定阈值/预注册约定 |
| A17 | C9 季度跟踪机制写入（触发条件 + 检查清单） | 复审计划步骤 5 | done | 已追加至 C9 文档；下一轮 2026-11-15 |
| A18 | D 层采集时点裁定（H-T1 的 M2 分档需 D 层数据 vs PROJECT_PLAN「D 层=P4」的源文档张力） | BLUEPRINT v2 §2.1 已知张力 | done | 已裁定（决策记录 2026-08-15）：D-mini（≤2h）随 P2；D-full（5–10h）于 G1∈{A,B1,B2} 后启动、C/B4 分支降级可选；PROJECT_PLAN L4/M2 行已同步 |
| A19 | H-T2/H-T4 消融执行（预注册登记已完成 2026-08-15；本机 6GB 或云端 48GB） | HYPOTHESES H-T2/H-T4；BLUEPRINT §3/G2 | done | **完成 2026-08-16**（AutoDL 3090 48GB，18 runs 全绿）：**H-T2 refuted**（方向反转：窄头更优，w32=-3.505 最优；饱和判据 4.80≫0.2）、**H-T4 refuted**（EMA 收益巨大，|Δ|=1715%）；证据 `experiments/ht2_ht4_2026-08-16/`；真实 B 层数据复验待办 |
| A20 | H-M1 开源接力观察（观察窗 2026-08-15 → 2027-02-15） | HYPOTHESES H-M1；BLUEPRINT §3/G4 | in_progress | 预注册判据：≥1 名持 GPU 外部研究者 fork/PR 并执行 M3 评测（或同类）；fork/PR/issue 动态为辅助信号 |
| A21 | C9 下一轮核验（2026-11-15 前后） | C9 跟踪机制；BLUEPRINT §3 | done | **提前轮（第二轮）+ 第三轮已执行**（2026-08-15 晚，用户指示）：见 C9 文档「第二轮/第三轮核验」小节；正式下一轮仍 2026-11-15（含三梯队终止率计数） |
| A22 | vla-evaluation-harness 对接评估：M3 评测包注册 RoboCasa 集成的可行性 + 统计层边界（harness 只保证可复现，ECE 分层/检验/失败归因仍自建） | WAM 生态学习批次（2026-08-15） | open | 学习报告 §五；结论后置 done |
| A23 | M3 设计借鉴储备（**不改冻结口径，采纳须走预注册**）：INTACT 四接口受控对照、ECE 时间轴分解（WorldRoamBench 度量）、复现即验收列、奖励模型判定器（过 judge 校准门）、stable-worldmodel A 基线参考 | WAM 生态学习批次 | open | 15 条精华模式 1–10/12；已落 M3 §7.2 注记 |
| A24 | 生态清单补录与核对：`jepa_agent_project_findings.md` 补 INTACT/LeWorldModel/W2-VLA/worldmodels_ros2 + JEPA-WAM 10780 区分；核对清单闭环（VLA-JEPA ECCV/LICENSE、W2-VLA LICENSE、JEPA-WAM 零样本声称、vla-eval 榜单口径） | WAM 生态学习批次 | open | 学习报告 §三 核对清单 |
| A25 | C9 核验新增「多种子稳定性」项（Seed Lottery 1/13 vs 0/21；上游 checkpoint ≥5 种子复现） | WAM 生态学习批次（2606.13856） | done | 检查清单已固化（2026-11-15 正式轮生效） |

## B. 需真实集群 / 硬件（本机 BLOCKED）

| # | 事项 | 来源决策 | 状态 | 前置条件 |
|---|---|---|---|---|
| B1 | 真实 NCCL 集群压测：拓扑基线→微基准→应用级动态计划→受控故障恢复；建立 GB/s 阈值 | NCCL 真实集群压测手册；决策「真实集群压测」条 | blocked（执行清单已产出） | 隔离 GPU 集群（≥2 节点），torchrun elastic；步骤见 `CLUSTER_VALIDATION_RUNBOOK.md` B1 |
| B2 | RDMA/rail 故障演练（分区、rendezvous、rail 失效） | RDMA/rail 守卫；决策「RDMA/rail」条 | blocked（执行清单已产出） | 专用 VLAN/节点池 + 双人审批 + TTL 窗口；见 RUNBOOK B2 |
| B3 | 生产监控阈值校准（0.85/5s/900s 等初始保护栏） | 生产阈值手册；决策「生产监控」条 | blocked（执行清单已产出） | 真实 cluster/job 基线 + SLO；见 RUNBOOK B3 |
| B4 | FP8 metadata / FP16 GradScaler / CUDA RNG / FSDP-ZeRO shard 扩展验证 | 混合精度恢复手册 | blocked（执行清单已产出） | 目标 GPU + Transformer Engine backend；见 RUNBOOK B4 |
| B5 | Gloo 验证升级为 GPU 全状态对比（AMP/TF32/lossy comm hook 独立阈值） | 决策「Gloo 边界」条 | blocked（执行清单已产出） | GPU 多卡环境；见 RUNBOOK B5 |
| B6 | K8s 隔离 Job 混沌演练（immutable manifest 实际 apply） | K8s 混沌契约；决策「K8s CI」条 | blocked（执行清单已产出） | 独立集群 + admission policy + GitOps 身份；见 RUNBOOK B6 |
| B7 | torchrun 故障恢复演练：checkpoint 提交协议与数据游标真实作业验证 | 决策「worker-group 重建」条 | blocked（执行清单已产出） | 真实调度器 + 共享存储；见 RUNBOOK B7 |
| B8 | 大规模抢占分波恢复参数校准（wave/concurrency） | 快速重建手册 | blocked（执行清单已产出） | rendezvous/对象存储/网络压测；见 RUNBOOK B8 |
| B9 | checkpoint 三层缓存（KV pointer/RDMA shard/verified cache）目标硬件验证 | 决策「checkpoint 加速」条 | blocked（执行清单已产出） | 目标 KV/对象存储/RDMA/rail 访问隔离；见 RUNBOOK B9 |

## C. 需外部系统 / 审批 / 部署环境

| # | 事项 | 来源决策 | 状态 | 前置条件 |
|---|---|---|---|---|
| C1 | GitHub workflow（failpoint-chaos-ci.yml 等）安装与 action SHA 审核 | 决策「CI workflow」条 | blocked | 仓库维护者安装 + 组织策略 |
| C2 | 生产对象存储版本化、保留策略、访问日志、备份 | 决策「审核账本」条 | blocked | 受控对象存储采购/配置 |
| C3 | 独立 WORM/透明日志锚定服务 + 密钥轮换 | 决策「防篡改」条 | blocked | 外部锚定服务 + HSM/密钥管理 |
| C4 | 生产 adapter mTLS、身份、速率限制、持久去重、服务端 allowlist | 决策「Alertmanager 联动」条 | blocked | 生产可观测平台 |
| C5 | RDMA/rail 基础设施自有受限 executor（mTLS/标签/TTL/审计） | 决策「双层模型」条 | blocked | 服务器端实现 + 审批窗口 |
| C6 | 严格 exactly-once 样本消费的事务性数据游标 | 决策「数据一致性」条 | blocked | 外部事务性数据系统 |
| C7 | 数据治理：训练评测、影子模式、灰度发布门控（HITL 资格≠自动训练） | 决策「HITL 双重审核」条 | blocked | 数据/发布平台 |
| C8 | 目标机器人实机：时延 p99/抖动/故障注入/影子验收（`AC-V-JEPA 双臂部署：实机时延与控制频率验收协议.md`） | AC-V-JEPA 验收协议 | blocked | 双臂硬件 + 受控场地 |
| C9 | AMI/JEPA 路线最新动态（2026-08-14 之后）更新核验 | 阶段一报告 | done | 已产出 `C9_AMI_JEPA_更新核验_2026-08-15.md`：ICML 2026 冯雁特邀演讲、AMI 拒绝 AGI 标签聚焦机器人世界模型、SBVA €30M 等后续投资、中国 JEPA 具身热度；原交付结论全部维持 |

## 记录规范

- 完成项：置 `done` 并在 `STATUS.md` 对应行补一句证据。
- 阻塞项：保持 `blocked`，前置条件不变不擅自取消。
- 新增项：注明来源（决策条目/手册/勘察），避免与既有条目重复。
