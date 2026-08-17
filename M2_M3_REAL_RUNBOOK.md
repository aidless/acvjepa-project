# M2 / M3 真实执行交接包（RUNBOOK）

> 建立：2026-08-17 ｜ 用途：真实 B/C 层数据与 RoboCasa 环境就绪后，**机械执行** M2（P1 域适配 + P2 动作条件后训练）→ M3（闭环评测）→ M4（判定报告），无需再作设计决策。
> 依据：`PROJECT_PLAN.md` M0–M5、`M3_MPC_EVALUATION_DESIGN.md` v1.2（预注册协议）、`决策记录.md`（口径冻结）、`实验记录与负结果报告_2026-08-17.md`。
> 前置（外部）：①真实 B 层数据（无标签视频，合法许可）；②RoboCasa 环境（专用 Isaac 镜像 + Python≤3.10 + 磁盘≥50GB，见 A27）；③GPU ≥24GB（AutoDL 3090/4090 档）。

---

## 阶段 0：前置检查清单（到位即勾）

- [ ] B 层真实视频：格式 MP4/帧序列，时间戳连续，无隐私/许可障碍；目标 ≥20–50h（H-T5 分档起步 5h 亦可）。
- [ ] C 层环境：RoboCasa 任务模板钉版（早餐台整理相关：拾取放置/门抽屉/桌面整理，10–30 模板起步）。
- [ ] GPU：≥24GB 显存 + 磁盘 ≥100GB；conda env `acvjepa`（py3.12 + torch 2.5.1+cu121，或专用 Isaac 镜像内另建 ≤3.10 环境只跑评测）。
- [ ] 权重：`weights/vjepa2.1-vitb-fpc64-384/model.safetensors`（SHA `77D2D116...C19D6035`），或云端已软链。
- [ ] 仓库：`git pull origin master` 到最新（含 `m3_mpc_eval.py` / `m3_vla_eval_dock.py` / 本 RUNBOOK）。

## 阶段 1：B 层数据装配

```bash
# 切窗（无动作/事件 → P1 域适配窗口；domain_adaptation_only 标记）
python video_to_windows.py --video <b_layer_clip> --root <data_b_root> --window-ms 1000 --stride-ms 500
# 端到端装配（C 层 episode → commit → windows + B 层窗口 + split 隔离校验）
python assemble_m2_dataset.py --simulator robocasa --jobs <c_layer_jobs> --b-root <data_b_root> \
  --repo <acvjepa_project> --out <m2_dataset> --img-size 384
# 验收：manifest.jsonl 行数、split 隔离（按 clip/job）、窗口键完整（context/future video+proprio）
```

## 阶段 2：P1 域适配（无动作，冻结骨干）

```bash
python train_p1_domain_adapt.py \
  --manifest <m2_dataset>/manifest_p1.jsonl \
  --output <p1_out> \
  --epochs 50 --per-rank-batch-size 4 --gradient-accumulation 4 \
  --init-from "vjepa2hf:<weights>/model.safetensors:frozen" --init-img-size 384 \
  --latent-dim 64            # G2 候选（合成数据方向反转；若 H-T2 真实复验仍支持窄头 → 用 64，否则 128）
  --ema-target ema           # H-T4 已 refuted：**必须保留 EMA**（sync 会崩溃）
```
- 验收：loss_latent_nll 单调下降；`p1-last.pt` 落盘；短时潜在预测优于最后帧基线（H-T5 分档 1h/5h/20h 曲线）。
- 可选：同数据按 `scripts/run_ht2_ht4_ablation.sh` 复验 H-T2/H-T4（真实数据解除合成限定）。

## 阶段 3：P2 动作条件后训练（M2 真实训练，产出 C 组模型）

```bash
python train_ac_vjepa_ddp.py \
  --manifest <m2_dataset>/manifest_p2.jsonl \
  --output <p2_out> \
  --epochs 30 --per-rank-batch-size 4 --gradient-accumulation 4 --num-workers 2 \
  --learning-rate 2e-4 --latent-dim 64 --init-from "vjepa2hf:<weights>/model.safetensors:frozen"
```
- 验收：`last.pt`；`z_t,a→z_{t+1}` 事件预测改进；**模型卡必填**（M3 §7.1）：checkpoint SHA、latent_dim、horizon、EMA τ=0.996、数据分档版本、训练 seed、split 版本。

## 阶段 4：RoboCasa 环境部署（专用环境，解 A27）

- 平台：AutoDL「Isaac Gym / 具身仿真」镜像（自带 ≤3.10 + IsaacGym），或官方容器；**不要**在 py3.12 的 acvjepa 环境硬装。
- 磁盘：≥50GB 空闲（注意当前实例磁盘已满 200G/200G，须清理或换实例）。
- 校验：`python -c "import robocasa; print(robocasa.__version__)"`；`python -c "from isaacgym import gymapi; print('ok')"`。
- 数据：按 `robocasa_adapter.py --simulator robocasa` 采集早餐台任务模板（10–30），反事实生成启用。

## 阶段 5：M3 闭环评测（v1.2 协议）

```bash
# 方式 A：vla-eval 底座（若可用）
#   M3ModelServer（scripts/m3_vla_eval_dock.py）注册为模型服务器（A/B/C 三基线 predict）；
#   RoboCasa 基准客户端接入（四方法契约）；统计层用 m3_mpc_eval.py 在 harness 之上跑。
# 方式 B：自建
python m3_mpc_eval.py --manifest <m2_dataset>/manifest_p3.jsonl \
  --model <p2_out>/last.pt --weights <weights>/model.safetensors \
  --task-templates <10-30 模板> --n-rollouts 20 --seeds 2026-2038 ...
```
- 执行前冻结：候选生成器种子、K/H 预算、六指标、ECE 15-bin + 时间轴三分段、失败归因三类、外部复现判定列（§7.3 四项 v1.2 协议）。
- 判定：§6 判定表（①→A ②→B4 ③→C ④→C3；分支映射见 BLUEPRINT §2.1/§2.4）。

## 阶段 6：M4 判定报告装配

- 汇总：H1/H0 判定 + 三负结果（合成级）与真实复验的对照 + 工程可复现说明 + 分支归属（G1）。
- 判定报告模板：`实验记录与负结果报告_2026-08-17.md` §5 前置清单逐项闭环。

---

## G1 条件决策草稿（M3 结果 → 决策记录动作，预演）

| M3 §6 结果模式 | 蓝图分支 | 动作（登记 `决策记录.md`） |
|---|---|---|
| C 显著优于 A 与 B | A（A1/A2/A3 归因细分） | 继续扩档；D 层按 A18 启动（G1∈{A,B1,B2}） |
| C≈B 且显著优于 A | **B4**（表征已够） | 基座转 VLA/策略学习；世界模型叙事收缩 |
| C 不显著优于 B 且不劣于 B | C（C1/C2/C3 细分） | 判数据不足（H-T5）→ 补数据重试；或架构自限（H-T2）→ 配置重设计 |
| C 显著劣于 B | C3 权重上升 | 负结果路线；预注册方法论样本 |

## 已知坑与纪律（执行前必读）

1. **DataLoader worker = 同名进程**：判重复进程看 ppid，勿凭 cmdline 杀（H-D1 教训）。
2. **远端操作走脚本文件**（本地写 → scp → bash），pwsh→ssh 内联转义易碎。
3. **HF 离线**：实例无 HF 直连 → `HF_HUB_OFFLINE=1` + 本地 config/权重（`build_hf_cache.sh`）。
4. **磁盘纪律**：训练产物及时回收；AutoDL 关机后数据盘仍计费。
5. **预注册纪律**：任何口径改动 → 决策记录登记 + 版本号；**不事后挑指标**。
6. **负结果同录**：真实数据复验无论正负写回 HYPOTHESES（解除合成限定或确认 refuted）。
