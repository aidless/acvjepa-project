# VERIFY_RESULTS — 接管验证结果（2026-08-15，最终全绿 48/48）

> 由 `verify_all.ps1` 生成；重跑命令：`.\verify_all.ps1`（幂等，产物在 `verify_artifacts/`）。
> 环境：Windows 11 沙箱 · Python 3.12.7 · torch 2.5.1+cu121（RTX 3060 Laptop 6GB）· Docker 29.6.2。
> 汇总：**48/48 PASS（全绿）**。Gloo 双进程语义回归通过 `scripts/manual_gloo_runner.py`
> 手工双进程运行（绕开 torchrun elastic agent 在本机页文件限制下不可行的问题）；
> **H 组 HF 真实权重训练冒烟**（`--init-from vjepa2hf:` → demo 数据 + 训练 + checkpoint 落盘）；
> **I 组 M2 数据装配**（RoboCasa 适配器契约 + B 层视频切窗 + 端到端装配，portable synthetic）。

## 通过项（48/48）

### 单元测试（5/5）
| 检查 | 结果 |
|---|---|
| unit.test_heterogeneous_microbatch_failpoints | PASS（6 用例，含 5 故障注入场景断言） |
| unit.test_shadow_canary_gate | PASS（4 用例） |
| unit.test_shadow_rca_and_pointcloud_pipeline | PASS（3 用例） |
| unit.test_resumable_ledger_and_training_input | PASS（2 用例） |
| unit.ac_vjepa_fault_injection_tests | PASS（6 用例） |

### 独立冒烟（22/22）
ac_vjepa_core（CPU 路径，见备注）、elastic_data_cursor_ledger、verified_checkpoint_cache、threadsafe_checkpoint_load_gate、multimodal_hitl_tamper_evident_ledger、rapid_recovery_alert_drill、**mixed_precision_elastic_recovery**、rdma_rail_chaos_guard、recovery_deployment_arbiter、heterogeneous_microbatch_chaos_framework（5 场景/17 断言）、checkpoint_cache_load_shedding_simulator（32→1 durable 读、16 writer→1 CAS）、checkpoint_integrity_corruption_demo、distributed_training_observability、run_failpoint_observability_drill、dr_policy_tuner、spsc_robot_pipeline、shadow_canary_gate、shadow_degradation_rca、hitl_quarantine_review、sim2real_hard_example_compiler（--demo）、sim2real_pointcloud_video_pipeline（--demo）、dynamic_nccl_update_plan_train（--smoke-test）。

### 配置校验（6/6）
validate.monitoring_config（18 alerts/10 rules/19 panels）、validate.local_compose、validate.kubernetes_chaos_lab、validate.kubernetes_chaos_ci、validate.failpoint_ci_config、update_production_dashboard。

### Gloo 双进程语义回归（4/4，经 manual_gloo_runner）
| 检查 | 结果 |
|---|---|
| ddp.train_ac_vjepa_gloo_2proc | PASS（train_ac_vjepa_ddp.py 2 rank，demo manifest 1 epoch） |
| ddp.topology_aware_update_plan_2proc | PASS（2:1 分配，global_samples=6，plan digest 一致） |
| ddp.test_dynamic_nccl_full_state_equivalence | PASS（118 cross-rank + 118 reference tensor 条目，atol/rtol 通过） |
| ddp.test_dynamic_nccl_acvjepa_integration | PASS（integration_test passed） |

### 容器与契约（3/3 + 契约脚本）
ddp.make_demo_data、docker.compose_config、**docker.local_chaos_demo（build→up 3 容器 Healthy→chaos-ci profile 容器内跑离线契约→down 全流程实测通过）**、contract.offline_chaos_contract_sh。

### 其他审计（本轮新增）
- **PPTX 一致性**：`verify_artifacts/ppt_audit_report.json` —— 12 页 ↔ slide_content.md 12 区块完全对齐；关键事实（10.3 亿融资/35 亿估值/2025-11 离开/2026-03 融资/LeBrun/Xie/四地/三论文/三来源）全部核对通过，无数量不匹配。

## 本机验证备注（环境适配，非逻辑改动）

1. **pytest 不可用**：全局 deepeval 插件损坏（`urllib3.packages.six.moves` 缺失）→ 全部改用 `python -m unittest`（与 `scripts/run_offline_chaos_contract.sh` 一致）。
2. **CUDA 冒烟不稳定**：`ac_vjepa_core` 在 CUDA 路径报 "CUDA error: unknown error"（原交付验证即 CPU 路径）→ 验证脚本以 `CUDA_VISIBLE_DEVICES=-1` 强制 CPU 运行；torch 2.5.1 Windows 忽略空串 `''`，须用无效设备号。
3. **Docker compose**：`--web.enable-lifecycle=false` 在 prometheus v3.5.0 报 "unexpected false" → 改为 `--no-web.enable-lifecycle`（等价语义）；实测 3 容器 Healthy。
4. **Windows 偶发文件锁**：chaos framework 的 `TemporaryDirectory` 清理偶发 WinError 32 → 加 `ignore_cleanup_errors=True`（Python 3.12 官方推荐）；np.load 句柄用 `payload.close()`；hitl 演示改用临时文件保证幂等。
5. **torchrun 不可用于 Gloo 回归**：torch 2.5.1 Windows wheel 无 libuv（`USE_LIBUV=0` 可修复 store 层），且 torchrun elastic agent 自身完整 import torch 导致 WinError 1455（页文件不足）→ 用 `scripts/manual_gloo_runner.py` 手工 spawn 两个 rank 进程（RANK/WORLD_SIZE env + gloo），语义回归 4/4 通过。
6. **容器内契约脚本**：chaos-ci 镜像按 .dockerignore 排除 `.github/workflows/`（CI 文件不入演示镜像），`run_offline_chaos_contract.sh` 对这两个校验在文件缺失时显式 SKIP（容器语义），仓库 checkout 场景保持严格校验。
7. **HF 真实权重训练冒烟（H 组）**：需 `weights/vjepa2.1-vitb-fpc64-384/model.safetensors`（438.9 MB，SHA 见 DATA_MANIFEST）；单进程 CPU 384px 冻结骨干，`--init-from vjepa2hf:<path>:frozen` → 1 epoch 4 步训练 → 校验 `last.pt` 落盘（约 414 MB，测后清理）；权重缺失时 SKIP 不阻塞其余项。
8. **M2 数据装配（I 组，portable synthetic）**：`robocasa_adapter.py` 契约冒烟（合成后端，无 RoboCasa 依赖）→ `scripts/make_synthetic_clips.py` 合成 B 层帧 → `video_to_windows.py` 切域适配窗口（零动作/事件 + domain_adaptation_only 标记）→ `assemble_m2_dataset.py` 端到端装配（C 层 episode→commit→windows + B 层窗口 + split 隔离校验 + DATA_MANIFEST 登记）。B 层窗口可被训练器 `WindowEpisodeDataset` 直接加载（provenance 存于 .pt）。

## 修复清单（接管期间对交付代码的改动，均已记录于决策记录）

| 文件 | 改动 | 原因 |
|---|---|---|
| elastic_data_cursor_ledger.py | `_local_path_from_uri` 用 urlparse 解析 `file://` | Windows 下 `file:///F:/...` 剥前缀得 `/F:/...` → 非法路径 `\\F:\...` |
| multimodal_hitl_tamper_evident_ledger.py | 同上（verify_local_artifacts） | 同上 |
| generate_pointcloud_pairs_ddp.py | 新增 `_local_uri_root()` 统一解析（2 处） | 同上 |
| heterogeneous_microbatch_chaos_framework.py | 4 场景补 `ledger.close()`；TemporaryDirectory 加 ignore_cleanup_errors | Windows 下 SQLite 文件锁导致临时目录清理失败 |
| test_shadow_rca_and_pointcloud_pipeline.py | np.load 后 `payload.close()` | Windows 下 npz 句柄未释放导致 TemporaryDirectory 清理失败 |
| hitl_quarantine_review.py | 顶层演示改用临时文件路径（幂等） | 硬编码 `/tmp/...` 在 Windows 非法且重复运行状态冲突 |
| docker-compose.local-chaos.yml | prometheus flag 改 `--no-web.enable-lifecycle` | prometheus v3.5.0 参数解析不兼容 |
| scripts/run_offline_chaos_contract.sh | CI workflow 校验在容器内文件缺失时显式 SKIP | chaos-ci 镜像排除 .github，原脚本容器内必失败 |
| verify_all.ps1 | 新增：unittest 运行器、CUDA 禁用、USE_LIBUV=0、manual_gloo_runner 替代 torchrun | 本机环境适配 |
| scripts/manual_gloo_runner.py | 新增：手工双进程 Gloo 启动器 | torchrun 在本机页文件限制下不可行 |
