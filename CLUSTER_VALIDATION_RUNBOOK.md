# 集群验证执行清单（CLUSTER VALIDATION RUNBOOK）

> BACKLOG B1–B9 的落地产物：把「需真实集群/硬件」的未决项转化为可执行的验证步骤。
> 目标环境假设：隔离 GPU 集群（≥2 节点，8× GPU/节点，NCCL，NVLink/IB，共享存储或对象存储），Linux。
> 每个阶段包含：前置条件 → 命令 → 验收标准 → 来源（决策记录/手册）。
> 安全约定：所有演练在受控测试窗口执行；每轮只注入一个故障；任何写操作需审批。

---

## B1. 真实 NCCL 集群压测（拓扑基线 → 微基准 → 应用级动态计划 → 受控故障恢复）

**来源**：`NCCL 真实集群压测与 Gloo 异构微批验证实施手册.md`；决策记录「真实集群压测」条。

前置条件：
- 2+ 节点，NCCL 可用，GPU-NIC 亲和（rail）拓扑确认；共享数据/检查点目录；torchrun elastic 可用。
- 记录硬件拓扑：`nvidia-smi topo -m`、`nvidia-smi -L`、IB 链路（`ibstat`）、`ncclTopoDumpFile`。

阶段化执行：
1. **拓扑基线**：跑官方 `nccl-tests`（all_reduce_bench / all_gather_bench），记录 p50/p95/p99 带宽与延迟，建立健康区间（不预设跨硬件阈值）。
2. **微基准**：`torch.distributed.benchmarks` 或自定义 allreduce p95，验证 NCCL 环境变量（async error handling、timeout）。
3. **应用级动态计划**：`torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=29500 train_ac_vjepa_ddp.py --manifest ... --output ...`（先 `--epochs 1` 小批量冒烟）。
4. **受控故障恢复**：kill 一个 worker → 验证 checkpoint/cursor/UpdatePlan 重建流程（见 B7），确认无部分提交。

验收：
- [ ] allreduce 带宽/延迟 p50/p95 落入健康基线区间（与拓扑预期一致）；
- [ ] 2:1 异构微批 loss-sum 缩放与单进程参考一致（全状态比较，见 B5）；
- [ ] 单节点/多节点吞吐数据已记录；
- [ ] 故障恢复后 cursor 无推进、无部分 checkpoint。

## B2. RDMA/rail 故障演练（分区、rendezvous、rail 失效）

**来源**：`rdma_rail_chaos_guard.py`；决策记录「RDMA/rail」「双层模型」条。

前置条件：专用 VLAN/节点池 + 基础设施授权；`rdma_rail_chaos_guard.py` 部署为**请求守卫**（仅干运行/mock 校验）；真实 executor 由基础设施独立实现（mTLS、标签、TTL、审计）。

执行：
1. 守卫干运行：`python rdma_rail_chaos_guard.py`（只校验请求：dry-run、环境隔离、静态范围、双人审批、TTL、checkpoint/健康、topology/work digest、finally rollback）。
2. 在**获批测试窗口**内由基础设施 executor 注入：rail 链路 down、rendezvous 分区、网卡 flapping；每轮一个故障。
3. 验证：训练组按 worker-group 失败语义重建（不进程内续跑）；告警只 freeze/取证。

验收：
- [ ] 守卫拒绝所有未审批/未隔离请求（烟雾测试已覆盖逻辑）；
- [ ] 真实 executor 事件有审计日志与 TTL；
- [ ] 每个故障单变量注入，恢复合同被验证。

## B3. 生产监控阈值校准（0.85 / 5s / 900s 等初始保护栏）

**来源**：`Kubernetes 生产监控阈值与 Rendezvous—GitOps 并发仲裁手册.md`；决策记录「生产监控」条。

执行：
1. 采集真实 cluster/job 基线：allreduce p95、恢复 p95、checkpoint commit 间隔、cursor 卡住窗口、告警率。
2. 依据 SLO 重新设定 `prometheus_rules.yml` 阈值（现为 0.85 成功率 / 5s 延迟 / 900s 恢复时限的初始值）；用 `update_production_dashboard.py` 同步面板。
3. `validate_monitoring_config.py` 回归（18 alerts / 10 rules / 19 panels 保持通过）。

验收：
- [ ] 阈值有 baseline 数据支撑并记录依据；
- [ ] 无告警风暴（告警与真实事件 1:1 对应）；
- [ ] 告警动作仍限于 allowlist（freeze/mark SUSPECT/取证/通知）。

## B4. FP8 / AMP / CUDA RNG / FSDP-ZeRO 扩展验证

**来源**：`mixed_precision_elastic_recovery.py`；决策记录「混合精度恢复」「Gloo 边界」条。

执行：
1. GPU/AMP 下重跑 `mixed_precision_elastic_recovery.py` smoke（bf16/fp16 scaler/fp8 metadata）；
2. `test_dynamic_nccl_full_state_equivalence.py` 扩展：AMP/TF32/lossy comm hook 独立阈值；CUDA RNG 状态纳入 checkpoint；
3. 若需解冻 300M–1B 骨干：FSDP/HSDP 分片 + EMA 目标编码器全状态字典生成/广播/校验流程。

验收：
- [ ] 混合精度恢复 fingerprint 相等（同一 COMMITTED checkpoint load 后、下一 forward 前）；
- [ ] FP8 metadata / GradScaler / CUDA RNG 纳入 checkpoint 并验证；
- [ ] FSDP 状态下 EMA 一致性校验通过（如启用）。

## B5. Gloo 验证升级为 GPU 全状态对比

**来源**：决策记录「Gloo 边界」条。

执行：在 GPU 多卡环境用 manual_gloo_runner/torchrun 跑 `test_dynamic_nccl_full_state_equivalence.py`，把「同六样本单进程参考」扩展到 GPU + AMP 上下文。

验收：
- [ ] 118 cross-rank + 118 reference 条目在 GPU 容差内通过（含 AMP 上下文独立阈值）；
- [ ] EMA/buffer/AdamW 全状态逐项一致。

## B6. K8s 隔离 Job 混沌演练（immutable manifest 实际 apply）

**来源**：`kubernetes-chaos-contract.yml`、`validate_kubernetes_chaos_lab/ci.py`、`render_kubernetes_chaos_contract.py`；决策记录「K8s CI」「GitOps」条。

前置条件：独立集群 + admission policy + 受保护 GitOps/部署身份（PR runner 不得持有）。

执行：
1. 本地：`python render_kubernetes_chaos_contract.py --image <approved-digest> --output <manifest>`；`validate_kubernetes_chaos_lab.py` / `validate_kubernetes_chaos_ci.py` 通过；
2. 在隔离 namespace `acvjepa-chaos-lab`（pod-security restricted、SA token 禁用）apply Job；容器只跑 `scripts/run_offline_chaos_contract.sh`；
3. 检查 Job 完成状态与 artifact（无 kubectl delete / RDMA / NCCL / secrets 出现）。

验收：
- [ ] Job 成功、artifact 完整、安全断言全部保持；
- [ ] 镜像为 immutable @sha256 digest，无法从 CI 重放。

## B7. torchrun 故障恢复演练（checkpoint 提交协议 + 数据游标真实作业）

**来源**：决策记录「worker-group 重建」「checkpoint ledger」「数据一致性」条。

执行：
1. torchrun elastic 启动作业，正常提交若干 checkpoint（ledger COMMITTED 记录推进 next_offset）；
2. 在 `after_step_before_commit` 窗口 kill 节点 → 验证重放相同 global range、cursor 不推进、无部分提交；
3. 演练 2:1 异构微批下的 rank 失效 → 新 group 重新预留同一连续 range（at-least-once）。

验收：
- [ ] 每个故障后 ledger 与 checkpoint 原子一致；
- [ ] 数据游标 rank 无关（world size/拓扑变化后仍正确重读）。

## B8. 大规模抢占分波恢复参数校准（wave / concurrency）

**来源**：`rapid_recovery_alert_drill.py`；决策记录「弹性恢复性能」条。

执行：在真实 rendezvous/对象存储/网络条件下，跑 `rapid_recovery_alert_drill.py` 校准：分波 admission、指数退避、确定性抖动、checkpoint 读并发上限（RTO_trusted 而非进程重启时间）。

验收：
- [ ] RTO_trusted 有实测分布；wave/concurrency 参数落盘并解释；
- [ ] 无 rendezvous/对象存储风暴。

## B9. checkpoint 三层缓存目标硬件验证（KV pointer / RDMA shard / verified cache）

**来源**：`verified_checkpoint_cache.py`、`checkpoint_cache_load_shedding_simulator.py`、`checkpoint_integrity_corruption_demo.py`；决策记录「checkpoint 加速」条。

执行：
1. 目标 KV（etcd 类）存小 pointer：CAS/fencing、revision 语义；
2. RDMA/节点缓存只读 + hash 校验，失效/损坏/TTL/契约不符回退 durable；
3. 高并发（32+ worker）下 single-flight、token budget、负缓存、完整性熔断实测；
4. 复跑 `checkpoint_cache_load_shedding_simulator.py`（32→1 durable、16 writer→1 CAS）在目标硬件上验证。

验收：
- [ ] 损坏缓存不造成无界 durable fallback，不推进 cursor/pointer；
- [ ] 冷 key 风暴下 durable 读不随 worker 数线性放大；
- [ ] pointer CAS fence 拒绝 stale 覆盖。

---

## 通用验收门槛（全部 B 项）

- 每个演练在**隔离资源池 + 获批窗口**执行，每轮单变量故障；
- 告警动作仅限 allowlist（freeze / mark SUSPECT / 取证 / 通知）；
- 训练/恢复/监控三方记录（ledger/指标/审计）对齐可复核；
- 完成后在 `决策记录.md` 追加条目并在 `BACKLOG.md` 置 done（附证据）。
