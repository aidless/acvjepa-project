# 根因驱动域随机化与 DDP 点云—视频对生成指南

## 1. 从根因分析到受约束的域随机化更新

影子模式发现性能退化后，正确流程不是“把所有随机化范围调大”。应先将 RCA 结果转换为**受审批边界约束的候选域随机化配置**：改变哪些场景更常被采样、哪些已有参数区间内的分位/分层被覆盖、哪些样本只进入压力评测；而不自动扩大材料、接触、相机或动作范围。

NVIDIA 的 Isaac Lab 教程展示了以环境配置中注册的 reset `EventTerm` 执行光照和相机位姿随机化的方式；每次 episode reset 应用新的随机化。该机制适合接收版本化、已审批的随机化配置，但不是跳过先验审核的理由。[1]

`dr_policy_tuner.py` 接收 `RCAReport` 和 `ApprovedDRConfig`，输出 `DRProposal`。`ApprovedDRConfig` 把每一个连续变量拆为：不可越过的 `approved_low/high`，以及当前可采样的 `active_low/high`；提案器不修改数值区间，只重排场景权重并写出建议。

| RCA 根因 | 允许的自动提案 | 不允许自动做的事 | CI/CD 动作 |
|---|---|---|---|
| `visual_domain_shift` | 提高现有弱光、室内色温、相机微扰、材质/纹理、部分遮挡、传感器噪声配置的采样权重 | 增大相机外参或曝光到批准范围外；改动软物体材料参数 | 创建视觉 DR 候选配置，进入仿真验证。 |
| `soft_physics_gap` | 重加权已有刚度、阻尼、摩擦、接触刚度、褶皱几何、低速动作回放分层 | 从一次难例推断全新的材料范围；开放式动作探索 | 创建物理数据/Sim-to-Real 工单。 |
| `sensor_or_calibration` | 添加有界延迟/噪声诊断 profile，并提高其评测覆盖 | 用随机化掩盖未修复的时间同步/标定故障 | 暂停发布，先关闭传感器事件。 |
| `uncertainty_miscalibration` | 生成隔离的压力集，用于不确定性校准评估 | 直接降低保持阈值或把压力集默认混入训练 | 阻断扩大，跑校准套件。 |
| `edge_runtime` / `engine_quantization` | 不改变物理/视觉主数据分布 | 以更多合成数据掩盖 p99/无效输出问题 | 暂停发布，修复 edge engine。 |

运行示例：

```bash
python3 dr_policy_tuner.py > dr_proposal.json
```

`dr_proposal.json` 不能直接进入生产数据生成。它应经过：配置 schema 校验、审批工单验证、仿真评测、数据质量检查和训练候选评测，随后才能成为新的 `SimJob` 编译输入。

## 2. DDP 数据并行生成架构

这里的 DDP 指**数据并行仿真生成**，不是梯度训练。每个 rank 拿到同一份稳定排序的 JSONL `SimJob` manifest，并以 `index % world_size == rank` 确定处理归属。这样同一 `job_id` 在一个 release 中只被一个 rank 生成，重跑时分片仍可复现。

```text
approved SimJob JSONL + approved DR config
  → torchrun / process group
       ├─ rank 0: jobs 0, W, 2W, ...
       ├─ rank 1: jobs 1, W+1, 2W+1, ...
       └─ rank N: jobs N, W+N, 2W+N, ...
  → 每 rank：仿真 rollout → RGB-D/点云/动作/接触 → quality gate → SHA-256
  → 每 rank：rank-N.jsonl
  → barrier
  → rank 0：合并、检查 job_id 唯一性、写入 dataset commit
  → 先上传 episode staging，再写 remote per-episode commit，最后写 dataset commit
```

`generate_pointcloud_pairs_ddp.py` 使用 `torch.distributed` 的进程组与 barrier；真实物理仿真由 `IsaacLabAdapter` 实现，便携式演示只使用显式标记为 **contract-test-only** 的 `SyntheticDeformableBackend`。不能把该测试后端产物当作具备物理有效性的软体训练集。

## 3. 两阶段对象存储提交协议

对象存储通常没有跨对象原子 rename。脚本因此把“训练可见”定义为 commit manifest 存在且哈希匹配，而不是对象文件已出现。

| 阶段 | 本地/远端位置 | 允许的动作 | 下游训练是否可见 |
|---|---|---|---|
| 生成 | 本地 `split/job_id/` | 写 `episode.npz`、`metadata.json`，运行 quality gate | 否。 |
| staging | `remote/.staging/release_id/job_id/` | 上传产物；记录种子、版本、哈希 | 否。 |
| episode commit | `remote/commits/release_id/job_id.json` | 写入产物 URI、质量状态、SHA-256 | 单 episode 审核工具可见。 |
| dataset commit | `remote/dataset_commits/release_id.json` | 合并各 rank 报告，确认 job 唯一性 | **是**，训练只读此清单。 |

对于 `file://` URI，脚本在同一文件系统使用 staging 再 `os.replace()` 作为本地原子提交模拟。对于 `s3://` URI，脚本要求预配置的 `aws` CLI，仅将产物写到 staging；写出小型 commit JSON 后，数据消费者应验证其中 SHA-256，再读取产物。脚本不包含、读取或输出任何云端凭据。

## 4. 运行方式

先使用已批准物理先验生成 `SimJob`：

```bash
python3 sim2real_hard_example_compiler.py \
  --demo --output approved_jobs.jsonl
```

两进程本地契约测试与本地对象存储模拟：

```bash
torchrun --standalone --nproc_per_node=2 generate_pointcloud_pairs_ddp.py \
  --manifest approved_jobs.jsonl \
  --output /shared/soft_pairs \
  --remote-uri file:///shared/object_store \
  --release-id soft-grasp-rca-001 \
  --backend contract --dist-backend gloo --max-points 1024
```

真实多节点 Isaac Lab 生成示例：

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=29500 \
  generate_pointcloud_pairs_ddp.py \
  --manifest approved_jobs.jsonl \
  --output /shared/soft_pairs \
  --remote-uri s3://approved-bucket/soft-pairs \
  --release-id soft-grasp-rca-001 \
  --backend isaac_lab --dist-backend nccl
```

多节点运行需满足以下前提：所有 rank 能访问同一 `--output` 路径或存在等效的集中 rank-report 汇聚机制；每个节点安装相同的固定仿真器/资产/驱动版本；远端 URI 已以最小权限方式预配置；消费者只读取最终 dataset commit。若共享文件系统不可用，应把 rank report 汇聚改为单独的对象存储收集阶段，而不是依赖 rank 0 读取其他节点的本地文件。

## 5. 质量、可追溯与发布门控

每个可训练 episode 应包含同步 RGB-D、机器人基坐标系点云、点掩码、实际执行动作、本体状态、接触、单调时间戳、相机标定、物理/视觉/传感器参数、simulator/asset 版本和 SHA-256。`accepted=true` 仅代表基础数据契约通过；训练前仍需按真实/合成来源、对象、场景、相机和任务拆分验证，并用独立 stress 集测试不确定性与保持机制。

分布式生成本身不应触发模型训练或部署。它只产生版本化数据集提交；训练 CI、仿真闭环、Jetson 性能、影子模式和受限灰度发布仍是后续的独立门槛。

## 参考资料

[1]: [NVIDIA, *Sim-to-Real Strategy 1: Domain Randomization*](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)

[2]: [NVIDIA, *Isaac Lab Documentation*](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/index.html)
