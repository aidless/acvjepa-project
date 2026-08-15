# 可恢复 Sim-to-Real 生成到 AC-VJEPA 增量微调：端到端自动化指南

## 1. 核心原则

节点重启、网络闪断和训练中断是分布式流水线的常态，而不是异常分支。可恢复设计的关键不是“让某个进程永不失败”，而是将每个阶段设计为**可重放、可验证、幂等、可跳过已完成结果**：作业以稳定键标识；worker 通过有限租约取得写权；产物先经质量/哈希校验；对象存储仅在 commit 写出后对训练可见；训练和发布只消费不可变 dataset commit。

PyTorch 的 `torchrun` 官方教程说明，弹性启动可在进程/节点故障后尝试重启工作进程，并从应用保存的 snapshot 恢复训练状态。[1] 这能够恢复训练计算，但不能替代仿真数据任务的租约、产物去重、对象存储两阶段提交或版本化数据集清单。

## 2. 自动化部署方式：两种可行路径

| 方案 | 运行体验与适用场景 | 成本与复杂度 | 局限 |
|---|---|---|---|
| **持久 CI/作业编排器 + 共享对象存储/数据库** | 面向多节点 GPU 仿真、长时训练和生产候选发布；支持 worker 自动重启、队列、审计和权限控制 | 配置较多，需要持久数据库、对象存储与 GPU runner | 是大规模/无人值守方案的必要运营成本。 |
| **实验室单机或小集群的共享账本 + 手动/CI 触发 worker** | 适合先验证 lease、提交和增量训练数据契约；可在共享 POSIX 存储上运行 | 成本和部署复杂度低 | 无跨区域高可用；SQLite 账本不适合高并发跨站点生产。 |

对于当前软体抓取实验，建议先以第二种方案验证端到端数据与模型契约；当需要多节点长时 Isaac Lab/RoboCasa 生成、对象存储同步和频繁训练时，再迁移到第一种方案。无论哪种方案，真实硬件 shadow/canary 不应由普通的云端训练 worker 直接触发，必须保留既有的独立安全与发布授权门控。

## 3. 可恢复生成：作业状态机

```text
REGISTERED/PENDING
  └─ acquire lease → LEASED (worker_id, attempt, lease_until)
       ├─ heartbeat
       ├─ generate → local quality gate → staging upload → hash verification
       │    └─ write episode commit → COMPLETE
       ├─ retryable failure → RETRY (bounded exponential backoff)
       └─ attempt limit / checksum conflict / quality rejection → QUARANTINED

LEASED + worker/node loss
  └─ lease expiry → PENDING/RETRY → later worker may reclaim
```

### 3.1 稳定键与幂等性

每个 job 的稳定键为：

> `SHA-256(job_id, simulator_version, physics, visual_randomization, sensor_randomization, action_perturbation, data_contract_version)`

相同键且相同 payload 的重复注册是无害的；相同键却 payload 不同是冲突，必须隔离。这样重启 worker 时可以安全跳过已完成 commit，而不会把“另一个版本的布料刚度分布”误认为同一 job。

### 3.2 续传与重试

| 失败点 | 检测 | 恢复动作 | 绝不做的事 |
|---|---|---|---|
| worker/节点挂掉 | 心跳停止且租约到期 | 新 worker 回收 lease，从 job 重新生成或验证已存在 commit | 直接相信旧 staging 文件已经完整。 |
| 仿真器进程崩溃 | 非零退出/异常 | 标记 `RETRY`，指数退避；超过限制转 `QUARANTINED` | 无限重试或把崩溃样本标为正常。 |
| 上传网络闪断 | 上传异常或 commit 不可读 | 仅重试幂等 staging 上传与 commit 验证，加入抖动 | 覆盖不同哈希的已完成 commit。 |
| rank 重启 | job 未达 `COMPLETE` | 由账本重新分配；`COMPLETE` 直接跳过 | 依赖 rank 临时内存判断完成。 |
| 最终清单中断 | dataset commit 缺失 | 从 episode commits/rank reports 可重建最终清单 | 训练扫描 staging 目录。 |

`resumable_simjob_ledger.py` 实现了共享 SQLite 的最小 lease ledger，适合实验室共享存储。`resumable_generate_worker.py` 将其接到现有点云—视频对生成器：本地质量通过后才上传；上传采用有界指数退避；远端 per-episode commit 和本地 SHA-256 成功后才写 `COMPLETE`。

对于大文件，目标对象存储应使用其支持的 multipart/分片上传与完成提交能力，并在最终 commit 中记录/校验对象哈希；本参考实现只抽象出 staging+commit 协议，并以预配置 `aws` CLI 执行上传。生产环境还应实现列出已上传 part、续传 upload ID、校验 ETag/服务端 checksum、清理孤儿 multipart upload 等存储提供商相关逻辑。

## 4. 数据集提交如何进入 AC-VJEPA 训练

### 4.1 只消费最终 dataset commit

生成器输出 `dataset-<release>.json`，其中每条 accepted episode 包括：本地/缓存路径、`episode.npz` 哈希、`metadata.json` 哈希、远端 episode commit URI、split 和 job ID。训练输入转换器只读取该不可变清单，不遍历 `/.staging/`、临时 rank reports 或“最新目录”。

```text
verified dataset commit
  → cache/prefetch（若数据仅在对象存储）
  → dataset_commit_to_acvjepa_windows.py
  → train_windows.jsonl + training_input_manifest.json
  → train_ac_vjepa_ddp.py (candidate checkpoint)
  → offline / sim / edge evaluation
  → shadow release candidate
  → canary only after existing automated gates + authorization
```

### 4.2 点云—视频对与当前轻量模型的兼容接入

当前 AC-VJEPA 训练器的最小训练契约是 `context_video`、`context_proprio`、`future_video`、`future_proprio`、`executed_actions` 和 `future_events`。`dataset_commit_to_acvjepa_windows.py` 从已校验的 `episode.npz` 构造滑动窗口，并保存所有兼容张量；同时将 `context_point_cloud_xyz` 和 `context_point_mask` 保留为附加张量，以便未来接入点云编码器。

当前视觉骨干不直接消费点云；因此该步骤不会虚假声称“点云已被模型利用”。有两条演进路径：一是维持现有 RGB 视频模型，把点云用作数据审计、几何评测和后续监督；二是在新的、独立版本化实验中引入点云编码器并融合到 `StateEncoder`，随后重新跑完整的离线、仿真、边缘和影子回归。两者不能混用同一模型版本标签。

### 4.3 训练输入的数据血缘

`training_input_manifest.json` 记录 dataset commit 的 SHA-256、窗口 manifest、上下文长度、预测时域和维度。checkpoint/模型注册表还应存储：父 checkpoint、训练代码 commit、动作 schema、预处理、DR config、物理先验、数据 commit、训练超参数、评测报告和 engine hash。任何不一致都应阻断 shadow 创建。

## 5. 端到端 DAG 与恢复点

| 阶段 | 输入 | 输出/检查点 | 自动继续条件 | 自动停止条件 |
|---|---|---|---|---|
| `GENERATE` | approved SimJob + DR config | episode commit | 租约有效、质量通过、哈希匹配 | job 冲突、质量拒绝、次数耗尽。 |
| `DATASET_COMMIT` | accepted episode commits | immutable dataset commit | job 唯一、hash 完整 | 缺 rank report/重复/缺 commit。 |
| `PREPARE_WINDOWS` | dataset commit + 本地缓存 | training manifest | episode hash 匹配、窗口完整 | staging/缺文件/契约不符。 |
| `TRAIN_CANDIDATE` | training manifest + parent checkpoint | candidate checkpoint | checkpoint 原子写入且可 resume | NaN/数据不符/训练失稳。 |
| `OFFLINE/SIM/EDGE_EVAL` | candidate + frozen sets | signed evaluation report | 全部必过门达标 | 任一安全/性能门失败。 |
| `SHADOW` | edge engine + release manifest | shadow evidence/RCA | 覆盖与延迟达标 | 无效输出/安全门失败/性能退化。 |
| `CANARY` | shadow-approved candidate | canary audit | 授权、白名单、身份稳定路由 | 自动回滚或人工暂停。 |

PyTorch 弹性训练应定期原子保存 `model`、`optimizer`、`scaler`、epoch、global step、数据 commit SHA 和训练输入 manifest SHA；重启时必须检查这些 hash 是否仍指向相同数据/动作/预处理契约。仅恢复模型权重但悄悄更换数据清单，会破坏实验可重现性。

## 6. 已验证的最小链路

本参考实现已执行以下最小本地链路：

1. 两进程数据并行生成将 3 个 SimJob 分片，写出点云—视频对、per-episode commit 和最终 dataset commit。
2. `dataset_commit_to_acvjepa_windows.py` 从最终 commit 生成 15 个 AC-VJEPA 训练窗口，并写入数据血缘 manifest。
3. 现有 `train_ac_vjepa_ddp.py` 在 CPU 上完成一轮增量微调烟雾测试并原子保存 `last.pt`。
4. 租约 worker 先完成 1 个 job，随后由新 worker 自动处理剩余 2 个 job，账本最终达到 `COMPLETE: 3`。

这些测试验证的是数据/训练/恢复契约，不是柔性物体物理真实性或真实机器人上线许可。

## 参考资料

[1]: [PyTorch, *Fault-tolerant Distributed Training with torchrun*](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)

[2]: [NVIDIA, *Sim-to-Real Strategy 1: Domain Randomization*](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/09-strategy1-dr-teleop.html)
