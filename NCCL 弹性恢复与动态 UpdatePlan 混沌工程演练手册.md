# NCCL 弹性恢复与动态 UpdatePlan 混沌工程演练手册

**作者：Manus AI**  
**日期：2026-08-15**  
**适用范围：多节点多 GPU 的 AC-VJEPA 同步 DDP/NCCL 训练；仅限隔离、授权、无机器人控制负载的测试集群。**

## 1. 核心结论

NCCL 通信故障与弹性恢复的关键不是“尽量让幸存 rank 继续”，而是维护一个更严格的不变量：**任何未确认的 update 都不能成为模型、EMA、optimizer 或数据游标的事实来源。** 对当前 PyTorch DDP/`torchrun` 架构而言，发生 worker、节点或成员变更时，最安全的恢复单元是整个 worker group：停止当前 group，放弃所有 in-flight collective 和本轮 `UpdatePlan`，从最后一个确认 checkpoint 重新 rendezvous，获取新 rank/world-size，重新验证拓扑与工作清单，再发起新的计划。[1]

> **不要混淆两个恢复层。** NCCL 原生 API 提供 communicator abort、shrink/grow 等能力；但当前 Python DDP 训练器不应在进程内直接调用这些原生接口来“缩小 communicator 后继续训练”。DDP bucket、参数副本、optimizer state、dataloader 分片、EMA、rank/world-size 与自定义 `UpdatePlan` 都需要一致重建。对于本项目，推荐将 NCCL 错误升级为受控 worker group 重启；只有在拥有专门原生通信运行时、完整重构所有上述状态并经过单独验证时，才讨论 `ncclCommShrink` 路线。[2]

NCCL 对异步网络错误的原则是：该 operation 通常不再 progress，communicator 应被 abort 并销毁；其后才可能创建新的 communicator。[2] PyTorch 的 watchdog 和 elastic agent 可用于把这一错误转化为可观测的进程终止与重启，但它们不会证明 optimizer/data state 已正确恢复。因此恢复验收必须把 **checkpoint 提交、epoch/plan 失效、全状态比较、数据血缘、拓扑/工作清单摘要与重启预算** 作为同等重要的门槛。

| 不变量 | 失败时必须发生什么 | 绝不可发生什么 |
|---|---|---|
| Collective 序列一致 | 停止当前 group，转储诊断，重启后重建 | 部分 rank 用旧 communicator 再发一个 collective |
| Optimizer/EMA 原子性 | 仅从确认 checkpoint 恢复 | 使用“已经 step 但未 checkpoint”的内存状态 |
| UpdatePlan 绑定性 | restart/membership/topology/work digest 变化后生成新计划 | 复用旧 `plan_version`、旧 `RANK` 或旧 work assignment |
| 样本账本正确性 | 恢复时重建可审计 data cursor/commit | 把不完整 update 的 sample 记为已提交 |
| 状态正确性 | 重启后运行全状态/参考对照与影子评测 | 仅因进程重新启动就宣称模型可发布 |

## 2. NCCL 通信异常：分类、边界与响应

### 2.1 先区分应用契约错误与 fatal 通信错误

NCCL 文档将 `ncclInvalidArgument` 解释为无效参数：该调用没有生效，communicator 可继续使用；而 `ncclInvalidUsage`、CUDA/system/internal error 以及异步网络错误对 communicator 是 fatal，应 abort 并重建。[2] 对 PyTorch DDP 而言，实际应对更保守：即使错误根因是应用分支、batch 不一致或调用顺序错误，也不应尝试让该 DDP process group 继续执行下一 update，因为未完成的 collective、梯度 bucket 和 CUDA stream 状态可能已经不可靠。

| 观测信号 | 首要假设 | 当前 DDP 处理边界 | 恢复动作 |
|---|---|---|---|
| `UpdatePlan` digest / schema / world-size 失败 | 控制面、rank 映射或数据合同分歧 | 尚未进入 backward，属于 fail-closed 逻辑错误 | 不做 NCCL 重试；保存 manifest，修复输入/代码后从同 checkpoint 重启 |
| `actual_samples != global_samples`、loader 耗尽 | 数据分桶/迭代器合同失败 | 可能发生在 collective 前或后；本轮不可提交 | 停止 group，审计 work IDs 和 data cursor，重新计划 |
| `TORCH_NCCL_DESYNC_DEBUG` 指向 rank 不同 collective 次数 | 应用分支、`no_sync` 使用或异常路径不对称 | 通信顺序已失配 | 不要增大 timeout；保留 flight recorder，修复调用对称性 |
| watchdog timeout / async NCCL error | 网络、GPU/驱动、rank 挂死或 collectives 失配 | communicator/stream 可能不可再用 | 由错误策略退出 worker group，重启后重新初始化 NCCL/DDP |
| CUDA/Xid/ECC 或 system error | GPU、驱动、NIC/HCA、PCIe/NVLink 或节点健康 | 不应以训练重试掩盖硬件告警 | 隔离节点，转交基础设施；仅在健康替代节点恢复 |
| rendezvous 不可达 / membership 变更 | 控制面或节点可用性问题 | rank/world-size 不再稳定 | 结束当前 group；新 rendezvous 后重新读环境变量与 checkpoint |

**边界条件一：超时不是网络故障的证明。** all-reduce 超时也可能由 rank 在 dataloader、Python GIL、CUDA kernel、OOM 异常分支或错误的 `no_sync()` 中先停住造成。必须用计划摘要、per-rank phase telemetry、PyTorch trace buffer、NCCL RAS 和系统健康证据联合分类，不能仅据 “NCCL timeout” 调大 timeout。

**边界条件二：网络恢复不等于 communicator 恢复。** 即使 RDMA 连接或路由很快恢复，已经观察到 async error 或 watchdog timeout 的 communicator 仍应视为不可用。让旧进程“再试一次 all-reduce”会把错误语义变成不可审计的偶发行为。

### 2.2 PyTorch watchdog 与错误处理配置

PyTorch ProcessGroupNCCL 提供异步错误策略、trace buffer、timeout dump、desync debug、per-collective timing 与 monitoring heartbeat。`TORCH_NCCL_DUMP_ON_TIMEOUT` 需要配合非零 `TORCH_NCCL_TRACE_BUFFER_SIZE`；monitoring 可在 watchdog heartbeat 停滞后中止进程，避免作业无限占用 GPU。[3]

对于由 `torchrun` 管理、并希望在故障后整个 worker group 重新创建的训练作业，应在**隔离演练环境**中验证 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` 的行为：其语义是 abort communicator 并终止进程。不要把这一建议机械复制到所有部署；应和 PyTorch/NCCL 版本、作业编排器、log collection、checkpoint 频率及组织的故障处置政策一起测试。`=2` 只 abort communicator 而不终止进程，适合拥有显式进程内恢复逻辑的系统；当前 Python DDP/动态计划代码并不具备该恢复器。默认 `=3` 会终止进程但不 abort communicator；同样需以实际版本行为和资源释放情况验证。[3]

建议的**演练诊断配置**应放进临时、版本化的作业 profile，而不是常规生产训练脚本：

| 目的 | 临时诊断能力 | 演练后应验证 |
|---|---|---|
| 找到 collective 尾部与顺序差异 | per-collective timing、desync debug、rank phase marker | 可定位问题出现于 plan、data、forward、backward 还是 AllReduce |
| 保存 timeout 证据 | trace buffer + timeout dump + 独立工件路径 | 错误前后的 collective、rank、epoch 和 plan 可关联 |
| 防止无穷 hang | watchdog monitoring + 已批准 heartbeat/等待上限 | 作业在预算内退出/重启，而不是无限等待 |
| 判断 NCCL job 状态 | RAS status/JSON/monitoring（若版本可用） | 识别 missing/unresponsive rank、collective count 与 communicator 状态 |
| 区分硬件/网络 | GPU ECC/Xid、温度、NIC/HCA/交换机错误计数 | 故障没有被误归为训练逻辑 |

NCCL RAS 自 2.24 起提供低开销的进程健康、communicator 状态和 outlier 观察；它可以显示 unresponsive/missing rank 与 collective count 不一致，但文档明确说明短暂 count mismatch 在作业仍进展时未必是错误。因此应连续观测“是否进展”，而不是把单次 RAS 快照当成故障判决。[4]

### 2.3 `torchrun` 弹性边界

`torchrun` 的失败与成员变化模型是 group 级别的：worker、agent 或节点故障会导致 worker group 重启；scale-up/scale-down 时现存 worker 被停止，新 group 以新的 `RANK` 和 `WORLD_SIZE` 启动。`RANK` 不稳定，且 elastic 运行不能硬编码 `WORLD_SIZE`。[1]

因此本项目的正确状态机如下：

```text
HEALTHY_UPDATE_BOUNDARY
  ├─ gather trusted topology + telemetry + verified work manifest
  ├─ broadcast/verify TopologyAwareUpdatePlan
  └─ execute one DDP update
       ├─ success → optimizer + EMA complete → atomic checkpoint → COMMIT → HEALTHY_UPDATE_BOUNDARY
       └─ any contract/NCCL/health failure → SUSPECT

SUSPECT
  ├─ block next plan and mark current update UNCOMMITTED
  ├─ best-effort collect trace/RAS/system evidence (bounded time)
  ├─ exit/abort current worker group according to orchestrator policy
  └─ RENDEZVOUS_RESTART

RENDEZVOUS_RESTART
  ├─ load last COMMITTED checkpoint only
  ├─ read new RANK/WORLD_SIZE/RESTART_COUNT; never reuse cached values
  ├─ gather fresh topology manifest and work manifest
  ├─ reject old epoch/digest/plan; re-shard cursor deterministically
  ├─ preflight full-state/checkpoint integrity
  └─ new plan → RESUMED_SHADOW_VALIDATION → HEALTHY_UPDATE_BOUNDARY or ESCALATED
```

### 2.4 必须原子提交的训练状态

“保存 checkpoint”本身不够。一个可恢复 commit 至少应包含学生模型、EMA target、optimizer、scaler、scheduler、global step、成功 update 的数据 cursor/采样随机数状态、数据集 commit、action/preprocess schema、world size、topology epoch/digest、work manifest digest、plan version、通信配置和前一 checkpoint hash。写入顺序应是：完成一次全局同步 update → 更新 EMA → 写临时 checkpoint → hash/读回验证 → 原子发布 checkpoint URI → 写 `COMMITTED` ledger 记录。只要失败发生在 ledger commit 前，本轮 update 都是 `UNCOMMITTED`。

这意味着在 world-size 改变后，不应承诺“精确保持每个 rank 的数据位置”；应承诺从同一全局数据 commit 和确定性全局 cursor 生成新分片。可以出现经过明确记录的 at-least-once 重读，但不能静默漏读、跳过或把未确认 batch 计入训练账本。若需要严格单样本一次性语义，必须在更高层实现事务性数据 cursor/manifest，而不是依赖 DDP sampler 的局部 index。

## 3. 混沌工程总体设计

### 3.1 演练原则

混沌工程的目的不是最大化破坏，而是验证一个可证伪假设：

> **假设：** 在任一批准的单故障模型下，系统将在预设检测窗口内停止未确认 update，保留足够诊断工件，不复用旧计划，且能从最后一个确认 checkpoint 组成新 worker group；恢复后的全状态、数据血缘和影子评测满足既定验收门槛。

每次演练只改变一个变量。先执行进程内安全 failpoint，再执行受控 worker 退出，之后才考虑由基础设施人员在测试 VLAN/节点池中实施的网络分区。不要把“网络分区 + 节点退出 + rendezvous 停机 + 高负载”叠加到同一轮；那会失去根因与恢复行为的可解释性。

| 演练级别 | 注入层 | 主要验证 | 执行主体 | 默认是否可自动运行 |
|---|---|---|---|---|
| L0 | 纯观察/基线 | 计划、checkpoint、RAS、trace、全状态对照可用 | 训练团队 | 是，低频预提交 |
| L1 | 进程内逻辑 failpoint | 未提交 update 被拒绝；旧 plan 不复用 | 测试脚本 | 是，仅隔离 CI/预生产 |
| L2 | 授权 worker 退出/迟滞 | watchdog、group restart、checkpoint 恢复 | 作业所有者 + 值班 | 否，需审批 |
| L3 | 数据面网络受控退化/分区 | NCCL async error、abort、证据与恢复 | 网络/基础设施值班 | 否，需变更窗口 |
| L4 | rendezvous/控制面分区或节点离开 | 新 group、rank/world size/topology epoch 重建 | 基础设施值班 | 否，需变更窗口 |
| L5 | 硬件告警/多域故障演练 | 人工升级、隔离与停止纪律 | 硬件/基础设施值班 | 否，不做常规自动化 |

### 3.2 统一实验信封（Experiment Envelope）

每轮开始前应生成一份不可变 experiment manifest，并由两类人确认：训练 owner 确认代码/数据/成功条件，基础设施 owner 确认节点池/网络范围/回滚动作。该 manifest 至少包括：演练 ID、环境和作业版本、批准的故障类别、目标 node/rank（仅逻辑标识）、窗口、最大持续时间、最大重启数、上一个已验证 checkpoint、数据 commit、topology/work manifest digest、停机阈值、证据输出位置和授权人。

| 阶段 | 进入条件 | 动作 | 成功信号 | 立即停止条件 |
|---|---|---|---|---|
| Preflight | 专用资源与 checkpoint 已确认 | 采集拓扑/RAS/环境/健康基线；跑一次无故障全状态测试 | baseline artifact 完整、无 ECC/Xid、计划 digest 一致 | 任一硬件/版本/数据血缘异常 |
| Arm | 两方批准，fault budget 未耗尽 | 打开临时 trace/timeout/RAS 工件收集 | marker 写入、无真实故障 | 工件库不可写或诊断配置不一致 |
| Inject | 到达指定安全 marker | 仅注入一类批准故障 | 注入事件可审计、未触及非目标资源 | 范围扩大、超过持续/错误预算 |
| Observe | 故障生效 | 观察 watchdog/RAS/agent/ledger | `UNCOMMITTED`、旧 group 退出、证据齐全 | GPU/网络健康告警、无界 hang |
| Recover | 新 group 建立 | 校验 checkpoint、epoch、计划、work manifest；跑恢复后验证 | 新 epoch、新 plan、无旧 rank 假设 | 旧 plan 被接受、状态/数据对照失败 |
| Close | 通过或安全失败 | 关闭临时诊断、收集工件、记录结论 | 执行恢复摘要与人工 sign-off | 未清理进程/临时网络规则 |

## 4. 针对动态 UpdatePlan 的具体混沌实验

### 4.1 控制面故障：应在 backward 前 fail closed

这些实验优先于网络分区，因为它们可在不触碰基础设施的条件下验证计划安全性。建议在测试代码里使用显式、默认关闭、带 experiment ID 的 failpoint；failpoint 只能抛出受控异常或使本 rank 退出，不能自行修改系统网络。

| ID | 安全注入点 | 预期系统行为 | 关键断言 |
|---|---|---|---|
| P1 | rank 0 生成计划后、广播前终止本次逻辑 | 无 rank 进入 backward；无 optimizer step | 不写 `COMMITTED`；下次新 epoch/plan 才能继续 |
| P2 | 某 rank 接收后篡改本地 plan bytes/digest（仅内存测试） | 摘要共识失败 | `UpdatePlan digest divergence`；无 DDP collective |
| P3 | work manifest 的 cost/locality/provenance 与 plan digest 不同 | `work_manifest_digest` 校验失败 | 不会将不同数据清单当成同一 work ID |
| P4 | topology epoch/restart count 改变但注入旧 plan | epoch 校验失败 | 旧 `RANK`/world-size/plan 不可复用 |
| P5 | loader 少提供一个 work item 或 batch size 错误 | 实际样本合同失败 | 无部分 optimizer step；case 进入数据合同调查 |
| P6 | rank 报告不健康 telemetry | planner 拒绝更新 | 不以削减某个 rank 的微批数替代成员重建 |

对于 P1—P6，成功不应定义为“异常被捕获后继续训练”，而是验证在优化器提交前停止。测试应检查：没有新的 checkpoint ledger commit；当前 plan version 仅存在于失败 manifest；数据 cursor 未被确认推进；旧 plan 的 topology/work digest 在 restart 后不再被接受。

### 4.2 2:1 异构微批与 collective 顺序故障

2:1 的核心风险在于 rank 0 有一个 `no_sync()` 微批，而 rank 1 立即进入最终同步 backward。演练必须区分“预期的等待”与“错误的 desync”。

| ID | 注入点 | 正确预期 | 失败判据 |
|---|---|---|---|
| H1 | rank 0 第一 `no_sync` 前插入有界延迟 | rank 1 在最终 collective 等待；rank 0 最终进入后完成 | 未超过 plan/超时预算且 collective count 继续推进不算故障 |
| H2 | rank 0 错误地把第一微批也做同步（测试分支） | RAS/desync debug 或 watchdog 识别序列失配 | rank 1 不可自行 step；整个 group 退出 |
| H3 | rank 1 在最终 backward 前受控退出 | rank 0 的同步 backward 不能提交 | `UNCOMMITTED`；agent 重启全组；新计划重建 |
| H4 | 最后一批实际大小与 `samples_per_micro_batch` 不符 | 在 backward 前拒绝 | global sample ledger 未变化 |
| H5 | rank 0/1 使用不同 work manifest 成本或血缘 | work digest 不同，计划拒绝 | 不发生“相同 ID 不同样本”的静默训练 |

H1 的关键是不能把同步等待误判为 NCCL 失败。应同时记录 per-rank `phase_start/phase_end`、collective timing、RAS collective count、计划的 `compute_budget_ms/network_guard_ms` 和 watchdog 上限；如果 counters 在移动且在预算内完成，则是 straggler/负载模型数据而非通信异常。

## 5. 网络分区与弹性恢复演练

### 5.1 分区模型必须拆分数据面与控制面

“网络断开”不是单一故障。RDMA/NCCL data plane、TCP bootstrap/RAS、rendezvous 控制面和对象存储/checkpoint 路径可能独立失效。每一类分区会触发不同恢复证据，必须独立演练。

| ID | 分区模型 | 允许的隔离范围 | 应观测的首个信号 | 正确恢复边界 |
|---|---|---|---|---|
| N1 | 单 node 的 NCCL/RDMA 数据面短暂丢包/高延迟 | 专用测试 VLAN、仅目标 HCA/rail、由网络值班控制 | AllReduce p99 上升；若超限则 async/watchdog | 不提交 in-flight update；恢复后仍重建 group，不“继续旧 collective” |
| N2 | 单 node 与其余 nodes 的 NCCL 数据面完全分区 | 专用节点池与测试 rail | RAS missing/unresponsive、NCCL system/async error 或 watchdog | 退出旧 group；用 checkpoint 重启，重建 rank/world size/plan |
| N3 | 单 rail 劣化/不可用、其它 rail 健康 | 测试 fabric 的单 rail | topology/rail 指标、通信 p95 上升、NIC 计数器变化 | 新 epoch 采集健康拓扑；不要训练进程自动改网络配置 |
| N4 | rendezvous/control-plane 不可用、data plane 仍可能短暂健康 | 专用 rendezvous 端点 | agent 无法形成/reform group | 当轮结束为未提交；恢复 endpoint 后新 group 初始化 |
| N5 | checkpoint/object-store 可读性短暂失败 | 仅测试 bucket/prefix | atomic publish/verify 失败 | 不将内存 state 标记为 commit；保留上一个 checkpoint |
| N6 | 误导性局部恢复：某些连接恢复但 rank/epoch 已变化 | 逻辑/测试 harness | old epoch/work digest 被拒绝 | 绝不让旧 plan 与新 group 混用 |

网络基础设施人员可以在专用测试边界使用组织批准的方法来实现 N1—N4；训练脚本只需等待明确的状态信号、超时或 worker 重启。**不要在训练代码中嵌入网络管理命令，也不要在文档里把破坏性网络操作写成可直接复制到共享网络的命令。**

### 5.2 恢复后的验证序列

当 agent 形成新 worker group 后，不应马上恢复正常吞吐。应先进入一个恢复验证窗口：加载 checkpoint、验证其 hash 与 ledger、读取新的 elastic 环境变量、构造新 topology manifest、比对当前 topology/work digest、运行一到数个受控 shadow update，然后再恢复常规计划。

```text
checkpoint hash + commit ledger verified
  → new WORLD_SIZE/RANK/RESTART_COUNT observed
  → fresh topology/work manifests verified
  → new TopologyAwareUpdatePlan broadcast + digest consensus
  → shadow update (no production promotion)
  → full-state / data-ledger assertions
  → normal training resumes
```

若 world size 保持不变且可构造同一固定输入，恢复验证应运行 `test_dynamic_nccl_full_state_equivalence.py` 的 NCCL 版本，并比较完整模型、EMA、optimizer state 与恢复前的 canonical reference。若 world size 改变，不应要求与旧世界规模的逐 bit 参数等价；应要求从同一确认 checkpoint 开始、无旧 plan 复用、样本账本一致、loss/梯度/参数健康在新规模基线内，并在后续固定 world-size 窗口内再次执行全状态验证。

## 6. 量化 SLO、停止条件与证据包

### 6.1 不预设跨集群通用秒数

检测/恢复时间取决于 watchdog、heartbeat、NCCL/网络 timeout、rendezvous、scheduler、checkpoint 大小和对象存储；因此不应把一个固定“30 秒恢复”写成通用承诺。应以健康基线和业务容忍的丢失工作窗口设置 SLO。例如：检测预算、诊断转储预算、group 终止预算、rendezvous 重建预算、checkpoint 加载/验证预算，以及最多允许损失的 committed updates。所有阈值需经过在目标集群的无故障基线与单故障演练验证。

| 度量 | 计算方式 | 成功判定示例（以项目基线设定） |
|---|---|---|
| 检测时间 | 注入 marker → first watchdog/RAS/agent error | 不超过批准的检测预算 |
| 停止时间 | first error → 所有旧 worker 已退出 | 不遗留旧 communicator/进程或无限 GPU 占用 |
| 恢复时间 | restart begin → 新 plan digest 共识 | 不超过环境专属恢复预算 |
| 丢失工作 | last committed step → 故障前 in-flight step 数 | 不超过 checkpoint 策略允许窗口 |
| 计划正确性 | 新旧 epoch、work/topology digest、actual samples | 旧计划零复用；新计划全 rank 一致 |
| 状态正确性 | 全 state/optimizer 比对或恢复 shadow 指标 | 固定规模时满足容差；变规模时满足 restart 合同 |
| 证据完整性 | trace、RAS、环境、checkpoint、ledger、系统健康 | 每轮都有可复核 experiment bundle |

### 6.2 每轮必须归档的证据包

> 最小证据包不是一段错误日志，而是一条从故障到恢复的可验证链。

包括：experiment manifest；授权与变更编号；注入 start/end marker；完整 `TopologyManifest` 与 `TopologyAwareUpdatePlan`；work manifest digest 与 data commit；checkpoint URI/hash/前后 ledger 记录；elastic run ID、restart count、rank/world-size；PyTorch trace/timeout dump；NCCL RAS 输出（若可用）；GPU/NIC/系统健康摘要；每 rank phase telemetry；全状态验证结果；恢复后的影子评测；清理/回滚确认。

## 7. 对现有代码的最小增强建议

以下伪代码描述**进程内逻辑 failpoint 和恢复合同记录**，不会触碰真实网络或其他作业。它可用于 L1/H 系列 CI/预生产演练；N3/N4 的真实网络或控制面故障仍交由授权基础设施流程实现。

```python
@dataclass(frozen=True)
class ExperimentEnvelope:
    experiment_id: str
    allowed_failpoint: str | None
    restart_budget: int
    checkpoint_hash: str
    topology_epoch: str
    work_manifest_digest: str


def failpoint(name: str, envelope: ExperimentEnvelope) -> None:
    if envelope.allowed_failpoint == name:
        # 仅在隔离测试 harness 中触发；不执行网络或节点操作。
        raise ControlledTrainingFault(
            f"experiment={envelope.experiment_id}, failpoint={name}"
        )

# 在安全边界调用：
manifest, plan = topology_aware_next_plan(...)
assert plan.topology_epoch == envelope.topology_epoch
assert plan.work_manifest_digest == envelope.work_manifest_digest
failpoint("after_plan_before_backward", envelope)
metrics = acvjepa_dynamic_update(...)
failpoint("after_step_before_commit", envelope)
atomic_checkpoint_and_commit(...)
```

关键规则是：`after_step_before_commit` 的注入必须使该 update 保持 `UNCOMMITTED`。恢复时只能加载上一个 checkpoint，不能试图使用已在内存中执行但未发布的 optimizer step。恢复测试必须断言新 `TORCHELASTIC_RESTART_COUNT` 或 topology epoch 已变化，并显式拒绝 envelope 中旧计划。

## 8. 实施顺序与选择

| 方案 | 适用情形 | 取舍 | 成本 | 设置复杂度 |
|---|---|---|---|---|
| 逻辑 failpoint + 双进程 Gloo/NCCL 语义回归 | 每次代码变更、快速发现 plan/状态机错误 | 不覆盖真实网络/硬件 | 低 | 低 |
| 专用 GPU 队列的单 worker 退出与恢复演练 | 验证 checkpoint、torchrun、全状态恢复链 | 需要作业重启和专用资源 | 中 | 中 |
| 专用 VLAN/节点池的 RDMA/rail/rendezvous 分区演练 | 验证真实 NCCL async error 和弹性边界 | 必须网络/基础设施授权；风险最高 | 高 | 高 |

建议以第一种作为每次合并前的语义守卫，以第二种作为发布候选的定期演练，以第三种作为基础设施变更、NCCL/CUDA/驱动升级或重大拓扑调整前后的专项可靠性演练。不要将真实网络分区作为频繁的自动化任务；它需要由受控实验编排和人工批准触发。

## 参考资料

[1] [PyTorch 2.13：torchrun（Elastic Launch）](https://docs.pytorch.org/docs/2.13/elastic/run.html)。

[2] [NVIDIA NCCL：Creating a Communicator、错误处理与 Fault Tolerance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)。

[3] [PyTorch 2.13：ProcessGroupNCCL Environment Variables](https://docs.pytorch.org/docs/2.13/torch_nccl_environment_variables.html)。

[4] [NVIDIA NCCL：RAS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html)。
