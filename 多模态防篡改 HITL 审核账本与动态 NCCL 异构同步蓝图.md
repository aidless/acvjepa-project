# 多模态防篡改 HITL 审核账本与动态 NCCL 异构同步蓝图

**作者：Manus AI**  
**日期：2026-08-15**  
**状态：研究与工程原型；不构成生产安全认证，也不连接或控制真实机械臂。**

## 1. 目标、边界与结论

本蓝图为 AC-VJEPA 的隔离难例闭环增加两层能力。第一层是**多模态、可核验、篡改可见**的 Human-in-the-Loop（HITL）审核账本：它把 RGB-D 视频、点云、机器人本体状态、实际执行动作、接触事件、标定信息、模型输出和审核补丁绑定为同一份证据包。第二层是**稳定成员集合内的 NCCL/DDP 异构同步训练**：它允许高吞吐 rank 在同一次同步更新中处理更多微批，但仍让全部 rank 只进行一次对齐的梯度 AllReduce 和一次共同的 optimizer step。

> **核心结论：** 哈希、Merkle 根与签名能够使证据或审核事件的事后修改、替换和重排变得可检测；它们并不能单独防止拥有数据库和密钥权限的攻击者删除全部本地记录。因此，生产部署必须把链头锚定复制到独立的 WORM/透明日志系统，并使用独立身份、密钥管理、对象存储版本控制和保留策略。[1]

> **核心训练结论：** 在同步 DDP 中，异构负载均衡不能意味着快 rank 额外执行 optimizer step。正确方式是：每个 update 前广播不可变 `UpdatePlan`；各 rank 执行不同数量的本地微批；仅各自的最后一微批开启 DDP 同步；用按全局有效样本数缩放的 loss 保持梯度等价于全局样本均值。NCCL 的每个 collective 必须由所有 rank 按一致的序列参与，否则可能出现挂起、崩溃或数据损坏。[2]

本方案的明确**非目标**包括：将审核补丁直接转化为机器人动作、放松 SafetyKernel 阈值、绕过双人审批、自动发布模型，或以“账本已验证”为理由继续执行物理任务。任何证据不一致、签名失效、传感器契约失败、NCCL 计划分歧或成员变化都必须保持 case/训练 update 的隔离状态。

| 资产 | 可信目标 | 本蓝图提供的控制 | 仍需外部控制 |
|---|---|---|---|
| 原始 RGB-D、点云、动作、接触、标定 | 证据未被静默替换 | SHA-256、统一清单、Merkle 证据根、重哈希 | 版本化对象存储、最小权限、加密、保留与备份 |
| 审核动作与修正补丁 | 责任可归属、不可静默改写 | Ed25519 签名、角色授权、前序哈希、双人同 patch 审核 | 企业身份系统、KMS/HSM、撤销、审计与合规流程 |
| 账本整体历史 | 删除/回滚可被发现 | 周期性链头锚定、独立验证器 | 外部 WORM/透明日志、多站点副本、告警响应 |
| `UpdatePlan` | 所有 rank 对同一 update 使用同一计划 | 数字张量广播、canonical JSON、SHA-256 共识摘要 | 集群身份、网络隔离、NCCL 监控、checkpoint/Rendezvous 机制 |
| 训练样本贡献 | 等价于全局有效样本平均梯度 | `W / N_global` 缩放、最终微批同步 | 正确的 padding/mask loss reducer、数据分片和评测门控 |

## 2. 威胁模型：应对什么，而不承诺什么

### 2.1 可检测的攻击与故障

该账本在私钥没有泄露、至少一份外部锚定存活并可访问的前提下，可检测以下情况：单一模态文件被替换或截断；同内容但不同来源/时间范围/解析契约的文件被冒充；审核 payload 被 SQL 直接更新；事件顺序被改写；未授权角色伪造验证或复核事件；以及本地链在已锚定位置之前被静默回滚。每一个 artifact leaf 同时包含 URI、内容哈希、字节数、时间区间、生产者和 schema version，因此“只替换对象存储路径”或“只保留同一字节内容”都无法维持原有证据根。

多模态校验不只检查文件哈希。它还需要验证**跨模态事实合同**：非标定模态是否在允许时间窗内重叠；RGB-D 与点云是否匹配同一标定版本；动作时序是否能和本体/接触事件对齐；接触事件和视觉/力觉观测是否存在不可解释矛盾；相机内外参、机器人模型与预处理版本是否明确。校验输出只记录 validator version、校验布尔结果和 report hash；完整报告保留在受控对象存储中，以支持之后用新验证器重跑而不改写旧审核事实。

| 校验维度 | 最小字段 | 失败时的动作 | 说明 |
|---|---|---|---|
| 内容完整性 | `content_sha256`、字节数 | 保持 `QUARANTINED` | 重新哈希与 manifest 比对；不可用“缺失文件”替代 |
| 时间同步 | `start_ns/end_ns`、统一时钟域 | 拒绝 validator attestation | 允许有限抖动，但要记录容差与设备时钟状态 |
| 几何/标定 | `calibration` artifact、版本、误差界 | 升级给标定/硬件责任人 | 点云—视频对齐失败不应被标作普通模型难例 |
| 行为一致性 | `executed_actions`、本体、接触 | 升级给安全/硬件调查 | 指令动作与实际执行动作必须分开留存 |
| 语义/物理交叉验证 | 视觉、点云、接触和事件边界 | 维持隔离，进入人工审核 | 仅是辅助检测，不能证明输入“非对抗” |
| 数据血缘 | episode commit、预处理/action schema、模型版本 | 停止进入训练候选池 | 防止跨版本样本被静默混用 |

### 2.2 明确不能解决的风险

哈希只能显示“当前字节不同于被承诺的字节”，无法判断传感器在采集时是否已被欺骗；跨模态模型也可能被协同对抗。数字签名提供来源鉴别与完整性验证，但不阻止密钥所有者恶意签名，也不自动提供角色撤销、密钥轮换或法律意义上的不可否认性。Ed25519 是 RFC 8032 定义的 EdDSA 实例；本原型用它签名事件哈希，而不是把私钥放进 SQLite。[3]

同理，`UpdatePlan` 的哈希共识只表明所有训练 rank 收到同一控制消息，并不证明 telemetry 本身真实。生产系统应使用节点健康代理、GPU/NCCL 指标来源鉴权、阈值保护和操作员告警；不能让单个 rank 通过伪造低吞吐诱导数据倾斜或训练停滞。

## 3. 审核证据的分层分布式账本设计

### 3.1 数据平面与审计平面分离

原始多模态文件可能体积大、含隐私或受删除请求约束。因此应存于受控对象存储，并使用 `object-version-id`/不可变 commit URI 以及服务器端版本保留；账本保存的是小而稳定的不可变证据引用。SQLite/WAL 适合单站点原型和边缘断连缓冲；生产“分布式账本”应理解为**多个独立验证者对同一追加日志的复制与锚定**，而不是让所有视频、点云写入区块链。

```text
采集节点/仿真节点
 └─ 写入版本化对象存储：rgbd, pointcloud, proprio, actions, contact, calibration
     └─ 生成 EvidenceManifest（每个 artifact 的内容哈希 + schema + 时间区间）
         └─ MerkleRoot = H(排序后的 artifact leaf)
             └─ 捕获网关签名 EVIDENCE_REGISTERED
                 └─ 验证器签名 VALIDATION_ATTESTED
                     └─ 审核者 A/B 签名同一 CorrectionPatch 哈希
                         └─ 追加事件哈希链 → 定期链头锚定 → 外部 WORM/透明日志
                             └─ 只向 curated-data 候选池提交资格；不触发实机动作/发布
```

对于每一个 case，`EvidenceManifest` 的 canonical payload 至少包含：`case_id`、不可变 `episode_commit_uri`、`capture_session_id`、`action_schema_version`、`preprocess_version` 和六类必需 artifact。每个 `EvidenceArtifact` 都包含 `modality`、`uri`、`content_sha256`、`byte_length`、`start_ns`、`end_ns`、`schema_version` 和 `producer_id`。对 artifact canonical JSON 取 SHA-256 形成 leaf；按 leaf 字典序构造二叉 Merkle 树；结果即 `evidence_root`。

### 3.2 事件格式与追加规则

每个事件的 unsigned canonical payload 包含序号、case、事件类型、payload JSON、`evidence_root`、签名者、创建时间和 `previous_event_hash`。事件哈希是该 payload 的 SHA-256，签名者对 `event_hash` 字节执行 Ed25519 签名。账本验证器按 sequence 重建 previous hash 链，并用已登记的公钥验证每一签名。

| 事件 | 允许角色 | 必填内容 | 审核/训练意义 |
|---|---|---|---|
| `EVIDENCE_REGISTERED` | `capture_gateway` | manifest hash、evidence root | 固定证据包；同 case ID 不允许不同 manifest 重注册 |
| `VALIDATION_ATTESTED` | `validator` | validator version、report hash、所有 check 通过 | 仅表明合同校验通过，未代表人工同意训练 |
| `REVIEW_SUBMITTED` | `reviewer` | patch hash、决定、rationale hash | 第一位审核者提交数据校正意见 |
| `REVIEW_CONFIRMED` | 另一 `reviewer` | **相同** patch hash、批准/拒绝 | 第二人独立确认；身份不得与第一人相同 |
| `REVIEW_REJECTED` | `reviewer` | 原因 hash、替代处置 | case 继续隔离或升级 |
| 外部锚定记录 | `anchor` | 已签名的 sequence、chain-head hash、时间 | 应复制到独立 WORM/透明日志；本地表只是交付队列/缓存 |

推荐把审核补丁本身作为版本化对象存储中的 JSON 文档，账本记录 `patch_sha256`。补丁只允许声明**数据处置**，例如事件边界修正、已批准的材料先验范围、仿真覆盖建议或“拒绝进入训练”。补丁的 `allowed_downstream` 只能是 `simjob_compiler` 和 `curated_dataset_manifest`；`robot_control`、`safety_threshold_change` 和 `direct_production_deploy` 必须永久在 forbidden set 中。

### 3.3 角色、密钥与复制

使用最小权限的五类身份：捕获网关、验证器、审核者、锚定器和只读验证器。每位审核者有独立私钥；私钥从不写进账本或日志。生产中应通过 KMS/HSM 对密钥签名、设置轮换与撤销，并将公钥注册、角色变更和撤销本身也作为可审计事件。一个独立服务或不同云账户的验证器定时拉取事件、重算 evidence root、验证签名和检查外部锚点；任何差异都应该产生不可自动关闭的告警。

NIST 将审计轨迹作为支持个人责任、事件重建、入侵检测和问题分析的技术控制，并指出数字签名可用于防止审计轨迹的未被察觉修改；其同时强调，签名并不防止删除或修改本身。[1] 因而，外部锚定、对象存储保留和多副本比“选择某一种数据库”更重要。

## 4. 动态 UpdatePlan 与异构同步 DDP

### 4.1 两时标控制与计划字段

动态训练应区分两个时间尺度。**快时标**是每次 optimizer update 内的固定计划，严格禁止调整；**慢时标**是在完整 update + 检查点边界上，根据滑动窗口 telemetry 重新计划。这避免了不同 rank 在同一组 NCCL collectives 中出现分支差异。

`RankTelemetry` 记录 `samples_per_second`、`p95_step_ms`、`p95_data_ms`、`p95_allreduce_ms`、`free_memory_gb` 和 health。rank 0 汇集固定长度数值 telemetry，以目标 update 时间计算每个 rank 的 `micro_batches`，上/下限裁剪后构造 `UpdatePlan`。所有 rank 的本地 batch size、全局样本数、loss scale、world size 和 plan version 均被写入计划并签名式摘要校验。

```text
全体 rank: all_gather(telemetry, local_batch_size)
rank 0: planner.plan(...) → UpdatePlan(v, world_size, global_samples, rank plans)
rank 0 → 全体 rank: broadcast(size) + broadcast(canonical JSON bytes)
全体 rank: parse + schema/数学验证 + all_gather(SHA-256(plan bytes))
全体 rank: 依各自 RankUpdatePlan 执行本地微批
  - 第 1..K_r-1 微批：DDP.no_sync()
  - 第 K_r 微批：普通 DDP backward → 一次梯度 AllReduce
全体 rank: clip → optimizer.step → EMA update → 记录实际全局样本数
```

NCCL 的 Broadcast 从 root rank 复制缓冲区给所有 rank；AllReduce 在所有 rank 间归约并把结果交给每个 rank。其文档明确要求每个 rank 调用 collective 时具有相同 count 和 datatype，并警告不满足会造成未定义行为，包括 hang、crash 或数据损坏。[2] 本实现因此只在**计划边界**调用控制面 broadcast，且在进行任何 backward 之前验证 plan digest 一致。

### 4.2 为什么 loss 要按 `W / N` 缩放

设 world size 为 `W`，rank `r` 有 `n_r` 个有效样本，全球有效样本总数 `N = Σ_r n_r`。DDP 默认将各 rank 梯度平均，即 `1/W * Σ_r g_r`。如果某 rank 用本地均值损失 `L̄_r`，直接反向则每个 rank 权重相同，不等于按样本数加权。

将每个本地微批的**均值损失**改为：

```text
scaled_loss = local_mean_loss × valid_local_samples × (W / N)
```

则本地 rank 在该 update 内累计的梯度是 `(W/N) × Σ_{i∈r} ∇ℓ_i`。DDP 平均后得到：

```text
(1/W) × Σ_r [(W/N) × Σ_{i∈r} ∇ℓ_i]
= (1/N) × Σ_all_samples ∇ℓ_i
```

也就是标准的全局样本平均梯度。该推导前提是 `local_mean_loss` 真实地对**有效样本**取均值。对带 padding 的时间序列、图像 token 或掩码视频，必须把 loss reducer 改为“有效 token 的 sum 与 count”，再使用 `W / total_valid_tokens` 缩放；不能以固定 batch size 代替有效元素数。

| 场景 | 正确做法 | 不可采用的做法 |
|---|---|---|
| 各 rank batch size 相同、微批数不同 | `N=Σ(K_r×B)`，每个微批缩放 `B×W/N` | 仍除以固定 accumulation steps |
| 各 rank batch size 不同 | 把 `samples_per_micro_batch` 写入计划并逐 batch 验证 | 假设每个 rank 贡献相同样本数 |
| 可变长度/掩码序列 | 使用有效 token `sum/count`；计划按有效预算或成本分桶 | 对 padding 后 mean 乘 nominal batch size |
| rank 健康变化 | 完成/丢弃当前 update，在 checkpoint 后重组世界 | 在 `no_sync` 循环中修改 world size |
| OOM/数据流耗尽 | fail closed，不进行部分 optimizer step | 让剩余 rank 自己 `step()` |

### 4.3 参考实现的关键行为

`dynamic_nccl_update_plan_train.py` 提供如下可运行核心：

1. `next_update_plan(...)`：固定长度 `all_gather` telemetry 和各 rank 微批大小；仅 rank 0 使用 `DynamicAccumulationPlanner`；以 size + uint8 JSON tensor 广播计划，避免从训练网络接收可反序列化的 pickle 对象。
2. `broadcast_update_plan(...)`：对计划的 schema、world size、全局样本数、rank 连续性和 `loss_sum_scale` 逐项 fail-closed 验证；全体 rank 对 canonical bytes 的 SHA-256 作 `all_gather`，保证开始 backward 前已获得完全相同的计划。
3. `acvjepa_dynamic_update(...)`：对除本地最后一微批外的所有 batch 调用 `ddp.no_sync()`；最后一微批以普通 `ddp(...)` 前向和 `backward()` 触发唯一一次梯度同步；全体 rank 再进行 clip、`optimizer.step()`、EMA target update 和实际样本数 AllReduce。
4. `test_dynamic_nccl_acvjepa_integration.py`：用真实的轻量 AC-VJEPA 前向、EMA 目标和损失运行双进程 Gloo 集成测试。GPU/NCCL 环境应使用相同逻辑，但需另行压力测试通信与故障路径。

以下是训练循环中最关键的同步部分；完整可执行版本在交付的 Python 文件中。

```python
for local_index in range(my_plan.micro_batches):
    batch = next(micro_batches)
    actual_samples = batch["context_video"].shape[0]
    is_final = local_index == my_plan.micro_batches - 1
    sync_context = nullcontext() if is_final else ddp.no_sync()

    with sync_context:
        prediction = ddp(
            batch["context_video"],
            batch["context_proprio"],
            batch["executed_actions"],
        )
        targets = ddp.module.target_latents(
            batch["future_video"], batch["future_proprio"]
        )
        losses = action_conditioned_jepa_loss(
            prediction, targets, batch["future_events"]
        )
        scaled_loss = losses.total * actual_samples * my_plan.loss_sum_scale
        scaler.scale(scaled_loss).backward()

# 全体 rank 仅在各自最后一微批共同完成 DDP AllReduce 后进入此处。
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(ddp.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
ddp.module.update_ema_target()
```

PyTorch 的分布式通信包提供 collective 操作，DDP 以同步数据并行方式包装模型；这段代码必须调用 `ddp(...)` 而不是 `ddp.module.predict(...)`，以确保 DDP 的 autograd hooks 被正确安装。[4]

## 5. 运行准则、故障处置与可审计闭环

### 5.1 账本运行准则

采集完成后先写对象存储并读取服务器返回的版本/commit URI，再流式计算每个 artifact 的 SHA-256。在所有必需模态、时间重叠、标定绑定和 schema 通过前，不应创建 `VALIDATION_ATTESTED`。若任务涉及硬件故障、传感器合同违例、操作者隐私或安全事件，则直接进入升级队列，不进入“可被数据修正的难例”流程。

双审核的语义是“两个身份对同一补丁承诺的同意”，而不是两人各自提交相似文本。审核者 B 必须引用相同 `patch_sha256`，并与审核者 A 有不同身份；否则资格检查返回 false。只有 validator 通过且两个不同 reviewer 批准时，`eligible_patch_for_data` 才能为真。该返回值仅允许编译 SimJob 或写入 curated dataset manifest；后续增量训练、离线评测、影子模式和灰度门控仍由现有独立流水线决定。

每日或每个固定事件数应执行：全链验证、抽样/全量对象重哈希、外部锚定可读性验证、签名公钥撤销检查、锚点与本地 chain head 的单调性检查。任何失败均冻结后续审核提交，保留取证副本并通知人类安全/数据治理负责人。

### 5.2 NCCL 训练运行准则

每次计划必须记录 `plan_version`、rank telemetry、rank micro-batch 数、local/global sample 数、checkpoint ID、数据集 commit、模型/动作/预处理 schema 和通信压缩配置。计划器不应每一微批重算计划；建议以完整 update 后的平滑 p95 telemetry 在十到数十次 update 的窗口上重新估计，并设置 maximum disparity、memory headroom、network health 与数据等待阈值。

rank 0 的规划权不等于单点信任：所有 rank 应独立执行结构验证，digest 不一致就停止 update。任何 NCCL watchdog error、rank 不健康、实际 global sample 数与计划不一致、OOM、数据流耗尽或 checkpoint 血缘不匹配，都应触发如下过程：停止下一次 update；保留当前 telemetry 和错误；从最后一个已确认 checkpoint 重新 rendezvous；在新成员集合上构建新的 `UpdatePlan`。禁止任何 rank 继续单独执行 `optimizer.step()`。

| 信号 | 自动动作 | 人工/恢复门槛 |
|---|---|---|
| plan digest 不一致、schema 失败 | 在 backward 前 fail closed | 检查控制面、序列化版本、rank 映射 |
| NCCL timeout / rank 健康失败 | 中止当前 update，不产出候选 checkpoint | 仅从最近确认 checkpoint 重建 process group |
| actual samples ≠ `global_samples` | 标为训练数据/loader 合同故障 | 修复 batch 与 padding 计数后重跑 update |
| telemetry 显示某 rank 持续慢 | 在下一计划边界下调其微批预算 | 核查 data I/O、GPU 温度、网络 p95 和资源争用 |
| 账本签名、锚定或多模态校验失败 | case 维持隔离，禁止生成训练补丁 | 完成调查并由有权限人员重新登记新的 evidence package |
| 两人审核不足或 patch 不一致 | 不进入数据候选池 | 获取独立复核或明确拒绝/升级 |

## 6. 已实现与验证

本次交付包含一个多模态账本原型和一个动态训练核心模块。原型使用 SQLite WAL 作为本地追加日志投影，使用 `cryptography` 的 Ed25519 实现签名，且为本地 `file://` artifact 提供重哈希验证；远程对象存储应由其版本/ETag API 或可信下载路径进行等价验证。它的烟雾测试覆盖：六类证据 artifact、Merkle evidence root、注册/验证/双审核事件、链头锚定、资格判定，以及对 SQL 直接篡改 event payload 的 fail-closed 检测。

动态训练模块提供常规的双进程 Gloo 烟雾测试，以及以轻量 AC-VJEPA 前向、目标编码、损失和 EMA 更新为对象的双进程集成测试。测试环境没有 GPU，因此验证的是 collective 顺序、计划广播、变长局部累积、梯度缩放和副本一致性，而**不是** NCCL 的带宽、拓扑、p99 或故障恢复性能。

| 交付物 | 验证命令 | 观测结果 |
|---|---|---|
| `multimodal_hitl_tamper_evident_ledger.py` | `python3 multimodal_hitl_tamper_evident_ledger.py` | 4 个签名事件、1 个锚定、1 个 evidence case；故意篡改被检测 |
| `dynamic_nccl_update_plan_train.py` | `torchrun --standalone --nproc_per_node=2 ... --smoke-test` | rank 0 规划 2 个微批、rank 1 规划 1 个微批；全局样本 6；副本保持一致 |
| `test_dynamic_nccl_acvjepa_integration.py` | `torchrun --standalone --nproc_per_node=2 test_dynamic_nccl_acvjepa_integration.py` | 真实 AC-VJEPA 路径通过；全局样本 6；rank 0 本地样本 4 |

## 7. 部署前检查清单

生产化前应将 Ed25519 私钥迁移到 KMS/HSM，建立审核者身份撤销与轮换，采用独立 WORM 或透明日志锚定，实施对象存储的不可变版本/保留策略，并部署与数据库分离的只读验证器。应在真实 NCCL 拓扑下测量控制面广播开销、AllReduce p50/p95/p99、GPU 显存、数据等待、straggler 等待时间和单更新有效样本吞吐。

上线前还应执行混沌实验：断开一个非 root rank、篡改 telemetry、制造 plan digest 差异、破坏一个对象 artifact、撤销一个 reviewer key、让两名审核者提交不同 patch hash、让 local data stream 提前耗尽，以及在 checkpoint 边界重组世界。成功标准不是“训练继续运行”，而是每种违例都**在错误的物理或训练行为发生前停止**，并留下可复核的错误证据。

## 参考资料

[1] [NIST SP 800-12，第 18 章：Audit Trails](https://csrc.nist.rip/publications/nistpubs/800-12/800-12-html/chapter18.html)。

[2] [NVIDIA NCCL：Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

[3] [RFC 8032：Edwards-Curve Digital Signature Algorithm (EdDSA)](https://www.rfc-editor.org/rfc/rfc8032.html)。

[4] [PyTorch 2.13：torch.distributed](https://docs.pytorch.org/docs/2.13/distributed.html)。
