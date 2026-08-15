# Action-conditioned V-JEPA：双臂机器人的代码集成、时延与控制部署蓝图

> **安全声明。** 本文是研究与原型工程蓝图，不是安全认证控制软件。真实双臂机器人必须保留厂家安全控制器、硬件急停、关节/速度/力限制、碰撞检测、独立看门狗和现场风险评估。轻量 V-JEPA 只能参与候选动作的预测、排序和风险感知；它不能直接输出力矩、电流或绕过安全层。

## 1. 设计总则：把“动作条件”限制在可审计的中层动作块

Action-conditioned V-JEPA 的关键不是把动作向量简单拼到视觉 token 后面，而是让训练与部署中的动作含义严格一致。对双臂机器人，世界模型应以**已执行的、限幅后的中层动作块**为条件，预测短时域潜在状态；而不是以 LLM 文字、未经安全裁剪的计划，或原始驱动器力矩为条件。

> \(\hat{s}_{t+1:t+H}=F_\theta(s_t,\operatorname{Enc}_a(a_{t:t+H-1}),g,c)\)

其中 \(s_t\) 是视觉潜在状态与本体状态的融合，\(a\) 是双臂动作块，\(g\) 是目标表征，\(c\) 是任务与安全上下文。V-JEPA 2 的公开路线也是先进行视频自监督预训练，再用动作条件模型进行后训练并纳入 MPC 规划；本蓝图沿用这一层次，但为轻量部署收缩了模型与动作时域。[1]

## 2. 推荐代码层级

```text
bimanual_ac_vjepa/
├── contracts/                    # 所有跨进程消息的 schema 与版本
│   ├── plan_spec.py              # LLM 高层任务规范；不可含关节/力矩
│   ├── state_estimate.py         # 潜在状态引用 + 可解释状态
│   ├── action_block.py           # 可审计的双臂中层动作块
│   ├── prediction_report.py      # 目标进展、风险、不确定性、时间戳
│   └── control_window.py         # 经安全批准的短期参考轨迹
├── data/
│   ├── synchronizer.py           # 多相机/本体/触觉/动作时钟对齐
│   ├── episode_store.py          # 原始轨迹、元数据、模型/环境版本
│   ├── action_normalizer.py      # 坐标系、尺度、臂型与动作语义标准化
│   └── dataset.py                # 上下文/未来窗口与掩码采样
├── perception/
│   ├── camera_pipeline.py        # 采集、时间戳、裁剪、去畸变
│   ├── scene_facts.py            # 对象、保护区、人/宠物、门状态等可解释事实
│   └── state_fuser.py            # 视觉潜在状态 + 本体/触觉融合
├── models/
│   ├── vjepa_backbone.py         # 冻结或部分适配的 V-JEPA 编码器
│   ├── target_encoder.py         # EMA/冻结目标编码器，仅训练时用
│   ├── action_tokenizer.py       # ActionBlock → 动作 token
│   ├── ac_predictor.py           # 动作条件潜在状态预测器
│   ├── risk_uncertainty_heads.py # 预测残差、OOD、事件/风险头
│   └── checkpoint_io.py          # 模型/动作定义/相机标定的版本联锁
├── planning/
│   ├── skill_library.py          # grasp/carry/place/open_door 等已注册技能
│   ├── candidate_generator.py    # 受 DSL、几何与技能限制的候选
│   ├── mpc.py                    # 批量 rollout、目标代价和动作选择
│   └── fallback_policy.py        # 预测超时/高不确定性时的保守策略
├── safety/
│   ├── safety_kernel.py          # 权限、速度、空间、对象、人/宠物、超时门控
│   ├── geometric_checker.py      # 确定性碰撞/保护区检查
│   ├── watchdog.py               # GPU/进程/消息新鲜度独立看门狗
│   └── recovery.py               # HOLD/RETREAT/ASK_USER 状态机
├── control/
│   ├── trajectory_bridge.py      # ControlWindow → 时间参数化末端轨迹
│   ├── ik_bridge.py              # 经验证的逆运动学接口
│   ├── vendor_driver.py          # 厂商 SDK / ros2_control 等低层桥接
│   └── realtime_loop.py          # 本地实时伺服与安全保持
├── runtime/
│   ├── sensor_worker.py          # 非阻塞采集
│   ├── inference_worker.py       # 固定形状批推理 / deadline
│   ├── planner_worker.py         # 滚动规划
│   ├── task_orchestrator.py      # LLM 低频任务图、子目标转换
│   └── supervisor.py             # 生命周期、健康检查、审计
└── tests/
    ├── replay_tests.py           # 离线确定性回放
    ├── latency_tests.py          # p50/p95/p99 时延与超时注入
    ├── safety_tests.py           # 人体靠近、过期状态、动作越界等
    └── sim_to_real_tests.py      # 传感器噪声、延迟、布局迁移
```

这一分层的目的，是让每个错误都有清晰归属：动作语义错在 `action_normalizer`，状态估计错在感知层，动作后果预测错在世界模型，候选选择错在规划器，最终安全拒绝/放行错在安全内核。不要将这些职责打包进一个“端到端 Agent”进程。

## 3. 动作条件机制：从双臂技能到动作 token

### 3.1 `ActionBlock` 的语义

每个动作块表示短时间内**两条机械臂实际可执行的参考变化**。它既比低层力矩稳定，也比“拿起牛奶”这类语言动作精确。推荐在机器人基座坐标系或经标定的任务坐标系中表达。

```python
@dataclass(frozen=True)
class ArmAction:
    # 均为经过 SafetyKernel 限幅后的参考，不是原始驱动器命令
    delta_position_xyz: np.ndarray      # 3, 基座/任务坐标系
    delta_rotation_6d: np.ndarray       # 6D rotation representation
    gripper_command: float              # 连续开合或归一化夹爪目标
    contact_mode: int                   # FREE_SPACE / GENTLE_CONTACT / HOLD
    speed_profile_id: int               # 经批准的速度档，不直接指定任意速度

@dataclass(frozen=True)
class ActionBlock:
    block_id: str
    dt_s: float                          # 例如一个固定、短的动作块周期
    left: ArmAction
    right: ArmAction
    base_action: np.ndarray | None       # 若有移动底盘，另行限幅
    executed_limits: dict                # 最终执行时的速度/力/空间限额
    robot_calibration_id: str
    action_schema_version: str
```

`ActionBlock` 的训练标签必须来自**实际执行后的命令与状态**，而不是规划器最初想执行的命令。若安全内核把速度裁剪、机器人未到位、驱动器发生保护或动作被中断，轨迹必须记录“实际执行动作”和中断原因；否则世界模型会在错误的因果对上训练。

### 3.2 动作标准化与跨本体边界

双臂模型要学习的不是某一台机器人的绝对关节编号，而是对场景状态有意义的动作。数据加载时应至少做到以下标准化：

| 项目 | 推荐表示 | 原因 |
|---|---|---|
| 双臂位移 | 基座/任务坐标系下的相对末端位姿变化 | 减少不同起始位姿造成的无关差异。 |
| 旋转 | 连续 6D 旋转表示 | 避免欧拉角不连续。 |
| 夹爪 | 归一化开合 + 接触模式 | 将不同夹爪行程映射到可学习的语义。 |
| 本体状态 | 关节位置/速度、末端位姿、夹爪力、近期动作 | 使视觉遮挡或相机模糊时仍可估计状态。 |
| 时间 | 统一重采样、记录时间戳与延迟 | 避免“动作发生在观察之后”的伪因果。 |
| 机器人身份 | 可选 embodiment token | 只有跨机器人训练时才需要；首轮单本体实验可省略。 |

硬件驱动器的关节目标或力矩不应直接作为跨本体动作 token。它们仍由 `vendor_driver` 和本地控制器使用，但世界模型的动作接口应保持在可解释的末端运动/夹爪层级。

### 3.3 轻量模型结构

```python
class ActionTokenizer(nn.Module):
    def __init__(self, action_dim: int, d_model: int):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.time_embedding = nn.Embedding(MAX_HORIZON, d_model)

    def forward(self, blocks: Tensor) -> Tensor:
        # blocks: [B, H, action_dim]，来自已执行/候选的规范化 ActionBlock
        return self.action_mlp(blocks) + self.time_embedding(time_index(blocks))

class LightACVJepa(nn.Module):
    def __init__(self, frozen_vjepa: nn.Module, d_visual: int, d_proprio: int,
                 d_action: int, d_model: int):
        super().__init__()
        self.visual_encoder = frozen_vjepa
        self.proprio_projector = MLP(d_proprio, d_model)
        self.state_projector = MLP(d_visual, d_model)
        self.action_tokenizer = ActionTokenizer(d_action, d_model)
        self.predictor = CausalTransformer(d_model=d_model, depth=SMALL_DEPTH)
        self.event_head = MLP(d_model, N_EVENTS)
        self.uncertainty_head = MLP(d_model, 1)

    def encode_state(self, video_context: Tensor, proprio: Tensor) -> Tensor:
        # 第一版可冻结编码器；只训练投影、预测器、动作头和不确定性头
        with torch.no_grad():
            visual_z = self.visual_encoder(video_context)
        return self.state_projector(pool(visual_z)) + self.proprio_projector(proprio)

    def predict_rollout(self, state_token: Tensor, action_blocks: Tensor) -> Prediction:
        a_tokens = self.action_tokenizer(action_blocks)
        pred_tokens = self.predictor(state_token, a_tokens)
        return Prediction(
            future_latents=pred_tokens,
            event_logits=self.event_head(pred_tokens),
            uncertainty=self.uncertainty_head(pred_tokens)
        )
```

训练时应保留目标编码器或其冻结版本，未来观测只用于构造潜在目标，不能泄漏到预测器输入。这与 VLA-JEPA 所强调的“未来帧仅作目标、不进入学生路径”的无信息泄漏状态预测原则一致。[3]

```python
def ac_vjepa_training_step(batch: EpisodeBatch, model: LightACVJepa,
                            target_encoder: EMATargetEncoder) -> Losses:
    # context: [B, T_ctx, C, H, W]
    # future:  [B, T_pred, C, H, W]
    # executed_actions: [B, T_pred, D]，注意：必须是实际执行动作
    state_t = model.encode_state(batch.context_video, batch.context_proprio)

    with torch.no_grad():
        target_z = target_encoder.encode(batch.future_video, batch.future_proprio)

    pred = model.predict_rollout(state_t, batch.executed_actions)
    latent_loss = masked_cosine_loss(pred.future_latents, target_z, batch.future_mask)
    event_loss = focal_or_bce(pred.event_logits, batch.future_events)
    uncertainty_loss = calibration_loss(pred.uncertainty, latent_error(pred, target_z))

    return latent_loss + LAMBDA_EVENT * event_loss + LAMBDA_U * uncertainty_loss
```

## 4. 训练—部署接口的一致性

| 训练中的字段 | 部署中的等价对象 | 必须保持一致的内容 |
|---|---|---|
| 相机帧、裁剪、色彩/归一化 | `camera_pipeline` 输出 | 相机内外参、时间窗口、分辨率、归一化和遮挡处理。 |
| 本体状态与触觉 | 低层驱动采样 | 坐标系、单位、延迟补偿、缺失值规则。 |
| `executed_actions` | 经安全门批准并由驱动执行的 `ActionBlock` | 动作顺序、时间步、归一化、限幅和中断标记。 |
| 未来潜在目标 | 部署后真实观测编码 | 使用同一视觉编码器版本及预处理。 |
| 事件标签 | 事后验证器的可观察断言 | 如抓取确认、门角、对象在目标区、双臂间隙。 |

训练/部署不一致是 Action-conditioned 世界模型最常见的隐性失效来源之一。例如训练动作是 10 Hz 的末端位姿增量、部署动作却是 100 Hz 的关节速度目标，模型在语义上已不再预测同一系统。若不得不改变控制频率，必须在 `action_normalizer` 中通过窗口聚合或插值显式重采样，并重新评估预测校准。

## 5. 在线滚动闭环：模型做前瞻，控制器做实时执行

```python
async def planning_tick(snapshot: SensorSnapshot) -> None:
    state = state_service.estimate(snapshot)

    # 过期或不完整状态不进入 GPU 规划
    if not state_is_fresh_and_safe(state):
        safety_kernel.request_hold(reason="stale_or_unsafe_state")
        return

    subgoal = task_orchestrator.current_subgoal()  # LLM 不在此关键路径调用
    candidates = skill_library.propose_action_blocks(
        subgoal=subgoal,
        facts=state.symbolic_facts,
        max_candidates=adaptive_budget.candidates()
    )

    # 将 N 条候选的 H 步 ActionBlock 组成固定形状 batch，单次 GPU 推理
    reports = ac_vjepa_runtime.rollout_batched(
        state=state,
        candidate_action_blocks=candidates,
        goal=goal_encoder.encode(subgoal),
        deadline=PLAN_DEADLINE
    )

    proposal = mpc.select(
        candidates=candidates,
        reports=reports,
        cost_fn=goal_progress_minus_risk_uncertainty
    )

    approved = safety_kernel.authorize(
        proposal=proposal,
        state=state,
        report=reports[proposal.id]
    )

    if approved.is_allowed:
        trajectory_queue.replace_latest(
            trajectory_bridge.make_window(approved, expires_at=approved.deadline)
        )
    else:
        recovery.transition(approved.rejection_reason)
```

重点是 `replace_latest`，而不是无界排队。运动控制器始终读取**最新、未过期且已批准**的短期轨迹。世界模型慢一拍时，系统不能继续积压旧计划；它应继续执行当前仍有效的局部安全段，或在安全段耗尽前进入保持/撤退。

## 6. 双臂任务的动作条件示例

以“左臂开冰箱门、右臂将牛奶放入冷藏区”为例：

```python
candidate_B = ActionBlockSequence([
    # 窗口 1：左臂接近门把手；右臂在安全姿态观测/准备
    block(left=move_to(handle_pregrasp, FREE_SPACE),
          right=hold_safe_pose(), dt=DT),
    # 窗口 2：左臂建立受限持门；右臂接近牛奶
    block(left=hold_door(angle_target, HOLD),
          right=move_to(milk_pregrasp, FREE_SPACE), dt=DT),
    # 窗口 3：左臂维持门角；右臂温和抓取
    block(left=hold_door(angle_target, HOLD),
          right=grasp(milk, GENTLE_CONTACT), dt=DT),
    # 窗口 4：左臂仍持门；右臂将已抓取物移向冷藏区入口
    block(left=hold_door(angle_target, HOLD),
          right=move_delta(to_fridge_entry, FREE_SPACE), dt=DT),
])
```

世界模型预测的不是“这段描述是否听起来合理”，而是对每个动作块后的潜在状态、预期可观察事件和不确定性：门是否稳定开到目标角、右夹爪是否保持对象、牛奶是否接近目标区、双臂是否接近互碰、玻璃杯保护区是否受到威胁。最终的门角约束、碰撞距离和夹爪力仍由确定性安全/控制模块验证。

## 7. 时延与控制频率：不是让 JEPA 跑到伺服频率

### 7.1 正确的多频率划分

双臂机器人并不需要、也不应让 V-JEPA 在每个电机伺服周期推理一次。相反，应采用**多频率异步架构**：高频环由确定性控制与安全组件负责；中频环负责状态更新、潜在预测和短期重规划；低频环负责语言任务和任务图管理。

| 环路 | 建议职责 | 典型频率区间（需实测确认） | 可否依赖 GPU/LLM | 超时后的安全行为 |
|---|---|---:|---|---|
| 硬件急停、驱动器限位、低层力/速度保护 | 关节限位、力矩/速度限制、碰撞/急停 | 厂商控制器/安全硬件定义，常为数百 Hz 至更高 | **绝不依赖** | 立即进入硬件安全模式。 |
| 本地轨迹跟踪/阻抗控制 | 跟踪已批准的末端/关节参考轨迹 | 约 100–500 Hz，随机器人 SDK 而定 | 不依赖模型推理 | 保持当前位置或执行预验证撤退段。 |
| 近场传感器与安全监督 | 力/触觉、人/宠物、轨迹过期、守护进程 | 约 50–250 Hz 或由安全硬件更高频执行 | 不依赖 | 中断当前窗口、保持/急停。 |
| 相机与状态估计 | 相机采集、预处理、对象/状态摘要、潜在状态更新 | 约 10–30 Hz，受相机和编码器限制 | 可用 GPU，但须有超时门限 | 状态过期时禁止新动作。 |
| AC-V-JEPA + MPC | 批量动作 rollout、候选排序、短窗口替换 | 首版建议 2–10 Hz | 可用 GPU；**禁止 LLM 同步阻塞** | 执行仍有效的安全缓冲段；耗尽则保持。 |
| LLM 任务编排 | 子目标切换、异常解释、用户澄清、技能选择 | 事件驱动，通常远低于 1–2 Hz | 可远程/本地，但不在控制关键路径 | 使用当前已批准子目标或请求人类接管。 |

表中频率是**系统设计范围**，并非对任意硬件的性能承诺。具体数值必须由“端到端 99 分位时延 + 最坏负载抖动 + 安全停机距离”共同确定。对于缓慢家居操控，先把世界模型/规划限制在 2–5 Hz、将本地控制保持在更高频，通常比强迫视频 Transformer 达到 100 Hz 更安全也更可复现。

### 7.2 时延预算应以 `p99` 和安全窗口倒推

记：

> \(T_{e2e}=T_{capture}+T_{sync}+T_{preprocess}+T_{encode}+T_{rollout}+T_{mpc}+T_{safety}+T_{queue}\)

系统不是看平均值，而是测量每个阶段的 **p50 / p95 / p99**。规划周期为 \(T_{plan}\)，当前已批准轨迹的剩余安全覆盖时间为 \(T_{safe}\)。部署前至少应满足：

> \(T^{p99}_{e2e} + T_{jitter} < \min(T_{plan},\; T_{safe}-T_{hold})\)

其中 \(T_{hold}\) 是从检测到问题到安全保持/撤退生效的实测时间。这个不等式将“模型应该多快”转化为机器人和安全策略实际允许的期限。

以下是**80M 级编码器、短视频窗口、慢速家居操控**的示例预算结构，数值仅用于建立测量目标，不能替代 profiling：

| 阶段 | 设计目标（p99） | 优化手段 | 逾期时的动作 |
|---|---:|---|---|
| 相机时间对齐、裁剪、归一化 | 受一个相机帧周期约束 | 零拷贝/固定缓冲、固定形状、CPU 线程绑定 | 标记状态不新鲜。 |
| 状态编码 | 不超过规划窗口中可分配的主要份额 | 短窗口、低分辨率、混合精度、特征缓存 | 不发新候选。 |
| 候选 rollout + MPC | 在 `PLAN_DEADLINE` 前完成 | 状态编码一次、候选批推理、固定候选数/时域、warm-start | 丢弃本次结果，不入队。 |
| 安全检查与轨迹桥接 | 显著小于模型规划时延 | CPU 规则/几何预计算、无网络往返 | 拒绝或保持。 |
| 伺服命令输出 | 满足机器人 SDK/控制器自身周期 | 预分配、无阻塞、无 GPU、无日志 I/O | 本地看门狗接管。 |

### 7.3 异步进程与双缓冲

```python
# 伪代码：三条独立时钟；真正的实时环运行在厂商控制器/实时进程中

async def sensor_loop():
    while supervisor.running:
        frame = camera.read_timestamped()
        proprio = robot.read_proprioception()
        sensor_ring.write(sync.align(frame, proprio))

async def world_model_loop():
    while supervisor.running:
        snapshot = sensor_ring.latest_complete_window()
        if snapshot is None:
            continue

        # 不与伺服共享锁；固定形状、预热后的 GPU 推理
        state = state_service.estimate(snapshot)
        if state.age_ms > LIMITS.max_state_age_ms:
            trajectory_queue.invalidate_new_plans("state_too_old")
            continue

        result = await with_deadline(
            planning_tick(snapshot), deadline=LIMITS.plan_deadline_ms
        )
        if result.timed_out or result.high_uncertainty:
            trajectory_queue.invalidate_new_plans(result.reason)
            continue

        # 原子替换：只写入已通过 SafetyKernel 的、带到期时间的短窗口
        trajectory_queue.atomic_replace(result.approved_window)

async def task_loop():
    # LLM 不运行在上两个实时/准实时循环中
    while supervisor.running:
        event = task_events.wait()
        if event.requires_semantic_replan:
            plan = await llm_orchestrator.update_plan(event)
            task_orchestrator.install_if_schema_valid(plan)

@realtime_process
def servo_loop():
    while robot.enabled:
        live = robot.read_fast_safety_state()
        window = trajectory_queue.read_current_if_valid(now())

        if fast_safety_violation(live) or window is None:
            robot.safe_hold_or_vendor_stop()
            continue

        setpoint = window.interpolate(now())
        command = local_controller.compute_limited_command(setpoint, live)
        robot.send(command)
```

`servo_loop` 不应等待相机、GPU、Python 垃圾回收、LLM、远端服务器或日志写盘。`world_model_loop` 的结果只是在下一个可替换时刻更新短期轨迹。若世界模型卡顿，机器人仍能依据已有安全窗口平稳保持，而不是失去控制。

## 8. 如何把推理延迟压到可用范围

### 8.1 先从模型结构减法开始

| 优化 | 实施方式 | 对延迟的影响 | 需验证的风险 |
|---|---|---|---|
| 冻结视觉骨干 | 在线只执行编码器前向，训练仅更新小预测器/动作头 | 减少训练成本；推理仍有视觉编码开销 | 领域差异可能造成状态失真。 |
| 滚动特征缓存 | 对重叠的相机窗口缓存历史帧特征，只编码新增帧 | 降低重复视觉编码 | 时间戳、相机丢帧和缓存失效必须处理。 |
| 状态仅编码一次 | 对 N 条候选共享 `s_t`，仅批量展开动作 token | 显著降低“候选数 × 编码器”的重复成本 | 所有候选必须引用同一状态版本。 |
| 固定形状批推理 | 固定最大候选数、时域与 token 尺寸；预热推理引擎 | 降低动态分配和编译抖动 | 空候选需 padding/mask，不能改变动作语义。 |
| 混合精度与图优化 | 经离线等价性验证后采用 bf16/fp16、编译/推理引擎优化 | 提升吞吐、降低显存 | 量化/精度改变可能破坏不确定性校准。 |
| 自适应候选预算 | 低风险状态少候选、短时域；高不确定性先观测/保持而非无限搜索 | 控制最坏时延 | 不可在高风险时因赶时延而放宽安全约束。 |
| 计划与感知分离 | 低频规划使用高质量状态；伺服使用本地反馈 | 避免世界模型进入高频环 | 状态过期门限必须严格。 |

### 8.2 推理主机与机器人底盘的部署形态

推荐将系统物理上分成三个故障域：

| 故障域 | 部署内容 | 失效后应发生什么 |
|---|---|---|
| **安全/控制域** | 厂商控制器、实时轨迹跟踪、紧急停止、低层限幅 | 保持或安全停机；不依赖上位机。 |
| **边缘推理域** | 相机预处理、状态估计、80M AC-V-JEPA、MPC、局部安全规则 | 不再更新轨迹；控制域消耗完当前安全窗口后保持。 |
| **任务/网络域** | LLM、数据库、可视化、远程日志、模型管理 | 任务暂停或降级；不影响高频保持/急停。 |

在“机器人底盘”上部署时，边缘推理机不一定必须放在机械臂控制器内部；关键是与控制域通过有界、可监测、低延迟的本地接口通信，并具备独立电源/进程健康检查。若边缘 GPU 的热设计、电源或抖动无法通过压力测试，应先让模型离线或影子运行，而不是提高自主权限。

## 9. 控制频率如何验证，而不是口头保证

### 9.1 必测的端到端指标

| 指标 | 测量方式 | 通过条件示例 |
|---|---|---|
| `state_age_ms` | 从相机曝光/本体采样到世界模型使用状态的实际年龄 | 小于针对任务定义的阈值；出现超限即不发新轨迹。 |
| `plan_latency_ms` | 从完整状态窗口到经安全批准的 `ControlWindow` | p99 必须低于 `PLAN_DEADLINE`；超时率在压力测试下可接受且触发安全降级。 |
| `servo_jitter` | 相邻控制命令的周期偏差 | 满足机器人厂商控制/安全规范，而非模型团队自定义均值。 |
| `hold_latency_ms` | 人/宠物靠近、力异常或看门狗触发到保持生效 | 在最坏负载与 GPU 卡顿下均达标。 |
| `stale_plan_rejections` | 过期轨迹被拒绝的次数与原因 | 应可审计；零容忍“过期窗口继续发送”。 |
| `prediction_calibration` | 预测不确定性与实际预测残差/失败事件的关系 | 高不确定性应对应更高风险，而非虚假自信。 |
| `closed_loop_success` | 未见布局/物体下的任务成功、违规、恢复 | 与无世界模型、冻结特征、规则基线在同一预算下比较。 |

### 9.2 最小性能验收流程

1. **离线回放。** 以记录的真实轨迹重放相机、本体与动作，确认状态估计、动作编码、预测和安全决策可重现。
2. **GPU 压力。** 同时运行目标候选数、最长窗口、相机最大帧率与日志负载，记录 p50/p95/p99；不可只测空闲 GPU 平均时延。
3. **超时注入。** 人为延迟模型、丢弃相机帧、暂停 GPU 进程、制造队列积压；系统必须拒绝新轨迹并平稳保持，而非使用过期计划。
4. **仿真闭环。** 对不同布局、光照、遮挡、动作延迟和抓取滑移进行稳定性测试；评估世界模型改善是否超过计算成本。
5. **影子实机。** 机器人执行经验证的确定性/遥操作控制；AC-V-JEPA 只预测与记录，不影响控制。
6. **受限自主。** 仅开放低速、低风险原子技能，且每个控制窗口都有可观察验收条件与人工急停。

## 10. 失败回退状态机

```text
NORMAL
  ├─ [state 新鲜 + 预测低不确定性 + 计划未超时] → 替换短期安全轨迹
  ├─ [状态过期 / GPU deadline miss] → HOLD_PENDING
  ├─ [高不确定性 / OOD] → CAUTIOUS_OBSERVE
  └─ [人接近 / 力异常 / 碰撞预测或硬件告警] → SAFE_HOLD

CAUTIOUS_OBSERVE
  ├─ [补充观测后置信恢复] → NORMAL
  ├─ [需语义澄清] → ASK_USER
  └─ [风险升级] → SAFE_HOLD

HOLD_PENDING / SAFE_HOLD
  ├─ [安全状态和新鲜观测恢复] → REPLAN
  ├─ [用户明确授权且安全规则允许] → REPLAN
  └─ [硬件/安全事件未清除] → 保持并请求人工接管
```

唯一允许继续执行旧计划的情形，是该计划对应的短期轨迹仍在有效期内、仍满足本地快速安全检查、且能在过期前安全保持。它不是“模型上一次说过可以”，而是显式带有状态版本、到期时间、约束摘要和撤退/保持路径的受控窗口。

## 11. 推荐的首个实机部署配置

| 项目 | 首版取舍 |
|---|---|
| 任务 | 单臂或双臂中仅一个运动、另一个保持的低速原子任务，例如“开门时放置物体”。 |
| 输入 | 单主相机 + 本体状态；视觉/动作时间同步先于增加更多传感器。 |
| 模型 | 冻结 80M V-JEPA 2.1 编码器 + 小动作 token MLP + 小预测器/风险头。 |
| 世界模型时域 | 短时域、固定长度 ActionBlock；不做长任务一口气 rollout。 |
| 规划 | 少量已注册技能候选 + 批量潜在 rollout + 保守 MPC 排序。 |
| 安全 | 厂商控制器、独立人/宠物检测、保护区、速度/力限幅、轨迹 TTL、看门狗、人工急停。 |
| 上线方式 | 离线回放 → 仿真闭环 → 影子预测 → 受限动作窗口；任何一步不达标即停止扩大权限。 |

## 12. 与公开工作的关系及边界

V-JEPA 2 已公开验证“视频预训练 + 动作条件后训练 + MPC”可在基本机器人任务中运作，并公布了相应代码/模型；VLA-JEPA 还提供了无信息泄漏的潜在状态预测与动作头微调思路。[1] [2] [3] 但这些工作并不自动给出某个双臂底盘上的确定性实时保证。真正的保证来自：目标机器人上的端到端基准、最坏负载测试、独立安全链、严格的时钟/动作语义一致性，以及在模型失败时不依赖模型也能安全停止的系统设计。

## 参考资料

[1]: [Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985](https://arxiv.org/html/2506.09985v1)

[2]: [Meta FAIR, *V-JEPA 2 官方开源仓库*](https://github.com/facebookresearch/vjepa2)

[3]: [Sun et al., *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model*, arXiv:2602.10098](https://arxiv.org/html/2602.10098v1)
