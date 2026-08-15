# 双臂机器人：从 LLM 高层指令到 JEPA 驱动控制信号的伪代码蓝图

> **定位。** 这是系统架构级伪代码，不可直接部署到真实机器人。真实部署必须接入经过验证的实时控制器、碰撞检查、硬件急停、速度/力限制、权限系统和现场风险评估。它刻意将自然语言、模型预测与底层执行隔离：LLM 永远不能直接驱动关节或夹爪。

## 1. 从语言到控制的转换链

```text
用户自然语言
  → LLM 任务编译器
  → 类型化 PlanSpec（子目标、动作 DSL、成功条件、约束）
  → 场景状态估计（可解释状态 + JEPA 潜在状态 z_t）
  → 候选双臂技能/轨迹
  → 动作条件 JEPA 世界模型预测 future_latents / risk / uncertainty
  → MPC / 采样规划器选择短时域 ControlWindow
  → 独立安全网关
  → 逆运动学 + 全身控制器
  → 低层受限命令 u_t
  → 真实观测与事后验证
  → 状态更新、继续/重规划/保持/人工确认
```

这里最关键的设计不是“模型更多”，而是**接口不可越权**：`PlanSpec` 不能含关节角；`PredictionReport` 不能绕过安全门；`ControlWindow` 不通过安全门就不能进入控制器；`OutcomeEvent` 必须以真实观测为准，不能被 LLM 文本覆盖。

## 2. 运行时对象与受限动作语言

```python
# ---------------------------
# 语义层：LLM 可见的对象
# ---------------------------
@dataclass
class PlanSpec:
    mission_id: str
    subgoals: list[Subgoal]
    constraints: SafetyConstraints
    action_grammar: set[str]        # 只允许预注册技能名称
    success_criteria: list[Predicate]
    risk_tier: str

@dataclass
class Subgoal:
    id: str
    intent: str                     # 例如 "return_milk_to_fridge"
    target_refs: list[str]          # milk, fridge_shelf 等稳定实体引用
    done_when: list[Predicate]
    prohibited_refs: list[str]      # glass_cup, knife, unknown_item

# ---------------------------
# 状态层：环境事实和潜在表征并行存在
# ---------------------------
@dataclass
class StateEstimate:
    state_id: str
    timestamp: Time
    latent_z: TensorRef             # JEPA 编码器输出，供预测与规划使用
    symbolic_facts: dict            # 供安全规则、LLM、日志读取
    object_tracks: list[ObjectTrack]
    human_pet_distance_m: float
    confidence: float
    freshness_ms: int

@dataclass
class PredictionReport:
    candidate_id: str
    future_latents: list[TensorRef]
    goal_progress: float
    constraint_risk: dict           # collision, fragile_contact, door_jam ...
    uncertainty: float              # ensemble / OOD / residual 等聚合值
    expected_observables: list[Predicate]
    model_version: str

# ---------------------------
# 控制层：只能通过已定义的技能和短控制窗口
# ---------------------------
@dataclass
class SkillCandidate:
    name: str                       # open_fridge / grasp / carry / place / wipe
    left_role: str                  # hold_door / observe / idle / carry
    right_role: str
    parameters: dict                # 目标实体、抓取候选、目标区域等

@dataclass
class ArmSetpoint:
    end_effector_pose: Pose         # 末端目标位姿，不暴露原始关节控制给 LLM
    gripper_mode: str               # open / close / hold / compliant_release
    velocity_cap: float
    impedance_profile: str          # "free_space", "gentle_contact", "hold"

@dataclass
class ControlWindow:
    horizon_ms: int                 # 仅允许很短的滚动窗口
    left: list[ArmSetpoint]
    right: list[ArmSetpoint]
    expected_observables: list[Predicate]
    rollback: SafetyRecovery
    source_candidate_id: str

@dataclass
class OutcomeEvent:
    state_before: str
    state_after: str
    observed_effects: list[Predicate]
    prediction_error: float
    failure_class: str | None
    evidence_refs: list[str]
```

动作 DSL 只包含经验证的技能，例如 `observe`、`open_fridge`、`hold_door`、`grasp`、`carry`、`place`、`wipe_region`、`retreat`、`ask_user` 与 `safe_stop`。任何 LLM 创造出来但未注册的动作名称都必须被拒绝，而不是自动映射为运动。

## 3. 顶层任务编译：LLM 只编译任务规范

```python
class TaskCompiler:
    def compile(self, user_request: str, policy: HomePolicy,
                scene_summary: dict) -> PlanSpec:
        # LLM 以 JSON schema / function calling 方式返回候选任务规范
        draft = llm.generate_structured(
            prompt="""
              将用户请求编译为家用机器人 PlanSpec。
              仅可使用给定动作语法和已知实体；必须列出成功条件、禁止对象、
              前置条件及需要用户确认的情况。不得输出任何关节、速度、力或原始控制命令。
            """,
            context={"request": user_request,
                     "home_policy": policy.public_rules,
                     "scene_summary": scene_summary},
            schema=PlanSpec.schema()
        )

        # 非语言、确定性的验证：禁止 LLM 越权或漏写安全条件
        validate_schema(draft, PlanSpec.schema())
        assert draft.action_grammar <= REGISTERED_SKILLS
        assert all(g.done_when for g in draft.subgoals)
        assert policy.permits(draft)
        return draft
```

对于“把牛奶放回冰箱、盘勺放水槽、玻璃杯不动、擦拭空白区域”，LLM 生成四个子目标，并将玻璃杯写入 `prohibited_refs`。它可建议“先处理牛奶”，但不能直接下达“左臂关节 3 转动若干角度”或“以某力抓住牛奶”。

## 4. 状态估计与 JEPA 世界模型接口

```python
class SceneStateService:
    def update(self, sensor_buffer: SensorBuffer,
               action_history: list[ExecutedAction]) -> StateEstimate:
        # 1. 可解释层：由检测、追踪、接触、力和规则融合产生
        symbolic = fuse_scene_facts(
            rgbd=sensor_buffer.head_rgbd,
            wrist_views=sensor_buffer.wrist_cameras,
            tactile=sensor_buffer.tactile,
            force_torque=sensor_buffer.wrist_force,
            robot_proprio=sensor_buffer.joint_state,
            person_pet=sensor_buffer.proximity
        )

        # 2. 预测层：JEPA 从时间窗口编码状态，而不是只看单帧
        z_t = jepa_encoder.encode(
            observations=sensor_buffer.recent_window(),
            actions=action_history[-K:]
        )

        return StateEstimate(
            state_id=new_version_id(),
            timestamp=now(),
            latent_z=z_t,
            symbolic_facts=symbolic,
            object_tracks=build_tracks(symbolic),
            human_pet_distance_m=symbolic["min_human_pet_distance_m"],
            confidence=estimate_state_confidence(symbolic, z_t),
            freshness_ms=0
        )

class ActionConditionedJepaWorldModel:
    def rollout(self, state: StateEstimate, action_tokens: list[ActionToken],
                goal_embedding: TensorRef, constraints: SafetyConstraints) -> PredictionReport:
        # 预测未来潜在状态，不要求生成未来每个像素
        trajectory, uncertainty = jepa_world_model.predict_latent_rollout(
            z0=state.latent_z,
            actions=action_tokens,
            goal=goal_embedding,
            context=encode_constraints(constraints)
        )

        return PredictionReport(
            candidate_id=action_tokens.id,
            future_latents=trajectory,
            goal_progress=goal_scorer(trajectory[-1], goal_embedding),
            constraint_risk=risk_heads(trajectory, action_tokens, state.symbolic_facts),
            uncertainty=uncertainty.aggregate(),
            expected_observables=observable_effect_head(trajectory, action_tokens),
            model_version=jepa_world_model.version
        )
```

世界模型的输入动作不应是未经约束的原始力矩流，而是由技能参数化器生成的、在动作 DSL 范围内的低维行动表示，例如“右臂以保守速度将当前抓取物移动到冷藏区入口，左臂保持冰箱门”。这样既降低学习难度，也使预测结果能与真实执行器接口对齐。

## 5. 从子目标到候选双臂轨迹

```python
class SkillAndTrajectoryGenerator:
    def propose(self, subgoal: Subgoal, state: StateEstimate,
                plan: PlanSpec) -> list[SkillCandidate]:
        # LLM 只能在声明式技能级别帮助提出顺序或角色分配
        semantic_options = skill_policy.propose_roles(
            subgoal=subgoal,
            facts=state.symbolic_facts,
            allowed_skills=plan.action_grammar
        )

        candidates = []
        for option in semantic_options:
            # 使用几何、IK、碰撞模型和对象抓取库将语义技能变为可行候选
            if not geometric_precheck(option, state.symbolic_facts):
                continue
            candidates.extend(
                trajectory_parameterizer.expand_to_candidates(option, state.symbolic_facts)
            )
        return candidates

class DualArmMPC:
    def select_window(self, candidates: list[SkillCandidate], state: StateEstimate,
                      subgoal: Subgoal, plan: PlanSpec) -> ControlWindow | Hold:
        scored = []
        for candidate in candidates:
            action_tokens = action_tokenizer.encode(candidate)
            prediction = world_model.rollout(
                state=state,
                action_tokens=action_tokens,
                goal_embedding=goal_encoder.encode(subgoal),
                constraints=plan.constraints
            )

            # 约束违反和高不确定性不能被“更接近目标”抵消
            if prediction.uncertainty > THRESHOLDS.max_prediction_uncertainty:
                continue
            if violates_hard_constraint(prediction.constraint_risk, plan.constraints):
                continue

            score = (
                W_GOAL * prediction.goal_progress
                - W_RISK * aggregate_risk(prediction.constraint_risk)
                - W_UNCERTAINTY * prediction.uncertainty
                - W_EFFORT * candidate.estimated_effort
            )
            scored.append((score, candidate, prediction))

        if not scored:
            return Hold(reason="no_low_risk_candidate")

        _, candidate, prediction = max(scored, key=lambda x: x[0])
        return trajectory_parameterizer.to_short_control_window(
            candidate=candidate,
            prediction=prediction,
            max_horizon_ms=CONTROL_POLICY.max_window_ms
        )
```

### 5.1 牛奶归还的具体候选

对于 `return_milk_to_fridge`，候选生成器可形成三种技能配置：

| 候选 | 左臂 | 右臂 | 世界模型主要检查项 |
|---|---|---|---|
| A | 空闲，之后再开门 | 先抓取牛奶 | 牛奶是否遮挡视野、二次姿态切换成本。 |
| B | `open_fridge + hold_door` | `grasp + carry + place_milk` | 门角稳定性、双臂最小间隙、牛奶滑移、玻璃杯距离。 |
| C | 同时从桌面中央穿越 | 同时从桌面中央穿越 | 双臂互碰、玻璃杯保护区、视野遮挡与不确定性。 |

通常 B 会因更低的双臂干涉与更清晰的状态观测而胜出，但这一决定来自当前状态下的预测与硬约束，而不是固化脚本。

## 6. 安全网关：在控制信号落地前做最后裁决

```python
class SafetyKernel:
    def authorize(self, window: ControlWindow, state: StateEstimate,
                  plan: PlanSpec, prediction: PredictionReport) -> ApprovedWindow | Hold:
        # 快速、确定性的输入检查
        if emergency_stop.is_pressed():
            return Hold("hardware_estop")
        if state.freshness_ms > LIMITS.max_state_age_ms:
            return Hold("stale_observation")
        if state.human_pet_distance_m < plan.constraints.min_human_pet_distance_m:
            return Hold("human_or_pet_too_close")
        if not objects_allowed(state.symbolic_facts, plan.constraints):
            return Hold("forbidden_or_unknown_object")

        # 预测风险与真实几何/接触约束同时检查
        if prediction.uncertainty > LIMITS.max_prediction_uncertainty:
            return Hold("world_model_uncertain")
        if prediction.constraint_risk["fragile_contact"] > LIMITS.fragile_risk:
            return Hold("fragile_object_risk")
        if not collision_checker.clearance_ok(window, state.symbolic_facts):
            return Hold("geometric_clearance_failed")
        if not permission_engine.permits(window, plan.risk_tier):
            return Hold("permission_or_confirmation_required")

        # 将任意控制候选裁剪到硬件包络内；上层不能扩大此包络
        safe_window = enforce_hardware_envelope(
            window,
            velocity_limit=LIMITS.max_cartesian_velocity,
            force_limit=LIMITS.max_gripper_force,
            joint_limit=robot.joint_limits,
            workspace=HOME_SAFE_WORKSPACE
        )
        return ApprovedWindow(safe_window)
```

安全内核与 LLM、JEPA 模型在进程/权限上隔离。它可以拒绝动作、缩短窗口、要求重新观测或触发硬件安全停止；但任何上层组件都不能通过“我认为这是安全的”来解除 `HOLD` 或绕过确认。

## 7. 从短控制窗口到低层受限控制信号

```python
class WholeBodyControlAdapter:
    def execute(self, approved: ApprovedWindow) -> ExecutionTrace:
        trace = ExecutionTrace.start(approved)

        # 只在窗口有效期内运行；每一个低层周期均重新检查硬约束
        for left_sp, right_sp in zip(approved.left, approved.right):
            live = sensors.read_fast()  # joint state, tactile, force/torque, proximity

            if fast_safety_violation(live):
                robot.safe_hold_or_retreat()
                return trace.finish(status="interrupted_by_fast_safety")

            # 逆运动学和运动学控制由确定性、经过验证的组件处理
            q_left_ref = inverse_kinematics.solve(
                arm="left", target_pose=left_sp.end_effector_pose,
                current=live.joints.left, collision_context=live
            )
            q_right_ref = inverse_kinematics.solve(
                arm="right", target_pose=right_sp.end_effector_pose,
                current=live.joints.right, collision_context=live
            )

            # 生成受限命令；实际驱动器可采用位置/速度/阻抗控制，
            # 但只能使用经过安全内核裁剪的参考轨迹和模式
            u_t = low_level_controller.track(
                q_left_ref=q_left_ref,
                q_right_ref=q_right_ref,
                left_gripper=left_sp.gripper_mode,
                right_gripper=right_sp.gripper_mode,
                impedance_left=left_sp.impedance_profile,
                impedance_right=right_sp.impedance_profile,
                velocity_cap=min(left_sp.velocity_cap, right_sp.velocity_cap)
            )
            robot.send_limited_command(u_t)
            trace.append(live, u_t)

        return trace.finish(status="window_completed")
```

在这一层，JEPA 已经不再直接“输出电机命令”。JEPA 的贡献是对候选动作序列的**未来状态、风险和不确定性进行前瞻评估**；低层控制器把被批准的末端位姿、夹爪模式和阻抗档位变为关节参考与受限驱动信号。这样即使世界模型误差上升，安全包络、碰撞检查和快速保持仍独立有效。

## 8. 顶层执行循环：长任务由短窗口重规划组成

```python
async def execute_home_task(user_request: str) -> TaskResult:
    # ---- 低频：语义编排 ----
    initial_state = scene_state_service.update(sensors.buffer(), history=[])
    plan = task_compiler.compile(
        user_request=user_request,
        policy=home_policy,
        scene_summary=project_for_llm(initial_state.symbolic_facts)
    )

    for subgoal in plan.subgoals:
        while not predicate_engine.all_true(subgoal.done_when):
            # ---- 中频：状态 → JEPA 预测 → MPC ----
            state = scene_state_service.update(sensors.buffer(), action_history)

            if state.confidence < THRESHOLDS.min_state_confidence:
                await request_more_observation_or_user_help(subgoal, state)
                continue

            candidates = skill_trajectory_generator.propose(subgoal, state, plan)
            proposal = dual_arm_mpc.select_window(candidates, state, subgoal, plan)

            if isinstance(proposal, Hold):
                outcome = recovery_manager.resolve(proposal, subgoal, state, plan)
                if outcome.requires_user:
                    return TaskResult.paused(outcome.user_message)
                continue

            # 重新取得中选候选的预测报告；防止控制窗口与风险证据脱钩
            prediction = proposal.prediction_report
            authorization = safety_kernel.authorize(proposal, state, plan, prediction)

            if isinstance(authorization, Hold):
                outcome = recovery_manager.resolve(authorization, subgoal, state, plan)
                if outcome.requires_user:
                    return TaskResult.paused(outcome.user_message)
                continue

            # ---- 高频：低层控制 ----
            trace = whole_body_control_adapter.execute(authorization)

            # ---- 事后校验：真实结果高于文本或预测 ----
            next_state = scene_state_service.update(sensors.buffer(), action_history)
            event = verifier.compare(
                before=state,
                after=next_state,
                expected=prediction.expected_observables,
                trace=trace
            )
            audit_log.append(plan, subgoal, proposal, prediction, event)

            if event.failure_class is not None:
                recovery_manager.register_failure(event)
                # 失败时停止依赖旧预测，下一轮从真实观测重建状态
                continue

            action_history.append(trace.to_executed_action())

    return TaskResult.completed(summary=reporter.summarize_from_audit_log())
```

## 9. 抓取滑移的异常处理伪代码

下面的分支说明系统不会因为高层计划仍然“逻辑正确”就继续执行危险动作。抓取滑移由真实触觉、视觉与力信号触发；JEPA 预测和 LLM 解释只参与恢复决策，不取代停止逻辑。

```python
def handle_grasp_slip(event: OutcomeEvent, state: StateEstimate,
                      subgoal: Subgoal, plan: PlanSpec) -> RecoveryResult:
    # 1. 先执行与模型无关的安全反应
    robot.freeze_cartesian_motion()
    robot.raise_to_prevalidated_safe_height()

    # 2. 重新读取真实环境，丢弃旧状态和旧世界模型 rollout
    refreshed = scene_state_service.update(sensors.buffer(), action_history=[])

    # 3. 若物体已落入危险区、状态不明或人靠近，则不尝试自主恢复
    if refreshed.symbolic_facts["milk_pose_confidence"] < THRESHOLDS.recovery_pose_confidence:
        return RecoveryResult.ask_user("牛奶位置不确定，已停止。请确认是否需要继续。")
    if refreshed.human_pet_distance_m < plan.constraints.min_human_pet_distance_m:
        return RecoveryResult.hold("工作区有人或宠物靠近")

    # 4. 生成不同于原抓取的保守候选；例如更大接触面、缩短搬运窗口
    alternatives = grasp_planner.generate_alternative_grasps(
        object_ref="milk", exclude=event.evidence_refs["failed_grasp_id"],
        prefer_stable_contact=True
    )

    # 5. 把替代候选重新送入 JEPA → MPC → SafetyKernel 正常闭环
    return RecoveryResult.replan(alternatives)
```

## 10. 模块级测试契约

| 模块 | 最小测试契约 | 失败时必须发生的行为 |
|---|---|---|
| LLM 任务编译器 | 给定模糊指令时只能输出 schema 合法、动作已注册、带约束的 `PlanSpec`。 | 拒绝执行，转为澄清问题。 |
| JEPA 状态/世界模型 | 对已知动作产生可校验的短期效果预测，并在分布外或遮挡时提高不确定性。 | 不能以低置信预测推动高风险动作。 |
| 候选规划器 | 不能产生违反动作 DSL、玻璃保护区或双臂互碰约束的候选。 | 返回 `Hold` 或要求重新观测。 |
| 安全内核 | 无论 LLM 或世界模型输出为何，均能拦截人/宠物过近、过期观测、碰撞、未知物和权限冲突。 | 强制保持、撤退或急停。 |
| 低层控制适配器 | 控制窗口过期、传感器突变或接触异常时，可在单窗口内中断。 | 执行安全保持并记录证据。 |
| 事后验证器 | 预测与真实状态冲突时，以真实观测为准并触发重规划。 | 不得把文本计划继续视为环境事实。 |

## 11. 实施要点

首先实现“LLM 输出 `PlanSpec` + 规则状态机 + 真实执行后校验”，即使 JEPA 模块暂时用仿真器或规则转移模型替代。随后以影子模式部署 JEPA：它先预测但不参与动作批准；只有在校准、风险提示和分布外检测达到预设标准后，才用于候选排序。最后再让世界模型参与短时域 MPC，始终保留独立的安全网关和人类接管路径。

这种实现顺序把系统的关键风险限制在可观察、可回归测试的接口中：**LLM 的错误被限制在任务规范层，JEPA 的误差被限制在候选排序层，低层危险由独立安全控制器拦截。**
