# AC-VJEPA 核心模块使用说明

## 文件与用途

| 文件 | 用途 |
|---|---|
| `ac_vjepa_core.py` | 可运行的研究原型核心：动作条件潜在预测、EMA 目标、训练损失、单飞 deadline 推理、延迟统计、局部保持到 LLM 托管的协调器。 |
| `LLM安全托管回退状态机.md` | 描述 `NORMAL → LOCAL_HOLD → LLM_SUPERVISION → REPLAN_PENDING` 的安全状态机。 |
| `ACVJEPA_实机时延与控制频率验收协议.md` | 上线前必须执行的时延、超时、故障注入与安全回退验收。 |

## 依赖与本地验证

该文件依赖 PyTorch。建议在项目隔离环境中安装与 CUDA 驱动匹配的 PyTorch 发行版，再运行：

```bash
python3 ac_vjepa_core.py
```

脚本的烟雾测试会执行一次合成数据训练步、EMA 更新、deadline 内正常推理，以及一次强制高不确定性回退。该交付已在 CPU PyTorch 环境中实际运行通过：训练步、EMA 更新、`NORMAL` 路径和 `LOCAL_HOLD → LLM_SUPERVISION` 路径均被触发。该测试只验证模块逻辑；其 CPU 时延不能外推为实机 GPU 时延，目标机器人仍须依据验收协议进行端到端 p99、抖动和故障注入测试。

## 最小训练调用

```python
model = ActionConditionedVJEPA(
    image_channels=3,
    proprio_dim=PROPRIO_DIM,
    action_dim=ACTION_BLOCK_DIM,
    latent_dim=128,
    event_dim=NUM_EVENTS,
)

prediction = model.predict(context_video, context_proprio, executed_action_blocks)
targets = model.target_latents(future_video, future_proprio)
losses = action_conditioned_jepa_loss(prediction, targets, event_targets)
losses.total.backward()
optimizer.step()
model.update_ema_target()
```

`executed_action_blocks` 必须代表经过安全限幅、最终由机器人实际执行的动作块，而不是 LLM 文本计划或未执行的候选轨迹。`future_video` 与动作的时间对齐、坐标系、相机预处理必须与部署时一致。

## 最小在线调用

```python
worker = SingleFlightInferenceWorker(model, device, LatencyMonitor())
coordinator = SafeHandoverCoordinator(worker, limits)

decision = coordinator.plan_or_handover(
    state=current_state_envelope,
    model_inputs={
        "context_video": video_tensor,
        "context_proprio": proprio_tensor,
        "action_blocks": candidate_action_blocks,
    },
    hardware_healthy=robot_health.is_ok(),
)
```

只有当 `decision.mode == NORMAL` 时，规划器才可将预测结果提交给 MPC 和独立安全网关。`LOCAL_HOLD` 表示控制域必须先原子化禁止新轨迹、确认安全保持；之后才允许：

```python
supervised = coordinator.confirm_local_hold(decision, hold_confirmed=True)
```

`supervised.mode == LLM_SUPERVISION` 时，LLM 只可返回下表中的受限选择。

| LLM 选择 | 允许的作用 | 不能做什么 |
|---|---|---|
| `OBSERVE` | 请求已注册的观察技能或更新状态。 | 直接移动机械臂。 |
| `ASK_USER` | 说明事实并请求澄清/确认。 | 降低风险阈值或解除保持。 |
| `SELECT_PREAPPROVED_SKILL` | 选择已注册、参数受约束的恢复技能。 | 创建新动作、指定关节/速度/力。 |
| `RETRY_AFTER_HEALTH` | 在系统已恢复健康后请求重新规划。 | 复用旧状态或旧轨迹。 |
| `END_TASK` | 安全终止并生成报告。 | 继续执行未完成的动作。 |

所有选择都必须经过 `RecoveryGate`，同时满足状态新鲜、硬件健康、动作已注册和独立安全门批准，才可进入 `REPLAN_PENDING`。

## 集成边界

`ac_vjepa_core.py` 不包含相机驱动、逆运动学、碰撞检测、厂商机械臂 SDK、实时线程、LLM API 或安全认证组件。应由外部系统提供：

1. 经时间同步的相机、本体、触觉与 `ActionBlock` 数据；
2. 受限候选动作生成器与 MPC；
3. 不依赖 Python/GPU/LLM 的实时安全与伺服控制域；
4. 对每一个 `ControlWindow` 的 TTL、状态版本、权限与审计日志；
5. 影子模式和故障注入验收后才开放的低速、短窗口自主执行。
