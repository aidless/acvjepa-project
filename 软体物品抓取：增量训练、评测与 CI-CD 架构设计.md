# 软体物品抓取：增量训练、评测与 CI/CD 架构设计

## 1. 目标与非目标

本架构的目标是让 AC-VJEPA 在布料、海绵、软包装等柔性物品抓取任务中，以**可追溯、可回归、可回滚**的方式吸收难例数据，同时不把模型迭代变成对实机控制安全链的替代。系统应提高短期状态预测、接触/抓取事件识别、误差—不确定性校准和受限规划质量；系统不应承诺在任何未知柔性物体上实现自主安全操作。

> **发布原则：数据和模型可以持续更新，安全控制权不可转移。** 厂商伺服、限幅、急停、碰撞/近场保护和轨迹 TTL 始终独立于训练、影子模型、灰度模型和 LLM。

## 2. 总体架构

```text
真实机器人 / 影子模式 / 遥操作示教
  └─ CandidateEpisode（状态、动作、残差、不确定性、安全上下文）
       └─ 审核、去重、隐私处理、物理先验确认
            ├─ 真实难例集
            ├─ 历史 replay 集
            ├─ Sim-to-Real 合成扩增集
            └─ 永不回流训练的 stress / 安全评测集
                 ↓
数据与配置版本锁定（DVC/对象存储/manifest/哈希）
                 ↓
离线训练（教师蒸馏 + adapter/预测器增量微调 + replay）
                 ↓
CI：单元/契约/数据/训练/评测/压缩/安全回归
                 ↓
模型注册表（权重、ONNX、TensorRT engine、schema、报告、签名）
                 ↓
CD：离线仿真 → 影子模式 → 低风险灰度 → 逐级扩大 / 自动回滚
                 ↓
仅在独立安全门批准后生成短期控制窗口
```

## 3. 数据层：版本、拆分与血缘

### 3.1 数据分区

| 分区 | 内容 | 训练用途 | 发布用途 |
|---|---|---|---|
| `replay_core` | 已验证硬物与常见软物任务 | 防遗忘 | 必须不退化。 |
| `real_soft_hard` | 审核过的真实柔性难例与遥操作示教 | 学习真实接触/形变 | 核心增量收益来源。 |
| `sim_counterfactual` | 由真实难例和批准物理先验生成的仿真反事实 | 扩大参数/视觉覆盖 | 仅辅助主训练。 |
| `validation_frozen` | 按对象、布局、相机、材质与时间隔离 | 模型选择 | 不参与训练。 |
| `stress_never_train` | 传感器故障、极端遮挡、热/延迟、接触异常、OOD 柔性物体 | 只测安全回退与鲁棒性 | 任何版本都不得回流。 |
| `shadow_live` | 生产镜像输入与候选预测 | 不训练直连 | 评估部署真实性与 p99。 |

### 3.2 每条 episode 必须随附的元数据

每条真实或合成 episode 至少包含：数据源、父难例 ID、传感器与机器人时间范围、相机/标定版本、动作 schema、实际执行动作、接触/力信号、目标与终态事件、模型与规划版本、仿真器与 seed（若为合成）、质量标签和隐私/审批状态。缺失任一关键字段的 episode 不进入训练。

软体物理先验（刚度、阻尼、摩擦、几何变体）必须绑定审查记录。仿真任务从已批准范围采样，不得由模型不确定性直接生成任意物理参数。Isaac Lab 的模块化仿真、传感器与域随机化能力可作为此数据生成层的候选基础设施。[1]

## 4. 增量训练架构

### 4.1 模型更新范围

在首次迭代中，优先更新动作条件预测器、低秩适配器、事件头和不确定性头；视觉骨干保持冻结或仅局部解冻。这样可以让软体交互能力向新任务适配，同时降低灾难性遗忘和重新验证整个视觉骨干的成本。

训练损失由以下部分构成：JEPA 潜在预测、教师—学生潜在蒸馏、事件蒸馏、预测误差—不确定性校准以及历史 replay 保持。高不确定性应触发更保守的运行时决策，因此候选模型不能只通过平均预测误差，也应通过不确定性排序与 `LOCAL_HOLD` 召回的评估。

### 4.2 训练工作流

```text
validate-data
  → build-manifest（内容哈希、split、版本兼容检查）
  → train-adapter/predictor（可恢复 checkpoint）
  → export FP32 / ONNX / TensorRT candidate
  → offline-evaluate
  → simulate-evaluate
  → edge-performance-evaluate
  → register-candidate 或 quarantine
```

## 5. 评测金字塔

| 层级 | 输入 | 核心指标 | 失败后的动作 |
|---|---|---|---|
| 单元/契约 | schema、动作归一化、时间同步、模型 I/O | 可解析、时间顺序、无 NaN、版本一致 | 阻断训练。 |
| 离线预测 | 固定真实/合成 episode | 潜在预测、事件 F1、残差、校准、OOD 排序 | 阻断候选注册。 |
| 回归防遗忘 | `replay_core` 与冻结验证集 | 原任务不劣化、风险/保持信号不退化 | 阻断发布。 |
| 故障注入 | 噪声、掉帧、冻结帧、NaN、deadline、GPU 忙 | 安全保持召回、过期窗口拒绝、无非法继续 | 阻断发布。 |
| 仿真闭环 | 柔性物体参数/视觉/动作扰动 | 任务、约束违规、恢复、候选排序 | 降级到影子或阻断。 |
| Jetson 性能 | 目标 engine、热稳态、真实管线负载 | 端到端 p50/p95/p99、温度、功耗、队列深度 | 降级模型/阻断 canary。 |
| 影子与 canary | 生产镜像/低风险真实任务 | 模型差异、安全回退、p99、人工接管 | 暂停、回滚或扩大。 |

发布门槛必须按技能、速度、对象类别与机器人配置建立版本化阈值，而不是在代码中固化一个通用成功率。只有自动指标、故障注入和人工/安全审查全部通过，候选才可进入下一级。

## 6. CI/CD 设计

### 6.1 持续集成（CI）

| 阶段 | 触发 | 自动任务 | 阻断条件 |
|---|---|---|---|
| `ci-code` | PR/提交 | 格式、类型、依赖扫描、单元测试、SPSC/安全状态机测试 | 任一失败。 |
| `ci-contract` | 数据/模型接口变更 | 动作 schema、时间同步、ONNX I/O、模型 metadata、版本兼容性 | 契约改变但缺迁移/评测。 |
| `ci-data` | 新 manifest | 去重、split 泄漏、完整性、许可证/隐私、物理先验审批 | 数据来源或字段不合格。 |
| `ci-train-smoke` | 训练代码/配置变更 | 小数据确定性训练、loss/EMA/checkpoint 测试 | 无法恢复或数值不稳定。 |
| `ci-safety` | 每个候选 | 故障注入、状态过期、NaN、超时、硬停止/LLM 托管契约 | 任一安全回退漏触发。 |
| `ci-export` | 候选模型 | ONNX 导出、TensorRT/ORT 图检查、静态 shape、数值差异 | 关键输出或 schema 不一致。 |

### 6.2 持续交付（CD）

模型通过 CI 后注册为 `CANDIDATE`，但不是立即生产版本。CD 使用逐级状态机：

```text
CANDIDATE
  → OFFLINE_APPROVED
  → SIM_APPROVED
  → EDGE_APPROVED
  → SHADOW
  → CANARY_LOW_RISK
  → CANARY_EXPANDED
  → PRODUCTION
  ↘ PAUSED / ROLLED_BACK / QUARANTINED
```

影子模式复制生产状态/候选动作到新模型，只记录预测与门控建议，不影响实机轨迹。一般 MLOps 实践也将影子部署定义为新模型接收真实流量但其预测不参与实际输出；灰度发布则以小比例、逐步扩大暴露并与基线比较。[2] 在机器人系统中，应把“流量百分比”改为更保守的**身份稳定的机器人/班次/场地白名单 + 低风险技能白名单**，而不是对同一机器人同一任务随机切换模型。

### 6.3 典型 CI/CD 工作流（伪 YAML）

```yaml
name: soft-grasp-model-release
on: [pull_request, workflow_dispatch]
jobs:
  code_and_contract:
    steps: [lint, unit_test, schema_compatibility, static_security]
  data_and_training:
    needs: code_and_contract
    steps: [validate_manifest, train_smoke, checkpoint_resume]
  evaluation:
    needs: data_and_training
    steps: [offline_regression, fault_injection, sim_closed_loop, calibration]
  edge_artifact:
    needs: evaluation
    steps: [onnx_export, trt_build_target, edge_p99_benchmark, artifact_sign]
  register_candidate:
    needs: edge_artifact
    steps: [attach_reports, approval_gate, create_shadow_release]
```

真实硬件 shadow/canary 任务通常需要受控实验场地、机器人状态和人工职责分离，不宜作为普通云端 CI runner 的无监督步骤。它们由 CD 编排器在满足前置门槛后调用，并将结果反写为不可篡改的发布审计记录。

## 7. 影子模式与灰度发布门控

### 7.1 影子模式

每个 `ShadowObservation` 同时记录基线与候选模型的：输入状态版本、延迟、潜在预测摘要、事件概率、不确定性、建议候选 ID、推理异常和事后真实事件。通过 `correlation_id` 将同一输入、同一候选动作和事后结果关联，防止误把不同时间窗口的预测当成模型差异。

影子门槛包含最小样本量、覆盖率（物体/场景/光照/技能）、候选 p99 与基线 p99 的相对差、无效输出率、风险/保持漏报、错误置信度和与事后事件的一致性。影子模型永远不获得控制写权限。

### 7.2 灰度发布

灰度版本仅可用于预注册的低速、低力、短 TTL、单原子技能，且要保持机器人身份稳定：同一 `robot_id + shift_id + task_template` 在一个观察窗口内始终路由到同一模型版本，以便归因与快速回滚。每个 canary 决策仍需经过独立安全门；候选模型不能自行绕过速度、力、保护区或人类在场要求。

自动回滚应在任一硬失败、安全回退漏触发、无效输出、p99 预算超限或统计窗口内劣于基线的核心指标时立即发生。扩大 canary 需要最小样本/覆盖量、全部硬门通过，并由授权人/发布策略批准。

## 8. 运维、审计与回滚

每个模型注册项应包含：父模型、训练数据 manifest 哈希、代码 commit、超参数、物理先验、评测报告、ONNX/TensorRT engine 哈希、JetPack/ORT/TRT 版本、签名、发布策略、灰度范围和回滚模型 ID。回滚应是控制面元数据切换到已验证基线，而不是在机器人上重装训练环境。

任何安全事件都会冻结扩大、保留完整证据链，并将候选版本转为 `PAUSED` 或 `QUARANTINED`。是否重新训练、修改先验或恢复 canary，应通过根因分析和相应的测试用例补充来决定。

## 参考资料

[1]: [NVIDIA, *Isaac Lab Documentation*](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/index.html)

[2]: [JFrog ML, *Shadow deployment vs. canary release of machine learning models*](https://www.qwak.com/post/shadow-deployment-vs-canary-release-of-machine-learning-models)

[3]: [Li & Silver, *Embodied Active Learning of Relational State Abstractions for Bilevel Planning*, CoLLAs 2023](https://proceedings.mlr.press/v232/li23a.html)
