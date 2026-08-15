# AC-VJEPA：难例驱动的 Sim-to-Real 自动化数据生成与合成流水线

## 1. 目标：从“不确定”到“可验证的数据资产”

难例流水线的目标不是用生成数据掩盖世界模型失败，而是将真实环境中持续高不确定性的交互，转换为**带有来源、物理假设、仿真参数、质量门槛和训练用途的可审计数据资产**。针对布料、海绵、软包装等柔性物体，真实物理过程往往无法被一次仿真完美复现；因此必须把仿真看作围绕审核过的真实记录建立反事实覆盖和数据课程的工具，而不是事实真值的替代。

Isaac Lab 提供模块化机器人学习、物理/渲染后端、传感器、向量化渲染和域随机化能力，可作为此类批量仿真任务的候选基础设施。[1] RoboCasa 则提供面向家居任务的仿真与合成轨迹参照，可用于家居操作场景的任务模板、对象布局和评测基线。[2]

## 2. 端到端闭环

```text
边缘端持续监测
  → 高不确定性 + 高残差 + 新颖度 + 安全事件形成 CandidateEpisode
  → 先 LOCAL_HOLD / 观察 / 人类示教，禁止自由物理探索
  → 难例审核、去重、聚类、隐私检查与软体物理先验确认
  → 将真实片段编译为种子化 SimJob manifests
  → 仿真批量运行：回放实际动作 + 有界反事实扰动 + 域随机化
  → 质量过滤与 sim-real 对齐评分
  → train / validation / stress 三个隔离 split
  → replay + 新难例 + 蒸馏的离线增量微调
  → 离线回放、故障注入、仿真闭环、影子模式、canary 与回滚发布
```

> **关键规则：模型未见过的柔性物体不触发“更多自主尝试”，而触发“更保守的安全状态 + 更高质量的数据采集”。**

## 3. 难例识别与去重

### 3.1 多信号触发器

不应只以单帧方差作为“难例”。建议为每个 episode 计算连续窗口分数：

> \(S_{hard}=w_u U_{persist}+w_r R_{persist}+w_n N+w_e E\)

其中 \(U_{persist}\) 是持续不确定性，\(R_{persist}\) 是预测潜在状态与新观测间的持续残差，\(N\) 是相对于已知回放库的表征新颖度，\(E\) 是抓取滑移、预期事件未发生、反复安全门拒绝等运行事件。任何人员近场、力异常、硬件告警、急停或安全禁区侵入，都优先归类为安全事件，而不是自动仿真任务。

具身主动学习研究已展示可用模型集成熵来选择信息量高的交互和专家查询，以提高交互和查询效率。[3] 该类结果支持“用不确定性做样本选择”，但不替代实机的安全约束与人工审核。

### 3.2 去重与聚类

将候选片段以视觉潜在状态、动作类型、对象类别、接触事件、相机位姿和预测残差特征进行分层聚类。每个簇只保留少量高质量代表样本，防止同一种遮挡、同一块布料或同一相机抖动淹没训练集。输出标签至少包含：`physics_novelty`、`visual_novelty`、`sensor_fault_suspected`、`safety_event`、`human_label_needed`。

| 候选类型 | 自动进入仿真编译？ | 下一步 |
|---|---|---|
| 高不确定性、无安全事件、动作/时间同步完整 | 可以，但仅基于已批准先验 | 回放+有界反事实 SimJob。 |
| 高残差且疑似相机掉帧/时间错位 | 不可以 | 先诊断传感器/同步链路。 |
| 柔软物体形变，且有人工示教/材质信息 | 可以，人工确认先验后 | 软体参数化仿真与真实 replay。 |
| 力异常、近场人员、碰撞、急停 | 不可以 | 安全事故审查与人工处理。 |
| 视觉新颖但物理预测正常 | 可作为视觉域随机化 | 不需扩张软体/接触参数。 |

## 4. 从真实难例拟合到仿真先验

### 4.1 先验不是随机数字

对柔性物体，`Young’s modulus`、泊松比、阻尼、密度、摩擦、接触刚度、初始褶皱/几何、夹爪摩擦和相机延迟都会影响行为。任何参数范围都必须来自下列至少一种来源：物体材料测量、供应商材料信息、经批准的物理实验、人工专家审核，或已验证的历史对象簇。`sim2real_hard_example_compiler.py` 因此要求 `SoftObjectPrior` 携带 `prior_id` 和 `approval_ticket`；它只从批准范围采样，不会从一次模型失败中自行发明物理区间。

### 4.2 两层随机化

| 随机化层 | 参数示例 | 作用 | 约束 |
|---|---|---|---|
| **任务条件化层** | 初始褶皱、目标区域、实际动作 replay、对象尺寸、接触时机 | 保持与真实难例的因果关联 | 优先围绕真实 episode 的已执行动作建立变化。 |
| **视觉/传感器层** | 光照、材质、相机外参小扰动、曝光/噪声、有限本体延迟 | 缓解外观与传感器域差 | 不能改变时间同步和动作 schema。 |
| **物理反事实层** | 阻尼、摩擦、材料刚度、接触刚度、几何变体 | 产生可解释的形变/接触覆盖 | 仅在经过审核的先验范围内采样。 |
| **压力测试层** | 有界动作扰动、遮挡、弱光、有限延迟 | 测试预测不确定性与安全回退 | 与训练 split 隔离，不将全部极端数据混入训练。 |

## 5. 自动化作业清单与数据契约

`sim2real_hard_example_compiler.py` 将审核后的 `CandidateEpisode + SoftObjectPrior` 编译为 JSONL `SimJob`。每个作业具有稳定 seed、父 episode、仿真器/版本、资产、物理参数、视觉/传感器随机化、动作扰动、数据来源和质量要求。这样 Isaac Lab、RoboCasa 或其他仿真适配器可以读同一份作业格式，而训练端能知道每条样本来自真实、仿真还是压力测试。

```json
{
  "parent_episode_id": "hard-soft-cloth-0001",
  "seed": 101,
  "split": "train",
  "simulator": "isaac_lab_adapter",
  "physics": {
    "young_modulus": "sampled_from_approved_prior",
    "damping": "sampled_from_approved_prior"
  },
  "action_perturbation": {
    "source": "executed_action_replay",
    "bounded_pose_noise": "approved_small_range"
  },
  "quality_requirements": {
    "must_record_actual_actions": true,
    "must_record_contacts": true,
    "must_record_simulator_seed": true
  }
}
```

仿真器适配器的输入/输出接口应为：

```text
SimJob manifest
  → reset(scene, robot, object, physics, sensor, seed)
  → replay_or_perturb_approved_action_trace()
  → record(RGB/RGB-D, proprio, contacts, actual actions, object state, terminal events)
  → validate_schema_and_physics()
  → EpisodeArtifact + QualityReport
```

“实际动作”在仿真中也必须记录：如果安全约束、接触反作用或控制器把计划动作裁剪，训练应看到裁剪后的动作，而不是原始请求动作。

## 6. 质量过滤与 sim-real 对齐

每个 EpisodeArtifact 进入训练前应通过三类检查。

| 检查 | 自动指标 | 失败处理 |
|---|---|---|
| 契约完整性 | 时间戳单调、模态齐全、schema/资产/种子存在、无 NaN | 拒绝并修复适配器。 |
| 物理可用性 | 接触记录合理、任务终态可解释、对象未穿透/爆炸、动作可执行 | 标为仿真故障，不作为学习样本。 |
| sim-real 相似性 | 真实/仿真特征距离、事件时间差、接触序列、图像/状态统计 | 低相似样本进入压力/诊断集；不直接进入主训练集。 |

主训练集不应被合成样本淹没。建议让真实审核轨迹保留较高权重，仿真用于填充材质、几何、光照和动作扰动组合；压力测试数据用于衡量不确定性校准与保持触发，而非全部用于拟合。具体配比应通过离线验证集和影子实机表现确定，而不是预设一个固定比例。

## 7. 增量微调与防遗忘

每个增量版本应训练在四个可追溯数据分区上：

1. **历史 replay**：保持硬物抓取、放置、开门等已有能力。
2. **真实难例**：由人工/遥操作或高质量被动记录获得的柔软物体交互。
3. **经过质量门的仿真扩增**：在真实难例附近做反事实覆盖。
4. **独立压力集**：只用于 OOD、传感器故障和安全回退评估，不回流训练。

使用旧模型蒸馏、适配器/小预测器优先微调、事件和不确定性校准损失，以及按对象/任务簇的均衡采样。训练完成后必须同时比较旧任务与新柔性任务的：多步预测、事件识别、误差—不确定性校准、MPC 候选排序、`LOCAL_HOLD` 召回/误触发、影子模式 p99。

## 8. 发布门控

```text
OFFLINE_CANDIDATE
  → 数据/许可证/隐私审查
  → 回放 + 难例 + 压力集评测
  → 仿真闭环与故障注入
  → 边缘 TensorRT/ORT 数值与 p99 验证
  → 影子模式（不影响机械臂）
  → 小范围、低速、短 TTL canary
  → 可回滚生产版本
```

任何数据、模型、动作 schema、相机标定、仿真版本或 TensorRT engine 变化，都应生成新版本标识。若柔软物体任务仍持续高不确定性，系统应保留保守模式并收集更高质量真实示教，而不是通过扩大合成数据规模强行解除保持。

## 参考资料

[1]: [NVIDIA, *Isaac Lab Documentation*](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/index.html)

[2]: [RoboCasa, *Large-Scale Simulation of Everyday Tasks for Generalist Robots*](https://robocasa.ai/)

[3]: [Li & Silver, *Embodied Active Learning of Relational State Abstractions for Bilevel Planning*, CoLLAs 2023](https://proceedings.mlr.press/v232/li23a.html)
