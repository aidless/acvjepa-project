# JEPA 技术事实核验

| 议题 | 可确认内容 | 边界与纠正 |
|---|---|---|
| JEPA 的提出 | LeCun 于 2022 年的《A Path Towards Autonomous Machine Intelligence》提出以预测世界状态表征为核心的架构方向；I-JEPA 于 2023 年成为图像实现，V-JEPA 于 2024 年扩展至视频。 | “JEPA”应理解为一个架构/训练范式家族，而非一个已完成、通用的世界模型。 |
| 训练目标 | I-JEPA 从图像上下文预测遮挡目标块的抽象表示；V-JEPA 则预测视频中被遮挡的时空区域的潜在表示。 | 这不是对原始像素的逐点重构，但“预测潜在表示”不自动等于已经学到了可干预的因果机制。 |
| 自监督与数据 | V-JEPA 预训练使用未标注视频数据，标签仅用于迁移至特定任务。 | 这与“只学习语言”的 LLM 确有本体差异，但视频观察本身并不提供动作、接触力、反事实和长期后果的全部信息。 |
| 已展示的能力 | 官方将 V-JEPA 定位为视频内容/情境感知的感知型研究模型；其预测器被称为早期物理世界模型。 | 官方亦明确把如何用于规划或序列决策称作“下一步”，不应把它表述为已经实现稳健的规划、因果推理或通用机器人控制。 |
| 与生成方法的关系 | 在高维像素/词元空间逐一生成会要求模型拟合许多与任务无关或本来不可预测的细节；潜在预测有机会学习更紧凑、语义性的表示并节省计算。 | 生成与预测不是绝对对立。生成式模型也可学习和使用潜在状态；世界模型通常仍需观测编码、状态预测、行动条件与规划/控制模块协同。 |

> Meta 对 V-JEPA 的谨慎原话是：到当时其工作“主要关于感知”，下一步才是展示此类预测器/世界模型如何用于规划或序列决策。这一限定是评估路线成熟度的关键。

## 来源

1. Yann LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022. https://openreview.net/pdf?id=BZ5a1r-kVsf
2. Meta AI, *I-JEPA: The first AI model based on Yann LeCun’s vision for more human-like AI*, 2023-06-13. https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/
3. Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023. https://arxiv.org/abs/2301.08243
4. Meta AI, *V-JEPA: The next step toward advanced machine intelligence*, 2024-02-15. https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
