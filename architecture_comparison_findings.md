# JEPA 与主流大模型：技术事实记录

## 比较范围与方法

本次比较针对两类公开技术范式，而非声称掌握 OpenAI 当前闭源模型的全部内部实现：

1. **主流生成式基础模型范式**：以 GPT-4 技术报告所公开的 Transformer、多模态输入、文本自回归输出和 post-training alignment 为代表。
2. **JEPA 范式**：以 I-JEPA 与 V-JEPA 论文/官方说明所公开的自监督潜在表征预测机制为代表。

## 已核验事实

| 维度 | 主流大模型（以 GPT-4 公开资料为例） | JEPA（I-JEPA/V-JEPA） |
|---|---|---|
| 明确公开的模型类型 | GPT-4 被描述为 Transformer-based multimodal model；可接受图像与文本输入，输出文本。 | I-JEPA 为 non-generative self-supervised image learning；V-JEPA 为仅用视频特征预测目标训练的视觉模型。 |
| 预训练目标 | GPT-4 技术报告明确：预测文档中的下一个 token。 | I-JEPA：从单个上下文块预测同一图像多个目标块的表示；V-JEPA：预测视频特征。 |
| 预测/损失所在空间 | token/输出信号空间；可直接生成语言文本，GPT-4 是文本输出。 | 学习到的 feature / latent representation space；并不直接重构像素。 |
| 监督与数据形态 | 公开报告未给出完整训练数据构成；公开陈述强调多模态输入与后训练对齐。 | I-JEPA 不依赖手工数据增强；V-JEPA 只从公开视频数据学习特征预测，不使用文本、重建、负样本或预训练图像编码器。 |
| 已经展示的能力 | 语言、代码、文本视觉理解及丰富的提示时推理/工具使用外层系统能力，但模型并非直接输出物理行动。 | 高质量视觉/视频表征、运动和外观相关下游任务；I-JEPA 官方称其预测器是受限的、早期的世界模型。 |
| 能力边界 | GPT-4 也明确称在许多真实世界情景中仍不及人类。 | 潜在表征预测尚不等同于动作条件的因果模型、长程规划或控制；这些是通向自主智能体仍需补齐的系统模块。 |

## 核心证据与来源

1. OpenAI, *GPT-4 Technical Report*, arXiv:2303.08774. GPT-4 是 Transformer-based model，预训练为预测下一个 token；post-training alignment 改善事实性与遵从性。https://arxiv.org/abs/2303.08774
2. Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243. I-JEPA 从上下文预测目标块的表示，属于非生成式图像自监督学习。https://arxiv.org/abs/2301.08243
3. Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video*, arXiv:2404.08471. V-JEPA 只以视频 feature-prediction objective 训练。https://arxiv.org/abs/2404.08471
4. Meta AI, *I-JEPA: The first AI model based on Yann LeCun’s vision for more human-like AI*, 2023-06-13. 官方将 I-JEPA 预测器称作在静态、部分可观察图像中建模空间不确定性的原始且受限世界模型。https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/
5. OpenAI, *GPT-4*, 2023-03-14. 公开说明 GPT-4 可接受图像/文本输入，生成文本输出，且在很多真实世界场景仍不及人类。https://openai.com/index/gpt-4-research/

## 表述注意

- “JEPA 是世界模型”在严格技术语境中应限定为世界模型的一个候选表征/预测部件；公开 I-JEPA 与 V-JEPA 并不等于完整、动作条件、通用的世界模型。
- “LLM 只会统计词频”不准确。next-token prediction 是训练目标；它不蕴含模型内部仅保存词频，也不蕴含模型不具备任何抽象表征。
- 不应根据 GPT-4 公开资料推断 OpenAI 现有闭源系统的完整内部架构、数据或推理实现，因为这些细节未被完全公开。

## 补充核验（浏览器阅读原始页面）

- GPT-4 技术报告原文摘要明确写道：其为 Transformer-based model，预训练为预测文档下一个 token；后训练对齐改善了事实性和对期望行为的遵从。报告同时把 GPT-4 描述为可接收图像与文本输入、产生文本输出的多模态模型，并承认其在许多真实世界场景中仍不如人类。
- V-JEPA 原始论文摘要明确写道：该系列视觉模型只使用 feature prediction objective，从公开视频的 200 万视频训练；不使用预训练图像编码器、文本、负样本、重建或其他监督。文中报告的是下游图像/视频表征任务成绩，而非端到端机器人规划或动作控制成绩。

因此，比较时不能把“V-JEPA 在视频表征任务成功”扩大为“JEPA 已实现通用物理因果理解”，也不能把“GPT-4 以 next-token 预训练”扩大为“主流系统只能处理文本”。两者都是对单一训练目标的不恰当过度外推。
