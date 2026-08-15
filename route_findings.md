# 路线互补性：语言模型、世界模型与具身智能

| 判断 | 证据与解释 |
|---|---|
| 纯文本、自回归 LLM 有系统性局限 | 缺少实时感知和直接交互数据，容易在未被训练文本充分覆盖的物理、空间、反事实和长程控制任务上失稳。这支持 LeCun 对“只靠 next-token 预测不足以形成自主智能体”的批评；Reuters 也记录了他认为 next-word/next-pixel 路线单独不足的说法。 |
| “LLM 是死胡同”过于绝对 | LLM 是优秀的语义压缩器、知识接口与高层任务分解器。PaLM-E 已将视觉和连续状态估计嵌入预训练语言模型，并在顺序机器人操控规划等多种具身任务进行端到端训练。这不证明语言模型已解决具身智能，但否定了“语言模型只能处理脱离世界的文本统计”这一强断言。 |
| 更合理的工程预测是混合架构 | 未来智能体可让视觉/触觉编码器提供状态估计，由动作条件世界模型预测局部未来，再用语言模型处理指令、常识、工具与高层计划，最后经搜索、优化或控制器执行。各模块可端到端协同训练；因此，JEPA 与 LLM 不是必然互斥的赛道。 |
| JEPA 的关键难题尚未被解决 | 要从视频表征走到可靠自主性，还要处理：动作条件预测、因果可辨识性、部分可观测、长时程记忆与信用分配、稀有风险、从模拟/视频到真实执行的迁移，以及在开放环境中可验证的安全控制。 |

> PaLM-E 的论文并未宣称已实现通用机器人智能；它的价值在于说明“语言模型 + 连续感知”已是可实验的技术路线，因此评估对象应是完整智能体系统，而不是孤立的 next-token 训练目标。

## 来源

1. Reuters, *Ex-Meta AI chief Yann LeCun's AMI raises $1.03 billion for alternative AI approach*, 2026-03-10. https://www.reuters.com/business/ex-meta-ai-chief-yann-lecuns-ami-raises-103-billion-alternative-ai-approach-2026-03-10/
2. Driess et al., *PaLM-E: An Embodied Multimodal Language Model*, 2023. https://arxiv.org/abs/2303.03378
3. Meta AI, *V-JEPA: The next step toward advanced machine intelligence*, 2024-02-15. https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/
