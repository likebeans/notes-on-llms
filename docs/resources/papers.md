---
title: 论文
description: LLM、RAG、Agent、微调和多模态方向的原始论文索引。
pageType: landing
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - resources
  - papers
---

# 论文

优先读摘要、方法图、实验设置和局限；不要一上来陷入所有公式。下面按“对本手册的解释价值”收录原始论文。

| 论文 | 年份 | URL | 为什么值得读 |
| --- | --- | --- | --- |
| Attention Is All You Need | 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Transformer 的起点，理解 attention、位置编码和并行训练。 |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 2018 | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | 理解预训练-微调范式和双向编码器。 |
| Language Models are Few-Shot Learners | 2020 | [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) | GPT-3 与 in-context learning 的代表论文。 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | 2020 | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | RAG 原始范式，理解参数化/非参数化记忆结合。 |
| LoRA: Low-Rank Adaptation of Large Language Models | 2021 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | 参数高效微调的经典方法。 |
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | 2022 | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) | 理解推理提示为什么能提升复杂任务表现。 |
| ReAct: Synergizing Reasoning and Acting in Language Models | 2022 | [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) | Agent 中“推理 + 行动”循环的关键思想。 |
| Toolformer: Language Models Can Teach Themselves to Use Tools | 2023 | [arXiv:2302.04761](https://arxiv.org/abs/2302.04761) | 理解模型如何学习何时调用工具。 |
| Direct Preference Optimization | 2023 | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) | DPO 的核心来源，适合对比 RLHF。 |
| Visual Instruction Tuning | 2023 | [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) | LLaVA 代表论文，理解视觉指令微调。 |
| Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context | 2024 | [arXiv:2403.05530](https://arxiv.org/abs/2403.05530) | 多模态与长上下文系统能力的代表性技术报告。 |

## 读论文的最小模板

1. 这篇论文要解决什么失败模式？
2. 它引入了什么新结构、训练方式或评估方式？
3. 它的对照实验是否足够说明问题？
4. 它的局限是什么？在生产系统里会怎么暴露？
5. 它和本手册哪一页相关：Prompt、RAG、Agent、训练还是多模态？
