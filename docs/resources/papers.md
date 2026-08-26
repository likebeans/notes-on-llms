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

<SourceList :items="[
  { title: 'Attention Is All You Need', href: 'https://arxiv.org/abs/1706.03762', note: '2017 · Transformer 的起点，理解 attention、位置编码和并行训练。' },
  { title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', href: 'https://arxiv.org/abs/1810.04805', note: '2018 · 理解预训练-微调范式和双向编码器。' },
  { title: 'Language Models are Few-Shot Learners', href: 'https://arxiv.org/abs/2005.14165', note: '2020 · GPT-3 与 in-context learning 的代表论文。' },
  { title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', href: 'https://arxiv.org/abs/2005.11401', note: '2020 · RAG 原始范式，理解参数化/非参数化记忆结合。' },
  { title: 'LoRA: Low-Rank Adaptation of Large Language Models', href: 'https://arxiv.org/abs/2106.09685', note: '2021 · 参数高效微调的经典方法。' },
  { title: 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models', href: 'https://arxiv.org/abs/2201.11903', note: '2022 · 理解推理提示为什么能提升复杂任务表现。' },
  { title: 'ReAct: Synergizing Reasoning and Acting in Language Models', href: 'https://arxiv.org/abs/2210.03629', note: '2022 · Agent 中“推理 + 行动”循环的关键思想。' },
  { title: 'Toolformer: Language Models Can Teach Themselves to Use Tools', href: 'https://arxiv.org/abs/2302.04761', note: '2023 · 理解模型如何学习何时调用工具。' },
  { title: 'Direct Preference Optimization', href: 'https://arxiv.org/abs/2305.18290', note: '2023 · DPO 的核心来源，适合对比 RLHF。' },
  { title: 'Visual Instruction Tuning', href: 'https://arxiv.org/abs/2304.08485', note: '2023 · LLaVA 代表论文，理解视觉指令微调。' },
  { title: 'Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context', href: 'https://arxiv.org/abs/2403.05530', note: '2024 · 多模态与长上下文系统能力的代表性技术报告。' }
]" />

## 读论文的最小模板

1. 这篇论文要解决什么失败模式？
2. 它引入了什么新结构、训练方式或评估方式？
3. 它的对照实验是否足够说明问题？
4. 它的局限是什么？在生产系统里会怎么暴露？
5. 它和本手册哪一页相关：Prompt、RAG、Agent、训练还是多模态？
