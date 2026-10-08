---
title: 视频
description: 官方课程、公开课和开发者演示视频索引。
pageType: landing
module: site
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - resources
  - videos
---

# 视频

视频适合建立直觉，但实现时仍要回到官方文档、论文或代码。这里优先收录官方课程页、大学公开课页和厂商开发者演示。

<SourceList :items="[
  { title: 'Stanford CS25: Transformers United', href: 'https://web.stanford.edu/class/cs25/', note: '官方课程页，适合从研究者视角理解 Transformer 和前沿应用。' },
  { title: 'LangChain for LLM Application Development', href: 'https://www.deeplearning.ai/courses/langchain', note: '2023 · DeepLearning.AI 官方课程，适合快速理解 LLM 应用链路。' },
  { title: 'AI Agents in LangGraph', href: 'https://www.deeplearning.ai/courses/ai-agents-in-langgraph', note: '2024 · DeepLearning.AI 官方课程，适合从状态图角度理解 Agent 工作流。' },
  { title: 'OpenAI DevDay 2024 | Structured outputs for reliable applications', href: 'https://www.youtube.com/watch?v=kE4BkATIl9c', note: '2024 · OpenAI 官方视频，适合学习 schema 约束输出。' },
  { title: 'OpenAI DevDay 2024 | Balancing accuracy, latency, and cost at scale', href: 'https://www.youtube.com/watch?v=Bx6sUDRMx-8', note: '2024 · OpenAI 官方视频，适合理解生产应用的质量、延迟和成本权衡。' },
  { title: 'OpenAI DevDay 2025', href: 'https://openai.com/devday/', note: 'OpenAI 官方活动入口；请核对页面所列年份与各场次资料。' }
]" />

## 看视频时怎么记笔记

- 只记三类东西：概念定义、工程流程、失败案例。
- 每看完一个视频，写一个“我能立刻做的小实验”。
- 如果视频涉及 API，一定回官方文档确认当前参数和版本。

## 按章节选观看目标

| 章节 | 观看时要回答的问题 | 看完后的动作 |
| --- | --- | --- |
| Prompt | 指令、示例、schema 分别约束什么 | 用固定样本比较两个提示版本 |
| RAG | 错误是没找到证据，还是没有用好证据 | 画出候选到答案的链路 |
| Agent | 哪些步骤由模型选，哪些由代码定 | 为一次工具超时设计恢复路径 |
| 训练 | 改的是目标、数据还是参数更新方式 | 写出基线与验证集设计 |
| 多模态 | 图像怎样进入模型，答案如何定位证据 | 保存一个小字或表格失败案例 |

旧演讲中的 SDK 代码只用于理解当时的实现。开始复制前，回到对应版本文档；课程标题、届次与活动页内容会变化，以资源页面的实际标注为准。
