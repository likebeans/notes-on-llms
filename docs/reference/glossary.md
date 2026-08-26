---
title: 术语表
description: LLM、RAG、Agent、训练微调和多模态常用术语速查。
pageType: article
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - reference
  - glossary
level: beginner
prerequisites: []
reviewed: '2026-08-26'
techVersion: 2026-08（概念速查）
---

# 术语表

每个术语尽量用一句话定义，再给出继续阅读入口。面试或写方案时，先说清“它解决什么问题”，再说技术细节。

| 术语 | 简明定义 | 深入阅读 |
| --- | --- | --- |
| Token | 模型处理文本的基本单位，可能是词、子词或字符片段。 | [Prompt 基础](/llms/prompt/basics) |
| Context Window | 单次请求中模型可看到的 token 上限，影响长文档、对话历史和工具结果放置。 | [上下文工程](/llms/prompt/context) |
| Embedding | 把文本、图片等映射成向量，用于相似度检索和聚类。 | [Embedding](/llms/rag/embedding) |
| RAG | 生成前先检索外部知识，把检索结果放入上下文再回答。 | [RAG 全景](/llms/rag/) |
| Chunk | 为检索把原文切成较小片段；大小、重叠和元数据会影响召回。 | [文档切分](/llms/rag/chunking) |
| Rerank | 对初步召回结果重新排序，提高最终上下文质量。 | [重排序](/llms/rag/rerank) |
| Tool Calling | 模型按 schema 生成工具调用参数，由系统执行外部函数或 API。 | [工具调用](/llms/agent/tool-calling) |
| Agent | 以模型为控制器，结合规划、记忆、工具和反馈执行多步任务的系统。 | [Agent 全景](/llms/agent/) |
| MCP | Model Context Protocol，用统一协议连接 AI 应用与外部工具/数据源。 | [MCP 概念](/llms/mcp/concepts) |
| SFT | Supervised Fine-Tuning，用高质量指令/答案数据让模型学习目标行为。 | [SFT](/llms/training/sft) |
| LoRA | Low-Rank Adaptation，通过低秩增量矩阵进行参数高效微调。 | [LoRA](/llms/training/lora)、[论文](https://arxiv.org/abs/2106.09685) |
| DPO | Direct Preference Optimization，用偏好对直接优化模型偏好。 | [DPO](/llms/training/dpo)、[论文](https://arxiv.org/abs/2305.18290) |
| RLHF | Reinforcement Learning from Human Feedback，用人类偏好训练奖励模型并优化策略。 | [RLHF](/llms/training/rlhf) |
| Multimodal | 同时处理文本、图像、音频或视频等多种模态。 | [多模态全景](/llms/multimodal/) |
| Evaluation Set | 用来比较模型/提示/检索版本的一组固定样本和期望结果。 | [评估指标](/reference/metrics) |

## 常见混淆

- RAG 不是“向量库”：向量库只是检索组件之一。
- Agent 不是“无限自动化”：越接近生产，越需要权限、审批和可观测性。
- 微调不是知识库：经常变化、需要引用的数据更适合 RAG。
- 长上下文不是免费午餐：上下文越长，越需要选择、压缩和评估。
