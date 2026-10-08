---
title: 术语表
description: LLM、RAG、Agent、训练微调和多模态常用术语速查。
pageType: article
module: site
updated: '2026-10-08'
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
| Agent | 由模型参与选择后续行动、结合工具和环境反馈完成任务的系统；不要求同时具备所有设计模式。 | [Agent 全景](/llms/agent/) |
| MCP | Model Context Protocol，用统一协议连接 AI 应用与外部工具/数据源。 | [MCP 概念](/llms/mcp/concepts) |
| SFT | Supervised Fine-Tuning，用高质量指令/答案数据让模型学习目标行为。 | [SFT](/llms/training/sft) |
| LoRA | Low-Rank Adaptation，通过低秩增量矩阵进行参数高效微调。 | [LoRA](/llms/training/lora)、[论文](https://arxiv.org/abs/2106.09685) |
| DPO | Direct Preference Optimization，用偏好对直接优化模型偏好。 | [DPO](/llms/training/dpo)、[论文](https://arxiv.org/abs/2305.18290) |
| RLHF | Reinforcement Learning from Human Feedback，利用人类反馈优化策略的一类方法；常见路线包含奖励模型与强化学习。 | [RLHF](/llms/training/rlhf) |
| Multimodal | 同时处理文本、图像、音频或视频等多种模态。 | [多模态全景](/llms/multimodal/) |
| Evaluation Set | 用来比较模型/提示/检索版本的一组固定样本和期望结果。 | [评估指标](/reference/metrics) |

## 常见混淆

- RAG 不是“向量库”：向量库只是检索组件之一。
- Agent 不是“无限自动化”：越接近生产，越需要权限、审批和可观测性。
- 微调不是知识库：经常变化、需要引用的数据更适合 RAG。
- 长上下文不是免费午餐：上下文越长，越需要选择、压缩和评估。

## 工程中容易漏掉的词

| 术语 | 定义与边界 | 对应主题 |
| --- | --- | --- |
| RRF | 按候选在各检索通道的名次融合排序，避免直接比较原始分数 | [检索](/llms/rag/retrieval) |
| Cross-encoder | 联合编码查询与候选文本并打分，常用于重排 | [重排](/llms/rag/rerank) |
| Faithfulness | 答案断言是否被给定证据支持；不等同事实绝对正确 | [指标](/reference/metrics) |
| Ground truth | 在明确标注规则下的参考答案或标签，也需要复核 | [评估](/llms/rag/evaluation) |
| Idempotency | 同一业务动作重复提交不产生额外副作用，需要明确动作标识与实现保证 | [异常恢复](/llms/agent/exception-handling) |
| Checkpoint | 可恢复的执行状态快照；不能只保存聊天摘要 | [Agent 记忆](/llms/agent/memory) |
| Tool schema | 工具参数与结果的结构契约；不是权限策略 | [工具调用](/llms/agent/tool-calling) |
| Capabilities | 协议参与方声明支持的能力，交换方式取决于规范版本；不等同用户授权 | [MCP 概念](/llms/mcp/concepts) |
| PEFT | 参数高效微调方法集合，LoRA 是其中一种 | [LoRA](/llms/training/lora) |
| QLoRA | 量化冻结基座并训练 LoRA 适配器的一种微调方法 | [LoRA](/llms/training/lora) |
| KV cache | 自回归解码复用历史键值状态的缓存，会占用推理显存 | [部署推理](/llms/training/serving) |
| TTFT | 首 token 延迟；与生成完整回答的延迟分开看 | [推理性能](/llms/training/csdn/llm-qps-tpm-concurrency) |
| Connector | 将视觉等模态的特征接到语言模型的连接模块 | [连接器](/llms/multimodal/connector) |
| OCR | 将图像中的文字识别为文本；布局和图表关系可能仍需额外处理 | [多模态 RAG](/llms/multimodal/rag-agent) |

## 一句话区分

- **SFT 与 LoRA**：前者描述监督学习目标，后者描述更新哪些参数，可以组合。
- **工具调用与 MCP**：前者是能力调用机制，后者是应用与外部能力间的协议接口。
- **Schema 通过与业务正确**：字段合法不代表内容真实、资源可访问或动作已成功。
- **检索相似度与概率**：排序分数不能未经校准就解释为“答案正确的概率”。
