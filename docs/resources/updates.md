---
title: 技术核验与更新索引
description: 2026 年 10 月的定向技术核验：连接官方文档、站内章节、版本边界和示例验证范围。
pageType: landing
module: site
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - updates
  - sources
---

# 技术核验与更新索引

本轮核验日期为 **2026-10-08**。这里记录哪些知识点重新对照了一手资料，以及阅读现有教程时需要注意的版本边界。核验日期不是论文发布日期，也不表示所有示例已经运行；每篇文章会分别说明资料复核和代码验证情况。

## 本轮重点

| 模块 | 更新内容 | 进入章节 |
| --- | --- | --- |
| RAG | 将上下文补充、图检索和页面视觉检索按任务与成本区分；明确评测分母和证据粒度 | [Embedding](/llms/rag/embedding)、[范式演进](/llms/rag/paradigms)、[评估](/llms/rag/evaluation) |
| Agent | 显式工具 schema、业务动作幂等、长任务状态持久化；区分多次尝试与持续成功 | [工具调用](/llms/agent/tool-calling)、[记忆](/llms/agent/memory)、[评测](/llms/agent/evaluation) |
| MCP | 区分新版逐请求元数据与旧版初始化流程；保留已运行教学依赖并说明迁移边界 | [概述](/llms/mcp/)、[核心概念](/llms/mcp/concepts)、[授权与高级功能](/llms/mcp/advanced) |
| Prompt | 结构化输出的拒答、未完成与空响应分支；上下文压缩边界和提示注入防护 | [高级提示](/llms/prompt/advanced)、[上下文工程](/llms/prompt/context)、[安全](/llms/prompt/security) |
| 训练 | GRPO 与可验证奖励的适用条件、PEFT 实现差异，以及前缀缓存与生成阶段的区别 | [RLHF](/llms/training/rlhf)、[LoRA](/llms/training/lora)、[部署](/llms/training/serving) |
| 多模态 | 页面与区域证据、视频时序信息、模型与 processor 配套，以及实际部署约束 | [架构](/llms/multimodal/architecture)、[多模态 RAG](/llms/multimodal/rag-agent)、[部署](/llms/multimodal/deployment) |

## 阅读时先确认三件事

1. **问题是否匹配**：全局归纳、精确条款检索、视频时序理解对应不同输入和评测；不要只因为技术更新就替换现有方案。
2. **版本是否匹配**：协议规范、SDK、模型权重和推理框架各有版本。一个旧版可运行示例不能证明新协议互通。
3. **证据是否足够**：论文或官方教程证明了特定设置下的结果；本地项目仍需自己的样本、预算和失败分析。GPU、外部模型 API 和生产 OAuth 未实跑的部分会明确标注。

## 可追溯的一手入口

- RAG：[Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)、[Microsoft GraphRAG 查询说明](https://microsoft.github.io/graphrag/query/overview/)、[Ragas Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)。
- Agent：[工具调用文档](https://developers.openai.com/api/docs/guides/function-calling)、[长任务执行环境实践](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)、[Agent 评测说明](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)。
- MCP：[2026-07-28 版本兼容规范](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)、[对应授权规范](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)。
- Prompt：[结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs)、[Compaction](https://developers.openai.com/api/docs/guides/compaction)、[提示注入防护设计](https://openai.com/index/designing-agents-to-resist-prompt-injection/)。
- 训练：[TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)、[PEFT 0.21.0 LoRA](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)。
- 多模态：[Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)、[vLLM 多模态输入](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)。

对应文章在具体结论旁保留引用；本页用于找入口，不代替正文里的假设、实验设置与版本说明。需要实际跑一遍检索、工具和协议边界时，进入 [可运行知识助手](/practice/knowledge-assistant)。
