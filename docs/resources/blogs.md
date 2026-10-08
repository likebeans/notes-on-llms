---
title: 博客
description: 官方博客、文档和工程指南索引。
pageType: landing
module: site
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - resources
  - blogs
---

# 博客

这里的“博客”泛指官方公告、开发者文档和工程指南。使用这些内容时，要特别注意发布日期、模型版本和 API 是否仍然有效。

如果你想阅读作者自己的技术博客原文，截至 2026-08-27 收录的 30 篇已同步到 [CSDN 全文镜像](/resources/csdn)，并按 Agent、Prompt、RAG、MCP、Training、Multimodal 与工程实践重新归类。

<SourceList :items="[
  { title: 'Function calling and other API updates', href: 'https://openai.com/index/function-calling-and-other-api-updates/', note: '2023 · OpenAI 官方博客，关联工具调用、结构化参数和 Agent 基础能力。' },
  { title: 'Introducing Structured Outputs in the API', href: 'https://openai.com/index/introducing-structured-outputs-in-the-api/', note: '2024 · OpenAI 官方博客，关联 schema 约束输出和可靠信息抽取。' },
  { title: 'New tools for building agents', href: 'https://openai.com/index/new-tools-for-building-agents/', note: '2025 · OpenAI 官方博客，关联 Responses API、内置工具和 Agents SDK。' },
  { title: 'Introducing AgentKit', href: 'https://openai.com/index/introducing-agentkit/', note: '2025 · OpenAI 官方博客，关联生产级 Agent 构建、部署和优化。' },
  { title: 'Introducing the Model Context Protocol', href: 'https://www.anthropic.com/news/model-context-protocol', note: '2024 · Anthropic 官方博客，关联 MCP 的动机、组件和生态。' },
  { title: 'What is the Model Context Protocol?', href: 'https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro', note: '2026 · MCP 官方文档，关联 Host、Client、Server 与工具连接。' },
  { title: 'Gemini 1.5', href: 'https://blog.google/innovation-and-ai/products/google-gemini-next-generation-model-february-2024/', note: '2024 · Google 官方博客，关联多模态、长上下文和 MoE 架构方向。' },
  { title: 'Transformers Documentation', href: 'https://huggingface.co/docs/transformers/en/index', note: '持续更新 · Hugging Face 官方文档，关联模型加载、推理、微调和开源生态。' },
  { title: 'OpenAI Cookbook', href: 'https://developers.openai.com/cookbook', note: '持续更新 · OpenAI Developers 官方资源，关联代码样例、评估、Agent、RAG 与 API 实践。' }
]" />

## 使用博客/文档的检查点

1. 是否是官方来源或项目维护者来源？
2. 发布日期是否仍适用于你使用的模型/API？
3. 示例代码是否有弃用接口？
4. 文中建议是否需要补充评估或安全边界？
5. 是否能和论文或官方仓库互相印证？

## 分清公告、教程和经验文章

公告适合确认某项能力何时发布；接口参数和兼容性以版本化开发文档为准；经验文章用于提出可检验的工程假设。三者承担不同职责，不能用发布时的演示替代现在的稳定性评估。

| 章节 | 阅读官方材料时重点核对 |
| --- | --- |
| Prompt | 支持的 schema、拒答、截断和错误行为 |
| RAG | 检索与重排的数据口径、权限和更新路径 |
| Agent | 工具结果格式、并行调用、取消与恢复 |
| MCP | 协议版本、传输、能力协商与身份边界 |
| 训练/多模态 | 模型卡、处理器、聊天模板与硬件条件 |

推荐把“资料里这样说”转成“在我的版本和样本下这样验证”，并把验证记录放回对应章节的实验笔记。
