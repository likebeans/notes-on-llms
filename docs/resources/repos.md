---
title: 开源项目
description: LLM 应用、Agent、MCP、训练和评估相关的官方或主流开源项目索引。
pageType: landing
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - resources
  - repos
---

# 开源项目

看仓库时，不要只看 star。更重要的是：是否还在维护、示例是否能跑、issue 里暴露了哪些工程边界。

| 项目 | 年份 | URL | 适合学习什么 |
| --- | --- | --- | --- |
| OpenAI Cookbook | 持续更新 | [GitHub](https://github.com/openai/openai-cookbook) | OpenAI API、RAG、评估、Agent 和结构化输出样例。 |
| OpenAI Evals | 2023 | [GitHub](https://github.com/openai/evals) | 如何组织模型评估样本、指标和回归。 |
| Model Context Protocol | 2024 | [GitHub 组织](https://github.com/modelcontextprotocol) | MCP 协议、SDK 与官方生态入口。 |
| Model Context Protocol servers | 2024 | [GitHub](https://github.com/modelcontextprotocol/servers) | MCP server 参考实现与社区资源入口。 |
| Hugging Face Transformers | 2018 | [GitHub](https://github.com/huggingface/transformers) | 模型加载、推理、微调和多模态模型生态。 |
| LangChain | 2022 | [GitHub](https://github.com/langchain-ai/langchain) | LLM 应用组件、链式编排和 Agent 工程模式。 |
| LangGraph | 2024 | [GitHub](https://github.com/langchain-ai/langgraph) | 用图建模 Agent 状态、控制流和多步任务。 |
| FAISS | 2017 | [GitHub](https://github.com/facebookresearch/faiss) | 向量索引、近似最近邻检索和性能权衡。 |
| LLaVA | 2023 | [GitHub](https://github.com/haotian-liu/LLaVA) | 视觉指令微调和多模态模型实践。 |

## 读仓库的顺序

1. README：确认项目定位、维护状态和最小示例。
2. examples/cookbook：找能直接跑的小样例。
3. docs：确认 API 当前版本。
4. tests：学习作者如何定义正确性。
5. issues：收集常见坑，尤其是生产部署、版本冲突和性能问题。
