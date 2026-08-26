---
title: 实践项目
description: 从入门、进阶到生产级的大模型项目练习清单。
pageType: landing
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - practice
  - projects
---

# 实践项目

学习 LLM 最容易掉进“看懂了但不会做”的坑。这里按难度列出项目，你可以把它们当成每个模块的验收题：做完一个，再回去补理论。

## 入门项目

::: tip 结构化摘要器
- **目标**：把长文本压缩成固定 JSON/Markdown 结构。
- **建议阅读**：[Prompt 基础](/llms/prompt/basics)、[上下文工程](/llms/prompt/context)。
- **验收标准**：同一输入多次运行时，结构稳定且字段可解析。
:::

::: tip 文档问答 Demo
- **目标**：上传 5-20 篇文档并回答问题。
- **建议阅读**：[RAG 全景](/llms/rag/)、[文档切分](/llms/rag/chunking)。
- **验收标准**：答案带引用，缺少证据时能明确说“不知道”。
:::

::: tip 简单工具调用
- **目标**：让模型调用计算器、搜索或内部 API。
- **建议阅读**：[工具调用](/llms/agent/tool-calling)。
- **验收标准**：工具参数可校验，失败时能重试或给出降级回答。
:::

## 进阶项目

::: info 可评估 RAG
- **目标**：建立 query 集、人工答案和自动指标。
- **建议阅读**：[RAG 评估](/llms/rag/evaluation)、[评估指标](/reference/metrics)。
- **验收标准**：每次改 chunk、召回或重排，都能看到指标变化。
:::

::: info 多步骤 Agent
- **目标**：让 Agent 拆任务、调用工具并生成执行记录。
- **建议阅读**：[规划](/llms/agent/planning)、[异常处理](/llms/agent/exception-handling)。
- **验收标准**：支持中途失败、重试、人工确认和执行回放。
:::

::: info MCP 小服务
- **目标**：把一个本地/业务能力封装成 MCP server。
- **建议阅读**：[MCP 快速入门](/llms/mcp/quickstart)、[核心概念](/llms/mcp/concepts)。
- **验收标准**：Host 能发现工具，并按权限边界调用。
:::

::: info LoRA 小实验
- **目标**：用小数据集微调一个轻量模型。
- **建议阅读**：[训练数据](/llms/training/data)、[LoRA](/llms/training/lora)。
- **验收标准**：有训练/验证拆分、基线对照和失败样本分析。
:::

## 生产级项目

::: warning 企业知识库 RAG
- **目标**：支持权限、增量索引、引用和反馈闭环。
- **建议阅读**：[RAG 生产实践](/llms/rag/production)、[Checklist](/reference/checklists)。
- **验收标准**：有召回评估、引用审计、成本/延迟监控和回滚策略。
:::

::: warning 客服/运营 Agent
- **目标**：多工具协作，关键动作需要确认。
- **建议阅读**：[Agent 安全](/llms/agent/safety)、[评估与监控](/llms/agent/evaluation-monitoring)。
- **验收标准**：有审批流、工具白名单、失败回放和人工接管路径。
:::

::: warning 模型评估平台
- **目标**：管理样本、指标、回归和模型版本。
- **建议阅读**：[训练评估](/llms/training/eval)、[评估指标](/reference/metrics)。
- **验收标准**：能比较版本，并阻止明显回归发布。
:::

::: warning 多模态知识助手
- **目标**：对图表、截图或视频片段做检索和问答。
- **建议阅读**：[多模态 RAG 与 Agent](/llms/multimodal/rag-agent)。
- **验收标准**：图文引用可追溯，错误样本可复盘。
:::

## 做项目时的记录模板

每个项目至少记录这些信息：

1. 目标：要解决谁的什么问题。
2. 数据：来源、规模、清洗规则、敏感信息处理。
3. 模型：版本、参数、上下文长度、调用成本。
4. 评估：样本集、指标、人工复核规则。
5. 失败案例：至少 10 个真实失败样本和原因分类。
6. 下一步：继续优化前，先说明最可能的瓶颈。

可复用模板见 [模板](/reference/templates)。
