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

| 项目 | 目标 | 建议阅读 | 验收标准 |
| --- | --- | --- | --- |
| 结构化摘要器 | 把长文本压缩成固定 JSON/Markdown 结构 | [Prompt 基础](/llms/prompt/basics)、[上下文工程](/llms/prompt/context) | 同一输入多次运行结构稳定 |
| 文档问答 Demo | 上传 5-20 篇文档并回答问题 | [RAG 全景](/llms/rag/)、[文档切分](/llms/rag/chunking) | 答案带引用，能指出“不知道” |
| 简单工具调用 | 让模型调用计算器、搜索或内部 API | [工具调用](/llms/agent/tool-calling) | 工具参数可验证，失败可重试 |

## 进阶项目

| 项目 | 目标 | 建议阅读 | 验收标准 |
| --- | --- | --- | --- |
| 可评估 RAG | 建立 query 集、人工答案和自动指标 | [RAG 评估](/llms/rag/evaluation)、[评估指标](/reference/metrics) | 每次改 chunk/召回能看到指标变化 |
| 多步骤 Agent | 让 Agent 拆任务、调用工具并生成执行记录 | [规划](/llms/agent/planning)、[异常处理](/llms/agent/exception-handling) | 支持中途失败、重试和人工确认 |
| MCP 小服务 | 把一个本地/业务能力封装成 MCP server | [MCP 快速入门](/llms/mcp/quickstart)、[核心概念](/llms/mcp/concepts) | Host 能发现工具并按权限调用 |
| LoRA 小实验 | 用小数据集微调一个轻量模型 | [训练数据](/llms/training/data)、[LoRA](/llms/training/lora) | 有训练/验证拆分和对照样本 |

## 生产级项目

| 项目 | 目标 | 建议阅读 | 验收标准 |
| --- | --- | --- | --- |
| 企业知识库 RAG | 支持权限、增量索引、引用、反馈闭环 | [RAG 生产实践](/llms/rag/production)、[Checklist](/reference/checklists) | 有召回评估、引用审计、成本/延迟监控 |
| 客服/运营 Agent | 多工具协作，关键动作需要确认 | [Agent 安全](/llms/agent/safety)、[评估与监控](/llms/agent/evaluation-monitoring) | 有审批流、工具白名单、失败回放 |
| 模型评估平台 | 管理样本、指标、回归和模型版本 | [训练评估](/llms/training/eval)、[评估指标](/reference/metrics) | 能比较版本并阻止回归发布 |
| 多模态知识助手 | 对图表、截图或视频片段做检索和问答 | [多模态 RAG 与 Agent](/llms/multimodal/rag-agent) | 图文引用可追溯，错误样本可复盘 |

## 做项目时的记录模板

每个项目至少记录这些信息：

1. 目标：要解决谁的什么问题。
2. 数据：来源、规模、清洗规则、敏感信息处理。
3. 模型：版本、参数、上下文长度、调用成本。
4. 评估：样本集、指标、人工复核规则。
5. 失败案例：至少 10 个真实失败样本和原因分类。
6. 下一步：继续优化前，先说明最可能的瓶颈。

可复用模板见 [模板](/reference/templates)。
