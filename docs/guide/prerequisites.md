---
title: 前置知识
description: 学习大模型应用开发前建议补齐的工程、机器学习和阅读基础。
pageType: article
module: site
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - guide
  - prerequisites
level: beginner
prerequisites: []
reviewed: '2026-08-26'
techVersion: 2026-08（基础概念）
---

# 前置知识

这份清单不是“入场考试”。它更像一张地图：你可以先用 API 做出东西，再按遇到的问题补课。真正影响学习速度的，通常不是会不会背 Transformer 公式，而是能不能把问题拆开、验证结果、读懂文档。

## 最小必备

| 能力 | 需要到什么程度 | 会在哪些模块用到 |
| --- | --- | --- |
| Python | 会写脚本、调用 API、处理 JSON/CSV、管理依赖与虚拟环境 | [Prompt](/llms/prompt/)、[RAG](/llms/rag/)、[Agent](/llms/agent/) |
| HTTP/API | 理解请求、响应、鉴权、状态码、超时与重试 | [Agent 工具调用](/llms/agent/tool-calling)、[MCP](/llms/mcp/) |
| Git | 能提交、分支、回滚小改动，读懂 diff | 所有实践项目 |
| 基础数据处理 | 能清洗文本、去重、切分、抽样检查 | [RAG 文档切分](/llms/rag/chunking)、[训练数据](/llms/training/data) |
| 实验记录 | 能记录输入、版本、指标和结论 | [评估](/llms/rag/evaluation)、[部署与评测](/llms/multimodal/deployment) |

## 按方向补的知识

### 应用开发方向

优先补工程能力。你需要知道模型输入输出如何被系统包起来，以及失败时如何定位问题。

1. API 调用：鉴权、限流、重试、结构化输出。
2. 文档处理：Markdown/PDF/网页抽取、chunk 策略、元数据。
3. 数据库：至少理解关系数据库、KV、向量检索的适用边界。
4. 可观测性：日志、trace、错误样本、延迟和成本统计。

推荐顺序：[Prompt 基础](/llms/prompt/basics) → [RAG 概述](/llms/rag/) → [Agent 工具调用](/llms/agent/tool-calling) → [MCP 快速入门](/llms/mcp/quickstart)。

### 算法工程方向

如果你想进入微调、对齐或多模态，需要补更多机器学习基础。

1. 线性代数与概率：向量、矩阵、分布、采样。
2. 深度学习：反向传播、优化器、过拟合、验证集。
3. NLP 基础：tokenization、embedding、语言模型目标。
4. Transformer：attention、位置编码、KV cache、长上下文。
5. 训练工程：数据质量、显存估算、分布式训练、推理 serving。

推荐顺序：[训练数据](/llms/training/data) → [SFT](/llms/training/sft) → [LoRA](/llms/training/lora) → [DPO](/llms/training/dpo) → [评估](/llms/training/eval)。

## 模块前置关系

| 模块 | 建议先掌握 | 不掌握会卡在哪里 |
| --- | --- | --- |
| [Prompt](/llms/prompt/) | 基本 API 调用、任务描述能力 | 难以稳定复现实验结果 |
| [RAG](/llms/rag/) | Embedding、文本切分、检索直觉 | 只会“接向量库”，不会调召回与可信度 |
| [Agent](/llms/agent/) | Prompt、工具调用、错误处理 | 容易做出能演示但不可靠的自动化 |
| [MCP](/llms/mcp/) | Agent 工具调用、协议/客户端概念 | 分不清 Host、Client、Server 的职责 |
| [训练与微调](/llms/training/) | ML/DL、数据集、评估 | 难以判断该微调还是该改数据/提示/RAG |
| [多模态](/llms/multimodal/) | Transformer、视觉编码、RAG/Agent | 难以理解图文对齐和多模态检索 |

## 快速补课资源

| 主题 | 建议资源 | 读法 |
| --- | --- | --- |
| Python | [Python 官方教程](https://docs.python.org/zh-cn/3/tutorial/) | 重点看数据结构、模块、异常、虚拟环境 |
| Transformer | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 与 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 先看图解，再回论文确认 Q/K/V 与 attention |
| API 设计 | [OpenAPI Specification](https://spec.openapis.org/oas/latest.html) | 理解 schema、参数、响应和错误建模 |
| 向量检索 | [FAISS 文档](https://faiss.ai/) | 关注 index 类型、召回/速度权衡 |
| 评估方法 | [OpenAI Evals](https://github.com/openai/evals) | 学习如何把主观体验变成可重复样本 |

## 常见误区

- “必须先懂全部深度学习才能开始”：应用开发可以先从 API 和评估做起。
- “RAG 等于向量数据库”：真正的难点通常在数据清洗、chunk、召回、重排和引用。
- “Agent 越自主越高级”：生产系统里，边界、审批和可观测性往往比自主性更重要。
- “微调能解决所有问题”：如果知识经常变、答案要引用来源，优先考虑 RAG 或工具调用。
- “指标越多越好”：先选能反映业务失败的少数指标，再扩展评估集。

## 下一步

- 想按角色学习：从 [学习路径](/guide/) 选择应用开发者、算法工程师或面试冲刺。
- 想直接看时间表：进入 [学习路线图](/guide/roadmap)。
- 想先做项目：进入 [实践项目](/practice/)。

## 用五个小任务定位短板

| 自测任务 | 通过时应展示什么 | 没通过先补什么 |
| --- | --- | --- |
| 读取一份 JSON 并拒绝缺少必填字段的输入 | 正常/异常两条输出和明确错误 | Python 类型、异常与数据校验 |
| 请求一个测试 API，并模拟超时 | 有超时上限和错误分类 | HTTP、幂等与重试边界 |
| 用三份文档回答一个有出处的问题 | 原文片段、页码和答案一一对应 | 文档解析和证据意识 |
| 比较两个提示版本 | 同一批输入的逐条对照，而非两张截图 | 控制变量和评估记录 |
| 解释一次矩阵乘法的输入输出形状 | 维度相容，并能算参数量 | 线性代数；进入微调前再深入 |

不必等所有任务都通过才开始。应用方向可以先完成前四项；训练方向应进一步理解梯度、训练/验证隔离和过拟合。每遇到一个不懂的公式，先核对符号、形状和量纲，再看它对应代码中的哪一步。
