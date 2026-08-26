---
title: 学习路径
description: 按应用开发、算法工程和面试准备三种目标组织的大模型学习路径。
pageType: path
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - guide
  - path
---

# 学习路径

这份手册不要求你线性读完。更好的方式是先选目标，再围绕一个能交付的小项目向外扩展：缺什么补什么，做完再复盘。

<LearningObjectives :items="[
  '根据应用开发、算法工程或面试准备选择合适路线。',
  '把每条路线拆成模块、项目和阶段性产出。',
  '理解本站内容状态、前置知识和阅读顺序的使用方式。'
]" />

## 路径一：应用开发者（3-6 个月）

适合目标：你想用现有模型构建 RAG、Agent、知识库、自动化工具或企业内部 AI 应用。

推荐顺序：

1. [Prompt 工程](/llms/prompt/)：学习如何描述任务、控制上下文和约束输出。
2. [RAG](/llms/rag/)：让答案能够引用外部知识，而不是只依赖模型记忆。
3. [Agent](/llms/agent/)：把模型接入工具、工作流和多步任务。
4. [MCP](/llms/mcp/)：用协议化方式连接外部系统和上下文。
5. 生产实践：补齐评估、监控、成本、权限和发布回滚。

阶段性产出：

| 阶段 | 应完成的作品 | 对应页面 |
| --- | --- | --- |
| 入门 | 一个结构化 Prompt/API demo | [Prompt 基础](/llms/prompt/basics) |
| 进阶 | 一个带引用的知识库问答 | [RAG 生产实践](/llms/rag/production) |
| 高级 | 一个可观察、可中断的工具 Agent | [Agent 工具调用](/llms/agent/tool-calling) |
| 集成 | 一个 MCP server/client 小样例 | [MCP 快速入门](/llms/mcp/quickstart) |

## 路径二：算法工程师（6-12 个月）

适合目标：你想理解模型训练、微调、对齐、评估和多模态模型的底层机制。

推荐顺序：

1. [前置知识](/guide/prerequisites)：补齐 Python、机器学习、NLP 与 Transformer。
2. [数据工程](/llms/training/data)：先学如何判断数据是否值得训练。
3. [SFT](/llms/training/sft) / [LoRA](/llms/training/lora)：从小模型和参数高效微调开始。
4. [DPO](/llms/training/dpo) / [RLHF](/llms/training/rlhf)：理解偏好对齐的目标和风险。
5. [评估](/llms/training/eval)：建立离线评测、红队和回归样本。
6. [多模态](/llms/multimodal/)：扩展到视觉编码、连接器和图文任务。

阶段性产出：

| 阶段 | 应完成的作品 | 对应页面 |
| --- | --- | --- |
| 数据 | 一份数据卡、清洗脚本和抽样报告 | [训练数据](/llms/training/data) |
| 微调 | 一次 LoRA/SFT 对照实验 | [LoRA](/llms/training/lora) |
| 对齐 | 一组偏好数据与 DPO 结果分析 | [DPO](/llms/training/dpo) |
| 部署 | 一份推理服务压测和评估报告 | [部署推理](/llms/training/serving) |

## 路径三：面试冲刺（2-6 周）

适合目标：你要准备 LLM 应用、算法、Agent、RAG 或平台方向面试，需要快速建立回答框架。

推荐顺序：

1. [术语速查](/reference/glossary)：先把基础词汇说准。
2. [RAG 面试题](/interviews/rag-questions)：准备检索、重排、引用和评估。
3. [Agent 面试题](/interviews/agent-questions)：准备工具调用、规划、记忆和安全。
4. [训练微调面试题](/interviews/training-questions)：准备 SFT、LoRA、DPO、RLHF、评估。
5. [系统设计](/interviews/system-design)：练知识库问答、客服 Agent、评估平台。
6. [代码题](/interviews/coding)：练文本切分、Top-k、缓存、重试和小评估脚本。

冲刺方法：

- 每个概念准备“定义、为什么需要、常见失败、如何评估”四句话。
- 每个项目准备“背景、架构、指标、失败案例、下一步”五段叙述。
- 遇到不会的问题，先拆成数据、检索、模型、工具、评估五层定位。

## 如何使用手册

### 状态标签

| 标签 | 含义 | 阅读建议 |
| --- | --- | --- |
| `verified` | 已按当前资料复核，可优先引用 | 适合作为主线材料 |
| `needs-review` | 已整理但仍需要更新或二次校验 | 阅读时留意技术版本 |
| `opinion` | 含作者判断或经验总结 | 适合启发，不要当成标准答案 |
| `historical` | 保留历史背景 | 用于理解演进，不代表最新实践 |

### 阅读顺序

1. 先看本页确定目标。
2. 再看 [前置知识](/guide/prerequisites) 补短板。
3. 用 [学习路线图](/guide/roadmap) 拆时间计划。
4. 每学完一个模块，到 [实践项目](/practice/) 选一个小项目验证。
5. 需要复习时，用 [参考手册](/reference/) 做速查。
