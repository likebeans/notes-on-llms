---
title: Agent 面试题
description: Agent 相关面试题精编
pageType: article
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - interviews
level: intermediate
prerequisites:
  - /llms/agent/
reviewed: '2026-08-26'
techVersion: 2026-08（Agent 设计与评估）
---

# Agent 面试题

这页用于快速检查你是否能把 Agent 从“会调用工具的聊天机器人”讲成一个可评估、可恢复、可上线的系统。回答时建议先给定义，再讲设计取舍，最后补一个失败案例。

## 核心问题

1. Agent 和普通 LLM 应用的边界是什么？什么时候不应该使用 Agent？
2. Tool calling、planning、reflection、memory 分别解决什么问题？它们之间有什么依赖关系？
3. 如果 Agent 调用工具失败，你会设计哪些重试、降级和人工介入策略？
4. 多 Agent 协作适合哪些场景？它相比单 Agent 增加了哪些观测和调试成本？
5. 如何防止 Agent 被 prompt injection 诱导调用越权工具？

## 追问角度

- 让候选人画出一次任务执行 trace：用户目标、计划、工具选择、工具参数、结果校验、最终输出。
- 追问“模型为什么知道该停下来”：是否有完成条件、预算、最大步数、置信度或人工确认。
- 追问“评估怎么做”：是否覆盖任务成功率、工具错误率、恢复率、人工接管率和安全拒绝率。

## 示例回答骨架

一个生产 Agent 至少要有四层边界：任务边界定义它能做什么；工具边界定义它能碰哪些系统；执行边界定义预算、超时和重试；评估边界定义怎样判断一次执行成功。Agent 的价值不是“自动循环”，而是在不确定任务中持续观察、选择行动并修正路径。风险也来自这里：每多一步行动，就多一次错误传播和越权机会。

相关复习：[Agent 技术全景](/llms/agent/) 与 [工具调用](/llms/agent/tool-calling)。
