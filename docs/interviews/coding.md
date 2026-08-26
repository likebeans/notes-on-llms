---
title: 代码题
description: LLM 相关代码面试题
pageType: article
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - interviews
level: intermediate
prerequisites:
  - /practice/
reviewed: '2026-08-26'
techVersion: 2026-08（LLM 工程代码题）
---

# 代码题

LLM 代码题通常不是算法竞赛，而是考你能否把不稳定模型输出包进可测试的工程边界。写代码前先说清输入、输出、错误语义和测试样例，会比直接调 API 更稳。

## 题型清单

1. 实现一个文档 chunker：保留标题层级、最大 token 限制和重叠窗口。
2. 实现一个 RAG trace logger：记录 query、候选文档、rerank 分数、上下文和回答。
3. 实现一个 tool schema validator：拒绝未知参数、缺失必填字段和危险默认值。
4. 实现一个 prompt injection filter：识别要求忽略系统指令、泄露密钥或越权调用工具的文本。
5. 实现一个评估 runner：读取样本、调用模型、运行 grader、输出失败样本报告。

## 答题要点

- 先写类型和边界：不要让模型输出直接进入数据库、shell 或外部 API。
- 保留原始输入和中间结果：LLM 系统 debug 依赖 trace，而不是只看最终回答。
- 测试失败路径：空输入、超长输入、格式错误、模型返回非 JSON、工具超时都要覆盖。
- 明确不可做的事：例如代码题里没有权限系统，就不要假装工具调用已经安全。

## 示例思路

如果题目要求实现 RAG trace logger，我会先定义 `TraceRecord`，包含 `query`、`retrievedDocuments`、`rerankedDocuments`、`contextChunks`、`answer`、`citations` 和 `labels`。写入时使用追加日志或事件表，不把大段文档重复存进主表；测试覆盖空召回、无答案、引用缺失和序列化失败。这个答案展示的是工程边界，而不是某个 SDK 的记忆。

相关复习：[实践项目](/practice/) 与 [模板](/reference/templates)。
