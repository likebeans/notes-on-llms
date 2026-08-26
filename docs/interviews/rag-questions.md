---
title: RAG 面试题
description: RAG 相关面试题精编
pageType: article
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - interviews
level: intermediate
prerequisites:
  - /llms/rag/
reviewed: '2026-08-26'
techVersion: 2026-08（RAG 设计与评估）
---

# RAG 面试题

RAG 面试通常考的不是“向量数据库用过没有”，而是能否把检索、证据、生成和评估拆开定位。回答时尽量少背工具名，多讲你如何定位失败来源。

## 核心问题

1. 为什么 RAG 不能简单等同于 embedding + top-k + prompt 拼接？
2. BM25、向量检索、混合检索和 rerank 各自适合什么问题？
3. chunk 太大或太小分别会带来什么失败？你会如何用实验选择 chunk 策略？
4. 如果答案幻觉了，怎样判断问题来自召回、重排、上下文组装还是模型生成？
5. 权限、增量更新、删除文档和索引版本如何影响生产 RAG？

## 追问角度

- 要求候选人给出一条完整 trace：query rewrite、retrieval、rerank、context、answer、citation、label。
- 追问无答案问题：系统是否能在证据不足时拒答，而不是给流畅但无根据的回答。
- 追问评估指标：top-k evidence hit、context relevance、faithfulness、citation correctness 和端到端任务成功率是否分开。

## 示例回答骨架

我会把 RAG 当成证据供应链：数据处理决定证据是否可用，检索决定证据是否进入候选集，重排决定正确证据是否排在前面，上下文组装决定模型实际看见什么，生成约束决定答案是否忠于证据。出现错误时，先看标准证据是否进入 top-k；如果没有，调索引和召回；如果进入但没被使用，调 rerank、上下文压缩和 prompt；如果证据足够仍回答错，再看模型与拒答策略。

相关复习：[RAG 技术全景](/llms/rag/) 与 [RAG 评估](/llms/rag/evaluation)。
