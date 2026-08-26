---
title: 模板
description: Prompt、RAG、工具调用、评估和项目复盘的可复用模板。
pageType: article
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - reference
  - templates
level: intermediate
prerequisites: []
reviewed: '2026-08-26'
techVersion: 2026-08（模板）
---

# 模板

模板的目标是减少空白页恐惧。使用时请按你的业务删改，不要把模板当成万能提示词。

## 1. 结构化任务 Prompt

```text
你要完成的任务：{一句话目标}

输入：
{用户内容或引用材料}

约束：
- 只基于输入和已给资料回答。
- 如果信息不足，说明缺什么。
- 输出必须符合下面的结构。

输出结构：
1. 结论：
2. 依据：
3. 风险/不确定性：
4. 下一步：
```

参考：[Prompt 基础](/llms/prompt/basics)、[上下文工程](/llms/prompt/context)。

## 2. RAG 回答模板

```text
问题：{query}

请根据检索片段回答。要求：
- 每个关键结论后标注来源编号。
- 如果片段没有支持，不要猜测。
- 最后列出“可能需要补充的资料”。

检索片段：
[1] {title, url, excerpt}
[2] {title, url, excerpt}
```

参考：[RAG 生产实践](/llms/rag/production)、[评估指标](/reference/metrics)。

## 3. 工具 schema 设计模板

```yaml
name: search_knowledge_base
description: Search approved internal documents by query and filters.
inputs:
  query:
    type: string
    description: User question rewritten as a search query.
  filters:
    type: object
    description: Optional source, date, owner or permission filters.
outputs:
  results:
    type: array
    description: Ranked passages with title, url, excerpt and score.
errors:
  - permission_denied
  - timeout
  - no_results
safety:
  requires_user_confirmation: false
  data_scope: approved_documents_only
```

参考：[Agent 工具调用](/llms/agent/tool-calling)、[MCP 核心概念](/llms/mcp/concepts)。

## 4. 评估样本模板

```yaml
id: rag_001
question: "用户会怎么问？"
expected_evidence:
  - source_id: "doc_123"
    quote: "支持答案的原文片段"
expected_answer_points:
  - "必须覆盖的关键点"
failure_tags:
  - retrieval_miss
  - hallucination
review_notes: "人工复核说明"
```

参考：[RAG 评估](/llms/rag/evaluation)、[训练评估](/llms/training/eval)。

## 5. 微调数据卡模板

```markdown
# 数据卡：{dataset_name}

- 目标行为：
- 数据来源：
- 样本规模：
- 清洗规则：
- 去重方法：
- 敏感信息处理：
- 训练/验证/测试拆分：
- 已知偏差：
- 不适用场景：
- 版本与负责人：
```

参考：[训练数据](/llms/training/data)、[LoRA](/llms/training/lora)。

## 6. 项目复盘模板

```markdown
## 背景
谁遇到了什么问题？为什么需要 LLM？

## 架构
数据、检索、模型、工具、评估分别怎么设计？

## 指标
上线前后关注哪些指标？基线是多少？

## 失败案例
列出至少 10 个失败样本，并归类原因。

## 取舍
为什么选择 Prompt/RAG/Agent/微调，而不是其他方案？

## 下一步
最值得做的一个实验是什么？如何判断成功？
```

参考：[实践项目](/practice/)、[Checklist](/reference/checklists)。
