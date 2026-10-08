---
title: 代码题
description: 用可运行 Python 示例练习 token 切分与 Top-k，并设计工具校验、trace 与评估错误边界。
pageType: article
module: site
updated: '2026-10-08'
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

代码题重点是输入输出契约和失败路径。先约定边界，再给实现；纯函数用固定样本验证，外部依赖用可控替身，最后才接真实模型或服务。

## 题型清单

| 题目 | 输入 / 输出 | 必测边界 |
| --- | --- | --- |
| Token 窗口切分 | token ID 序列 → 若干窗口 | 空输入、重叠非法、尾块 |
| 去重后的 Top-k | 文档 ID 和分数 → 有序候选 | 重复 ID、同分、空集合 |
| 工具参数校验 | 不可信 JSON → 类型明确的参数 | 未知字段、缺字段、越权资源 |
| RAG trace 记录 | 各阶段事件 → 可关联记录 | 无召回、超时、日志脱敏 |
| 评估 runner | 固定样本和评分器 → 逐条结果与统计 | 解析失败、评分器失败、无答案 |

提示注入检测可以作为附加题，但不能把识别几个关键词写成安全保证。更有价值的题目是：让不可信文本无法改变工具授权范围，并证明越权参数被服务端拒绝。

## 示例一：按 token 窗口切分

下面是完整的 Python 3 标准库示例。它假设分词已经完成，只负责窗口边界；实际文档还需要保留结构、来源和 tokenizer 版本。`size` 与 `overlap` 的单位都是 token，不能直接换成字符数。

```python
def token_windows(tokens: list[int], size: int, overlap: int) -> list[list[int]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

assert token_windows([], 4, 1) == []
assert token_windows(list(range(7)), 4, 1) == [[0, 1, 2, 3], [3, 4, 5, 6]]
assert token_windows(list(range(5)), 4, 1) == [[0, 1, 2, 3], [3, 4]]
try:
    token_windows([1], 4, 4)
except ValueError:
    pass
else:
    raise AssertionError("invalid overlap must fail")
```

时间和空间与输出 token 总量成正比；重叠越大，重复输出越多。这个基线不理解表格和标题，适合作为 [文档切分](/llms/rag/chunking) 实验的对照组。

## 示例二：去重后取 Top-k

约定：分数越大越好；同一 ID 保留最高分；同分按 ID 排序；非有限分数是输入错误。不能把不同检索器的原始分数直接塞进这个函数当作已经校准的分数。

```python
import math

def top_k_unique(rows: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    if k < 0:
        raise ValueError("k must be non-negative")
    best: dict[str, float] = {}
    for doc_id, score in rows:
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        best[doc_id] = max(best.get(doc_id, -math.inf), score)
    return sorted(best.items(), key=lambda row: (-row[1], row[0]))[:k]

assert top_k_unique([("b", 0.7), ("a", 0.7), ("b", 0.9)], 2) == [
    ("b", 0.9), ("a", 0.7)
]
assert top_k_unique([], 2) == []
assert top_k_unique([("a", 1.0)], 0) == []
```

设输入数为 `n`、去重后为 `u`，当前实现时间为 `O(n + u log u)`，额外空间为 `O(u)`。当 `k` 很小、候选很大时，可讨论堆选择；先保持同分和重复项语义一致。

## 示例三：为评估 runner 设计错误语义

不要求现场接模型，先设计一条记录：`sample_id`、配置版本、预测结果、业务得分、错误类型、耗时和 token。模型调用失败、JSON 解析失败与评分器失败必须区分；评分器没跑完不能默认给 0.5 分。

验收至少包括：一条正常样本、一条无答案样本、一条模型超时、一条格式错误，以及一条评分器报错。报告有效评分数量和失败数量，不能默默删掉失败请求再报告“平均准确率”。参考 [评估指标](/reference/metrics)。

## 答题要点

- 参数合法不等于有权限：资源归属由认证身份与服务端策略决定。
- 日志记录证据 ID、版本和必要摘要，原始提示与个人信息按策略脱敏或限制留存。
- 对写工具，超时可能意味着结果未知；重试需幂等或结果确认。
- 注明哪些是可运行实现，哪些是接口草图，哪些需要业务系统补齐。

## 示例思路

实现 RAG trace logger 时，用同一个 `trace_id` 串起检索、重排、生成和引用事件。证据正文放受权限控制的存储，日志记录 ID、版本和摘要。通过“候选有正确片段但最终上下文没有它”的用例，证明日志能够定位组装阶段的问题。

继续练习：[实践项目](/practice/) · [模板](/reference/templates)。
