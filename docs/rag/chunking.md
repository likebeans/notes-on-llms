---
title: 文档切分
description: RAG 中的文档切分策略与最佳实践
---

# 文档切分

## 🎯 本篇目标

> 掌握文档切分的核心策略，理解不同切分方式的适用场景。

## 📊 核心概念

### 为什么需要切分？

- LLM 上下文窗口有限
- 检索需要细粒度匹配
- 大文档包含多个主题

### 切分策略对比

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 固定大小 | 按字符/Token 数量切分 | 简单、可控 | 可能切断语义 |
| 递归切分 | 按分隔符层级切分 | 保留结构 | 配置复杂 |
| 语义切分 | 按语义相似度切分 | 语义完整 | 计算开销大 |
| 文档结构 | 按标题/段落切分 | 逻辑清晰 | 依赖文档格式 |

## 💻 最小实践

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "]
)

chunks = splitter.split_text(document)
```

## ⚠️ 常见坑与排雷

- **Chunk 太大**：检索不精准，噪音多
- **Chunk 太小**：丢失上下文，答案不完整
- **Overlap 过小**：边界信息丢失
- **忽略元数据**：无法追溯来源

## 📚 延伸阅读

- [Embedding](/rag/embedding)
- [检索策略](/rag/retrieval)
