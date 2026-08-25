---
title: 基础提示技术
description: Zero-shot、Few-shot、思维链等核心技术
pageType: article
module: prompt
updated: '2025-12-27'
contentStatus: needs-review
tags:
  - prompt
level: beginner
prerequisites: []
reviewed: '2026-08-25'
techVersion: 待复核（2026-08）
---

# 基础提示技术

> 掌握与LLM对话的基础技能——提示词工程是"艺术与科学的结合"

## 📖 核心原理

### 模型响应机制

模型并非通过独立"问答"模块回答问题，其输出是基于庞大训练数据，对**"跟随输入提示词后最可能出现的词语序列"**的预测。

::: tip 提示词的本质
提示词是一种自然语言文本，用于描述AI模型应执行的任务。核心目标是通过提供清晰的"路线图"，充分释放模型内在能力，在用户意图与模型生成过程间架起桥梁。
:::

### 提示词技术分类

| 类别 | 技术 | 复杂度 |
|------|------|--------|
| **基础范式** | Zero-shot、Few-shot | ⭐ |
| **推理增强** | CoT、Self-Consistency | ⭐⭐ |
| **结构控制** | 角色设定、格式约束 | ⭐⭐ |
| **高级推理** | ToT、GoT、ReAct | ⭐⭐⭐ |

---

## 🎯 Zero-shot Prompting

### 概念

直接向模型提问，不提供任何示例。是大多数任务的**首选起点**，完全依赖模型预训练阶段的知识和指令遵循能力。

```
用户：将以下句子翻译成英文：今天天气很好。
模型：Today the weather is very nice.
```

### 适用场景

- 简单、定义明确的任务
- 模型能力足够（大模型效果更好）
- 快速原型验证
- **推理模型（o1系列）首选**——原生推理能力无需示例

### 清晰性与特异性

| 类型 | 示例 | 效果 |
|------|------|------|
| **模糊提示** | "谈谈人工智能" | ❌ 输出无效、发散 |
| **明确提示** | "解释人工智能在医疗保健领域的影响" | ✅ 聚焦、有价值 |

**最佳实践**：
- 核心指令置于**开头**
- 用分隔符（`###`或`"""`）区分指令与上下文信息
- 避免指令与内容混淆

### 优化技巧

```python
# ❌ 模糊指令
prompt_bad = "写点东西"

# ✅ 明确指令
prompt_good = """
请写一段关于人工智能的介绍，要求：
- 长度：100-150字
- 风格：科普向
- 重点：AI的应用场景
"""
```

---

## 🎯 Few-shot Prompting

### 概念

提供几个示例，让模型学习模式后完成任务。当Zero-shot效果不佳时，通过**"在上下文中学习"**引导模型理解任务要求（非永久性训练）。

::: warning 推理模型注意
对于o1/o3等推理模型，通常**不需要**Few-shot示例。如果确实需要，必须确保示例与指令**高度一致**，否则可能导致糟糕结果。
:::

```
用户：
情感分类任务：

输入：这家餐厅的菜太好吃了！
输出：正面

输入：等了两小时才上菜，体验很差。
输出：负面

输入：这部电影让我看哭了，太感人了。
输出：
模型：正面
```

### 示例数量选择

| 示例数 | 效果 | Token成本 |
|--------|------|-----------|
| 1-2个 | 基础 | 低 |
| 3-5个 | 良好 | 中 |
| 5-10个 | 最佳 | 高 |

### 示例选择原则

1. **多样性**：覆盖不同情况
2. **代表性**：反映真实分布
3. **相关性**：与目标任务相似
4. **质量**：示例本身要正确

::: tip 有趣发现
研究表明，示例的**格式比标签正确性更重要**。即使标签错误，只要格式一致，模型仍能学习任务结构。
:::

```python
def select_examples(query, example_pool, k=5):
    """选择与查询最相关的示例"""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_emb = model.encode([query])
    example_embs = model.encode([e['input'] for e in example_pool])
    
    similarities = cosine_similarity(query_emb, example_embs)[0]
    top_indices = similarities.argsort()[-k:][::-1]
    
    return [example_pool[i] for i in top_indices]
```

---

## 🧠 思维链（Chain-of-Thought）

> 来源：[微调高级推理大模型（COT）综合指南](https://dd-ff.blog.csdn.net/article/details/153210150)

### 概念

让模型逐步展示推理过程，而不是直接给出答案。CoT是提示工程的**里程碑技术**，通过中间推理步骤解锁复杂推理能力。

### 为什么CoT有效？

| 理论 | 解释 |
|------|------|
| **可变计算量** | 每个中间Token都经过完整Transformer处理，"购买"更多计算时间 |
| **逻辑展开** | 将复杂推理"展开"在时间轴上逐步解决 |
| **语义锚定** | 自然语言作为思维载体，锚定问题的语义逻辑 |

::: danger 推理模型警告
对于o1/o3等原生推理模型，**避免使用CoT提示**！这些模型已在内部执行高质量推理，外部CoT会干扰其原生过程。
:::

### Zero-shot CoT

```
用户：小明有5个苹果，给了小红2个，又买了3个，请问现在有几个？
请一步一步思考。

模型：让我逐步分析：
1. 小明最初有5个苹果
2. 给了小红2个，剩余：5 - 2 = 3个
3. 又买了3个，现在有：3 + 3 = 6个
答案：6个苹果
```

### Few-shot CoT

```
用户：
问题：一个商店有24个苹果，卖掉了8个，又进货了12个，现在有多少？
思考：
- 初始：24个
- 卖掉后：24 - 8 = 16个
- 进货后：16 + 12 = 28个
答案：28个

问题：小明有5个苹果，给了小红2个，又买了3个，现在有几个？
思考：
```

### CoT最佳实践

| 技巧 | 说明 |
|------|------|
| **明确触发词** | "让我们一步一步思考" |
| **结构化步骤** | 使用编号或分隔符 |
| **最终答案标记** | 明确标注"答案：" |
| **错误检查** | 提示模型验证结果 |

---

## 🎯 Self-Consistency

### 概念

多次采样，取众数作为最终答案。

```python
import openai
from collections import Counter

def self_consistency(prompt: str, n_samples: int = 5) -> str:
    """自洽性推理"""
    answers = []
    
    for _ in range(n_samples):
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7  # 允许一定随机性
        )
        # 提取答案
        answer = extract_answer(response.choices[0].message.content)
        answers.append(answer)
    
    # 投票取众数
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

### 适用场景

- 数学推理
- 逻辑判断
- 事实性问答

---

## 📋 角色设定（Role Prompting）

### 概念

为模型设定特定角色，影响其回答风格和专业度。

```
你是一位资深的Python开发专家，拥有10年经验。
请以专业但易懂的方式回答以下问题：

用户问题：什么是装饰器？
```

### 常用角色模板

| 角色 | 适用场景 |
|------|----------|
| 专家 | 技术问答 |
| 教师 | 概念解释 |
| 编辑 | 文本优化 |
| 批评家 | 质量评估 |
| 助手 | 通用任务 |

---

## 📝 结构化输出

### 概念

明确指定输出格式，便于后续程序处理。

```python
STRUCTURED_PROMPT = """
请分析以下文本，返回JSON格式：

文本：{text}

返回格式：
{{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0-1.0,
  "keywords": ["关键词1", "关键词2"],
  "summary": "一句话总结"
}}
"""

# OpenAI JSON模式
response = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}  # 强制JSON输出
)
```

### 格式选项

| 格式 | 适用场景 | 优势 |
|------|----------|------|
| **JSON** | API集成、数据处理 | 结构化、易解析 |
| **Markdown** | 文档生成、报告 | 可读性好 |
| **列表** | 步骤说明、要点总结 | 清晰简洁 |
| **表格** | 对比分析、数据展示 | 信息密度高 |

---

## ⚡ 模型选择指南

### 系统1 vs 系统2

| 任务类型 | 推荐模型 | 提示策略 |
|----------|----------|----------|
| 简单问答、聊天 | GPT-4o | Zero-shot/Few-shot |
| 创意写作 | GPT-4o | 角色设定 + 格式约束 |
| 数学推理 | o1/o3 | **简洁直接，不用CoT** |
| 代码调试 | o1/o3 | 明确约束条件 |
| 复杂规划 | o1/o3 | 清晰目标描述 |

### 提示策略选择流程

```
任务 → 是否简单？
  │
  ├─ 是 → Zero-shot（首选）
  │
  └─ 否 → 模型类型？
           │
           ├─ 传统模型 → Few-shot + CoT
           │
           └─ 推理模型 → 简洁直接 + 明确约束
```

---

## 🔗 相关阅读

- [高级提示技术](/llms/prompt/advanced) - ReAct、ToT等
- [上下文工程](/llms/prompt/context) - 动态上下文管理
- [提示词概述](/llms/prompt/) - 技术全景

> **相关文章**：
> - [从指令到智能：提示词与上下文工程](https://dd-ff.blog.csdn.net/article/details/152799914)
> - [掌握AI推理：从"提示工程"到"推理架构"](https://dd-ff.blog.csdn.net/article/details/154479954)
> - [OpenAI Prompt Engineering与Prompt Caching实战](https://dd-ff.blog.csdn.net/article/details/154450002)

> **外部资源**：
> - [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
> - [Prompt Engineering Guide](https://www.promptingguide.ai/)
