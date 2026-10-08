---
title: 基础提示技术
description: Zero-shot、Few-shot、思维链等核心技术
pageType: article
module: prompt
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - prompt
level: beginner
prerequisites: []
reviewed: '2026-08-26'
techVersion: 2026-08（基础范式，部分模型细节待复核）
---

# 基础提示技术

> 掌握与LLM对话的基础技能——提示词工程是"艺术与科学的结合"

## 2026 阅读提示

提示工程已经不只是“把一句话写漂亮”。在真实系统里，它更像一层轻量协议：定义任务、输入边界、输出格式、失败处理和评估样本。读这一页时，不要只记 Zero-shot、Few-shot、CoT 这些名字，而要问三个问题：

1. 这个提示是在补充模型不知道的信息，还是在约束模型本来就会的能力？
2. 输出会不会进入数据库、工具调用、检索链路或用户决策？如果会，就必须有结构校验和失败兜底。
3. 这个提示是否有固定样本集验证？如果没有，所谓“效果更好”很可能只是一次对话里的错觉。

一个稳妥的起点是：先写最小 Zero-shot baseline，再加入格式约束和少量高质量示例；当任务需要多步推理、工具调用或证据引用时，再引入分步骤提示、ReAct 或 RAG 模板。不要一开始就堆复杂技巧。

## 📖 核心原理

### 模型响应机制

自回归语言模型根据当前上下文估计下一个 token 的分布，再按解码策略生成序列；这不等于始终选取全局“最可能的词语序列”。指令微调、推理训练、工具结果和解码设置共同影响回答。写 prompt 的作用是明确任务与条件，而非向模型写入永久知识。

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
- **推理型模型首选**——先给清晰目标和约束，再根据失败样本决定是否补示例

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
对于具备更强推理能力的模型，通常先尝试清晰的 Zero-shot 指令和输出约束。如果确实需要 Few-shot，示例必须与指令高度一致，并覆盖关键边界条件，否则可能把模型带向错误模式。
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

没有“5–10 个最佳”的通用结论。以零样例为基线，逐步增加能够修复已知错误的样例，比较准确率、格式错误率、拒答行为和 token 成本。重复展示同一种简单正例，可能不如一个边界反例有效。

把开发样例与测试集分开；相同文档的改写、近重复输入也可能泄漏答案。分类任务还要检查类别平衡、顺序敏感性，以及检索示例是否改变测试分布。

### 示例选择原则

1. **多样性**：覆盖不同情况
2. **代表性**：反映真实分布
3. **相关性**：与目标任务相似
4. **质量**：示例本身要正确

::: tip 示例质量
格式、标签空间和输入分布都可能影响上下文学习，但这不意味着错误标签可安全使用。工程上仍应核验标签，并用打乱顺序、移除样例和类别平衡对照查找偏差。
:::

```python
def select_examples(query, example_pool, k=5):
    """教学基线；生产中缓存模型和示例向量。"""
    if k <= 0 or not example_pool:
        return []
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    
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

CoT 原始方法通过中间步骤示例引导模型处理多步任务。下面的算术过程是可验证的解题说明，不应理解为模型隐藏推理的完整记录；面向用户可要求关键公式、简短理由和验证结果。

### 为什么CoT有效？

| 理论 | 解释 |
|------|------|
| **可变计算量** | 每个中间Token都经过完整Transformer处理，"购买"更多计算时间 |
| **逻辑展开** | 将复杂推理"展开"在时间轴上逐步解决 |
| **语义锚定** | 自然语言作为思维载体，锚定问题的语义逻辑 |

::: tip 推理模型的适用边界
对于在内部执行推理的模型，优先写目标、约束与验收要求，不必强迫逐步吐出思维链。[OpenAI 推理最佳实践](https://developers.openai.com/api/docs/guides/reasoning-best-practices) 将此类提示称为不必要；不能由此推断所有分步任务说明都会有害。
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
from collections import Counter

def vote_answers(normalized_answers):
    # 模型调用和答案规范化由上层完成；只对最终短答案投票。
    if not normalized_answers:
        return {"status": "no_valid_answer", "answer": None}
    counts = Counter(normalized_answers).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return {"status": "tie", "answer": None}
    return {"status": "selected", "answer": counts[0][0]}
```

先从每次响应中提取结构化最终答案并规范化单位/数值，解析失败单独记录。投票一致不等于正确：同一模型可重复同一偏差。适合有可比较短答案的任务，开放写作或缺少资料的事实问题不能只靠多数投票；采样开销也需计入每次成功任务成本。

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

“系统1/系统2”是帮助理解延迟与推理预算的类比，不是严格模型分类。下面的历史模型只作示例，实际选择以当前任务评测和可用型号为准。

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

## 从提示试验到回归用例

把一个失败输入变成可判定测试：输入、必须满足的要点、禁止添加的事实、允许的拒答和机器可验证格式。固定模型与生成参数，比较 baseline、少量示例和结构约束，每次只改变一个主要因素；概率性任务需要重复运行并报告波动。

| 失败表现 | 优先修改 | 验收 |
| --- | --- | --- |
| 标签漂移 | 明确类别定义与边界示例 | 按类别统计混淆矩阵 |
| 合法 JSON 但字段不对 | Schema 与业务验证 | 格式/字段/业务错误分别计数 |
| 材料不足仍补全 | 明确空值与拒答规则，加入无答案例 | 有答案误拒绝率与无答案误接受率 |
| 角色设定后编造权威 | 把角色改成受众、任务范围与证据要求 | 所有事实可回溯到输入或来源 |

角色标签不能赋予模型真实资格、工具权限或新知识；结构化输出的完整处理见[高级提示技术](/llms/prompt/advanced)。

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
