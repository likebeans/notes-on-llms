---
title: 高级提示技术
description: ReAct、思维树、自我反思等高级技术
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

# 高级提示技术

> 复杂任务的提示词策略——从手动推理引导到原生推理调用

## ⚠️ 重要提示：推理模型的范式转变

在学习高级提示技术前，需要理解**AI模型的"大分流"**：

| 模型类型 | 代表 | CoT等技巧 | 最佳策略 |
|----------|------|-----------|----------|
| **系统1模型** | GPT-4o | ✅ 有效 | 使用CoT、ReAct等技巧引导推理 |
| **系统2模型** | o1/o3 | ❌ 反而有害 | 简洁直接的指令，信任原生推理 |

::: warning 关键区别
对于GPT-4o等传统模型，CoT等技术是**模拟**"系统2"思考。  
对于o1等推理模型，推理是**原生**能力，外部CoT提示会**干扰**其内部更优的推理过程。
:::

---

## 🔄 ReAct（推理+行动）

### 概念

结合推理（Reasoning）和行动（Acting），让模型在思考和执行之间交替。

```
问题：苹果公司CEO是谁？他出生在哪一年？

Thought 1: 我需要先查找苹果公司的CEO
Action 1: Search("Apple Inc. current CEO")
Observation 1: Tim Cook is the CEO of Apple Inc.

Thought 2: 现在我知道CEO是Tim Cook，需要查找他的出生年份
Action 2: Search("Tim Cook birth year")
Observation 2: Tim Cook was born on November 1, 1960.

Thought 3: 我已获得所有信息
Final Answer: 苹果公司CEO是Tim Cook，他出生于1960年。
```

### ReAct实现

```python
REACT_PROMPT = """
回答以下问题，使用Thought/Action/Observation格式：

可用工具：
- Search(query): 搜索信息
- Calculator(expression): 计算数学表达式

问题：{question}

Thought 1:"""

def react_loop(question: str, max_steps: int = 5):
    """ReAct推理循环"""
    prompt = REACT_PROMPT.format(question=question)
    
    for step in range(max_steps):
        response = llm.generate(prompt)
        
        if "Final Answer:" in response:
            return extract_answer(response)
        
        # 解析Action
        action = parse_action(response)
        
        # 执行Action
        observation = execute_action(action)
        
        # 更新提示
        prompt += response + f"\nObservation {step+1}: {observation}\n\nThought {step+2}:"
    
    return "无法得出结论"
```

---

## 🌳 思维树（Tree of Thoughts）

### 概念

探索多个推理路径，选择最优解。

```
问题：24点游戏 - 用 1, 2, 3, 4 组成24

┌─ 路径1: (1+2+3)*4 = 24 ✓
│
├─ 路径2: (1+3)*(2+4) = 24 ✓
│
├─ 路径3: 1*2*3*4 = 24 ✓
│
└─ 路径4: (4-1)*(2+3+1) = ? ✗ 不对
```

### ToT实现

```python
def tree_of_thoughts(problem: str, branching: int = 3, depth: int = 3):
    """思维树搜索"""
    
    def generate_thoughts(state: str) -> list:
        """生成多个思考分支"""
        prompt = f"""
问题：{problem}
当前状态：{state}

请生成{branching}个不同的下一步思考，每行一个：
"""
        response = llm.generate(prompt)
        return response.strip().split('\n')
    
    def evaluate_thought(thought: str) -> float:
        """评估思考的质量"""
        prompt = f"""
问题：{problem}
思考步骤：{thought}

请评估这个思考是否有助于解决问题，返回0-1之间的分数：
"""
        score = float(llm.generate(prompt))
        return score
    
    # BFS搜索
    queue = [("", 0)]  # (state, depth)
    best_solution = None
    
    while queue:
        state, d = queue.pop(0)
        
        if d >= depth:
            continue
        
        thoughts = generate_thoughts(state)
        
        for thought in thoughts:
            score = evaluate_thought(thought)
            new_state = state + "\n" + thought
            
            if is_solution(new_state, problem):
                if best_solution is None or score > best_solution[1]:
                    best_solution = (new_state, score)
            else:
                queue.append((new_state, d + 1))
    
    return best_solution
```

---

## 🔍 自我反思（Self-Reflection）

### 概念

让模型批评和改进自己的输出。

```
第一次回答：
Python是一种编程语言。

自我反思：
这个回答太简短了，没有提供有用的信息。应该包括：
- Python的特点
- 主要应用场景
- 学习建议

改进后的回答：
Python是一种高级、解释型编程语言，以简洁易读著称。
主要应用于Web开发、数据分析、机器学习、自动化脚本等领域。
对初学者友好，是入门编程的首选语言。
```

### 实现

```python
def self_reflect(question: str, max_iterations: int = 3) -> str:
    """自我反思改进"""
    
    # 初始回答
    response = llm.generate(f"请回答：{question}")
    
    for i in range(max_iterations):
        # 自我批评
        critique = llm.generate(f"""
请批评以下回答的不足之处：

问题：{question}
回答：{response}

需要改进的地方：""")
        
        # 检查是否满意
        if "没有明显问题" in critique or "回答完整" in critique:
            break
        
        # 改进回答
        response = llm.generate(f"""
根据批评改进回答：

问题：{question}
原回答：{response}
批评：{critique}

改进后的回答：""")
    
    return response
```

---

## 📊 结构化输出

### JSON输出

```python
STRUCTURED_PROMPT = """
请分析以下文本的情感，返回JSON格式：

文本：{text}

返回格式：
{{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0-1.0,
  "keywords": ["关键词1", "关键词2"],
  "summary": "一句话总结"
}}
"""

# 使用OpenAI的JSON模式
response = openai.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}
)
```

### Pydantic结构化

```python
from pydantic import BaseModel
from openai import OpenAI

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    keywords: list[str]
    summary: str

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format=SentimentResult
)

result: SentimentResult = response.choices[0].message.parsed
```

---

## 🎭 多角色讨论

### 概念

模拟多个角色讨论，获得更全面的观点。

```python
MULTI_ROLE_PROMPT = """
请模拟三位专家讨论以下问题：

问题：{question}

专家A（支持者）：
[阐述支持观点]

专家B（反对者）：
[阐述反对观点]

专家C（中立者）：
[综合分析，给出平衡结论]

最终结论：
[基于讨论的综合答案]
"""
```

---

## 🧠 推理模型最佳实践

### 何时使用推理模型

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| **数学推理** | o1/o3 | 原生强推理能力 |
| **代码调试** | o1/o3 | 多步逻辑分析 |
| **复杂规划** | o1/o3 | 分解和协调能力 |
| **聊天对话** | GPT-4o | 速度快、成本低 |
| **创意写作** | GPT-4o | 灵活性和多样性 |
| **实时交互** | GPT-4o | 低延迟要求 |

### o-系列提示技巧

```python
# ❌ 错误：使用CoT提示
bad_prompt = """
请一步一步思考这个问题：
1. 首先分析问题的条件
2. 然后列出可能的解决方案
3. 最后选择最优解

问题：如何设计一个分布式缓存系统？
"""

# ✅ 正确：简洁直接
good_prompt = """
设计一个分布式缓存系统。

要求：
- 支持百万级QPS
- 99.9%可用性
- 数据一致性保障

请给出详细的架构设计。
"""

# ✅ 使用developer消息（替代system消息）
messages = [
    {"role": "developer", "content": "你是分布式系统架构专家"},
    {"role": "user", "content": good_prompt}
]
```

### 混合架构：规划者+执行者

```python
class HybridReasoningSystem:
    """混合推理系统：o1规划 + GPT-4o执行"""
    
    def __init__(self):
        self.planner = "o1"  # 规划者
        self.executor = "gpt-4o"  # 执行者
    
    async def solve_complex_task(self, task: str):
        # 1. 规划者制定计划（使用简洁提示）
        plan = await self.call_model(
            model=self.planner,
            prompt=f"将以下任务分解为具体步骤：{task}"
        )
        
        # 2. 解析计划步骤
        steps = self.parse_steps(plan)
        
        # 3. 执行者逐步执行
        results = []
        for step in steps:
            result = await self.call_model(
                model=self.executor,
                prompt=step
            )
            results.append(result)
        
        # 4. 整合结果
        return self.synthesize_results(results)
```

---

## 🔗 相关阅读

- [基础提示技术](/llms/prompt/basics) - Zero-shot、Few-shot
- [上下文工程](/llms/prompt/context) - 动态上下文管理
- [提示工程全景](/llms/prompt/) - 完整技术架构
- [Agent规划](/llms/agent/planning) - ReAct在Agent中的应用

> **相关论文**：
> - [ReAct](https://arxiv.org/abs/2210.03629)
> - [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
> - [Self-Refine](https://arxiv.org/abs/2303.17651)

> **相关文章**：
> - [掌握AI推理：从"提示工程"到"推理架构"](https://dd-ff.blog.csdn.net/article/details/154479954)
> - [从指令到智能：提示词与上下文工程](https://dd-ff.blog.csdn.net/article/details/152799914)
> - [OpenAI Prompt Engineering与Prompt Caching实战](https://dd-ff.blog.csdn.net/article/details/154450002)
