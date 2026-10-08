---
title: 高级提示技术
description: ReAct、思维树、自我反思等高级技术
pageType: article
module: prompt
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - prompt
level: beginner
prerequisites: []
reviewed: '2026-10-08'
reviewScope: Responses 结构化输出字段与拒答分支；本地类型校验与网络调用分别标注
exampleStatus: not-run
techVersion: Responses 结构化输出文档核验 2026-10-08；网络示例未实跑；Pydantic 2 本地校验
---

# 高级提示技术

> 复杂任务的提示词策略——从手动推理引导到原生推理调用

## ⚠️ 重要提示：推理模型的范式转变

高级提示需要额外模型调用、搜索或验证。先确认现有错误来自任务分解或证据获取，再选择相应策略；如果问题只是字段不合法，优先用结构化输出与验证器。

| 需求 | 候选方法 | 必须支付的成本 |
| --- | --- | --- |
| 需要外部观察才能继续 | ReAct / 工具循环 | 工具延迟、失败恢复和权限治理 |
| 存在多个可验证候选路径 | ToT / 有界搜索 | 分支数量与评估器成本 |
| 已有答案能被独立检查 | 反思 + 外部验证 | 检查器覆盖与误判风险 |
| 接口需要固定字段 | Schema 约束 | Schema 设计、失败分支与业务验证 |

推理模型通常不需要追加“展示全部思维链”。[OpenAI 官方建议](https://developers.openai.com/api/docs/guides/reasoning-best-practices) 优先清晰目标、约束和输出要求，而不是断言所有 CoT 写法都会干扰推理。对用户提供可验证步骤或简要理由即可。

---

## 🔄 ReAct（推理+行动）

### 概念

结合推理（Reasoning）和行动（Acting），让模型在思考和执行之间交替。

```
历史演示问题：某份资料记载苹果公司 CEO 为 Tim Cook，他出生在哪一年？

Thought 1: 我需要先查找苹果公司的CEO
Action 1: Search("Apple Inc. Tim Cook biography")
Observation 1: 示例资料记载 CEO 为 Tim Cook；实际使用应核对资料日期。

Thought 2: 现在我知道CEO是Tim Cook，需要查找他的出生年份
Action 2: Search("Tim Cook birth year")
Observation 2: Tim Cook was born on November 1, 1960.

Thought 3: 我已获得所有信息
Final Answer: 根据示例资料，人物为 Tim Cook，出生于1960年；实际答案需带来源日期。
```

### ReAct实现

下例保留论文风格的文本循环，是伪代码；`llm`、解析器与执行器由应用注入。生产工具调用使用明确 schema 和允许列表，先校验参数与授权，再执行；工具 Observation 视作数据，不因其中出现指令而扩大权限。需要最大步数、超时、错误状态与幂等键。[ReAct 原论文](https://arxiv.org/abs/2210.03629) 讨论推理与行动交替，不提供这些业务保障。

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

显式维护多个候选状态，生成下一步后评估、剪枝或回溯。模型评分只是搜索启发，不保证最优解；适合能定义状态、合法动作和终止验证的任务。

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

下例为宽度未限制的 BFS 示意，`llm` 与 `is_solution` 需实现；深度 d、分支 b 的节点数可按 b 的幂增长。工程上增加 beam 宽度、去重、预算和可验证终止条件。24 点应由算术检查器确认数值、运算与数字使用次数，不能仅凭模型打分通过。[ToT 原论文](https://arxiv.org/abs/2305.10601)

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

让模型根据明确 rubric 检查草稿，最好结合测试、检索或规则提供外部反馈。同一模型可能保留原错误，也可能把正确答案改错，因此每轮都应记录验证结果并保留最好版本；“回答完整”等文本不能作为可靠停止信号。

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

以下展示提示交互的伪代码，`llm.generate` 不是通用 SDK；实际停止条件应由独立验证器和最大迭代预算共同决定。

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

自然语言“请输出 JSON”、JSON mode、Schema 约束是三种不同强度的接口。JSON mode 主要约束语法，Structured Outputs 可在支持的模型与 Schema 子集上约束字段结构；均需处理拒答、截断和服务错误。见 [OpenAI Structured Outputs 官方文档](https://developers.openai.com/api/docs/guides/structured-outputs)。

业务流程应为：响应完成状态检查 → 拒答分支 → JSON/Schema 解析 → 业务规则验证 → 执行后续动作。不能把半截 JSON 自动补全后当真实答案，也不能在重试结构修复时重复执行已经产生副作用的动作。

### Responses 与 Chat Completions 的字段不要混用

**2026-10-08 文档核验；以下网络调用未实跑。** Responses 使用 `text.format`，Chat Completions 使用 `response_format`；调用外部动作则用函数工具的参数 schema。三者不是改个方法名即可迁移。严格结构保证适用 schema 的形状，拒答、未完成响应和业务真实性仍要分别处理。[Structured Outputs 官方接口](https://developers.openai.com/api/docs/guides/structured-outputs)

以下是 Responses 的最小抽取示例。运行需要兼容的 `openai` SDK、`OPENAI_API_KEY` 和支持该接口的 `OPENAI_MODEL`；姓名允许为空，避免缺资料时被迫捏造：

```python
import json
import os
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model=os.environ["OPENAI_MODEL"],
    input="从这段资料抽取姓名；没有姓名就填 null。资料：一位匿名读者留言。",
    text={"format": {
        "type": "json_schema", "name": "person", "strict": True,
        "schema": {
            "type": "object",
            "properties": {"name": {"type": ["string", "null"]}},
            "required": ["name"], "additionalProperties": False,
        },
    }},
)
if response.status != "completed":
    raise RuntimeError(f"响应未完成：{response.status}")
refusals = [part.refusal for item in response.output
            if item.type == "message" for part in item.content
            if part.type == "refusal"]
if refusals:
    raise RuntimeError("模型拒答：" + "; ".join(refusals))
if not response.output_text:
    raise RuntimeError("没有可解析的结构化结果")
person = json.loads(response.output_text)
print(person)  # 仍需结合原文做语义验收；这里不会执行外部写操作。
```

本地 Pydantic 校验器可以表达跨字段业务规则，但这些 Python 函数不会自动成为模型的 schema 约束。即便 SDK 根据类型生成 JSON Schema，下游仍必须运行本地校验。

### Pydantic结构化

下面是可本地执行的 Pydantic 2 校验示例，不依赖 API。实际接入支持 Schema 的模型时可复用类型，但要检查供应商支持的 Schema 子集。删去未经校准的 `confidence` 自报数值，以证据字段和明确失败状态表达边界。

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class SentimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "insufficient_input"]
    sentiment: Literal["positive", "negative", "neutral"] | None
    keywords: list[str] = Field(max_length=10)
    summary: str

    @model_validator(mode="after")
    def check_status(self):
        if (self.status == "ok") != (self.sentiment is not None):
            raise ValueError("ok 需要情感标签；资料不足时标签必须为空")
        return self

raw = '{"status":"ok","sentiment":"positive","keywords":["好吃"],"summary":"用户满意"}'
result = SentimentResult.model_validate_json(raw)
print(result.model_dump())
```

结构验证不检查“用户是否真的满意”。分类边界、原文依据和资料不足的判断还需语义评估。拒答不是缺字段的成功响应，必须保留独立状态。

---

## 🎭 多角色讨论

### 概念

模拟多个角色可以检查不同标准，但同一模型的三个角色不是三个独立专家，也不能通过投票消除共同偏差。让每个角色引用证据、提出可检验反例，再由规则或人工作最终判断。

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

以下型号为历史示例，体现“复杂推理预算与交互延迟”的取舍，不是实时推荐表。比较成功率、p95、token 及工具调用总成本后再选择。

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
# 对照提示：步骤较多，是否有益需要实测
bad_prompt = """
请一步一步思考这个问题：
1. 首先分析问题的条件
2. 然后列出可能的解决方案
3. 最后选择最优解

问题：如何设计一个分布式缓存系统？
"""

# 基线提示：直接描述目标与验收条件
good_prompt = """
设计一个分布式缓存系统。

要求：
- 支持百万级QPS
- 99.9%可用性
- 数据一致性保障

请给出详细的架构设计。
"""

# 在支持 developer 角色的 API 中放置应用指令；角色支持以接口文档为准
messages = [
    {"role": "developer", "content": "你是分布式系统架构专家"},
    {"role": "user", "content": good_prompt}
]
```

### 混合架构：规划者+执行者

下面为接口骨架，`call_model`、`parse_steps` 与汇总器需实现。步骤间有依赖时必须传递前一步产物和失败状态，不能只把孤立步骤文字发给执行模型。计划输出还需 schema、允许动作及预算检查。

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

## 高级策略的验收

使用相同测试集和成本预算，比较单次调用、追加验证、候选搜索三种方案。记录正确率、从错到对和从对到错的比例、平均/最坏调用次数、工具失败率与延迟。只有新增机制修复了明确错误，且没有超出预算，才值得保留。

对 ReAct 查工具是否提供了新证据；对 ToT 查剪枝是否丢掉正确路径；对反思查反馈是否可验证；对结构化输出分别统计解析失败、Schema 失败和业务失败。不要把“文本更长、角色更多、看起来更会思考”当作收益。

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
