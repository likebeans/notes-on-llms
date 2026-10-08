---
title: 规划与推理
description: Agent 规划与推理机制 - ReAct、Plan-and-Execute
pageType: article
module: agent
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - agent
level: advanced
prerequisites:
  - /llms/prompt/
  - /llms/rag/
reviewed: '2026-08-25'
techVersion: 待复核（2026-08）
---

# 规划与推理

> 把目标转换成有依赖、有反馈、有停止条件的行动

## 🎯 核心概念

### 什么是Agent规划？

::: tip 定义
**Agent规划** 是智能体将复杂任务分解为可执行步骤，并动态调整执行策略的能力。它是Agent从"被动响应"到"主动解决问题"的关键。
:::

### 规划的核心挑战

| 挑战 | 描述 | 解决方案 |
|------|------|----------|
| **任务分解** | 如何将复杂任务拆分为子任务 | 层次化规划、递归分解 |
| **依赖管理** | 子任务之间的执行顺序 | DAG图、拓扑排序 |
| **动态调整** | 根据执行结果修正计划 | 反馈循环、重规划 |
| **资源约束** | 时间、Token、API调用限制 | 预算管理、优先级排序 |

---

## 📋 规划模式概述

> 来源：[Agentic Design Patterns - Planning](https://github.com/ginobefun/agentic-design-patterns-cn)

### 规划的核心定义

**规划**是指智能体能够**制定一系列行动**，使系统从**初始状态**迈向**目标状态**。

- **你定义「What」**：目标和约束条件
- **智能体规划「How」**：自主规划实现路径

### 规划 vs 提示链

| 模式 | 步骤定义 | 特点 |
|------|----------|------|
| **提示链** | 预先设定 | 步骤固定，适合已知流程 |
| **规划** | 动态生成 | 步骤即时生成，适合探索性任务 |

### 适应性：规划的关键特征

规划的核心是**灵活应变**：

- 初步计划只是**出发点**，而非僵硬的指令
- 智能体能够**接纳新信息**，在遇到阻碍时**调整路线**

```
初始计划：预订A酒店 → 联系B餐饮 → 安排C交通

执行中发现：❌ A酒店已满

智能体反应：
✅ 不是失败，而是适应
✅ 重新评估可选方案
✅ 制定替代计划
```

### 灵活性 vs 可预测性

| 场景 | 推荐方式 |
|------|----------|
| 「如何做」**需要探索** | 使用**规划型智能体** |
| 「如何做」**已经明确** | 使用**固定流程智能体** |

### 三大应用场景

| 场景 | 示例 | 规划内容 |
|------|------|----------|
| **流程自动化** | 新员工入职 | 创建账户 → 分配培训 → 协调部门 |
| **自主导航** | 机器人寻路 | 生成从A到B的最优路径 |
| **信息整合** | 研究报告 | 收集 → 归纳 → 结构化 → 打磨 |

### 实际案例：Deep Research

研究型助手可以采用下列工作流。这是概念示例，不代表所有厂商产品都会要求用户审阅计划，也不承诺搜索来源数量：

```
用户输入："研究AI在医疗领域的应用"
    │
    ▼
① 分解目标，生成研究计划
    │
    ▼
② 用户审阅并修改计划（协作）
    │
    ▼
③ 执行迭代搜索与分析循环
   - 动态调整查询
   - 核对相关来源和证据
   - 主动发现知识盲点
    │
    ▼
④ 生成带引用的结构化报告
```

### 规划与其他模式的关系

规划是「指挥官」，可以调用和整合其他所有模式：

| 执行阶段 | 使用模式 |
|----------|----------|
| 选择处理路径 | 🔀 路由 |
| 同时收集多源信息 | ⚡ 并行化 |
| 调用外部API | 🔧 工具使用 |
| 顺序处理和整合 | 📝 提示链 |
| 检查质量，迭代改进 | 🔄 反思 |

---

## 🔄 ReAct框架详解

> 来源：[解码AI智能体的大脑：Function Calling与ReAct深度对决](https://dd-ff.blog.csdn.net/article/details/153210207)

### 核心理念

**ReAct（Reasoning and Acting）** 在生成过程里交替组织推理与行动，并使用环境观测调整下一步。原始研究强调二者配合，不意味着自然语言解释就是真实内部计算的完整记录。[ReAct 原论文](https://arxiv.org/abs/2210.03629)

### ReAct循环

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   Thought ──→ Action ──→ Observation ──→ ...   │
│      ↑                                    │     │
│      └────────────────────────────────────┘     │
│                                                 │
│              直到任务完成                        │
└─────────────────────────────────────────────────┘
```

### 完整示例

下列注释为人工构造的流程示意，`Search` 和 `Observation` 不是本页实际执行的检索记录。

```python
# 问题："《星际穿越》的导演是谁？他执导的另一部电影的主演又是谁？"

# === 第一轮 ===
# Thought 1: 我需要先找到《星际穿越》的导演
# Action 1: Search("《星际穿越》 导演")
# Observation 1: 《星际穿越》由克里斯托弗·诺兰执导

# === 第二轮 ===
# Thought 2: 好了，导演是诺兰。现在我需要找他另一部电影
# Action 2: Search("克里斯托弗·诺兰 电影作品")
# Observation 2: 诺兰执导过《盗梦空间》、《蝙蝠侠》三部曲、《敦刻尔克》等

# === 第三轮 ===
# Thought 3: 我选择《盗梦空间》，需要找它的主演
# Action 3: Search("《盗梦空间》 主演")
# Observation 3: 《盗梦空间》主演是莱昂纳多·迪卡普里奥

# === 最终答案 ===
# Thought 4: 我已经收集到所有需要的信息
# Final Answer: 《星际穿越》的导演是克里斯托弗·诺兰，
#              他执导的《盗梦空间》主演是莱昂纳多·迪卡普里奥。
```

### ReAct实现

这是控制结构伪代码：`llm.generate`、工具 `.run` 和 `_parse_response` 需自行实现。生产系统使用结构化调用与独立校验，不依赖解析自由文本 Thought，也不要求存储模型私有思考。

```python
from typing import List, Dict, Any

class ReActAgent:
    """ReAct Agent实现"""
    
    def __init__(self, llm, tools: List):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_iterations = 10
    
    def run(self, question: str) -> str:
        """执行ReAct循环"""
        history = []
        
        for i in range(self.max_iterations):
            # 1. 生成Thought和Action
            prompt = self._build_prompt(question, history)
            response = self.llm.generate(prompt)
            
            # 2. 解析响应
            thought, action, action_input = self._parse_response(response)
            history.append({"thought": thought, "action": action, "input": action_input})
            
            # 3. 检查是否完成
            if action == "Final Answer":
                return action_input
            
            # 4. 执行Action获取Observation
            if action in self.tools:
                observation = self.tools[action].run(action_input)
            else:
                observation = f"工具 {action} 不存在"
            
            history.append({"observation": observation})
        
        return "达到最大迭代次数，任务未完成"
    
    def _build_prompt(self, question: str, history: List[Dict]) -> str:
        """构建提示词"""
        prompt = f"""回答以下问题，使用Thought/Action/Observation格式：

问题: {question}

可用工具: {list(self.tools.keys())}

"""
        for item in history:
            if "thought" in item:
                prompt += f"Thought: {item['thought']}\n"
                prompt += f"Action: {item['action']}\n"
                prompt += f"Action Input: {item['input']}\n"
            if "observation" in item:
                prompt += f"Observation: {item['observation']}\n"
        
        return prompt
```

### ReAct优缺点

| 优点 | 缺点 |
|------|------|
| ✅ 动作和观测可追踪 | ❌ 多轮交互，延迟高 |
| ✅ 适合复杂多步任务 | ❌ Token消耗大 |
| ✅ 能根据新观测修订动作 | ❌ 实现相对复杂 |
| ✅ 便于调试和信任建立 | ❌ 可能陷入循环 |

---

## 📋 Plan-and-Execute模式

### 核心思想

与ReAct的"边想边做"不同，**Plan-and-Execute**采用"先规划，后执行"的策略：

1. **Planning阶段**：分析任务，生成完整执行计划
2. **Execution阶段**：按计划逐步执行
3. **Replanning阶段**：根据执行结果调整计划

### 工作流程

```
用户任务
    │
    ▼
┌─────────────┐
│   Planner   │ ──→ 生成任务计划（步骤列表）
└─────────────┘
    │
    ▼
┌─────────────┐
│  Executor   │ ──→ 执行当前步骤
└─────────────┘
    │
    ├── 成功 ──→ 下一步骤
    │
    └── 失败 ──→ Replanner ──→ 调整计划
```

### 实现示例

重规划必须替换待执行队列。对正在 `for step in plan` 遍历的变量重新赋值，并不会让已有迭代器转向新列表。下面使用显式队列避免这个问题。

```python
from collections import deque


def plan_and_execute(task, make_plan, execute_step, replan, verify, max_steps=10):
    pending = deque(make_plan(task))
    history = []
    for _ in range(max_steps):
        if not pending:
            return {"status": "completed" if verify(task, history) else "incomplete",
                    "history": history}
        step = pending.popleft()
        result = execute_step(step, history)
        history.append({"step": step, "result": result})
        if result["status"] != "succeeded":
            pending = deque(replan(task, history, list(pending)))
    completed = not pending and verify(task, history)
    return {"status": "completed" if completed else "budget_exhausted",
            "history": history, "remaining": list(pending)}
```

这是可复用的循环骨架，不包含模型和工具实现。约定 `make_plan` 与 `replan` 返回步骤列表，`execute_step` 真正执行受控工具并返回含 `status` 的结果，`verify` 检查最终业务目标。仅让 LLM 写出“请选择工具并执行”不会发生真实执行。工具超时、权限和结果未知应由执行器明确处理，再决定是否允许重规划。

每个步骤还应有稳定 ID、前置条件、成功证据和幂等键。已经完成的步骤不能因为重规划而再次退款或发信。计划清空仅表示没有待执行项，只有最终证据通过才表示任务完成。

### Plan-and-Execute vs ReAct

| 维度 | ReAct | Plan-and-Execute |
|------|-------|------------------|
| **规划时机** | 边想边做 | 先规划后执行 |
| **适用任务** | 探索性任务 | 明确目标的任务 |
| **效率** | 灵活但可能迂回 | 结构化高效 |
| **可控性** | 中等 | 高（计划可审核） |
| **纠错方式** | 实时调整 | 重规划 |

---

## 🧠 思维链（Chain of Thought）

### CoT基础

**思维链（CoT）** 是让模型在回答前先展示推理过程的技术：

```python
# 普通提示
prompt = "计算 23 × 17 的结果"
# 模型可能直接给出错误答案

# CoT提示
prompt = """计算 23 × 17 的结果。
让我们一步步思考："""

# 模型输出：
# 23 × 17
# = 23 × (20 - 3)
# = 23 × 20 - 23 × 3
# = 460 - 69
# = 391
```

### CoT变体

| 变体 | 描述 | 适用场景 |
|------|------|----------|
| **Zero-shot CoT** | "让我们一步步思考" | 简单推理 |
| **Few-shot CoT** | 提供推理示例 | 复杂推理 |
| **Self-Consistency** | 多次采样取众数 | 提高准确性 |
| **Tree of Thoughts** | 探索多条推理路径 | 创造性任务 |

### LangGraph中的规划

以下仅展示图与状态的连接方式，`llm`、`execute_task` 等业务适配器需补齐；进入生产前应加入检查点、失败路由和上述完成条件。

> 来源：[LangGraph深度解析（一）](https://dd-ff.blog.csdn.net/article/details/151024355)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

class PlanExecuteState(TypedDict):
    task: str
    plan: List[str]
    current_step: int
    results: Annotated[List[str], operator.add]
    final_answer: str

def planner(state: PlanExecuteState) -> PlanExecuteState:
    """规划节点"""
    task = state["task"]
    plan = llm.generate(f"为任务'{task}'创建执行计划")
    return {"plan": plan.split("\n"), "current_step": 0}

def executor(state: PlanExecuteState) -> PlanExecuteState:
    """执行节点"""
    current_step = state["current_step"]
    step = state["plan"][current_step]
    result = execute_with_tools(step)
    return {
        "results": [result],
        "current_step": current_step + 1
    }

def should_continue(state: PlanExecuteState) -> str:
    """判断是否继续"""
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "executor"

def synthesize(state: PlanExecuteState) -> PlanExecuteState:
    """综合结果"""
    answer = llm.generate(
        f"根据以下结果回答问题：{state['results']}"
    )
    return {"final_answer": answer}

# 构建图
graph = StateGraph(PlanExecuteState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("synthesize", synthesize)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue)
graph.add_edge("synthesize", END)

app = graph.compile()
```

---

## 规划失败如何验收

至少准备四类任务：一步可完成、需要依赖顺序、执行中出现新信息、永远无法完成。检查简单任务是否过度规划，后继步骤是否只在前置证据满足后启动，以及无法完成时是否在预算内返回原因。对同一失败动作重复调用，应触发停机或升级，而不是无限“再试一下”。

把状态持久化和副作用恢复交给[记忆系统](/llms/agent/memory)与[异常处理](/llms/agent/exception-handling)，把候选推理方法放在[推理技术](/llms/agent/reasoning)比较，避免规划器同时承担所有职责。

## 🔗 相关阅读

- [Agent概述](/llms/agent/) - 了解Agent整体架构
- [工具调用](/llms/agent/tool-calling) - 行动执行详解
- [记忆系统](/llms/agent/memory) - 状态管理

> **相关文章**：
> - [解码AI智能体的大脑：Function Calling与ReAct深度对决](https://dd-ff.blog.csdn.net/article/details/153210207)
> - [LangGraph深度解析（一）：核心原理到生产级工作流](https://dd-ff.blog.csdn.net/article/details/151024355)
> - [未来的认知架构：深入剖析自主AI研究智能体](https://dd-ff.blog.csdn.net/article/details/150217636)

> **外部资源**：
> - [ReAct论文](https://arxiv.org/abs/2210.03629) - ReAct框架原始论文
> - [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - CoT原始论文
> - [LangGraph Planning](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)
