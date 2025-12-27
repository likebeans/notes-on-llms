---
title: 规划与推理
description: Agent 规划与推理机制 - ReAct、Plan-and-Execute
---

# 规划与推理

> 让AI像人类一样思考、规划、行动

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

Google/OpenAI的Deep Research是规划型智能体的典型应用：

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
   - 分析数百个来源
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

**ReAct（Reasoning and Acting）** 由Google Research提出，核心思想是模仿人类解决问题的方式，将**推理(Reasoning)**和**行动(Acting)**显式地结合起来。

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
| ✅ 推理过程透明可见 | ❌ 多轮交互，延迟高 |
| ✅ 适合复杂多步任务 | ❌ Token消耗大 |
| ✅ 纠错能力强 | ❌ 实现相对复杂 |
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

```python
from typing import List

class PlanAndExecuteAgent:
    """Plan-and-Execute Agent"""
    
    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools
    
    def run(self, task: str) -> str:
        # 1. 生成计划
        plan = self.create_plan(task)
        print(f"📋 计划: {plan}")
        
        results = []
        for i, step in enumerate(plan):
            print(f"\n🔄 执行步骤 {i+1}: {step}")
            
            # 2. 执行单个步骤
            result = self.execute_step(step, results)
            results.append({"step": step, "result": result})
            
            # 3. 检查是否需要重规划
            if self.needs_replan(step, result):
                remaining_steps = plan[i+1:]
                plan = self.replan(task, results, remaining_steps)
                print(f"🔄 重规划: {plan}")
        
        # 4. 综合结果
        return self.synthesize(task, results)
    
    def create_plan(self, task: str) -> List[str]:
        """生成执行计划"""
        prompt = f"""为以下任务创建执行计划，返回步骤列表：

任务: {task}

可用工具: {[t.name for t in self.tools]}

要求：
1. 每个步骤应该是具体可执行的
2. 步骤之间有清晰的逻辑顺序
3. 步骤数量控制在5个以内

输出格式（JSON数组）：
["步骤1", "步骤2", ...]
"""
        response = self.planner.generate(prompt)
        return json.loads(response)
    
    def execute_step(self, step: str, previous_results: List) -> str:
        """执行单个步骤"""
        context = "\n".join([
            f"步骤: {r['step']}\n结果: {r['result']}" 
            for r in previous_results
        ])
        
        prompt = f"""执行以下步骤：

当前步骤: {step}

之前的执行结果:
{context}

可用工具: {[t.name for t in self.tools]}

请选择合适的工具并执行。
"""
        return self.executor.generate(prompt)
```

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
