---
title: 多智能体系统
description: 多智能体协作架构 - 从单Agent到Multi-Agent
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

# 多智能体系统

> 让多个AI智能体协作完成复杂任务

## 🎯 核心概念

### 为什么需要多智能体？

> 来源：[LangGraph多智能体系统权威指南](https://dd-ff.blog.csdn.net/article/details/151153365)

::: tip 单Agent的局限
当任务复杂度增加时，单个Agent容易产生**认知过载**：
- 工具太多导致选择困难
- 上下文窗口被撑满
- 单点故障风险高
- 难以并行处理
:::

### 多智能体的优势

| 优势 | 说明 |
|------|------|
| **专业化分工** | 每个Agent专注特定领域 |
| **并行处理** | 多个Agent同时工作 |
| **模块化** | 独立开发、测试、维护 |
| **容错性** | 单个Agent失败不影响整体 |
| **可扩展** | 按需添加新Agent |

---

## 📋 多智能体协作模式概述

> 来源：[Agentic Design Patterns - Multi-Agent Collaboration](https://github.com/ginobefun/agentic-design-patterns-cn)

### 核心原理

多智能体协作基于**任务分解原则**：将高层次目标拆解为若干独立的子问题，然后将每个子问题分配给拥有相应工具、数据权限或推理能力的智能体来处理。

### 系统三要素

| 要素 | 说明 |
|------|------|
| **角色与职责** | 明确每个智能体的专长和任务范围 |
| **通信通道** | 智能体之间如何交换信息 |
| **交互协议** | 引导协同工作的任务流程 |

### 六种协作形式

| 形式 | 特点 | 适用场景 |
|------|------|----------|
| **顺序交接** | A完成 → 输出给B → B完成 | 流水线式工作流 |
| **并行处理** | A、B、C同时工作 → 合并结果 | 可并行的独立子任务 |
| **辩论与共识** | 不同视角讨论 → 达成共识 | 需要多角度分析的决策 |
| **层级结构** | 管理者分配 → 执行者完成 → 汇总 | 协调多个专业智能体 |
| **专家团队** | 研究员+撰稿人+编辑协作 | 软件开发、内容创作 |
| **评审者模式** | 创作 → 评审 → 修改 → 输出 | 代码生成、学术写作 |

### 六种通信架构

| 架构 | 特点 | 优势 | 劣势 |
|------|------|------|------|
| **单智能体** | 独立运行，无交互 | 简单易管理 | 能力受限 |
| **网络化** | 点对点去中心化 | 可分散部分调度职责 | 通信与一致性开销大；仍可能有共享依赖故障 |
| **监督者** | 中心枢纽协调 | 职责清晰 | 可能瓶颈 |
| **监督者作为工具** | 提供资源而非指挥 | 灵活 | 设计复杂 |
| **层级** | 多层监督者结构 | 可扩展 | 结构复杂 |
| **自定义** | 完全定制化 | 最大灵活性 | 需深入理解 |

### 七大应用场景

| 场景 | 智能体分工 |
|------|------------|
| **复杂研究** | 搜索 → 总结 → 趋势分析 → 报告 |
| **软件开发** | 需求分析 → 编码 → 测试 → 文档 |
| **创意内容** | 调研 → 文案 → 设计 → 排期 |
| **财务分析** | 数据获取 → 情绪分析 → 技术分析 → 建议 |
| **客户支持** | 一线处理 → 专家升级 |
| **供应链** | 供应商 ↔ 制造商 ↔ 分销商协同 |
| **网络运维** | 排查 → 定位 → 修复 |

### 架构选择考量

| 考虑因素 | 影响 |
|----------|------|
| **任务复杂度** | 复杂任务需要层级或网络化 |
| **智能体数量** | 数量多时需要监督者 |
| **自治程度** | 高自治选网络化，低自治选监督者 |
| **鲁棒性要求** | 按故障域、重试、状态恢复和共享依赖评估，不能只看拓扑 |

---

## 🏗️ 三种协作架构

### 架构对比

| 架构 | 特点 | 适用场景 |
|------|------|----------|
| **网络模式** | Agent间直接通信 | 平等协作、灵活交互 |
| **监督者模式** | 中心协调所有Worker | 任务分配、流程控制 |
| **层级模式** | 多层管理结构 | 大型复杂项目 |

### 1. 监督者模式（Supervisor）

以下为图结构示意，`llm`、`do_research`、`write_code`、`run_tests` 由业务实现；进入模型前应把内部事件转成 SDK 接受的消息角色。路由返回值必须做枚举校验，不能直接信任模型给出的节点名。

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Annotated
import operator

class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    next_worker: str

def supervisor(state: SupervisorState) -> SupervisorState:
    """监督者：决定下一步由哪个Worker执行"""
    prompt = f"""根据当前任务状态，选择下一个Worker：
- researcher: 信息收集
- coder: 编写代码
- tester: 测试验证
- FINISH: 任务完成

当前消息：{state['messages'][-1]}"""
    
    response = llm.generate(prompt).strip()
    if response not in {"researcher", "coder", "tester", "FINISH"}:
        raise ValueError("监督者返回了未知 Worker")
    return {"next_worker": response}

def researcher(state): 
    result = do_research(state["messages"])
    return {"messages": [{"role": "researcher", "content": result}]}

def coder(state):
    result = write_code(state["messages"])
    return {"messages": [{"role": "coder", "content": result}]}

def tester(state):
    return {"messages": [{"role": "tester", "content": run_tests(state)}]}

# 构建图
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("tester", tester)

def route(state) -> str:
    if state["next_worker"] == "FINISH":
        return END
    return state["next_worker"]

graph.add_conditional_edges("supervisor", route)
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")
graph.add_edge("tester", "supervisor")
graph.set_entry_point("supervisor")
```

### 2. 网络模式（Network）

```python
# Agent间可直接通信
def agent_a(state) -> dict:
    if needs_help_from_b(state):
        return {"next": "agent_b", "message": "需要你的帮助"}
    return {"result": "完成"}

def agent_b(state) -> dict:
    if needs_verification(state):
        return {"next": "agent_c", "message": "请验证"}
    return {"next": "agent_a", "message": "已处理"}
```

---

## 委派契约与共享状态

先用单 Agent 和固定工作流建立基线。只有子任务能分开取证、使用不同工具或独立验收时，才值得引入多个 Agent。把同一提示复制给更多角色，可能只增加重复错误和汇总成本。

每次委派应明确目标、输入快照、允许修改的资源、预算、预期产物和验收标准。子 Agent 返回结论、证据、产物地址与未解决项；主 Agent 负责整合与最终验收，不能把“子 Agent 说完成了”当作完成证据。

对共享文件或数据库，先划定单一写入者或使用版本检查。消息有唯一 ID，重复接收不重复执行；增加最大委派深度、轮数和费用上限，避免 A 委派 B、B 再委派 A 的循环。通信协议参见 [A2A](/llms/agent/a2a)，它不会自动提供业务一致性。

验收按同样预算比较单 Agent 与多 Agent：任务成功率、端到端耗时、总 Token、重复工作量和合并冲突数。再演练一个 Worker 失败或返回矛盾证据的情况，检查监督者是否能明确处理分歧。

## 🔄 AutoGen多智能体团队

> 来源：[AutoGen多智能体团队实战指南](https://dd-ff.blog.csdn.net/article/details/149090900)

### RoundRobinGroupChat

以下是 AgentChat 接口片段，需安装对应版本并初始化 `model_client`。轮询团队是按顺序发言，不等于并行执行；顶部 `await` 需放在支持它的 notebook 或异步入口内。

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat

# 创建专业化Agent
poet = AssistantAgent(
    name="poet",
    model_client=model_client,
    system_message="你是一位诗人，擅长创作诗歌。"
)

critic = AssistantAgent(
    name="critic", 
    model_client=model_client,
    system_message="你是文学评论家，擅长改进诗歌。"
)

# 创建轮询团队
team = RoundRobinGroupChat(
    participants=[poet, critic],
    max_turns=10
)

# 运行任务
result = await team.run(task="创作一首关于春天的诗")
```

### SelectorGroupChat（动态选择）

```python
from autogen_agentchat.teams import SelectorGroupChat

# 模型动态选择下一个发言者
team = SelectorGroupChat(
    participants=[researcher, writer, editor],
    model_client=model_client,
    selector_prompt="根据对话选择下一个Agent..."
)
```

---

## 🎭 协作设计模式

### 反思模式（Reflection）

```text
# 生成者 + 评审者循环
generator → reviewer → refiner → reviewer → ... → APPROVED
```

### 分工模式

```text
# 并行执行子任务
planner → [executor1, executor2, executor3] → aggregator → reporter
```

### 辩论模式

```text
# 正反方辩论，裁判评估
pro_agent ←→ con_agent → judge → conclusion
```

---

## 🔗 相关阅读

- [Agent概述](/llms/agent/) - Agent整体架构
- [记忆系统](/llms/agent/memory) - 多Agent状态共享
- [规划与推理](/llms/agent/planning) - 任务分解协调

> **相关文章**：
> - [LangGraph多智能体系统权威指南](https://dd-ff.blog.csdn.net/article/details/151153365)
> - [AutoGen多智能体团队实战指南](https://dd-ff.blog.csdn.net/article/details/149090900)
> - [精通人机协同：LangGraph交互式智能体](https://dd-ff.blog.csdn.net/article/details/151149262)

> **外部资源**：
> - [LangGraph Multi-Agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
> - [AutoGen Teams](https://microsoft.github.io/autogen/docs/tutorial/teams)
