---
title: 记忆系统
description: Agent 记忆与状态管理 - 短期/长期记忆机制
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
reviewed: '2026-10-08'
reviewScope: 长任务恢复、持久状态与摘要边界；未逐一验证 LangGraph/AutoGen 接口
exampleStatus: not-run
techVersion: 长任务状态原理复核 2026-10-08；LangGraph/AutoGen 片段未做当前版本集成测试
---

# 记忆系统

> 让AI拥有"记忆"，实现跨对话的持续智能

## 🎯 核心概念

### 为什么Agent需要记忆？

::: tip 核心问题
模型权重不会因一次普通对话自动更新。连续对话依赖应用或服务端保存并再次提供历史；检查点、数据库和会话 API 都属于系统状态，而不是模型自然“记住了”。没有这些机制，Agent难以：
- 记住之前的对话内容
- 跟踪任务执行进度
- 学习用户偏好
- 在中断后恢复工作
:::

### 记忆类型

| 类型 | 作用域 | 生命周期 | 典型用途 |
|------|--------|----------|----------|
| **短期记忆** | 单次对话 | 可随线程持久化；保留期由系统定义 | 对话上下文、中间结果 |
| **长期记忆** | 跨对话 | 持久化存储 | 用户偏好、历史知识 |
| **工作记忆** | 单次任务 | 任务完成即清理 | 任务状态、执行计划 |
| **情景记忆** | 特定场景 | 按需召回 | 过往对话摘要 |

---

## Agentic Design Patterns 视角

> 来源：[Agentic Design Patterns - Memory Management](https://github.com/ginobefun/agentic-design-patterns-cn)

### 双组件记忆系统

常见设计是区分**短期与长期存储**的双组件记忆系统：

| 组件 | 存储位置 | 作用 |
|------|----------|------|
| **短期记忆** | LLM上下文窗口 | 保存最近交互数据，维持对话流程 |
| **长期记忆** | 外部数据库/向量存储 | 高效语义检索持久化信息 |

### 框架实现

| 框架 | 组件 | 功能 |
|------|------|------|
| **Google ADK** | `Session` | 管理对话线程 |
| **Google ADK** | `State` | 存储临时数据（`user:`/`app:`/`temp:`前缀） |
| **Google ADK** | `MemoryService` | 与长期知识库交互 |
| **LangChain / LangGraph** | 消息状态 + checkpointer | 保存线程级历史；旧版 Memory 类需按版本迁移 |
| **LangGraph** | `Store` | 跨会话保存语义事实、情景经历 |

### 六大应用场景

| 场景 | 记忆需求 |
|------|----------|
| **聊天机器人** | 短期维持对话流，长期记忆用户偏好 |
| **任务导向智能体** | 短期跟踪步骤进度，长期访问用户数据 |
| **个性化体验** | 长期存储用户偏好和历史行为 |
| **学习与优化** | 长期保存成功策略和错误经验 |
| **RAG信息检索** | 长期记忆作为知识库 |
| **自主系统** | 短期感知环境，长期存储地图和行为 |

### 使用场景

✅ 需要在对话中**维持上下文**  
✅ 需要**跟踪多步骤任务进度**  
✅ 需要**个性化交互**（回忆用户偏好）  
✅ 需要基于过去经验**学习或自适应**

---

## 📝 短期记忆（对话上下文）

### 基本实现

裁剪必须保留协议完整性：一次工具调用及其返回值通常应一起保留或一起移除。只按消息条数切片可能留下没有对应调用的工具结果。下面的示例适用于纯文本 user/assistant 消息，工具轨迹应改用按轮次或调用组裁剪。

```python
class ConversationMemory:
    def __init__(self, count_tokens, max_tokens=4000):
        self.messages = []
        self.count_tokens = count_tokens
        self.max_tokens = max_tokens

    def add_message(self, role, content):
        candidate = self.messages + [{"role": role, "content": content}]
        while self.count_tokens(candidate) > self.max_tokens:
            index = next((i for i, msg in enumerate(candidate)
                          if msg["role"] != "system"), None)
            if index is None or index == len(candidate) - 1:
                raise ValueError("固定上下文或最新消息超出预算，需缩短输入")
            candidate.pop(index)
        self.messages = candidate

    def get_messages(self):
        return [dict(message) for message in self.messages]
```

`count_tokens(messages)` 需注入目标模型的计数器或服务端计数接口，并预留工具定义、输出及推理预算。只保留 system 消息仍超限时必须退出，不能陷入无限裁剪循环。

### 滑动窗口策略

下例只适用于每轮恰好一条用户消息与一条助手消息的纯文本对话。真实工具会话可能一轮多条消息，不能照搬 `window_size * 2`。

```python
class SlidingWindowMemory:
    """滑动窗口记忆 - 只保留最近N轮对话"""
    
    def __init__(self, window_size: int = 10):
        self.messages = []
        self.window_size = window_size
        self.system_message = None
    
    def add_message(self, role: str, content: str):
        if role == "system":
            self.system_message = {"role": role, "content": content}
        else:
            self.messages.append({"role": role, "content": content})
            # 保持窗口大小（每轮2条消息：user + assistant）
            max_messages = self.window_size * 2
            if len(self.messages) > max_messages:
                self.messages = self.messages[-max_messages:]
    
    def get_messages(self) -> list:
        result = []
        if self.system_message:
            result.append(self.system_message)
        result.extend(self.messages)
        return result
```

### 摘要记忆

以下为接口示意，`llm.generate` 与 `_format_messages` 需实现。摘要要保留原始消息 ID、用户约束、未完成动作和关键来源；摘要生成失败时保留原始历史。反复摘要可能累积遗漏，事实记录与任务状态应另存为结构化数据。

```python
class SummaryMemory:
    """摘要记忆 - 压缩历史对话为摘要"""
    
    def __init__(self, llm, summary_threshold: int = 20):
        self.llm = llm
        self.messages = []
        self.summary = ""
        self.summary_threshold = summary_threshold
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # 消息过多时生成摘要
        if len(self.messages) > self.summary_threshold:
            self._summarize()
    
    def _summarize(self):
        """将早期对话压缩为摘要"""
        # 保留最近的消息
        recent = self.messages[-10:]
        to_summarize = self.messages[:-10]
        
        # 生成摘要
        prompt = f"""请将以下对话历史压缩为简洁摘要：
        
之前的摘要：{self.summary}

新的对话：
{self._format_messages(to_summarize)}

输出简洁的摘要（保留关键信息）："""
        
        self.summary = self.llm.generate(prompt)
        self.messages = recent
    
    def get_context(self) -> str:
        """获取完整上下文"""
        context = f"对话摘要：{self.summary}\n\n" if self.summary else ""
        context += self._format_messages(self.messages)
        return context
```

---

## 💾 长期记忆（跨对话持久化）

> 来源：[精通状态智能体：LangGraph内存机制综合指南](https://dd-ff.blog.csdn.net/article/details/151118407)

### LangGraph双轨制记忆

LangGraph 区分线程级检查点与跨线程 Store。下面的内存实现只适合进程内演示，进程退出会丢失；要实现重启恢复必须换用持久化后端。示例中的 `State`、节点和消息需按应用定义。[官方短期记忆文档](https://docs.langchain.com/oss/python/langchain/short-term-memory)

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# 短期记忆：通过thread_id追踪单次对话
checkpointer = InMemorySaver()

# 跨线程记忆接口：这里仍是内存存储，不具备磁盘持久性
store = InMemoryStore()

# 创建带记忆的图
graph = StateGraph(State)
# ... 添加节点和边 ...

app = graph.compile(
    checkpointer=checkpointer,
    store=store
)

# 使用thread_id进行对话（短期记忆）
config = {"configurable": {"thread_id": "conversation_123"}}
result = app.invoke({"messages": [user_message]}, config)

# 存储用户偏好（长期记忆）
store.put(
    namespace=("users", "user_001"),
    key="preferences",
    value={"language": "zh", "style": "formal"}
)
```

### 检查点与时间旅行

> 来源：[LangGraph时间旅行深度解析](https://dd-ff.blog.csdn.net/article/details/151151727)

“回到检查点”是从指定状态继续或创建分支，并不撤销检查点之后的外部动作。仅把旧 `values` 写入当前状态也不等价于精确恢复，因为 reducer 可能将旧列表追加到现有列表。

应选择目标检查点对应的配置，按固定版本的恢复/分叉 API 操作，并重新核对外部世界。退款、发信、文件写入要通过幂等记录避免重复。需要了解检查点、线程和 replay 的边界时，参见 [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)。

### 向量化长期记忆

以下为历史 Chroma 集成风格，包导入与持久化选项需按固定版本核对。`user_id` 必须来自已认证会话，不能直接相信模型或用户填写的路径。检索分数可能表示距离而非相似度，阈值方向须按后端定义。

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorMemory:
    """向量化长期记忆 - 支持语义检索"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name=f"memory_{user_id}",
            embedding_function=self.embeddings,
            persist_directory=f"./memory/{user_id}"
        )
    
    def store(self, content: str, metadata: dict = None):
        """存储记忆"""
        self.vectorstore.add_texts(
            texts=[content],
            metadatas=[metadata or {}]
        )
    
    def recall(self, query: str, k: int = 5) -> list:
        """召回相关记忆"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def recall_with_score(self, query: str, k: int = 5) -> list:
        """召回并返回相关度分数"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results]
```

---

## 🔄 AutoGen状态管理

> 来源：[AutoGen状态管理实战：从内存到持久化](https://dd-ff.blog.csdn.net/article/details/149097602)

### 智能体状态序列化

以下为 AutoGen AgentChat 的接口片段，需提供兼容版本的 `model_client` 并在异步函数中运行；持久化文件需做访问控制与原子写入，恢复时应核对 Agent 名称、工具定义和状态版本。

```python
from autogen_agentchat.agents import AssistantAgent

# 创建智能体
agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="你是一个有帮助的助手"
)

# 对话后保存状态
await agent.run(task="帮我分析这份数据")
state = await agent.save_state()

# state包含：
# - llm_messages: 对话历史
# - model_context: 模型上下文
# - 自定义状态数据

# 持久化到文件
import json
with open("agent_state.json", "w") as f:
    json.dump(state, f)

# 恢复状态
with open("agent_state.json", "r") as f:
    saved_state = json.load(f)

new_agent = AssistantAgent(
    name="assistant", model_client=model_client,
    system_message="你是一个有帮助的助手"
)
await new_agent.load_state(saved_state)

# 继续之前的对话
await new_agent.run(task="继续上次的分析")
```

### 团队状态管理

```python
from autogen_agentchat.teams import RoundRobinGroupChat

# 创建团队
team = RoundRobinGroupChat(
    participants=[agent1, agent2],
    max_turns=10
)

# 执行任务
result = await team.run(task="完成这个项目")

# 保存团队状态（递归包含所有成员状态）
team_state = await team.save_state()

# 恢复团队状态
await team.load_state(team_state)
```

---

## 🧠 上下文工程

> 来源：[LangGraph上下文工程权威指南](https://dd-ff.blog.csdn.net/article/details/151118698)

### 三种上下文类型

| 类型 | 传递方式 | 生命周期 | 用途 |
|------|----------|----------|------|
| **静态运行时上下文** | config参数 | 单次运行 | 用户配置、权限 |
| **动态运行时上下文** | State对象 | 单次运行 | 对话历史、中间结果 |
| **跨对话持久化上下文** | Store | 跨运行 | 用户偏好、学习数据 |

### 实现示例

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore

class State(TypedDict):
    messages: list  # 动态运行时上下文
    user_context: dict  # 从Store加载的持久化上下文

def load_user_context(state: State, config: dict, store: BaseStore) -> State:
    """加载用户持久化上下文"""
    user_id = config["configurable"]["user_id"]
    
    # 从Store获取用户偏好
    preferences = store.get(("users", user_id), "preferences")
    history_summary = store.get(("users", user_id), "history_summary")
    
    return {
        "user_context": {
            "preferences": preferences.value if preferences else {},
            "history": history_summary.value if history_summary else {}
        }
    }

def update_user_context(state: State, config: dict, store: BaseStore) -> State:
    """更新用户持久化上下文"""
    user_id = config["configurable"]["user_id"]
    
    # 更新对话摘要
    new_summary = summarize_conversation(state["messages"])
    store.put(("users", user_id), "history_summary", {"text": new_summary})
    
    return state
```

---

## 记忆写入与遗忘同样重要

不要把所有检索文本和模型结论自动存成用户事实。建议每条长期记忆保存内容、来源、主体、写入时间、有效期和状态；用户明确更新偏好时使旧值失效，矛盾证据未解决时保留冲突，避免最后一次模型猜测覆盖事实。

记忆的访问过滤必须先于语义检索结果注入：租户、用户与资源权限由应用提供。删除一条记忆时，要同时处理结构化存储、向量索引、摘要和相关缓存，否则“已忘记”的事实仍可能被召回。

验收使用跨用户同名资料、偏好更新、事实过期、摘要后继续任务、进程重启五组案例。分别测正确召回、错误召回和敏感信息跨租户泄漏；更长的历史和更多记忆并不自动提升回答质量。

## 长任务恢复：摘要之外还要保存什么

**核验范围：2026-10-08，长任务状态设计；下文框架示例未逐一做当前 SDK 集成测试。** 长上下文解决一次调用能读取多少信息，持久状态解决重启后系统知道什么已发生。Anthropic 的长任务实践用需求清单、进度文件和环境检查跨窗口续接，并将其定位为特定编码场景的工程方案，不能据此断言多 Agent 一定优于单 Agent。[原始工程实践](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

| 信息 | 应保存的证据 | 恢复时怎样处理 |
| --- | --- | --- |
| 用户目标与硬约束 | 原始请求引用、最新修订、范围 | 新指令覆盖旧摘要；冲突需显式解决 |
| 已完成工作 | 产物路径/版本与验证结果 | 检查产物仍存在、验证对应当前版本 |
| 未完成工作 | 下一个可执行步骤、阻塞依赖 | 从真实环境重新确认，不能只信“已完成”文字 |
| 外部副作用 | 操作 ID、幂等键、最终回执 | 未知状态先查询，避免重复创建或发送 |
| 授权与凭据 | 独立授权记录和到期时间 | 恢复时重新校验有效性，不把摘要当凭证 |

一个有效的恢复测试是：在写入工具已成功、模型尚未收到结果时终止进程，然后重启。合格系统应关联已有操作并继续汇报；若只恢复聊天摘要，可能重复提交。另在摘要压缩前后各问一次“当前禁止做什么、哪项尚未验证”，检查否定约束和证据引用是否丢失。

模型供应商的 compaction 可减少历史 token，但不取代这些应用记录；具体接口和返回项处理见 [上下文压缩](/llms/prompt/context#接口压缩与应用状态)。

## 📊 记忆策略选择

### 决策流程图

```
开始
  │
  ▼
需要跨对话记忆？
  │
  ├── 否 ──→ 使用滑动窗口/摘要记忆
  │
  └── 是
        │
        ▼
    需要语义检索？
        │
        ├── 否 ──→ 使用KV存储（Redis/PostgreSQL）
        │
        └── 是 ──→ 使用向量数据库（Chroma/Pinecone）
```

### 策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **滑动窗口** | 简单高效 | 丢失早期信息 | 短对话 |
| **摘要压缩** | 保留关键信息 | 需要额外LLM调用 | 长对话 |
| **KV存储** | 快速精确 | 无语义理解 | 结构化数据 |
| **向量存储** | 语义召回 | 成本较高 | 知识密集型 |
| **混合策略** | 兼顾多种需求 | 实现复杂 | 生产环境 |

---

## 🔗 相关阅读

- [Agent概述](/llms/agent/) - 了解Agent整体架构
- [规划与推理](/llms/agent/planning) - 任务状态管理
- [多智能体](/llms/agent/multi-agent) - 多Agent状态共享

> **相关文章**：
> - [精通状态智能体：LangGraph内存机制综合指南](https://dd-ff.blog.csdn.net/article/details/151118407)
> - [LangGraph时间旅行深度解析](https://dd-ff.blog.csdn.net/article/details/151151727)
> - [LangGraph上下文工程权威指南](https://dd-ff.blog.csdn.net/article/details/151118698)
> - [AutoGen状态管理实战](https://dd-ff.blog.csdn.net/article/details/149097602)
> - [构建弹性AI代理：LangGraph中的持久化](https://dd-ff.blog.csdn.net/article/details/151113741)

> **外部资源**：
> - [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
> - [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
