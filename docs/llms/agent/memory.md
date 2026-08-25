---
title: 记忆系统
description: Agent 记忆与状态管理 - 短期/长期记忆机制
pageType: article
module: agent
updated: '2025-12-27'
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

# 记忆系统

> 让AI拥有"记忆"，实现跨对话的持续智能

## 🎯 核心概念

### 为什么Agent需要记忆？

::: tip 核心问题
LLM本身是**无状态**的——每次调用都是独立的。没有记忆系统，Agent无法：
- 记住之前的对话内容
- 跟踪任务执行进度
- 学习用户偏好
- 在中断后恢复工作
:::

### 记忆类型

| 类型 | 作用域 | 生命周期 | 典型用途 |
|------|--------|----------|----------|
| **短期记忆** | 单次对话 | 会话结束即失效 | 对话上下文、中间结果 |
| **长期记忆** | 跨对话 | 持久化存储 | 用户偏好、历史知识 |
| **工作记忆** | 单次任务 | 任务完成即清理 | 任务状态、执行计划 |
| **情景记忆** | 特定场景 | 按需召回 | 过往对话摘要 |

---

## � Agentic Design Patterns 视角

> 来源：[Agentic Design Patterns - Memory Management](https://github.com/ginobefun/agentic-design-patterns-cn)

### 双组件记忆系统

标准解决方案是实现区分**短期与长期存储**的双组件记忆系统：

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
| **LangChain** | `ConversationBufferMemory` | 自动注入对话历史 |
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

## �📝 短期记忆（对话上下文）

### 基本实现

```python
class ConversationMemory:
    """基础对话记忆"""
    
    def __init__(self, max_tokens: int = 4000):
        self.messages = []
        self.max_tokens = max_tokens
    
    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """超出限制时裁剪早期消息"""
        while self._count_tokens() > self.max_tokens:
            # 保留系统消息，删除最早的用户/助手消息
            for i, msg in enumerate(self.messages):
                if msg["role"] != "system":
                    self.messages.pop(i)
                    break
    
    def get_messages(self) -> list:
        return self.messages.copy()
```

### 滑动窗口策略

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

LangGraph提供**短期记忆(Checkpointer)**和**长期记忆(Store)**两种机制：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# 短期记忆：通过thread_id追踪单次对话
checkpointer = MemorySaver()

# 长期记忆：通过user_id和namespace存储持久化数据
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

```python
# 获取对话历史的所有检查点
checkpoints = list(app.get_state_history(config))

# 回到之前的状态（时间旅行）
previous_state = checkpoints[2]  # 第3个检查点
app.update_state(config, previous_state.values)

# 从该点继续对话
result = app.invoke({"messages": [new_message]}, config)
```

### 向量化长期记忆

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

new_agent = AssistantAgent(name="assistant", ...)
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
            "preferences": preferences,
            "history": history_summary
        }
    }

def update_user_context(state: State, config: dict, store: BaseStore) -> State:
    """更新用户持久化上下文"""
    user_id = config["configurable"]["user_id"]
    
    # 更新对话摘要
    new_summary = summarize_conversation(state["messages"])
    store.put(("users", user_id), "history_summary", new_summary)
    
    return state
```

---

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
