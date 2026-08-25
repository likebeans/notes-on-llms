---
title: 上下文工程
description: Context Engineering - 从提示词到上下文管理
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

# 上下文工程

> 管理AI的"工作记忆"——从"优化单个指令"到"设计整个信息环境"的范式转变

## 🎯 核心概念

> 来源：[LangGraph上下文工程权威指南](https://dd-ff.blog.csdn.net/article/details/151118698)、[从指令到智能：提示词与上下文工程](https://dd-ff.blog.csdn.net/article/details/152799914)

### 什么是上下文工程？

::: tip 定义
**上下文工程（Context Engineering）** 是一门系统性设计、构建和管理"模型推理期间提供给LLM的完整信息负载"的学科。核心是构建动态系统，在恰当时间、以正确格式，为模型提供完成任务所需的全部信息。
:::

### 为什么需要上下文工程？

随着业界从简单聊天机器人转向复杂、可靠、可扩展的AI系统，单纯提示词工程显现局限性：

| 挑战 | 提示词工程 | 上下文工程 |
|------|------------|------------|
| **知识时效** | 依赖模型训练数据 | 动态接入实时知识源 |
| **长对话** | 上下文窗口易溢出 | 记忆管理与压缩 |
| **工具使用** | 无法交互外部世界 | 集成API、数据库、代码执行 |
| **个性化** | 无跨会话记忆 | 持久化用户偏好 |

**核心思维转变**：
- **从**："你问了什么"（提示词考古学）
- **到**："当你提问时，模型知道什么"（信息环境设计）

### 提示词工程 vs 上下文工程

| 维度 | 提示词工程 | 上下文工程 |
|------|------------|------------|
| **范围** | 单次提示文本 | 完整输入环境 |
| **动态性** | 相对静态 | 高度动态 |
| **组成** | 指令+示例 | 指令+检索+记忆+工具+状态 |
| **复杂度** | 中等 | 高 |

---

## 📊 上下文三层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      上下文三层架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  静态上下文 (Static)                                         │ │
│  │  - 系统提示词、角色定义、规则约束                              │ │
│  │  - 编译时确定，运行时不变                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  动态上下文 (Dynamic)                                        │ │
│  │  - 对话历史、检索结果、工具输出                               │ │
│  │  - 运行时组装，会话内变化                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  持久化上下文 (Persistent)                                   │ │
│  │  - 用户偏好、长期记忆、知识库                                 │ │
│  │  - 跨会话保存                                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 动态上下文组装

### 上下文组装器

```python
class ContextAssembler:
    """动态上下文组装器"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.components = []
    
    def add_system_prompt(self, prompt: str, priority: int = 100):
        """添加系统提示（高优先级）"""
        self.components.append({
            "type": "system",
            "content": prompt,
            "priority": priority
        })
    
    def add_retrieved_docs(self, docs: list, priority: int = 80):
        """添加检索文档"""
        content = "\n\n".join([f"[文档{i+1}] {doc}" for i, doc in enumerate(docs)])
        self.components.append({
            "type": "retrieval",
            "content": f"相关资料：\n{content}",
            "priority": priority
        })
    
    def add_conversation_history(self, history: list, priority: int = 60):
        """添加对话历史"""
        content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        self.components.append({
            "type": "history",
            "content": f"对话历史：\n{content}",
            "priority": priority
        })
    
    def add_user_memory(self, memory: dict, priority: int = 70):
        """添加用户记忆"""
        content = "\n".join([f"- {k}: {v}" for k, v in memory.items()])
        self.components.append({
            "type": "memory",
            "content": f"用户信息：\n{content}",
            "priority": priority
        })
    
    def assemble(self) -> str:
        """组装最终上下文"""
        # 按优先级排序
        sorted_components = sorted(
            self.components, 
            key=lambda x: x["priority"], 
            reverse=True
        )
        
        # Token预算分配
        result = []
        current_tokens = 0
        
        for comp in sorted_components:
            comp_tokens = count_tokens(comp["content"])
            if current_tokens + comp_tokens <= self.max_tokens:
                result.append(comp["content"])
                current_tokens += comp_tokens
        
        return "\n\n---\n\n".join(result)
```

### 使用示例

```python
assembler = ContextAssembler(max_tokens=4000)

# 静态上下文
assembler.add_system_prompt("你是一个专业的技术顾问")

# 动态上下文
assembler.add_retrieved_docs(search_results)
assembler.add_conversation_history(chat_history[-10:])

# 持久化上下文
assembler.add_user_memory({"偏好": "简洁回答", "专业": "Python"})

# 组装
final_context = assembler.assemble()
```

---

## 📝 对话历史管理

### 滑动窗口策略

```python
def sliding_window(history: list, max_messages: int = 10) -> list:
    """保留最近N轮对话"""
    return history[-max_messages:]
```

### 摘要压缩策略

```python
async def summarize_history(history: list, llm) -> str:
    """将历史对话压缩为摘要"""
    if len(history) <= 5:
        return format_messages(history)
    
    # 早期对话生成摘要
    early_history = history[:-5]
    recent_history = history[-5:]
    
    summary = await llm.generate(f"""
请将以下对话历史压缩为简洁摘要：

{format_messages(early_history)}

摘要要求：保留关键信息和用户意图
""")
    
    return f"[历史摘要] {summary}\n\n[最近对话]\n{format_messages(recent_history)}"
```

### Token预算管理

```python
class TokenBudgetManager:
    """Token预算管理"""
    
    def __init__(self, total_budget: int = 8000):
        self.total = total_budget
        self.allocations = {
            "system": 500,      # 系统提示
            "retrieval": 2000,  # 检索内容
            "history": 2000,    # 对话历史
            "memory": 500,      # 用户记忆
            "response": 3000    # 预留给响应
        }
    
    def allocate(self, component: str, content: str) -> str:
        """分配Token并裁剪"""
        budget = self.allocations.get(component, 500)
        tokens = count_tokens(content)
        
        if tokens <= budget:
            return content
        
        # 裁剪策略
        return truncate_to_tokens(content, budget)
```

---

## 🛠️ 系统提示设计

### 系统提示的作用

**系统提示（System Prompt）** 是上下文工程的基石，为模型在整个会话期间设定总体角色、行为准则、约束条件和目标。

```python
# 客户服务机器人系统提示示例
SYSTEM_PROMPT = """
你是一家名为'ACME公司'的客户服务助理。

## 行为准则
- 语气始终保持礼貌和共情
- 优先使用提供的文档上下文回答
- 禁止透露个人身份信息（PII）

## 能力边界
- 可以：查询订单状态、解答产品问题、处理退换货
- 不可以：修改用户账户、处理支付信息

## 响应格式
- 简洁明了，控制在3段以内
- 需要更多信息时，主动询问用户
"""
```

### 系统提示最佳实践

| 原则 | 说明 |
|------|------|
| **清晰直接** | 语言处于"恰当高度"——足够具体以引导行为，保持灵活以适应不同情况 |
| **结构化组织** | 用XML标签或Markdown标题组织不同部分（如`<role>`、`<constraints>`） |
| **避免过度拟合** | 不要硬编码过多逻辑，保留模型推理空间 |
| **分隔符分离** | 用`###`或`---`区分指令与上下文信息 |

---

## 🔍 检索增强（RAG）

### 检索上下文注入

```python
async def inject_retrieval_context(
    query: str,
    retriever,
    reranker=None,
    top_k: int = 5
) -> str:
    """注入检索上下文"""
    
    # 检索
    docs = await retriever.search(query, top_k=top_k * 2)
    
    # 重排序（可选）
    if reranker:
        docs = reranker.rerank(query, docs, top_k=top_k)
    else:
        docs = docs[:top_k]
    
    # 格式化
    context = "以下是相关参考资料：\n\n"
    for i, doc in enumerate(docs):
        context += f"[来源{i+1}] {doc.title}\n{doc.content}\n\n"
    
    return context
```

### 查询改写

```python
async def rewrite_query(original_query: str, history: list, llm) -> str:
    """基于历史改写查询"""
    
    prompt = f"""
根据对话历史，改写用户查询使其更完整：

对话历史：
{format_messages(history[-3:])}

原始查询：{original_query}

改写后的查询（保持原意，补充上下文）：
"""
    return await llm.generate(prompt)
```

---

## 💾 长期记忆

### 记忆存储

```python
class MemoryStore:
    """长期记忆存储"""
    
    def __init__(self, vector_db):
        self.vector_db = vector_db
    
    async def save_memory(self, user_id: str, memory: dict):
        """保存记忆"""
        embedding = await embed(memory["content"])
        await self.vector_db.upsert({
            "id": f"{user_id}_{memory['key']}",
            "embedding": embedding,
            "metadata": {
                "user_id": user_id,
                "type": memory["type"],
                "content": memory["content"],
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def recall_memories(
        self, 
        user_id: str, 
        query: str, 
        top_k: int = 5
    ) -> list:
        """检索相关记忆"""
        query_embedding = await embed(query)
        results = await self.vector_db.search(
            embedding=query_embedding,
            filter={"user_id": user_id},
            top_k=top_k
        )
        return results
```

---

## 🤖 工具集成与智能体行为

上下文工程使LLM超越单纯文本生成，进化为可与外部世界交互的**智能体（Agent）**。

### 工具集成工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                  工具集成工作流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 工具定义                                                │
│     在系统提示中描述可用工具、功能及参数格式                   │
│                         ↓                                   │
│  2. 模型决策                                                │
│     模型根据用户查询决定是否调用工具                          │
│                         ↓                                   │
│  3. 工具执行                                                │
│     系统执行工具调用，获取结果                                │
│                         ↓                                   │
│  4. 结果注入                                                │
│     将工具输出注入上下文，模型基于此生成最终回答               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 工具定义示例

```python
TOOLS_CONTEXT = """
## 可用工具

### search_database
- 功能：查询产品数据库
- 参数：{"query": "搜索关键词", "limit": 最大返回数量}
- 返回：产品列表

### send_email
- 功能：发送邮件通知
- 参数：{"to": "收件人", "subject": "主题", "body": "正文"}
- 返回：发送状态

### execute_code
- 功能：执行Python代码
- 参数：{"code": "Python代码字符串"}
- 返回：执行结果

当需要使用工具时，请使用以下格式：
<tool_call>{"name": "工具名", "params": {...}}</tool_call>
"""
```

---

## ⚠️ 上下文工程陷阱

### "迷失在中间"问题

LLM**倾向于忽略长上下文中中间部分的信息**，呈现"U形性能曲线"——开头和结尾的信息得到最多关注。

| 位置 | 关注度 | 建议 |
|------|--------|------|
| **开头** | 高 | 放置最重要的指令和约束 |
| **中间** | 低 | 放置次要参考信息 |
| **结尾** | 高 | 放置用户查询和核心任务 |

**缓解策略**：
- 将最关键信息放在上下文开头或结尾
- 对于RAG场景，将最相关文档放在首尾位置
- 使用结构化标记（XML标签）突出重要部分

### 信息过载

::: warning 注意
设计再精妙的提示词，若淹没在无关聊天记录或格式混乱的文档片段中，也无法发挥作用。
:::

**解决方案**：
- 实施严格的Token预算管理
- 对检索结果进行重排序和过滤
- 对话历史使用摘要压缩

---

## 🔗 相关阅读

- [基础提示技术](/llms/prompt/basics) - Zero-shot、Few-shot
- [高级提示技术](/llms/prompt/advanced) - ReAct、ToT
- [Agent记忆](/llms/agent/memory) - Agent记忆系统
- [提示工程全景](/llms/prompt/) - 完整技术架构

> **相关文章**：
> - [LangGraph上下文工程权威指南](https://dd-ff.blog.csdn.net/article/details/151118698)
> - [从指令到智能：提示词与上下文工程](https://dd-ff.blog.csdn.net/article/details/152799914)
> - [掌握AI推理：从"提示工程"到"推理架构"](https://dd-ff.blog.csdn.net/article/details/154479954)
