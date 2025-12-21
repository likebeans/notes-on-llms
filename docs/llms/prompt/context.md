---
title: 上下文工程
description: Context Engineering - 从提示词到上下文管理
---

# 上下文工程

> 管理AI的"工作记忆"

## 🎯 核心概念

> 来源：[LangGraph上下文工程权威指南](https://dd-ff.blog.csdn.net/article/details/151118698)

### 什么是上下文工程？

::: tip 定义
**上下文工程（Context Engineering）** 是管理AI模型输入的整体信息环境的技术，包括动态组装提示词、知识检索、记忆管理和工具输出等。
:::

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

## 🔗 相关阅读

- [基础提示技术](/llms/prompt/basics) - Zero-shot、Few-shot
- [高级提示技术](/llms/prompt/advanced) - ReAct、ToT
- [Agent记忆](/llms/agent/memory) - Agent记忆系统

> **相关文章**：
> - [LangGraph上下文工程权威指南](https://dd-ff.blog.csdn.net/article/details/151118698)
> - [从指令到智能：提示词与上下文工程](https://dd-ff.blog.csdn.net/article/details/152799914)
