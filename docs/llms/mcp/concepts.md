---
title: MCP核心概念
description: Tools、Resources、Prompts深度解析
---

# MCP核心概念

> 理解MCP的三大支柱

## 🔧 Tools（工具）

### 概念

**工具**是MCP中最核心的概念，允许AI模型执行具体操作。

```python
from fastmcp import FastMCP

mcp = FastMCP("tools-demo")

@mcp.tool()
def search_database(query: str, limit: int = 10) -> list:
    """搜索数据库
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量上限
    
    Returns:
        匹配的记录列表
    """
    # 实现搜索逻辑
    results = db.search(query, limit=limit)
    return results
```

### 工具设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **单一职责** | 一个工具做一件事 | `search_users` 而非 `manage_users` |
| **清晰命名** | 动词+名词 | `get_weather`, `send_email` |
| **详细描述** | docstring要详尽 | 包含参数说明和返回值 |
| **类型标注** | 使用类型注解 | `def add(a: int, b: int) -> int` |

### 同步与异步工具

```python
# 同步工具 - 简单操作
@mcp.tool()
def calculate(expression: str) -> float:
    return eval(expression)

# 异步工具 - I/O操作
@mcp.tool()
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### 复杂参数类型

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskInput(BaseModel):
    title: str = Field(..., description="任务标题")
    description: Optional[str] = Field(None, description="任务描述")
    priority: Priority = Field(Priority.MEDIUM, description="优先级")
    tags: list[str] = Field(default_factory=list, description="标签列表")

@mcp.tool()
def create_task(task: TaskInput) -> dict:
    """创建新任务"""
    return {
        "id": generate_id(),
        "title": task.title,
        "description": task.description,
        "priority": task.priority.value,
        "tags": task.tags,
        "created_at": datetime.now().isoformat()
    }
```

---

## 📁 Resources（资源）

### 概念

**资源**提供结构化数据访问，类似REST API的GET端点。

```python
# 静态资源
@mcp.resource("config://app")
def get_app_config() -> dict:
    """获取应用配置"""
    return {
        "name": "MyApp",
        "version": "1.0.0",
        "environment": "production"
    }

# 动态资源（带参数）
@mcp.resource("user://{user_id}")
def get_user(user_id: str) -> dict:
    """获取用户信息"""
    return db.get_user(user_id)

# 文件资源
@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, 'r') as f:
        return f.read()
```

### 资源URI设计

| 模式 | 示例 | 说明 |
|------|------|------|
| **协议前缀** | `config://settings` | 类型标识 |
| **路径参数** | `user://{id}` | 动态资源 |
| **文件路径** | `file://path/to/file` | 文件访问 |
| **数据库** | `db://table/{id}` | 数据库记录 |

### 资源与工具的区别

| 特性 | Resources | Tools |
|------|-----------|-------|
| **操作类型** | 只读 | 读写 |
| **副作用** | 无 | 可能有 |
| **幂等性** | 是 | 不一定 |
| **用途** | 获取数据 | 执行操作 |

---

## 📝 Prompts（提示模板）

### 概念

**提示模板**是预定义的提示词，引导AI完成特定任务。

```python
@mcp.prompt()
def code_review(code: str, language: str = "python") -> str:
    """代码审查模板"""
    return f"""请审查以下{language}代码：

```{language}
{code}
```

请从以下方面进行评审：
1. **代码质量**：可读性、命名规范、代码结构
2. **潜在问题**：bug、安全漏洞、边界情况
3. **性能**：时间复杂度、空间复杂度
4. **改进建议**：具体的优化方案

输出格式：
- 使用Markdown格式
- 问题按严重程度排序
- 提供具体的修改建议"""
```

### 复杂提示模板

```python
@mcp.prompt()
def data_analysis(
    data_description: str,
    analysis_goal: str,
    output_format: str = "report"
) -> str:
    """数据分析模板"""
    
    format_instructions = {
        "report": "生成详细的分析报告",
        "summary": "生成简洁的摘要",
        "json": "以JSON格式输出结果"
    }
    
    return f"""# 数据分析任务

## 数据描述
{data_description}

## 分析目标
{analysis_goal}

## 分析要求
1. 数据质量检查
2. 描述性统计
3. 趋势分析
4. 异常检测
5. 结论与建议

## 输出要求
{format_instructions.get(output_format, format_instructions["report"])}"""
```

### 带上下文的提示

```python
@mcp.prompt()
async def contextual_qa(question: str) -> str:
    """带上下文的问答模板"""
    # 获取相关上下文
    context = await retrieve_relevant_docs(question)
    
    return f"""基于以下上下文回答问题：

## 上下文
{context}

## 问题
{question}

## 回答要求
1. 仅基于提供的上下文回答
2. 如果上下文不足，说明无法回答
3. 引用来源"""
```

---

## 🔄 三者协作

### 典型工作流

```
用户请求
    │
    ▼
┌─────────────────┐
│  读取资源       │  ← Resources
│  获取上下文     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  应用提示模板   │  ← Prompts
│  构建完整提示   │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  AI处理         │
│  决定行动       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  调用工具       │  ← Tools
│  执行操作       │
└─────────────────┘
    │
    ▼
  返回结果
```

### 完整示例

```python
from fastmcp import FastMCP

mcp = FastMCP("integrated-demo")

# 资源：提供数据
@mcp.resource("customer://{customer_id}")
async def get_customer(customer_id: str) -> dict:
    return await db.get_customer(customer_id)

# 工具：执行操作
@mcp.tool()
async def send_email(to: str, subject: str, body: str) -> bool:
    """发送邮件"""
    return await email_service.send(to, subject, body)

@mcp.tool()
async def create_ticket(customer_id: str, issue: str) -> dict:
    """创建工单"""
    return await ticket_service.create(customer_id, issue)

# 提示：引导AI
@mcp.prompt()
def customer_support(customer_info: str, issue: str) -> str:
    """客户支持模板"""
    return f"""你是客户支持专家。

客户信息：
{customer_info}

客户问题：
{issue}

请：
1. 分析问题
2. 提供解决方案
3. 如需要，使用工具执行操作"""
```

---

## 🔗 相关阅读

- [MCP快速入门](/mcp/quickstart) - 5分钟创建服务
- [高级功能](/mcp/advanced) - 中间件、认证
- [MCP概述](/mcp/) - 协议全貌
