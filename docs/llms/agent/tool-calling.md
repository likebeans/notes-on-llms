---
title: 工具调用详解
description: Agent 工具调用机制 - Function Calling与MCP协议
---

# 工具调用详解

> 让AI从"聊天"变成"做事"的核心能力

## 🎯 核心概念

### 什么是工具调用？

::: tip 定义
**工具调用（Tool Calling）** 是让LLM能够识别用户意图，并生成结构化指令来调用外部函数或API的能力。它是AI Agent从"对话系统"进化为"行动系统"的关键技术。
:::

### 工具调用的价值

| 能力 | 无工具调用 | 有工具调用 |
|------|------------|------------|
| **数据获取** | 只能使用训练数据 | 可实时查询外部数据 |
| **计算能力** | 数学推理易出错 | 调用计算器精确计算 |
| **系统交互** | 无法操作系统 | 可发邮件、操作数据库 |
| **知识边界** | 受限于训练截止日期 | 可搜索最新信息 |

---

## 🔧 Function Calling详解

> 来源：[解码AI智能体的大脑：Function Calling与ReAct深度对决](https://dd-ff.blog.csdn.net/article/details/153210207)

### 工作流程

```
用户请求 → LLM分析 → 生成JSON指令 → 执行函数 → 返回结果 → LLM整合回复
```

### 五步执行流程

```python
# 1. 定义工具（函数描述）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 2. 发送请求，模型决策
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto"  # 让模型自主决定是否调用工具
)

# 3. 模型返回结构化调用指令
# response.choices[0].message.tool_calls[0]:
# {
#     "id": "call_abc123",
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "arguments": '{"city": "北京", "unit": "celsius"}'
#     }
# }

# 4. 执行函数
import json
args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
result = get_weather(**args)  # {"temperature": 25, "condition": "晴"}

# 5. 将结果返回给模型，生成最终回复
messages.append(response.choices[0].message)
messages.append({
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": json.dumps(result)
})

final_response = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)
# "北京今天天气晴朗，气温25摄氏度，适合外出活动。"
```

### 函数描述的重要性

::: warning 关键洞察
函数描述是告诉模型"如何理解用户输入"和"如何构造正确的函数调用"的关键信息。描述越清晰，调用越准确。
:::

```python
# 好的函数描述
{
    "name": "search_products",
    "description": "搜索商品。支持按名称、类别、价格范围筛选。返回匹配的商品列表。",
    "parameters": {
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，如商品名称或描述"
            },
            "category": {
                "type": "string",
                "enum": ["电子产品", "服装", "食品", "家居"],
                "description": "商品类别，可选"
            },
            "max_price": {
                "type": "number",
                "description": "最高价格限制，单位：元"
            }
        },
        "required": ["query"]
    }
}
```

### 并行工具调用

```python
# 模型可以一次返回多个工具调用
# 用户："北京和上海今天天气怎么样？"

# 模型返回：
tool_calls = [
    {"function": {"name": "get_weather", "arguments": '{"city": "北京"}'}},
    {"function": {"name": "get_weather", "arguments": '{"city": "上海"}'}}
]

# 并行执行
import asyncio

async def execute_tools(tool_calls):
    tasks = [execute_tool(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)
```

---

## 🔌 MCP协议（Model Context Protocol）

> 来源：[FastMCP快速入门指南](https://dd-ff.blog.csdn.net/article/details/148854073)

### 什么是MCP？

::: tip 定义
**MCP（Model Context Protocol）** 是一种标准化的AI模型通信协议，用于连接LLM与外部工具和数据源。它提供了统一的接口规范，使得工具开发更加标准化。
:::

### MCP vs Function Calling

| 特性 | Function Calling | MCP |
|------|------------------|-----|
| **标准化** | 各厂商实现不同 | 统一协议规范 |
| **工具发现** | 需手动定义 | 支持动态发现 |
| **传输方式** | HTTP/WebSocket | Stdio/SSE/HTTP |
| **状态管理** | 无内置支持 | 支持会话状态 |
| **生态系统** | 厂商锁定 | 跨平台通用 |

### FastMCP快速入门

```python
# 安装
# pip install fastmcp

from fastmcp import FastMCP

# 创建MCP服务器
mcp = FastMCP("天气服务")

# 定义工具
@mcp.tool()
def get_weather(city: str, unit: str = "celsius") -> dict:
    """获取指定城市的天气信息
    
    Args:
        city: 城市名称
        unit: 温度单位，celsius或fahrenheit
    
    Returns:
        包含温度和天气状况的字典
    """
    # 实际实现会调用天气API
    return {"city": city, "temperature": 25, "condition": "晴"}

@mcp.tool()
def search_news(query: str, limit: int = 5) -> list:
    """搜索新闻
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量
    
    Returns:
        新闻列表
    """
    return [{"title": f"关于{query}的新闻", "url": "..."}]

# 运行服务器
if __name__ == "__main__":
    mcp.run()
```

### MCP资源与提示模板

```python
# 定义资源（静态数据）
@mcp.resource("config://app")
def get_app_config() -> str:
    """获取应用配置"""
    return json.dumps({"version": "1.0", "env": "production"})

# 定义提示模板
@mcp.prompt()
def analyze_data(data_type: str) -> str:
    """生成数据分析提示词"""
    return f"""请分析以下{data_type}数据，提供：
    1. 主要趋势
    2. 异常点
    3. 建议措施"""
```

---

## 🛠️ OpenAI Agent工具

> 来源：[OpenAI Agent工具全面开发者指南](https://dd-ff.blog.csdn.net/article/details/154445828)

### 六种核心工具

| 工具 | 功能 | 适用场景 |
|------|------|----------|
| **file_search** | 托管式RAG | 知识库问答、文档分析 |
| **code_interpreter** | Python代码执行 | 数据分析、可视化 |
| **web_search** | 实时网络搜索 | 获取最新信息 |
| **computer_use** | 计算机操作 | 自动化任务 |
| **mcp** | MCP协议集成 | 连接外部服务 |
| **function** | 自定义函数 | 业务逻辑集成 |

### file_search：托管式RAG

```python
from openai import OpenAI

client = OpenAI()

# 1. 创建向量存储
vector_store = client.vector_stores.create(name="知识库")

# 2. 上传文件
file = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="assistants"
)

# 3. 添加到向量存储
client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file.id
)

# 4. 创建带file_search的Assistant
assistant = client.assistants.create(
    name="知识助手",
    model="gpt-4-turbo",
    tools=[{"type": "file_search"}],
    tool_resources={
        "file_search": {"vector_store_ids": [vector_store.id]}
    }
)
```

### code_interpreter：安全沙箱执行

```python
# 创建带代码解释器的Assistant
assistant = client.assistants.create(
    name="数据分析师",
    model="gpt-4-turbo",
    tools=[{"type": "code_interpreter"}],
    instructions="你是一个数据分析专家，使用Python进行数据处理和可视化。"
)

# 用户可以上传数据文件，Assistant会自动分析
# "请分析这份销售数据，生成趋势图"
```

---

## 🔒 工具调用安全

> 来源：[AI智能体的牢笼：大模型沙箱技术深度解析](https://dd-ff.blog.csdn.net/article/details/151970698)

### 安全风险

| 风险类型 | 描述 | 防护措施 |
|----------|------|----------|
| **提示注入** | 恶意输入触发危险操作 | 输入验证、权限隔离 |
| **数据泄露** | 敏感信息被工具暴露 | 数据脱敏、访问控制 |
| **资源滥用** | 无限循环消耗资源 | 执行超时、资源限制 |
| **系统破坏** | 恶意代码执行 | 沙箱隔离、只读权限 |

### 沙箱技术选型

| 技术 | 隔离级别 | 性能 | 适用场景 |
|------|----------|------|----------|
| **Docker** | 容器级 | 高 | 通用隔离 |
| **gVisor** | 内核级 | 中 | 高安全需求 |
| **Firecracker** | 微虚拟机 | 高 | 多租户环境 |
| **WebAssembly** | 字节码级 | 极高 | 轻量级隔离 |

### 安全最佳实践

```python
class SafeToolExecutor:
    """安全的工具执行器"""
    
    def __init__(self):
        self.allowed_tools = {"get_weather", "search_news"}
        self.max_execution_time = 30  # 秒
        self.rate_limiter = RateLimiter(max_calls=100, period=60)
    
    def execute(self, tool_name: str, arguments: dict) -> dict:
        # 1. 工具白名单检查
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"工具 {tool_name} 未授权")
        
        # 2. 参数验证
        self.validate_arguments(tool_name, arguments)
        
        # 3. 速率限制
        if not self.rate_limiter.allow():
            raise RateLimitError("调用频率超限")
        
        # 4. 超时执行
        with timeout(self.max_execution_time):
            result = self.tools[tool_name](**arguments)
        
        # 5. 输出过滤
        return self.filter_sensitive_data(result)
```

---

## 📊 LangGraph工具集成

> 来源：[精通LangGraph中的工具使用](https://dd-ff.blog.csdn.net/article/details/151148039)

### 工具定义与绑定

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 使用@tool装饰器定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    
    Returns:
        计算结果
    """
    return str(eval(expression))

@tool
def web_search(query: str) -> str:
    """搜索网络信息
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果摘要
    """
    # 实际实现调用搜索API
    return f"关于'{query}'的搜索结果..."

# 绑定工具到模型
tools = [calculator, web_search]
model_with_tools = model.bind_tools(tools)

# 创建ReAct Agent
agent = create_react_agent(model, tools)
```

### 自定义工具节点

```python
from langgraph.graph import StateGraph, END

def tool_node(state):
    """执行工具调用"""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_calls = last_message.tool_calls
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # 查找并执行工具
        tool = next(t for t in tools if t.name == tool_name)
        result = tool.invoke(tool_args)
        
        results.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call["id"]
        })
    
    return {"messages": results}

# 构建图
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue)
```

---

## 🔗 相关阅读

- [Agent概述](/llms/agent/) - 了解Agent整体架构
- [规划与推理](/llms/agent/planning) - ReAct循环详解
- [安全与沙箱](/llms/agent/safety) - 工具执行安全

> **相关文章**：
> - [解码AI智能体的大脑：Function Calling与ReAct深度对决](https://dd-ff.blog.csdn.net/article/details/153210207)
> - [OpenAI Agent工具全面开发者指南](https://dd-ff.blog.csdn.net/article/details/154445828)
> - [FastMCP快速入门指南](https://dd-ff.blog.csdn.net/article/details/148854073)
> - [function_call的流程和作用](https://dd-ff.blog.csdn.net/article/details/147471435)
> - [精通LangGraph中的工具使用](https://dd-ff.blog.csdn.net/article/details/151148039)

> **外部资源**：
> - [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
> - [MCP协议规范](https://modelcontextprotocol.io/)
> - [LangChain Tools文档](https://python.langchain.com/docs/modules/tools/)
