---
title: MCP协议知识体系
description: Model Context Protocol - AI工具调用的标准协议
---

# MCP协议知识体系

> 连接AI模型与外部世界的标准协议

## 🎯 什么是MCP？

### 核心定义

::: tip 定义
**MCP（Model Context Protocol）** 是由Anthropic提出的开放协议，用于标准化AI模型与外部工具、数据源之间的通信方式，类似于AI领域的"USB协议"。
:::

### 为什么需要MCP？

| 问题 | 传统方案 | MCP方案 |
|------|----------|---------|
| **工具集成** | 每个工具单独适配 | 统一协议接入 |
| **多模型支持** | 重复开发 | 一次开发，多处复用 |
| **安全控制** | 各自实现 | 协议层标准化 |
| **生态互通** | 碎片化 | 标准化生态 |

---

## 🏗️ MCP架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP 架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐         ┌─────────────┐         ┌───────────┐ │
│   │  MCP Client │ ◄─────► │  MCP Server │ ◄─────► │  外部服务  │ │
│   │  (AI模型)   │   协议   │  (工具提供) │   实现   │  (API等)  │ │
│   └─────────────┘         └─────────────┘         └───────────┘ │
│                                                                  │
│   传输层: stdio | HTTP/SSE | WebSocket                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 角色 | 说明 |
|------|------|------|
| **MCP Client** | 请求方 | AI应用/模型，发起工具调用 |
| **MCP Server** | 提供方 | 暴露工具、资源、提示模板 |
| **Transport** | 传输层 | stdio/HTTP/SSE等通信方式 |

---

## 🔧 三大核心能力

### 1. Tools（工具）

允许模型执行操作，如API调用、数据处理等。

```python
from mcp.server import Server
from mcp.types import Tool

server = Server("my-server")

@server.tool()
async def search_web(query: str) -> str:
    """搜索网络信息
    
    Args:
        query: 搜索关键词
    """
    # 实现搜索逻辑
    results = await perform_search(query)
    return results
```

### 2. Resources（资源）

提供结构化数据访问，如文件、数据库记录等。

```python
@server.resource("file://{path}")
async def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, 'r') as f:
        return f.read()

@server.resource("config://settings")
async def get_settings() -> dict:
    """获取配置信息"""
    return {"theme": "dark", "language": "zh"}
```

### 3. Prompts（提示模板）

预定义的提示词模板，引导模型行为。

```python
@server.prompt()
async def summarize_document(document: str) -> str:
    """文档摘要提示模板"""
    return f"""请对以下文档进行摘要：

{document}

要求：
- 保留关键信息
- 长度控制在200字以内
- 使用简洁的语言"""
```

---

## 🚀 快速开始

### 安装

```bash
# Python SDK
pip install mcp

# 或使用 FastMCP（更简单的封装）
pip install fastmcp
```

### 创建MCP服务器

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

# 创建服务器
server = Server("demo-server")

# 添加工具
@server.tool()
async def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b

@server.tool()
async def get_weather(city: str) -> str:
    """获取天气信息"""
    return f"{city}今日晴，气温25°C"

# 运行服务器
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 使用FastMCP（推荐）

```python
from fastmcp import FastMCP

# 创建服务器
mcp = FastMCP("my-service")

@mcp.tool()
def calculate(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

@mcp.resource("greeting://{name}")
def greet(name: str) -> str:
    """个性化问候"""
    return f"你好，{name}！"

# 运行
mcp.run()
```

---

## 📡 传输协议

| 协议 | 场景 | 特点 |
|------|------|------|
| **stdio** | 本地进程 | 简单、安全、无网络 |
| **HTTP/SSE** | Web服务 | 支持流式、跨域 |
| **WebSocket** | 实时交互 | 双向通信 |

### stdio模式

```python
# 服务器
from mcp.server.stdio import stdio_server

async with stdio_server() as (read, write):
    await server.run(read, write)
```

### HTTP/SSE模式

```python
from fastmcp import FastMCP

mcp = FastMCP("web-service")

# 使用HTTP传输
mcp.run(transport="sse", port=8000)
```

---

## 🔌 客户端集成

### Claude Desktop配置

```json
{
  "mcpServers": {
    "my-tools": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {
        "API_KEY": "xxx"
      }
    }
  }
}
```

### 编程方式调用

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()
            
            # 列出工具
            tools = await session.list_tools()
            print(tools)
            
            # 调用工具
            result = await session.call_tool("add", {"a": 1, "b": 2})
            print(result)  # 3
```

---

## 🛡️ 安全最佳实践

### 认证与授权

```python
from fastmcp import FastMCP
from fastmcp.auth import BearerAuth

mcp = FastMCP("secure-service")

# 配置Bearer Token认证
mcp.auth = BearerAuth(
    public_key="path/to/public_key.pem"
)

@mcp.tool()
async def sensitive_operation(ctx, data: str) -> str:
    # 检查权限
    if "admin" not in ctx.user.roles:
        raise PermissionError("需要管理员权限")
    return process(data)
```

### 输入验证

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(..., min_length=1, max_length=100)
    limit: int = Field(10, ge=1, le=100)

@mcp.tool()
def search(params: SearchParams) -> list:
    """安全的搜索工具"""
    return do_search(params.query, params.limit)
```

---

## 🌐 生态系统

### 主流MCP服务器

| 服务 | 功能 |
|------|------|
| **Filesystem** | 文件读写 |
| **GitHub** | 代码仓库操作 |
| **Slack** | 消息发送 |
| **PostgreSQL** | 数据库查询 |
| **Web Search** | 网络搜索 |

### 托管平台

| 平台 | 特点 |
|------|------|
| **Composio** | 200+预置工具 |
| **Zapier MCP** | 连接6000+应用 |
| **MCP.so** | 社区市场 |

---

## 🔗 章节导航

| 章节 | 内容 |
|------|------|
| [快速入门](/mcp/quickstart) | 5分钟创建MCP服务 |
| [核心概念](/mcp/concepts) | Tools/Resources/Prompts |
| [高级功能](/mcp/advanced) | 中间件、认证、代理 |
| [最佳实践](/mcp/best-practices) | 生产环境指南 |

---

## 🌐 外部资源

| 资源 | 说明 |
|------|------|
| [MCP官方文档](https://modelcontextprotocol.io/) | 协议规范 |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | Python实现 |
| [FastMCP](https://github.com/jlowin/fastmcp) | 简化封装 |
| [MCP服务器列表](https://github.com/modelcontextprotocol/servers) | 官方服务器集合 |
