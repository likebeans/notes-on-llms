---
title: MCP快速入门
description: 5分钟创建你的第一个MCP服务
---

# MCP快速入门

> 5分钟创建你的第一个MCP服务

## 🚀 环境准备

### FastMCP 版本说明

| 版本 | 说明 |
|------|------|
| **FastMCP 1.0** | 已并入官方 MCP Python SDK（2024年） |
| **FastMCP 2.0** | 活跃维护版本，功能更强大（推荐） |

### 安装依赖

```bash
# 方式1：使用官方SDK（包含FastMCP 1.0）
pip install mcp

# 方式2：使用FastMCP 2.0（推荐，功能更强大）
pip install fastmcp

# 或使用uv（更快）
uv add fastmcp
```

::: tip FastMCP 2.0 优势
FastMCP 2.0 超越了基本协议实现，提供了完整的MCP生态工具包：Client、代理、服务器组合、中间件等。
:::

### 依赖管理（uv推荐）

FastMCP 与 `uv` 深度集成，支持多种依赖管理方式：

```bash
# 指定Python版本
fastmcp run server.py --python 3.11

# 添加单个包
fastmcp run server.py --with pandas

# 添加多个包
fastmcp run server.py --with pandas --with numpy --with httpx

# 从requirements.txt安装
fastmcp run server.py --with-requirements requirements.txt

# 指定项目目录
fastmcp run server.py --project /path/to/project
```

### fastmcp.json 配置文件（推荐）

```json
{
  "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
  "source": {
    "path": "server.py",
    "entrypoint": "mcp"
  },
  "environment": {
    "type": "uv",
    "python": ">=3.10",
    "dependencies": ["pandas", "requests", "httpx"]
  }
}
```

使用配置文件运行：
```bash
fastmcp run fastmcp.json
```

---

## 📝 创建第一个MCP服务器

### 使用FastMCP（最简方式）

```python
# server.py
from fastmcp import FastMCP

# 创建服务器实例
mcp = FastMCP("my-first-mcp")

# 添加工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加
    
    Args:
        a: 第一个数字
        b: 第二个数字
    
    Returns:
        两数之和
    """
    return a + b

@mcp.tool()
def greet(name: str) -> str:
    """问候用户
    
    Args:
        name: 用户名称
    """
    return f"你好，{name}！欢迎使用MCP！"

# 运行服务器
if __name__ == "__main__":
    mcp.run()
```

### 运行服务器

```bash
python server.py
```

---

## 🧪 测试服务器

### 使用FastMCP客户端测试

```python
# test_client.py
import asyncio
from fastmcp import Client

async def main():
    # 连接到服务器
    async with Client("python server.py") as client:
        # 列出可用工具
        tools = await client.list_tools()
        print("可用工具:", [t.name for t in tools])
        
        # 调用工具
        result = await client.call_tool("add", {"a": 5, "b": 3})
        print(f"5 + 3 = {result}")
        
        result = await client.call_tool("greet", {"name": "开发者"})
        print(result)

asyncio.run(main())
```

### 使用MCP Inspector测试

```bash
# 安装Inspector
npx @modelcontextprotocol/inspector python server.py
```

---

## 🔌 集成到Claude Desktop

### 配置文件位置

| 系统 | 路径 |
|------|------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

### 配置示例

```json
{
  "mcpServers": {
    "my-first-mcp": {
      "command": "python",
      "args": ["D:/path/to/server.py"],
      "env": {}
    }
  }
}
```

### 重启Claude Desktop

配置完成后，重启Claude Desktop即可使用新添加的工具。

---

## 📦 添加更多功能

### 添加资源

```python
@mcp.resource("config://settings")
def get_settings() -> dict:
    """获取应用配置"""
    return {
        "theme": "dark",
        "language": "zh-CN",
        "version": "1.0.0"
    }

@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
```

### 添加提示模板

```python
@mcp.prompt()
def code_review(code: str, language: str = "python") -> str:
    """代码审查提示模板"""
    return f"""请审查以下{language}代码：

```{language}
{code}
```

请从以下方面进行评审：
1. 代码质量
2. 潜在bug
3. 性能问题
4. 改进建议"""
```

---

## 🎯 完整示例

```python
# complete_server.py
from fastmcp import FastMCP
import json
from datetime import datetime

mcp = FastMCP("complete-demo")

# ===== 工具 =====
@mcp.tool()
def calculate(expression: str) -> float:
    """安全计算数学表达式"""
    allowed = set('0123456789+-*/.() ')
    if not all(c in allowed for c in expression):
        raise ValueError("表达式包含非法字符")
    return eval(expression)

@mcp.tool()
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
async def fetch_url(url: str) -> str:
    """获取URL内容"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# ===== 资源 =====
@mcp.resource("app://info")
def app_info() -> dict:
    """应用信息"""
    return {
        "name": "Complete Demo",
        "version": "1.0.0",
        "author": "Developer"
    }

# ===== 提示模板 =====
@mcp.prompt()
def summarize(text: str, max_length: int = 100) -> str:
    """文本摘要模板"""
    return f"""请将以下文本总结为不超过{max_length}字的摘要：

{text}

摘要："""

if __name__ == "__main__":
    mcp.run()
```

---

## ⚠️ 常见问题

### 1. 服务器无法启动

```bash
# 检查Python路径
which python  # Linux/Mac
where python  # Windows

# 确保使用绝对路径
python /absolute/path/to/server.py
```

### 2. Claude Desktop未识别工具

- 确保配置文件JSON格式正确
- 检查command和args路径是否正确
- 重启Claude Desktop

### 3. 工具调用报错

```python
# 添加错误处理
@mcp.tool()
def safe_tool(param: str) -> str:
    try:
        # 业务逻辑
        return result
    except Exception as e:
        return f"错误: {str(e)}"
```

---

## 🔧 FastMCP 2.0 Client

FastMCP 2.0 不仅是 Server 框架，还提供了完整的 **Client** 实现：

```python
from fastmcp import Client

async def main():
    # 连接本地Server
    async with Client("python server.py") as client:
        # 调用工具
        result = await client.call_tool("add", {"a": 1, "b": 2})
        print(result)
    
    # 连接远程Server（HTTP/SSE）
    async with Client("http://localhost:8000/sse") as client:
        tools = await client.list_tools()
        print(tools)
```

### Client 支持的传输方式

| 传输 | 连接方式 | 适用场景 |
|------|----------|----------|
| **Stdio** | `Client("python server.py")` | 本地开发 |
| **SSE** | `Client("http://host:port/sse")` | 远程服务 |
| **Websocket** | `Client("ws://host:port/ws")` | 实时通信 |

---

---

## 🔧 安装到客户端

FastMCP 提供 `fastmcp install` 命令快速安装到各种客户端：

```bash
# 安装到 Claude Desktop
fastmcp install claude-desktop server.py

# 安装到 Cursor
fastmcp install cursor server.py

# 安装到 Claude Code
fastmcp install claude-code server.py

# 带依赖安装
fastmcp install claude-desktop server.py --with pandas --with requests

# 使用 fastmcp.json 安装
fastmcp install claude-desktop fastmcp.json

# 生成 MCP JSON 配置
fastmcp install mcp-json server.py --name "My Server"
```

### 支持的客户端

| 客户端 | 安装方式 |
|--------|----------|
| **Claude Desktop** | 直接修改配置文件 |
| **Claude Code** | 内置MCP管理系统 |
| **Cursor** | 通过deeplink确认 |
| **MCP JSON** | 生成标准JSON配置 |

---

## 🔗 下一步

- [核心概念](/llms/mcp/concepts) - 深入理解Tools/Resources/Prompts
- [实战项目](/llms/mcp/practice) - 完整可运行的示例
- [高级功能](/llms/mcp/advanced) - 中间件、认证、代理
- [MCP概述](/llms/mcp/) - 了解MCP全貌

> **外部资源**：
> - [FastMCP 官方文档](https://gofastmcp.com/)
> - [MCP 官方文档](https://modelcontextprotocol.io/)
