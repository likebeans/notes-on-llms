---
title: MCP快速入门
description: 固定 FastMCP 2.12.5，验证工具、资源、模板与 stdio 链路，区分协议发现和业务成功。
pageType: article
module: mcp
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - mcp
level: intermediate
prerequisites:
  - /llms/agent/tool-calling
reviewed: '2026-10-08'
reviewScope: 本轮对照新版协议边界；前轮已验证 FastMCP 2.12.5 内存与 stdio 示例
exampleStatus: partial
techVersion: FastMCP 2.12.5 内存/stdio 教学基线；新版 MCP 2026-07-28 仅文档对照
---

# MCP快速入门

这一页只做一件事：跑通一个可发现、可调用、能返回明确错误的本地 MCP 服务。先用程序化 Client 验证协议，再接入模型应用，便于区分服务端问题与模型选择工具的问题。

::: info 教学基线与当前规范
2026-10-08 已核验 MCP 2026-07-28 的新版生命周期，但本页继续固定 FastMCP 2.12.5，便于复现已验证的内存与 stdio 链路。这里的运行结果不证明支持新协议。迁移差异见 [核心概念](/llms/mcp/concepts#_2026-07-28-生命周期变化)；升级时同时记录客户端、服务端、依赖版本及线上实际协议版本。
:::

## 🚀 环境准备

### 先分清协议与 SDK

MCP 是协议，`fastmcp` 是一种 Python 实现。独立包的 `from fastmcp import FastMCP` 与官方 Python SDK 的导入路径、版本和生命周期不能混用。本例固定使用 **Python 3.11+、FastMCP 2.12.5**，作为可复现的教学基线，不代表最新版本或生产选型建议。升级时同时核对服务端、客户端和依赖锁文件。

```bash
mkdir mcp-notes-demo
cd mcp-notes-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'fastmcp==2.12.5'
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`。下面的文件都保存在 `mcp-notes-demo`。具体 API 可对照 [FastMCP v2 快速入门](https://gofastmcp.com/v2/getting-started/quickstart)。

## 📝 创建第一个MCP服务器

保存为 `server.py`。工具读取固定的演示数据，不访问本机文件，不需要模型密钥。

```python
from fastmcp import FastMCP

mcp = FastMCP("notes-demo")
NOTES = {
    "rag": "RAG 使用检索到的外部证据辅助生成。",
    "mcp": "MCP 定义模型应用与外部能力之间的协议接口。",
}

@mcp.tool
def lookup_note(topic: str) -> dict[str, str]:
    """Read a demo note. Supported topics: rag, mcp. No writes or network calls."""
    topic = topic.strip().lower()
    if topic not in NOTES:
        raise ValueError("topic must be rag or mcp")
    return {"topic": topic, "text": NOTES[topic]}

@mcp.resource("notes://topics")
def topics() -> str:
    return "rag, mcp"

@mcp.prompt
def explain_topic(topic: str) -> str:
    return f"查询主题 {topic} 的演示笔记，解释其含义，并标出笔记未覆盖的信息。"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## 🧪 测试服务器

保存为 `check_client.py`，通过内存传输验证接口。它会检查发现、正常调用、非法参数、资源读取和提示模板；不需要先启动另一个进程。

```python
import asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError
from server import mcp

async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools} == {"lookup_note"}
        result = await client.call_tool("lookup_note", {"topic": "rag"})
        assert not result.is_error
        assert "RAG" in result.content[0].text
        try:
            await client.call_tool("lookup_note", {"topic": "unknown"})
        except ToolError:
            pass
        else:
            raise AssertionError("unknown topic must fail")
        resources = await client.read_resource("notes://topics")
        assert "mcp" in resources[0].text
        prompt = await client.get_prompt("explain_topic", {"topic": "rag"})
        assert prompt.messages
    print("MCP smoke check passed")

asyncio.run(main())
```

```bash
python check_client.py
```

通过内存测试只证明协议接口与业务函数能协作，不证明远程鉴权或网络传输正常。接着把 `Client(mcp)` 改成 `Client("server.py")` 再运行一次，验证 stdio 子进程链路。客户端使用 `async with` 管理初始化与关闭，详见 [Client 文档](https://gofastmcp.com/v2/clients/client)。

## 🔌 集成到模型应用

支持启动本地 MCP 子进程的 Host，通常需要配置可执行文件与参数。以下为常见 `mcpServers` 配置形状，具体存放位置和字段以所用 Host 文档为准，它不是 MCP 协议规定的统一配置文件。

```json
{
  "mcpServers": {
    "notes-demo": {
      "command": "/absolute/path/mcp-notes-demo/.venv/bin/python",
      "args": ["/absolute/path/mcp-notes-demo/server.py"]
    }
  }
}
```

替换为本机绝对路径。Host 启动服务后，先检查是否发现 `lookup_note`，再问“查询 rag 笔记”。如果工具能被手动调用但模型未选择它，应检查工具描述和 Host 暴露策略，而不是反复改传输配置。

## 📦 添加更多功能

本例同时展示三种原语：`lookup_note` 是工具，`notes://topics` 是只读资源，`explain_topic` 是可选提示模板。它们不会自动串联：Host 决定是否读取资源、如何选择模板，以及何时把工具交给模型。概念与职责见 [核心概念](/llms/mcp/concepts)。

需要远程连接时，可在另一个终端启动本机 HTTP 演示：

```bash
fastmcp run server.py:mcp --transport http --host 127.0.0.1 --port 8000
```

然后让 Client 连接 `http://127.0.0.1:8000/mcp`。这里的 `http` 是 FastMCP 对 Streamable HTTP 的参数名；生产部署还需身份、权限、TLS 和运行隔离。不要把这个无鉴权演示直接暴露到公网。参见 [运行服务](https://gofastmcp.com/v2/deployment/running-server) 与 [高级功能](/llms/mcp/advanced)。

## ⚠️ 常见问题

| 现象 | 优先检查 | 验证方法 |
| --- | --- | --- |
| Host 无法启动进程 | Python 路径、虚拟环境、文件路径 | 在相同环境运行指定命令 |
| stdio 无法解析消息 | 是否向 stdout 写了日志 | 日志改写 stderr；stdout 只承载协议 |
| 找不到工具 | 初始化失败、工具未注册、Host 缓存 | 先运行 `list_tools()` |
| 参数错误 | 名称、类型、枚举、业务范围 | 对照发现结果里的 schema |
| 内存测试通过而远程失败 | URL、传输、网络、鉴权 | 分开验证 stdio 与 HTTP |
| 服务在终端等待 | stdio 正在等客户端请求 | 让 Client 启动服务，而非期待聊天界面 |

## 🎯 验收与下一步

完成后应留下：依赖版本、服务源码、测试输出和一条失败调用。能够说明“工具发现成功”和“任务完成成功”的区别，再进入 [实践示例](/llms/mcp/practice)。
