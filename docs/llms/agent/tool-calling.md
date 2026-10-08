---
title: 工具调用详解
description: Agent 工具调用机制 - Function Calling与MCP协议
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
reviewScope: Responses strict、调用结果关联；MCP 新旧版本边界；未调用真实模型 API
exampleStatus: not-run
techVersion: Responses 函数调用文档核验 2026-10-08；API 示例未实跑；FastMCP 2.12.5 接口示意
---

# 工具调用详解

> 让AI从"聊天"变成"做事"的核心能力

## 2026 阅读提示

工具调用不是让模型“更聪明”的魔法，而是让模型在受控边界内产生可执行动作。评估工具调用时，重点不只是模型能不能生成 JSON，而是整条链路是否安全、可恢复、可审计：

1. 工具 schema 是否足够窄，能拒绝未知参数、危险默认值和越权操作？
2. 工具结果是否会被二次校验，而不是直接相信外部系统或模型解释？
3. 失败时系统能否重试、降级、请求人工确认或安全停止？
4. 每一次调用是否留下 trace：用户目标、工具名、参数、结果、错误和最终回答？

如果一个 Agent 出问题，常见根因并不是“模型不会调用工具”，而是工具权限太宽、参数校验太松、完成条件不清、或错误恢复完全依赖模型临场发挥。读下面的流程时，建议把每一步都映射到日志和测试用例。

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

### 工具使用的六个步骤

> 来源：[Agentic Design Patterns - Tool Use](https://github.com/ginobefun/agentic-design-patterns-cn)

```
① 工具定义 → ② LLM决策 → ③ 生成调用 → ④ 工具执行 → ⑤ 结果返回 → ⑥ LLM处理
```

| 步骤 | 说明 |
|------|------|
| **① 工具定义** | 向LLM描述可用工具：名称、用途、参数类型和说明 |
| **② LLM决策** | LLM接收用户请求和工具定义，判断是否需要调用工具 |
| **③ 生成调用** | LLM生成结构化输出（JSON）：工具名称 + 参数 |
| **④ 工具执行** | 框架捕获调用请求，执行实际的外部函数 |
| **⑤ 结果返回** | 工具执行结果返回给智能体 |
| **⑥ LLM处理** | LLM用工具结果生成最终回复，或决定下一步行动 |

### 函数调用 vs 工具调用

| 术语 | 范围 | 说明 |
|------|------|------|
| **函数调用** | 狭义 | 调用预定义的代码函数 |
| **工具调用** | 广义 | 包含函数、API、数据库、甚至其他智能体 |

工具可以是：传统函数、复杂API、数据库查询、代码执行器、其他智能体、外部知识库等。

### 六大应用场景

| 场景 | 工具类型 | 典型流程 |
|------|----------|----------|
| **获取实时信息** | 天气API、股票API | 用户问天气 → 调用API → 返回数据 |
| **数据库交互** | 库存API、订单API | 查询库存 → 执行SQL → 返回结果 |
| **精确计算** | 计算器、数据分析库 | 获取数据 → 执行计算 → 返回结果 |
| **发送通知** | 邮件API、消息API | 提取信息 → 调用API → 发送成功 |
| **执行代码** | Python解释器 | 接收代码 → 沙箱执行 → 返回输出 |
| **控制设备** | IoT控制API | 解析指令 → 调用设备API → 确认执行 |

---

## 🔧 Function Calling详解

> 来源：[解码AI智能体的大脑：Function Calling与ReAct深度对决](https://dd-ff.blog.csdn.net/article/details/153210207)

### 工作流程

```
用户请求 → LLM分析 → 生成JSON指令 → 执行函数 → 返回结果 → LLM整合回复
```

### 五步执行流程

以下示例使用 Responses API 展示“发送工具定义 → 收到调用 → 应用执行 → 回传结果 → 模型继续”。运行前需安装兼容版本的 `openai`，设置 `OPENAI_API_KEY` 和支持函数调用的 `OPENAI_MODEL`。天气函数**固定返回模拟数据**，示例不连接真实天气服务，也未在本页执行 API 测试。

```python
import json
import os
from openai import OpenAI

client = OpenAI()
model = os.environ["OPENAI_MODEL"]

def get_weather(city, unit):
    return {"city": city, "unit": unit, "temperature": 25,
            "condition": "晴", "simulated": True}

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "返回指定城市的模拟摄氏天气，仅用于接口演示。",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius"]}
        },
        "required": ["city", "unit"],
        "additionalProperties": False
    }
}]
items = [{"role": "user", "content": "演示查询北京天气，注明数据为模拟。"}]
for _ in range(4):
    response = client.responses.create(model=model, input=items, tools=tools)
    # 保留所有返回 item，包括模型继续运行所需的上下文项。
    items.extend(response.output)
    calls = [item for item in response.output if item.type == "function_call"]
    if not calls:
        print(response.output_text)
        break
    for call in calls:
        try:
            args = json.loads(call.arguments)
            if call.name != "get_weather" or not isinstance(args, dict):
                raise ValueError("未知工具或参数结构")
            if set(args) != {"city", "unit"}:
                raise ValueError("参数字段不符合约定")
            if not isinstance(args["city"], str) or not 1 <= len(args["city"]) <= 100:
                raise ValueError("城市名称无效")
            if args["unit"] != "celsius":
                raise ValueError("不支持的单位")
            result = get_weather(**args)
        except (ValueError, TypeError) as exc:
            result = {"error": str(exc)}
        items.append({"type": "function_call_output", "call_id": call.call_id,
                      "output": json.dumps(result, ensure_ascii=False)})
else:
    raise RuntimeError("达到模型轮数上限，尚未得到最终回答")
```

严格 schema 约束的是参数形状，不验证业务事实、用户身份或权限；真实执行器仍要检查资源范围、幂等性、超时和输出大小。模型可能返回零个、一个或多个调用，结果必须对应原 `call_id`，不能写死示例 ID。Responses 与 Chat Completions 的字段层级不同，迁移时不可只替换方法名。[官方函数调用文档](https://developers.openai.com/api/docs/guides/function-calling)

### 严格参数与执行状态（2026-10-08 核验）

当前 Responses 文档说明：省略 `strict` 时会尝试转换成严格模式，不兼容时可能回退为 best effort；Chat Completions 默认行为不同。因此可复现接口应显式写 `strict: true`，每层对象关闭额外字段，所有属性列入 `required`，可空值用 `null` 联合类型表达。不能把“请求成功”当成严格约束确已生效。[Function calling 官方约束](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)

上例的轮数限制是教学停止条件。真实写操作还应持久化 `operation_id`、幂等键、参数摘要和执行状态：`pending → running → succeeded / failed / unknown`。网络超时落到 `unknown` 时，先查外部操作记录；不要让模型换一个调用 ID 重新提交同一笔动作。`call_id` 关联模型消息，业务幂等键关联真实副作用，两者职责不同。

验收时至少注入三个故障：合法 JSON 但无资源权限、上游已提交但响应丢失、模型返回多个有先后依赖的调用。期望分别是拒绝执行、查状态后恢复、按依赖顺序调度；不能用并发或自动重试掩盖这些分支。

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

下例是独立于厂商 SDK 的概念示意，`execute_tool` 需实现。仅并行执行彼此独立的调用；对同一记录写入或有前置依赖的调用，按顺序执行。`gather` 也不自带并发上限，取消与部分失败策略见[并行化](/llms/agent/parallelization)。

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

| 比较项 | Function Calling | MCP |
| --- | --- | --- |
| 所处层次 | 模型 API 表达工具定义与调用意图 | 应用与工具/资源服务之间的协议 |
| 工具定义 | 由应用提交给模型，格式依厂商 API | 客户端可向服务器发现工具，再适配给模型 |
| 执行方 | 自定义函数通常由应用执行 | MCP 服务器执行工具，客户端接收结果 |
| 能否组合 | 可以把 MCP 工具转换成模型工具定义 | 不能替代模型的选择能力或业务授权 |

本节 FastMCP 代码只演示工具接口风格；可复现实验统一见 [MCP 快速入门](/llms/mcp/quickstart)。截至 2026-10-08 已核对的 MCP 2026-07-28 规范采用逐请求版本与能力元数据，旧版初始化握手属于兼容流程；不能通过更换版本字符串让旧 SDK 自动支持新生命周期。新版变化与固定教学基线见 [MCP 版本边界](/llms/mcp/#版本与边界)。

### FastMCP快速入门

以下是 `fastmcp` 包接口风格的演示，需要固定安装版本后验证。天气和新闻均为桩数据；启动成功只说明服务能够暴露工具，不代表接入了真实外部数据。

```python
# 安装
# pip install fastmcp==2.12.5

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
    return [{"title": f"关于{query}的模拟新闻", "simulated": True}]

# 运行服务器
if __name__ == "__main__":
    mcp.run()
```

### MCP资源与提示模板

```python
# 定义资源（本例返回固定配置；资源也可以是动态读取）
import json
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

下表列常见能力类别，不是完整且永久不变的工具清单；具体工具名、模型支持和参数以选用 API 的文档为准。

| 工具 | 功能 | 适用场景 |
|------|------|----------|
| **file_search** | 托管式RAG | 知识库问答、文档分析 |
| **code_interpreter** | Python代码执行 | 数据分析、可视化 |
| **web_search** | 实时网络搜索 | 获取最新信息 |
| **computer_use** | 计算机操作 | 自动化任务 |
| **mcp** | MCP协议集成 | 连接外部服务 |
| **function** | 自定义函数 | 业务逻辑集成 |

### file_search：托管式RAG

旧版 Assistants API 的官方移除日期为 **2026-08-26**，不再把 `client.assistants.create` 作为新项目示例。[官方弃用记录](https://developers.openai.com/api/docs/deprecations)

先创建向量存储、上传文件并等待索引处理完成，再发起检索。下面假设 `OPENAI_VECTOR_STORE_ID` 指向已就绪且当前用户有权访问的存储；它不是一个真实示例 ID。

```python
import os
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model=os.environ["OPENAI_MODEL"],
    input="总结资料中的主要结论，并给出文件引用。",
    tools=[{"type": "file_search",
            "vector_store_ids": [os.environ["OPENAI_VECTOR_STORE_ID"]]}],
    include=["file_search_call.results"]
)
print(response.output_text)
```

除了最终文本，还应检查响应中的引用与实际检索结果；无命中文档时不可假装找到了证据。[File search 官方指南](https://developers.openai.com/api/docs/guides/tools-file-search)

### code_interpreter：安全沙箱执行

```python
response = client.responses.create(
    model=os.environ["OPENAI_MODEL"],
    input="用 Python 计算 1 到 100 的整数平方和，并说明计算方法。",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}]
)
print(response.output_text)
```

这里沿用前一段的 `client` 和模型配置，模型需支持该工具。托管执行环境的生命周期、文件访问与费用由服务定义；生产系统仍需校验上传数据、生成文件与业务结果，不能把“在沙箱里运行”理解为“结果必然正确”。[Code Interpreter 官方指南](https://developers.openai.com/api/docs/guides/tools-code-interpreter)

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
| **gVisor** | 用户态应用内核 | 依系统调用与负载而变 | 减少应用直接接触宿主内核 |
| **Firecracker** | 微虚拟机 | 高 | 多租户环境 |
| **WebAssembly** | 运行时与宿主导入能力边界 | 依运行时与负载而变 | 受限模块执行 |

### 安全最佳实践

下面是执行器职责的伪代码：`RateLimiter`、`timeout`、`self.tools` 和校验/过滤函数需实现。同步函数超时不能只靠上下文管理器名称表达，应由可终止的进程或服务请求实现。

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

窄接口通常比任意代码字符串更容易验证。计算器不要使用 `eval(expression)` 执行用户输入，下面将能力限制为两个整数相乘。

```python
import os
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def multiply(a: int, b: int) -> int:
    """将两个绝对值不超过一百万的整数相乘。"""
    if type(a) is not int or type(b) is not int:
        raise ValueError("仅接受整数")
    if abs(a) > 1_000_000 or abs(b) > 1_000_000:
        raise ValueError("输入超过允许范围")
    return a * b

agent = create_agent(model=os.environ["LANGCHAIN_MODEL"], tools=[multiply])
```

`LANGCHAIN_MODEL` 需设置为所安装 provider 集成支持的模型标识，并配置对应凭证。当前官方入口使用 `create_agent`，旧教程中的 `langgraph.prebuilt.create_react_agent` 应结合项目固定版本迁移。[LangChain Agents 文档](https://docs.langchain.com/oss/python/langchain/agents)

### 自定义工具节点

自建 LangGraph 图时，核心边为 `START → model → tools → model`；只有模型发出了工具调用才走工具节点，否则结束。工具返回后通常回到模型，而不是继续绕工具节点。

状态应保留消息追加/合并规则；每个工具结果关联调用 ID。还要处理未知工具、参数错误、异常、最大轮数和消息裁剪。优先使用固定版本框架提供的工具节点，再围绕它补充业务授权与审计，避免重新实现一套不完整协议。

## 调用链的验收

测试应覆盖无需工具、单调用、多调用、无效参数、权限拒绝、工具超时和工具返回恶意指令。通过标准包括：调用与结果 ID 一一关联；失败没有伪装成功；未知工具无法执行；读到的外部内容不能授予新权限；结果过大时有可追溯的截断说明。最终答案必须与真实工具结果一致。

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
