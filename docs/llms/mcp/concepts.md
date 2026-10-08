---
title: MCP核心概念
description: 区分 Host、Client、Server，解释工具、资源、模板、能力协商、采样与权限边界。
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
reviewScope: MCP 2026-07-28 生命周期与旧协议兼容；新协议未做互通测试
exampleStatus: not-run
techVersion: MCP 2026-07-28 生命周期已核验；2025-06-18 旧基线与 FastMCP 2.12.5 实验保留
---

# MCP核心概念

MCP 标准化的是 AI 应用与外部能力之间的通信契约。理解它时先区分三件事：谁负责用户与模型、谁维护协议连接、谁执行真实业务操作。

## 🍳 厨房类比

可以把 Host 看作负责接单与调度的前台，Client 看作与某个厨房通信的连接，Server 看作提供菜单与具体能力的厨房。这个类比只帮助区分职责，不能据此推导授权关系。

| 角色 | 负责什么 | 不应假设什么 |
| --- | --- | --- |
| Host | 用户交互、模型上下文、能力选择与授权体验 | 接入 Server 后所有返回内容都可信 |
| Client | 与一个 Server 建立连接、协商能力、交换消息 | 每个 Server 都实现全部协议特性 |
| Server | 暴露工具/资源/模板，执行访问控制与业务逻辑 | 请求来自模型就自动有权限 |

## 🔌 服务器生命周期

**2025-06-18 教学基线**的典型连接顺序是 `initialize` → 协商版本与 capabilities → `notifications/initialized` → 正常请求 → 关闭连接。请求带 ID，用于匹配响应；通知不需要响应。双方只能调用已经协商且支持的能力。详见 [初始化生命周期](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle)。

SDK 的 `lifespan` 回调则是进程/应用资源管理：例如打开连接池、退出时释放资源。它与协议初始化不是同一层。连接池可以在进程内复用，用户身份、会话和审批状态不能随意放进跨用户共享的全局变量。

### 2026-07-28 生命周期变化

2026-10-08 核验：新版不再以初始化握手建立核心请求语义，而是在每次请求的 `_meta` 中声明版本、身份和能力；HTTP 同时有版本请求头。不支持请求版本时返回带兼容版本信息的错误。`initialize` 仍属于 2025-11-25 及更早版本的兼容流程。[Versioning and Compatibility](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)

因此排查连接时先问“双方实际支持哪个协议世代”，再查工具列表。新版服务并不自动接受旧版客户端；FastMCP 2.12.5 的教学实验也不能证明新协议兼容。核心协议无状态不意味着数据库任务、审批或用户会话无需存储，这些仍由应用负责。

## 🔧 Tools（工具）

工具把计算、检索或操作暴露为可发现的接口。Client 使用 `tools/list` 获取定义，通过 `tools/call` 发起调用；模型可以提出名称与参数，但真实调用由应用和 Server 执行。

一个好工具应说明：用途、参数含义、输出、可能的副作用和失败语义。例如 `search_policy(query, limit)` 比 `do_everything(command)` 更容易验证和约束。输入 schema 解决结构问题，服务端权限解决“谁能对哪个资源做什么”，二者不可替代。

### 工具结果与错误

- **正常结果**：返回内容；采用支持的协议版本时，可声明输出 schema 并返回结构化内容。
- **工具执行失败**：工具可用，但参数业务范围或上游执行失败，通常通过带 `isError` 的工具结果表示。
- **协议错误**：未知方法、无效协议请求等，使用 JSON-RPC 错误。

不要将空结果、无权限和上游超时全部返回成“没有找到”。具体 schema 与错误约定见 [Tools 规范](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)。

### 同步与异步工具

异步声明不自动让阻塞操作变为非阻塞。数据库、HTTP、CPU 计算分别需要适合的客户端、线程/进程执行方式和超时预算。每个工具还要限制返回大小，避免一次调用耗尽上下文或占满进程内存。

## 📁 Resources（资源）

资源用 URI 标识可读取的上下文，例如 `policy://travel/2026`。资源列表、参数化 URI 模板和实际读取是不同操作；返回 URI 不意味着该内容已自动进入模型上下文，也不意味着它可以像网页一样公开访问。

### 资源与工具的区别

| 需求 | 优先考虑 | 判断依据 |
| --- | --- | --- |
| 展示某份已知制度正文 | Resource | 有明确标识，适合由应用选择读取 |
| 根据问题查找若干制度 | Tool | 需要执行带参数的搜索 |
| 复用一套制度解读步骤 | Prompt | 用户选择并传入模板参数 |
| 提交报销申请 | Tool | 有业务副作用，需要授权与状态确认 |

工具也可以只读，资源也需要权限；不要用“有没有副作用”作为唯一分类标准。依据见 [Resources 规范](https://modelcontextprotocol.io/specification/2025-06-18/server/resources)。

## 📝 Prompts（提示模板）

Prompts 是 Server 提供的可发现、可参数化模板。Host 可以让用户选择模板并获取消息内容，但模板本身不自动执行工具，也不是可覆盖应用安全策略的系统指令。

设计模板时，把任务目标、需要的证据和输出要求写清楚。外部插入的文档内容按不可信数据处理；不要在模板中嵌入密钥或把用户可控字段拼成高权限指令。接口定义见 [Prompts 规范](https://modelcontextprotocol.io/specification/2025-06-18/server/prompts)。

## 🔄 三者协作

以制度助手为例：用户选择“解释制度”模板，Host 获取模板；随后读取制度资源或允许模型调用搜索工具；Host 选择要送入模型的证据，最后生成答案。如果涉及提交申请，另走写入工具、权限检查和确认流程。

协议提供这些接口，但不规定必须按上述顺序运行。可运行的三原语示例集中在 [快速入门](/llms/mcp/quickstart)，避免多个版本不同的示例混用。

## 🔄 Sampling（采样）

在本页 2025-06-18 基线中，Sampling 允许 Server 请求 Client 代表它调用模型，但必须由 Client 声明支持并受 Host 策略控制。Server 不能因此任意选择用户未授权的模型、上下文或费用预算，也不能假定每个 Host 都支持此功能。参见 [Sampling 规范](https://modelcontextprotocol.io/specification/2025-06-18/client/sampling)。

### Roots 与其他客户端能力

Roots 表达客户端希望 Server 工作的根范围，是上下文约定，不是操作系统沙箱。Server 仍要自行校验路径与权限；恶意进程不会因为收到 roots 就失去其他文件访问能力。详见 [Roots 规范](https://modelcontextprotocol.io/specification/2025-06-18/client/roots)。其他可选能力也应先发现、再使用，并为不支持的 Host 提供明确降级。

## 验证你是否理解了协议边界

1. 列出一次连接协商的版本和 capabilities，解释关闭一项能力后应用怎样降级。
2. 用同一查询分别设计 resource、tool 与 prompt，说明为什么选其中一种。
3. 模拟工具业务失败和协议请求错误，检查 Client 是否能区分。
4. 给一个有效 JSON 但无权限的请求，确认 Server 仍拒绝它。

本页保留旧基线以配合可运行示例，并单列新版生命周期变化；Sampling 等跨端交互也必须按对应版本规范实现，不能复用旧握手流程推导新版行为。下一步阅读 [高级功能](/llms/mcp/advanced)。
