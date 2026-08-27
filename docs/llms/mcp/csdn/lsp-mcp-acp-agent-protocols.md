---
title: "从 LSP 到 MCP、ACP：AI Agent 时代的协议体系设计详解"
description: "CSDN 原文全文镜像：AI Agent 时代，协议会变得越来越重要。AI ClientIDEAgentToolResourcePrompt这些组件之间如果没有统一协议，就会变成大量脆弱的胶水代码。JSON-RPC：消息格式底座stdio / HTTP / S……"
pageType: article
module: mcp
updated: '2026-05-20'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "mcp"
  - "人工智能"
  - "软件工程"
  - "架构"
  - "大模型"
level: intermediate
prerequisites:
  - "/llms/mcp/"
  - "/llms/agent/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-05-20，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-05-20。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/161224325](https://blog.csdn.net/m0_63309778/article/details/161224325)
- 站内分区：MCP / Agent 协议体系
:::

<p><img src="https://i-blog.csdnimg.cn/direct/971d8a0375004f7fa8ff2f62b1e9e32a.png" alt="在这里插入图片描述" /></p> 
<h3>前言</h3> 
<p>过去我们做软件系统集成&#xff0c;常见的是 REST API、RPC、WebSocket、消息队列这些技术。但进入 AI Agent 时代之后&#xff0c;我们会发现新的问题出现了&#xff1a;</p> 
<blockquote> 
<p>不是系统之间简单地调用接口&#xff0c;而是 AI 客户端、IDE、Agent、外部工具、聊天平台之间需要协同工作。</p> 
</blockquote> 
<p>例如&#xff1a;</p> 
<ul><li>Claude Desktop 想调用你本地写的“查公司内部 Wiki”的工具。</li><li>Cursor / VS Code 想让一个 coding agent 接管项目重构任务。</li><li>一个企业 AI 助手想同时接入微信、Slack、Discord、Telegram。</li><li>Agent 在 IDE 中修改文件时&#xff0c;需要向客户端请求权限、返回进度、支持取消任务。</li><li>AI 模型想知道有哪些工具可以调用、工具参数是什么、返回结果是什么。</li></ul> 
<p>这些场景背后&#xff0c;本质上都需要一套“通信协议”。</p> 
<p>这篇文章围绕几个关键协议和架构模式展开&#xff1a;</p> 
<ol><li><strong>JSON-RPC 2.0</strong>&#xff1a;底层消息格式。</li><li><strong>stdio Transport</strong>&#xff1a;本地进程间通信方式。</li><li><strong>LSP</strong>&#xff1a;IDE 与语言服务器之间的经典协议。</li><li><strong>MCP</strong>&#xff1a;AI 客户端发现和调用外部工具的协议。</li><li><strong>ACP</strong>&#xff1a;IDE 与 AI coding agent 之间的协议。</li><li><strong>Gateway Plugin</strong>&#xff1a;把 Agent 接入微信、Slack、Discord 等聊天平台的插件架构。</li></ol> 
<p>如果用一句话概括它们的关系&#xff1a;</p> 
<blockquote> 
<p>JSON-RPC 是消息格式&#xff0c;stdio/HTTP/SSE 是传输通道&#xff0c;LSP 是 IDE 时代的语言能力协议&#xff0c;MCP 是 AI 调用外部工具的协议&#xff0c;ACP 是 IDE 驱动 Agent 的协议&#xff0c;Gateway Plugin 是 Agent 接入多聊天平台的适配层。</p> 
</blockquote> 
<hr /> 
<h2>一、先理解 JSON-RPC&#xff1a;这些协议的共同底座</h2> 
<p>无论是 LSP、MCP 还是 ACP&#xff0c;它们都大量借鉴或直接使用了 <strong>JSON-RPC 2.0</strong>。</p> 
<p>JSON-RPC 是一种非常轻量的远程调用协议。它不像 REST 那样围绕 URL 和 HTTP Method 设计&#xff0c;而是围绕“方法名 &#43; 参数 &#43; 请求 ID”设计。</p> 
<p>JSON-RPC 2.0 官方规范中定义了 Request、Response、Notification、Error 等核心结构&#xff1b;其中 Notification 是没有 <code>id</code> 的请求&#xff0c;服务端不需要返回响应。(<a href="https://www.jsonrpc.org/specification?utm_source&#61;chatgpt.com" title="JSON-RPC 2.0 Specification" rel="nofollow">jsonrpc.org</a>)</p> 
<h3>1.1 JSON-RPC 请求格式</h3> 
<p>一个最简单的 JSON-RPC 请求长这样&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">1</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"tools/list"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>字段含义&#xff1a;</p> 
<table><thead><tr><th>字段</th><th>含义</th></tr></thead><tbody><tr><td><code>jsonrpc</code></td><td>协议版本&#xff0c;固定为 <code>&#34;2.0&#34;</code></td></tr><tr><td><code>id</code></td><td>请求 ID&#xff0c;用来匹配响应</td></tr><tr><td><code>method</code></td><td>要调用的方法名</td></tr><tr><td><code>params</code></td><td>方法参数</td></tr></tbody></table>
<p>服务端响应&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">1</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"tools"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>如果出错&#xff0c;则返回&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">1</span><span class="token punctuation">,</span>
<span class="token string-property property">"error"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"code"</span><span class="token operator">:</span> <span class="token operator">-</span><span class="token number">32601</span><span class="token punctuation">,</span>
<span class="token string-property property">"message"</span><span class="token operator">:</span> <span class="token string">"Method not found"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<h3>1.2 Notification&#xff1a;不需要响应的消息</h3> 
<p>JSON-RPC 中还有一种特殊消息&#xff0c;叫 Notification。</p> 
<p>它没有 <code>id</code>&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"notifications/initialized"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>因为没有 <code>id</code>&#xff0c;服务端就不需要返回响应。</p> 
<p>这类消息很适合用于&#xff1a;</p> 
<ul><li>初始化完成通知</li><li>进度更新</li><li>日志推送</li><li>状态变更</li><li>文件变化通知</li></ul> 
<p>JSON-RPC 2.0 规范明确说明&#xff0c;没有 <code>id</code> 的 Request 就是 Notification&#xff0c;服务端不能回复 Notification。(<a href="https://www.jsonrpc.org/specification?utm_source&#61;chatgpt.com" title="JSON-RPC 2.0 Specification" rel="nofollow">jsonrpc.org</a>)</p> 
<h3>1.3 为什么这些协议喜欢用 JSON-RPC&#xff1f;</h3> 
<p>因为它非常适合本地工具、IDE、Agent 之间通信。</p> 
<p>它的优点是&#xff1a;</p> 
<ol><li><strong>简单</strong>&#xff1a;一个 JSON 对象就能表达一次调用。</li><li><strong>语言无关</strong>&#xff1a;Python、TypeScript、Go、Rust、Java 都能轻松实现。</li><li><strong>传输无关</strong>&#xff1a;可以跑在 stdio、WebSocket、HTTP、TCP 上。</li><li><strong>天然支持双向通信</strong>&#xff1a;客户端可以请求服务端&#xff0c;服务端也可以发送通知。</li><li><strong>适合长生命周期进程</strong>&#xff1a;例如 IDE 启动一个 language server&#xff0c;持续通信。</li></ol> 
<p>这也是为什么 LSP、MCP、ACP 都和 JSON-RPC 有很深的关系。</p> 
<hr /> 
<h2>二、再理解 stdio&#xff1a;为什么这些协议喜欢用标准输入输出&#xff1f;</h2> 
<p>很多人第一次看到 MCP 或 ACP 的时候&#xff0c;会疑惑&#xff1a;</p> 
<blockquote> 
<p>为什么不用 HTTP&#xff0c;而是用 stdio&#xff1f;</p> 
</blockquote> 
<p>stdio 指的是&#xff1a;</p> 
<ul><li>stdin&#xff1a;标准输入</li><li>stdout&#xff1a;标准输出</li><li>stderr&#xff1a;标准错误</li></ul> 
<p>也就是说&#xff0c;客户端可以启动一个子进程&#xff0c;然后通过这个子进程的输入输出流通信。</p> 
<p>例如&#xff1a;</p> 


```text
Claude Desktop / Cursor / IDE
|
| 启动子进程
v
python mcp_server.py
|
| stdin / stdout
v
JSON-RPC 消息
```

 
<h3>2.1 stdio 的优势</h3> 
<h4>1. 本地部署简单</h4> 
<p>不需要开放端口。</p> 
<p>例如你写一个 MCP Server&#xff1a;</p> 


```bash
python wiki_mcp_server.py
```

 
<p>AI 客户端直接启动这个进程即可。</p> 
<p>不用考虑&#xff1a;</p> 
<ul><li>端口冲突</li><li>防火墙</li><li>HTTPS 证书</li><li>反向代理</li><li>外网访问</li></ul> 
<h4>2. 安全边界更清晰</h4> 
<p>stdio 通信通常发生在本机父子进程之间。</p> 
<p>相比暴露 HTTP 服务&#xff0c;它的攻击面更小。</p> 
<h4>3. 非常适合 IDE / 桌面客户端</h4> 
<p>LSP 早就大量采用类似模式&#xff1a;</p> 


```text
VS Code 启动 pyright / rust-analyzer / gopls
然后通过 JSON-RPC 通信
```

 
<p>ACP 也借鉴了类似思想。ACP 官方相关文档说明&#xff0c;它是基于 JSON-RPC 的协议&#xff0c;客户端通常会以子进程方式启动 agent&#xff0c;并通过 stdio 通信&#xff0c;但协议本身也可以适配其他双向流。(<a href="https://docs.rs/agent-client-protocol-schema/latest/aarch64-unknown-linux-gnu/agent_client_protocol_schema/?utm_source&#61;chatgpt.com" title="agent_client_protocol_schema - Rust" rel="nofollow">docs.rs</a>)</p> 
<h3>2.2 stdio 的缺点</h3> 
<p>stdio 也不是万能的。</p> 
<p>它不适合&#xff1a;</p> 
<ul><li>多客户端共享同一个服务</li><li>跨机器访问</li><li>云端部署</li><li>需要负载均衡的服务</li><li>需要浏览器直接访问的服务</li></ul> 
<p>这时就更适合 HTTP、SSE、WebSocket 或 Streamable HTTP。</p> 
<p>MCP 当前规范中也明确提到&#xff0c;MCP 使用 JSON-RPC 编码消息&#xff0c;并定义了 stdio 和 Streamable HTTP 两类标准传输机制。(<a href="https://modelcontextprotocol.io/specification/2025-06-18/basic/transports?utm_source&#61;chatgpt.com" title="Transports" rel="nofollow">模型上下文协议</a>)</p> 
<hr /> 
<h2>三、LSP&#xff1a;IDE 协议设计的经典范式</h2> 
<p>在讲 MCP 和 ACP 之前&#xff0c;必须先讲 LSP。</p> 
<p>LSP 全称是 <strong>Language Server Protocol</strong>&#xff0c;即语言服务器协议。</p> 
<p>它解决的问题是&#xff1a;</p> 
<blockquote> 
<p>不同编辑器都需要代码补全、跳转定义、悬浮文档、诊断报错&#xff0c;但如果每个语言都给每个编辑器单独适配一遍&#xff0c;成本会爆炸。</p> 
</blockquote> 
<p>以前的情况是&#xff1a;</p> 


```text
Python 插件需要适配 VS Code
Python 插件需要适配 Vim
Python 插件需要适配 Emacs
Python 插件需要适配 JetBrains

Go 插件也要适配 VS Code
Go 插件也要适配 Vim
Go 插件也要适配 Emacs
……
```

 
<p>这就是 N × M 的集成问题。</p> 
<p>LSP 的思路是&#xff1a;</p> 
<blockquote> 
<p>编辑器只实现 LSP Client&#xff0c;语言能力只实现 LSP Server&#xff0c;中间用统一协议通信。</p> 
</blockquote> 


```text
VS Code / Vim / Emacs / Cursor
|
| LSP
v
Python Language Server / Go Language Server / Rust Analyzer
```

 
<p>LSP 官方规范描述的是编辑器和语言服务器之间的通信协议&#xff0c;当前稳定规范是 3.17。(<a href="https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/?utm_source&#61;chatgpt.com" title="Language Server Protocol Specification - 3.17" rel="nofollow">GitHub Microsoft</a>)</p> 
<h3>3.1 LSP 的核心流程</h3> 
<p>典型 LSP 流程&#xff1a;</p> 


```text
1. initialize
2. initialized
3. textDocument/didOpen
4. textDocument/didChange
5. textDocument/completion
6. textDocument/hover
7. textDocument/definition
8. textDocument/publishDiagnostics
```

 
<p>例如编辑器打开一个 Python 文件&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">1</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"initialize"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"processId"</span><span class="token operator">:</span> <span class="token number">12345</span><span class="token punctuation">,</span>
<span class="token string-property property">"rootUri"</span><span class="token operator">:</span> <span class="token string">"file:///project"</span><span class="token punctuation">,</span>
<span class="token string-property property">"capabilities"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>语言服务器返回自己支持的能力&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">1</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"capabilities"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"hoverProvider"</span><span class="token operator">:</span> <span class="token boolean">true</span><span class="token punctuation">,</span>
<span class="token string-property property">"definitionProvider"</span><span class="token operator">:</span> <span class="token boolean">true</span><span class="token punctuation">,</span>
<span class="token string-property property">"completionProvider"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"resolveProvider"</span><span class="token operator">:</span> <span class="token boolean">true</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>之后编辑器就知道&#xff1a;</p> 


```text
这个语言服务器支持 hover、definition、completion。
```

 
<h3>3.2 LSP 的价值</h3> 
<p>LSP 最核心的价值是解耦&#xff1a;</p> 


```text
编辑器不需要懂 Python / Go / Rust 的语义
语言服务器不需要关心 VS Code / Vim / Emacs 的 UI
```

 
<p>编辑器只负责展示&#xff1a;</p> 
<ul><li>补全列表</li><li>错误波浪线</li><li>跳转结果</li><li>悬浮文档</li></ul> 
<p>语言服务器只负责计算&#xff1a;</p> 
<ul><li>代码语义</li><li>类型信息</li><li>诊断错误</li><li>定义位置</li><li>引用位置</li></ul> 
<p>这就是协议的力量。</p> 
<p>LSP 通过统一协议把“编辑器”和“语言智能”解耦&#xff0c;减少了不同编辑器重复实现语言能力的成本。相关介绍也强调&#xff0c;LSP 让语言相关能力可以实现一次&#xff0c;并在多个支持 LSP 的编辑器中复用。(<a href="https://nabeelvalley.co.za/blog/2025/26-03/the-language-server-protocol/?utm_source&#61;chatgpt.com" title="Getting Started with the Language Server Protocol" rel="nofollow">nabeelvalley.co.za</a>)</p> 
<h3>3.3 LSP 对 AI Agent 协议的启发</h3> 
<p>LSP 的成功给 MCP / ACP 很大启发。</p> 
<p>因为 AI Agent 时代也遇到了类似问题&#xff1a;</p> 


```text
Claude Desktop 想接各种工具
Cursor 想接各种 agent
VS Code 想接各种 coding agent
聊天平台想接各种 AI 后端
```

 
<p>如果每个客户端都和每个工具、每个 agent 单独适配&#xff0c;就会再次变成 N × M 的集成灾难。</p> 
<p>所以我们需要新的标准协议&#xff1a;</p> 


```text
MCP：统一 AI 调用工具的方式
ACP：统一 IDE 调用 Agent 的方式
Gateway Plugin：统一聊天平台接入 Agent 的方式
```

 
<hr /> 
<h2>四、MCP&#xff1a;让 AI 客户端发现并调用外部工具</h2> 
<p>MCP 全称是 <strong>Model Context Protocol</strong>。</p> 
<p>它解决的问题是&#xff1a;</p> 
<blockquote> 
<p>AI 客户端如何标准化地发现外部工具、读取外部资源、调用外部能力&#xff1f;</p> 
</blockquote> 
<p>MCP 官方文档中将 tools 描述为允许模型与外部系统交互的机制&#xff0c;例如查询数据库、调用 API 或执行计算&#xff1b;每个 tool 有唯一名称和描述其参数的 metadata/schema。(<a href="https://modelcontextprotocol.io/specification/2025-06-18/server/tools?utm_source&#61;chatgpt.com" title="Tools" rel="nofollow">模型上下文协议</a>)</p> 
<h3>4.1 MCP 的典型场景</h3> 
<p>假设你写了一个公司内部 Wiki 查询服务&#xff1a;</p> 


```text
wiki_search(query: str) -> str
```

 
<p>你希望 Claude Desktop、Cursor 或其他 AI 客户端能够调用它。</p> 
<p>没有 MCP 时&#xff0c;你可能要为每个客户端单独写插件。</p> 
<p>有 MCP 后&#xff0c;你可以写一个 MCP Server&#xff1a;</p> 


```text
AI Client
|
| MCP
v
Wiki MCP Server
|
v
公司内部 Wiki / 知识库 / 数据库
```

 
<p>AI 客户端只需要知道&#xff1a;</p> 
<ol><li>这个 MCP Server 暴露了哪些工具&#xff1f;</li><li>每个工具叫什么&#xff1f;</li><li>参数 schema 是什么&#xff1f;</li><li>如何调用&#xff1f;</li><li>返回结果是什么&#xff1f;</li></ol> 
<h3>4.2 MCP 的核心角色</h3> 
<p>MCP 中通常有三个角色&#xff1a;</p> 


```text
MCP Host：AI 应用本体，例如 Claude Desktop、Cursor
MCP Client：Host 内部负责协议通信的客户端
MCP Server：暴露工具、资源、Prompt 的服务
```

 
<p>可以理解为&#xff1a;</p> 


```text
Claude Desktop = Host
Claude 内部 MCP 连接模块 = Client
你写的 wiki_mcp_server.py = Server
```

 
<h3>4.3 MCP 的核心能力</h3> 
<p>MCP Server 通常可以暴露三类能力&#xff1a;</p> 
<table><thead><tr><th>能力</th><th>说明</th></tr></thead><tbody><tr><td>Tools</td><td>可被模型调用的函数&#xff0c;例如查数据库、发请求、计算</td></tr><tr><td>Resources</td><td>可被读取的资源&#xff0c;例如文件、文档、上下文</td></tr><tr><td>Prompts</td><td>可复用的提示词模板</td></tr></tbody></table>
<p>你给出的几个核心方法非常典型&#xff1a;</p> 


```text
initialize
tools/list
tools/call
resources/list
resources/read
prompts/list
```

 
<p>其中最常用的是&#xff1a;</p> 


```text
initialize
tools/list
tools/call
```

 
<h3>4.4 MCP initialize&#xff1a;协议握手</h3> 
<p>初始化请求示例&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">1</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"initialize"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"protocolVersion"</span><span class="token operator">:</span> <span class="token string">"2024-11-05"</span><span class="token punctuation">,</span>
<span class="token string-property property">"capabilities"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"clientInfo"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"claude-desktop"</span><span class="token punctuation">,</span>
<span class="token string-property property">"version"</span><span class="token operator">:</span> <span class="token string">"1.0.0"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>服务端响应&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">1</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"protocolVersion"</span><span class="token operator">:</span> <span class="token string">"2024-11-05"</span><span class="token punctuation">,</span>
<span class="token string-property property">"capabilities"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"tools"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"resources"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"prompts"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"serverInfo"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"company-wiki-mcp"</span><span class="token punctuation">,</span>
<span class="token string-property property">"version"</span><span class="token operator">:</span> <span class="token string">"1.0.0"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>这个过程类似 LSP 的 initialize。</p> 
<p>它的作用是&#xff1a;</p> 
<ul><li>客户端告诉服务端自己是谁。</li><li>服务端告诉客户端自己支持什么能力。</li><li>双方确认协议版本。</li><li>双方交换 capabilities。</li></ul> 
<h3>4.5 MCP tools/list&#xff1a;列出可用工具</h3> 
<p>客户端请求&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">2</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"tools/list"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>服务端返回&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">2</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"tools"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"search_company_wiki"</span><span class="token punctuation">,</span>
<span class="token string-property property">"description"</span><span class="token operator">:</span> <span class="token string">"搜索公司内部 Wiki，适合查询制度、流程、项目文档"</span><span class="token punctuation">,</span>
<span class="token string-property property">"inputSchema"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"object"</span><span class="token punctuation">,</span>
<span class="token string-property property">"properties"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"query"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"string"</span><span class="token punctuation">,</span>
<span class="token string-property property">"description"</span><span class="token operator">:</span> <span class="token string">"搜索关键词"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"required"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token string">"query"</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>这个结果非常重要。</p> 
<p>因为模型并不是直接“知道”你的工具&#xff0c;而是通过 <code>tools/list</code> 获取工具列表。</p> 
<p>模型看到工具描述后&#xff0c;才能判断&#xff1a;</p> 


```text
用户问的是公司制度问题，应该调用 search_company_wiki。
```

 
<h3>4.6 MCP tools/call&#xff1a;执行工具</h3> 
<p>客户端调用&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">3</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"tools/call"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"search_company_wiki"</span><span class="token punctuation">,</span>
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"query"</span><span class="token operator">:</span> <span class="token string">"年假申请流程"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>服务端返回&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">3</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"text"</span><span class="token punctuation">,</span>
<span class="token string-property property">"text"</span><span class="token operator">:</span> <span class="token string">"年假申请需要在 OA 系统提交请假单，直属领导审批后生效。"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>这就是 MCP 的核心闭环&#xff1a;</p> 


```text
发现工具 → 理解工具 → 调用工具 → 返回结果 → 模型组织回答
```

 
<h3>4.7 MCP 的本质</h3> 
<p>MCP 不是简单的函数调用。</p> 
<p>它真正解决的是&#xff1a;</p> 
<blockquote> 
<p>AI 应用和外部工具之间的标准化连接问题。</p> 
</blockquote> 
<p>以前&#xff1a;</p> 


```text
Claude 接 Wiki 要写一套
Cursor 接 Wiki 要写一套
自研 Agent 接 Wiki 又要写一套
```

 
<p>现在&#xff1a;</p> 


```text
只要实现一个 Wiki MCP Server
多个 MCP Client 都可以接入
```

 
<p>所以 MCP 可以理解为&#xff1a;</p> 


```text
AI 工具生态里的 USB-C 接口
```

 
<hr /> 
<h2>五、ACP&#xff1a;让 IDE 可以唤起和控制 Agent</h2> 
<p>ACP 全称是 <strong>Agent Client Protocol</strong>。</p> 
<p>它解决的问题和 MCP 不一样。</p> 
<p>MCP 是&#xff1a;</p> 


```text
AI 调用工具
```

 
<p>ACP 是&#xff1a;</p> 


```text
IDE 调用 Agent
```

 
<p>ACP 官方介绍中明确说&#xff0c;它标准化了代码编辑器和 coding agents 之间的通信&#xff1b;相关文档也说明 ACP 面向 agent-editor 集成&#xff0c;而如果你的目标是让 agent 调用外部工具&#xff0c;则应该看 MCP。(<a href="https://github.com/agentclientprotocol/agent-client-protocol?utm_source&#61;chatgpt.com" title="agentclientprotocol/agent-client-protocol: A ...">GitHub</a>)</p> 
<h3>5.1 ACP 的典型场景</h3> 
<p>用户在 IDE 里说&#xff1a;</p> 


```text
帮我重构这个函数。
```

 
<p>这时不是简单调用一个工具&#xff0c;而是让一个 Agent 接管任务。</p> 
<p>Agent 可能需要&#xff1a;</p> 
<ol><li>读取当前文件。</li><li>分析项目结构。</li><li>修改多个文件。</li><li>运行测试。</li><li>返回 diff。</li><li>请求用户确认。</li><li>支持取消任务。</li><li>支持中途 steer&#xff0c;比如“慢一点”“更保守一点”“不要改接口”。</li></ol> 
<p>这和 MCP 的“调用一个工具”完全不是一个层级。</p> 
<h3>5.2 ACP 和 LSP 的关系</h3> 
<p>ACP 很像 LSP。</p> 
<p>LSP 是&#xff1a;</p> 


```text
IDE <-> Language Server
```

 
<p>ACP 是&#xff1a;</p> 


```text
IDE <-> Coding Agent
```

 
<p>LSP 解决的是&#xff1a;</p> 


```text
补全、跳转、诊断、hover
```

 
<p>ACP 解决的是&#xff1a;</p> 


```text
任务执行、文件修改、进度更新、权限请求、取消、中途引导
```

 
<p>所以可以这样理解&#xff1a;</p> 


```text
LSP 是 IDE 接入语言智能的协议。
ACP 是 IDE 接入 AI Agent 的协议。
```

 
<h3>5.3 ACP 的核心流程</h3> 
<p>根据 ACP 官方概览&#xff0c;典型流程包括&#xff1a;客户端发送 <code>session/prompt</code>&#xff0c;Agent 发送 <code>session/update</code> 通知进度&#xff0c;必要时 Agent 向客户端请求文件操作或权限&#xff0c;客户端可发送 <code>session/cancel</code> 中断任务&#xff0c;最终 <code>session/prompt</code> 返回 stop reason。(<a href="https://agentclientprotocol.com/protocol/overview?utm_source&#61;chatgpt.com" title="Overview - Agent Client Protocol" rel="nofollow">agentclientprotocol.com</a>)</p> 
<p>一个简化流程&#xff1a;</p> 


```text
1. initialize
2. session/new
3. session/prompt
4. session/update
5. session/cancel
6. session/prompt response
```

 
<p>不同版本和实现中方法名可能略有差异。例如你提到的 <code>newSession</code>、<code>prompt</code>、<code>cancel</code>、<code>/steer</code>、<code>/queue</code> 更像是概念层或某些实现中的命令表达&#xff1b;当前公开文档中更常见的是 <code>session/prompt</code>、<code>session/update</code>、<code>session/cancel</code> 这类命名。ACP 仍处于发展阶段&#xff0c;部分实现和命名可能会演进。(<a href="https://goose-docs.ai/docs/guides/acp-clients/?utm_source&#61;chatgpt.com" title="Using goose in ACP Clients | goose | Your open source AI agent" rel="nofollow">goose-docs.ai</a>)</p> 
<h3>5.4 ACP initialize&#xff1a;客户端注册</h3> 
<p>示例&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">1</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"initialize"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"protocolVersion"</span><span class="token operator">:</span> <span class="token string">"0.1.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"clientInfo"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"cursor"</span><span class="token punctuation">,</span>
<span class="token string-property property">"version"</span><span class="token operator">:</span> <span class="token string">"1.0.0"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"capabilities"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"fileSystem"</span><span class="token operator">:</span> <span class="token boolean">true</span><span class="token punctuation">,</span>
<span class="token string-property property">"terminal"</span><span class="token operator">:</span> <span class="token boolean">true</span><span class="token punctuation">,</span>
<span class="token string-property property">"diff"</span><span class="token operator">:</span> <span class="token boolean">true</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>Agent 返回&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">1</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"agentInfo"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"my-coding-agent"</span><span class="token punctuation">,</span>
<span class="token string-property property">"version"</span><span class="token operator">:</span> <span class="token string">"1.0.0"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"capabilities"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"streaming"</span><span class="token operator">:</span> <span class="token boolean">true</span><span class="token punctuation">,</span>
<span class="token string-property property">"fileEdit"</span><span class="token operator">:</span> <span class="token boolean">true</span><span class="token punctuation">,</span>
<span class="token string-property property">"terminalCommand"</span><span class="token operator">:</span> <span class="token boolean">true</span><span class="token punctuation">,</span>
<span class="token string-property property">"cancellation"</span><span class="token operator">:</span> <span class="token boolean">true</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>这里交换的是&#xff1a;</p> 


```text
客户端能提供什么上下文？
Agent 能执行什么任务？
双方支持哪些能力？
```

 
<h3>5.5 ACP 创建会话</h3> 
<p>一个 coding agent 往往是会话化的。</p> 
<p>因为它需要记住&#xff1a;</p> 
<ul><li>当前任务目标</li><li>当前项目路径</li><li>已经读取过的文件</li><li>已经修改过的文件</li><li>当前计划</li><li>用户中途补充的约束</li></ul> 
<p>示例&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">2</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"session/new"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"workspace"</span><span class="token operator">:</span> <span class="token string">"file:///Users/me/project"</span><span class="token punctuation">,</span>
<span class="token string-property property">"mode"</span><span class="token operator">:</span> <span class="token string">"edit"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>返回&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">2</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"sessionId"</span><span class="token operator">:</span> <span class="token string">"sess_123"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<h3>5.6 ACP prompt&#xff1a;发送任务</h3> 
<p>用户在 IDE 里说&#xff1a;</p> 


```text
帮我把这个函数拆成三个小函数，并补充单元测试。
```

 
<p>对应请求&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">3</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"session/prompt"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"sessionId"</span><span class="token operator">:</span> <span class="token string">"sess_123"</span><span class="token punctuation">,</span>
<span class="token string-property property">"prompt"</span><span class="token operator">:</span> <span class="token string">"帮我把这个函数拆成三个小函数，并补充单元测试。"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>Agent 开始工作。</p> 
<p>这时它可能会不断发送进度通知&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"session/update"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"sessionId"</span><span class="token operator">:</span> <span class="token string">"sess_123"</span><span class="token punctuation">,</span>
<span class="token string-property property">"message"</span><span class="token operator">:</span> <span class="token string">"正在分析项目结构..."</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>再比如&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"session/update"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"sessionId"</span><span class="token operator">:</span> <span class="token string">"sess_123"</span><span class="token punctuation">,</span>
<span class="token string-property property">"message"</span><span class="token operator">:</span> <span class="token string">"已找到目标函数，准备修改 service.py 和 test_service.py"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>最后返回&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">3</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"stopReason"</span><span class="token operator">:</span> <span class="token string">"completed"</span><span class="token punctuation">,</span>
<span class="token string-property property">"summary"</span><span class="token operator">:</span> <span class="token string">"已完成函数拆分，并新增 3 个单元测试。"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<h3>5.7 ACP cancel&#xff1a;中断任务</h3> 
<p>Agent 任务可能很长。</p> 
<p>用户可能中途发现方向不对&#xff0c;于是点击取消。</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">4</span><span class="token punctuation">,</span>
<span class="token string-property property">"method"</span><span class="token operator">:</span> <span class="token string">"session/cancel"</span><span class="token punctuation">,</span>
<span class="token string-property property">"params"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"sessionId"</span><span class="token operator">:</span> <span class="token string">"sess_123"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<p>Agent 收到后应该尽量停止当前任务。</p> 
<p>返回&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"jsonrpc"</span><span class="token operator">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">4</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"cancelled"</span><span class="token operator">:</span> <span class="token boolean">true</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<h3>5.8 steer&#xff1a;中途引导</h3> 
<p>你提到的 <code>/steer</code> 很关键。</p> 
<p>它不是简单的“新任务”&#xff0c;而是对当前任务的方向修正。</p> 
<p>例如用户说&#xff1a;</p> 


```text
/steer 慢一点，先不要改文件，先给我解释计划。
```

 
<p>Agent 应该调整策略&#xff1a;</p> 


```text
从直接修改模式 → 计划说明模式
```

 
<p>或者&#xff1a;</p> 


```text
/steer 保守一点，不要改公共接口。
```

 
<p>Agent 应该把这个约束加入当前会话。</p> 
<p>这类能力是 coding agent 和普通 chat bot 的重要区别。</p> 
<h3>5.9 queue&#xff1a;排队消息</h3> 
<p>当 Agent 正在执行长任务时&#xff0c;用户可能又发来一句&#xff1a;</p> 


```text
顺便把测试也补上。
```

 
<p>如果立即打断当前任务&#xff0c;可能会导致状态混乱。</p> 
<p>所以可以设计 <code>/queue</code>&#xff1a;</p> 


```text
当前任务继续执行
新消息进入队列
当前任务完成后再处理
```

 
<p>这对于长任务 Agent 非常重要。</p> 
<hr /> 
<h2>六、MCP 和 ACP 的核心差异</h2> 
<p>MCP 和 ACP 很容易混淆。</p> 
<p>其实它们解决的问题完全不同。</p> 
<table><thead><tr><th>对比项</th><th>MCP</th><th>ACP</th></tr></thead><tbody><tr><td>全称</td><td>Model Context Protocol</td><td>Agent Client Protocol</td></tr><tr><td>主要对象</td><td>AI 客户端与外部工具</td><td>IDE 与 coding agent</td></tr><tr><td>核心问题</td><td>让模型发现并调用工具</td><td>让 IDE 唤起、控制、协作 Agent</td></tr><tr><td>典型客户端</td><td>Claude Desktop、Cursor、自研 Agent Host</td><td>Cursor、VS Code、Zed 等 IDE</td></tr><tr><td>典型服务端</td><td>Wiki MCP Server、DB MCP Server、Git MCP Server</td><td>Coding Agent、Deep Agent、Goose</td></tr><tr><td>通信风格</td><td>工具发现与调用</td><td>会话、任务、进度、文件操作、权限</td></tr><tr><td>类比</td><td>AI 时代的工具 USB-C</td><td>AI Agent 时代的 LSP</td></tr><tr><td>主动/被动</td><td>工具被 AI 调用&#xff0c;偏被动</td><td>Agent 被 IDE 唤起执行任务&#xff0c;偏主动</td></tr></tbody></table>
<p>可以这样记&#xff1a;</p> 


```text
MCP：让 Agent 会“用工具”。
ACP：让 IDE 会“用 Agent”。
```

 
<p>或者更直接&#xff1a;</p> 


```text
MCP 是 Tool Protocol。
ACP 是 Agent Protocol。
```

 
<p>LangChain 文档中也明确区分了两者&#xff1a;ACP 用于 coding agents 和 code editors / IDEs 通信&#xff1b;如果需要 agent 调用外部服务器上的工具&#xff0c;应参考 MCP。(<a href="https://docs.langchain.com/oss/python/deepagents/acp?utm_source&#61;chatgpt.com" title="Agent Client Protocol (ACP)" rel="nofollow">LangChain 文档</a>)</p> 
<hr /> 
<h2>七、Gateway Plugin&#xff1a;把 Agent 接入聊天平台</h2> 
<p>MCP 和 ACP 主要面向 AI 客户端、IDE、工具生态。</p> 
<p>但企业里还有另一个常见需求&#xff1a;</p> 
<blockquote> 
<p>我希望用户可以在微信、Slack、Discord、Telegram 里直接和 Agent 对话。</p> 
</blockquote> 
<p>这时候就需要 Gateway Plugin。</p> 
<p>它不是一个像 MCP / ACP 那样的统一协议标准&#xff0c;更像是一种插件化架构模式。</p> 
<p>你的设计是这样的&#xff1a;</p> 


```text
gateway/
platforms/
wechat/
slack/
discord/
telegram/
...
```

 
<p>每个平台一个插件包。</p> 
<p>每个插件实现统一接口&#xff1a;</p> 


```python
<span class="token keyword">class</span> <span class="token class-name">PlatformPlugin</span><span class="token punctuation">:</span>
<span class="token keyword">def</span> <span class="token function">receive_message</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> Message<span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>

<span class="token keyword">def</span> <span class="token function">send_message</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> msg<span class="token punctuation">:</span> Message<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token boolean">None</span><span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>

<span class="token keyword">def</span> <span class="token function">authenticate</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token builtin">bool</span><span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>
```

 
<h3>7.1 为什么需要 Gateway Plugin&#xff1f;</h3> 
<p>因为不同聊天平台的接入方式完全不同。</p> 
<p>Slack 使用 Events API 时&#xff0c;Slack 会把订阅的事件发送给你的应用&#xff1b;Slack 文档中也说明 Events API 支持通过 Socket Mode 或指定公开 HTTP endpoint 接收事件。(<a href="https://docs.slack.dev/apis/events-api/?utm_source&#61;chatgpt.com" title="The Events API | Slack Developer Docs" rel="nofollow">Slack 开发者文档</a>)</p> 
<p>Telegram Bot API 则支持 <code>getUpdates</code> 和 webhook 两种互斥的更新接收方式&#xff0c;收到的是 JSON 序列化的 Update 对象。(<a href="https://core.telegram.org/bots/api?utm_source&#61;chatgpt.com" title="Telegram Bot API" rel="nofollow">core.telegram.org</a>)</p> 
<p>不同平台差异包括&#xff1a;</p> 
<table><thead><tr><th>平台</th><th>接收消息方式</th><th>发送消息方式</th><th>特殊能力</th></tr></thead><tbody><tr><td>Slack</td><td>Events API / Socket Mode</td><td>Web API</td><td>Thread、Channel、App Mention</td></tr><tr><td>Telegram</td><td>getUpdates / Webhook</td><td>Bot API</td><td>Chat ID、Inline Keyboard</td></tr><tr><td>Discord</td><td>Gateway / HTTP API</td><td>REST / Gateway</td><td>Guild、Channel、Interaction</td></tr><tr><td>微信</td><td>回调 / 轮询 / 企业微信 API</td><td>平台 API</td><td>企业身份、群聊、审批流</td></tr></tbody></table>
<p>如果业务层直接适配每个平台&#xff0c;就会非常混乱。</p> 
<p>所以需要 Gateway Plugin 做一层抽象。</p> 
<h3>7.2 Gateway Plugin 的核心目标</h3> 
<p>Gateway Plugin 解决的是&#xff1a;</p> 


```text
不同聊天平台输入输出不统一的问题
```

 
<p>它把平台消息统一转换为内部标准消息&#xff1a;</p> 


```python
<span class="token decorator annotation punctuation">@dataclass</span>
<span class="token keyword">class</span> <span class="token class-name">Message</span><span class="token punctuation">:</span>
platform<span class="token punctuation">:</span> <span class="token builtin">str</span>
conversation_id<span class="token punctuation">:</span> <span class="token builtin">str</span>
user_id<span class="token punctuation">:</span> <span class="token builtin">str</span>
text<span class="token punctuation">:</span> <span class="token builtin">str</span>
attachments<span class="token punctuation">:</span> <span class="token builtin">list</span>
raw<span class="token punctuation">:</span> <span class="token builtin">dict</span>
```

 
<p>然后交给 Agent Runtime&#xff1a;</p> 


```text
Slack Message
Telegram Update
Discord Event
Wechat Message
|
v
Gateway Plugin
|
v
统一 Message 对象
|
v
Agent Runtime
```

 
<p>Agent 返回结果后&#xff0c;再由对应插件转换回平台格式&#xff1a;</p> 


```text
Agent Response
|
v
Gateway Plugin
|
v
Slack / Telegram / Discord / WeChat
```

 
<h3>7.3 Gateway Plugin 架构示例</h3> 


```text
+----------------------+
|      Slack Plugin    |
+----------------------+
|
+----------------------+
|    Telegram Plugin   |
+----------------------+
|
+----------------------+
|     WeChat Plugin    |
+----------------------+
|
v
+----------------------+
|   Gateway Core       |
| - auth               |
| - routing            |
| - session mapping    |
| - rate limit         |
| - retry              |
+----------------------+
|
v
+----------------------+
|   Agent Runtime      |
| - memory             |
| - tools              |
| - skills             |
| - model              |
+----------------------+
```

 
<h3>7.4 Gateway Plugin 的关键模块</h3> 
<h4>1. 平台认证</h4> 


```python
<span class="token keyword">def</span> <span class="token function">authenticate</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token builtin">bool</span><span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>
```

 
<p>不同平台认证方式不同&#xff1a;</p> 
<ul><li>Slack&#xff1a;OAuth / Bot Token</li><li>Telegram&#xff1a;Bot Token</li><li>Discord&#xff1a;Bot Token</li><li>企业微信&#xff1a;CorpID / Secret / AgentID</li></ul> 
<p>插件层负责处理平台认证&#xff0c;不要把这些细节暴露给 Agent Runtime。</p> 
<h4>2. 消息接收</h4> 


```python
<span class="token keyword">def</span> <span class="token function">receive_message</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> Message<span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>
```

 
<p>不同平台收到的原始消息结构完全不同。</p> 
<p>插件要把它转成统一 Message。</p> 
<p>例如 Slack 原始事件&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"message"</span><span class="token punctuation">,</span>
<span class="token string-property property">"user"</span><span class="token operator">:</span> <span class="token string">"U123"</span><span class="token punctuation">,</span>
<span class="token string-property property">"channel"</span><span class="token operator">:</span> <span class="token string">"C123"</span><span class="token punctuation">,</span>
<span class="token string-property property">"text"</span><span class="token operator">:</span> <span class="token string">"帮我总结一下今天的工作"</span>
<span class="token punctuation">}</span>
```

 
<p>统一成&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"platform"</span><span class="token operator">:</span> <span class="token string">"slack"</span><span class="token punctuation">,</span>
<span class="token string-property property">"conversation_id"</span><span class="token operator">:</span> <span class="token string">"C123"</span><span class="token punctuation">,</span>
<span class="token string-property property">"user_id"</span><span class="token operator">:</span> <span class="token string">"U123"</span><span class="token punctuation">,</span>
<span class="token string-property property">"text"</span><span class="token operator">:</span> <span class="token string">"帮我总结一下今天的工作"</span>
<span class="token punctuation">}</span>
```

 
<h4>3. 消息发送</h4> 


```python
<span class="token keyword">def</span> <span class="token function">send_message</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> msg<span class="token punctuation">:</span> Message<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token boolean">None</span><span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>
```

 
<p>Agent 返回的统一结构&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"conversation_id"</span><span class="token operator">:</span> <span class="token string">"C123"</span><span class="token punctuation">,</span>
<span class="token string-property property">"text"</span><span class="token operator">:</span> <span class="token string">"今天你主要完成了三个事项……"</span>
<span class="token punctuation">}</span>
```

 
<p>Slack 插件转换成 Slack Web API 调用。</p> 
<p>Telegram 插件转换成 <code>sendMessage</code>。</p> 
<p>Discord 插件转换成 Discord 消息发送。</p> 
<h4>4. 会话映射</h4> 
<p>同一个用户可能在多个平台出现&#xff1a;</p> 


```text
Slack user U123
Telegram user T456
企业微信 user W789
```

 
<p>Gateway 需要维护身份映射&#xff1a;</p> 


```text
platform_user_id -> internal_user_id
```

 
<p>这样才能共享 Agent 记忆和上下文。</p> 
<h4>5. 平台能力降级</h4> 
<p>不同平台支持的能力不同。</p> 
<p>例如&#xff1a;</p> 
<ul><li>Slack 支持 Thread。</li><li>Telegram 支持 Inline Keyboard。</li><li>Discord 支持 Slash Command。</li><li>微信可能有自己的卡片消息。</li></ul> 
<p>Agent Runtime 不应该关心这些差异。</p> 
<p>Gateway Plugin 应该做能力适配&#xff1a;</p> 


```python
<span class="token keyword">class</span> <span class="token class-name">PlatformCapabilities</span><span class="token punctuation">:</span>
supports_thread<span class="token punctuation">:</span> <span class="token builtin">bool</span>
supports_markdown<span class="token punctuation">:</span> <span class="token builtin">bool</span>
supports_file_upload<span class="token punctuation">:</span> <span class="token builtin">bool</span>
supports_buttons<span class="token punctuation">:</span> <span class="token builtin">bool</span>
```

 
<p>如果平台不支持某种能力&#xff0c;就降级成普通文本。</p> 
<hr /> 
<h2>八、把 MCP、ACP、Gateway Plugin 放到一个系统里</h2> 
<p>在一个完整 AI Agent 系统中&#xff0c;这三者可以同时存在。</p> 
<p>例如你要做一个企业 AI 助手&#xff1a;</p> 


```text
用户入口：
- IDE
- Claude Desktop
- 企业微信
- Slack

Agent 能力：
- 查询内部 Wiki
- 查询数据库
- 读取代码仓库
- 修改代码
- 生成日报
```

 
<p>这时候架构可以是&#xff1a;</p> 


```text
                  +-------------------+
|   Claude Desktop  |
+-------------------+
|
| MCP
v
+-------------------+    +-------------------+
|   Wiki MCP Server |    |   DB MCP Server   |
+-------------------+    +-------------------+

+-------------------+
|   Cursor / IDE    |
+-------------------+
|
| ACP
v
+-------------------+
|   Coding Agent    |
+-------------------+

+-------------------+     +-------------------+
| Slack / WeChat    | --> | Gateway Plugin    |
| Telegram / Discord|     | Platform Adapter  |
+-------------------+     +-------------------+
|
v
+-------------------+
|   Agent Runtime   |
+-------------------+
```

 
<h3>8.1 分工关系</h3> 
<table><thead><tr><th>模块</th><th>解决什么问题</th></tr></thead><tbody><tr><td>MCP</td><td>Agent 如何调用外部工具</td></tr><tr><td>ACP</td><td>IDE 如何驱动 Agent 执行代码任务</td></tr><tr><td>Gateway Plugin</td><td>聊天平台如何接入 Agent</td></tr><tr><td>LSP</td><td>IDE 如何获得语言能力</td></tr><tr><td>JSON-RPC</td><td>请求、响应、通知的消息格式</td></tr><tr><td>stdio / HTTP / SSE</td><td>消息传输通道</td></tr></tbody></table>
<h3>8.2 一个完整调用链示例</h3> 
<p>用户在 Slack 里问&#xff1a;</p> 


```text
公司年假怎么申请？
```

 
<p>流程&#xff1a;</p> 


```text
1. Slack Plugin 收到 message event
2. Gateway 转成统一 Message
3. Agent Runtime 接收用户问题
4. Agent 判断需要查询内部制度
5. Agent 通过 MCP 调用 wiki_search 工具
6. Wiki MCP Server 返回制度内容
7. Agent 组织回答
8. Gateway 调用 Slack Plugin 发回消息
```

 
<p>调用链&#xff1a;</p> 


```text
Slack
-> Gateway Plugin
-> Agent Runtime
-> MCP Client
-> Wiki MCP Server
-> Agent Runtime
-> Gateway Plugin
-> Slack
```

 
<p>用户在 Cursor 里说&#xff1a;</p> 


```text
帮我重构 order_service.py，并补充测试。
```

 
<p>流程&#xff1a;</p> 


```text
1. Cursor 通过 ACP 创建 session
2. Cursor 发送 session/prompt
3. Coding Agent 分析项目文件
4. Agent 请求文件读写权限
5. Agent 修改代码
6. Agent 返回进度 update
7. Agent 返回最终 diff 和总结
```

 
<p>调用链&#xff1a;</p> 


```text
Cursor
-> ACP
-> Coding Agent
-> File System / Tools
-> ACP Update
-> Cursor UI
```

 
<hr /> 
<h2>九、为什么这些协议都喜欢“initialize &#43; capabilities”&#xff1f;</h2> 
<p>你会发现 LSP、MCP、ACP 都有一个共同点&#xff1a;</p> 


```text
第一步都是 initialize。
```

 
<p>这是为什么&#xff1f;</p> 
<p>因为这类协议都不是一次性 HTTP API&#xff0c;而是长连接、长生命周期、双向协作。</p> 
<p>双方需要先交换能力。</p> 
<p>例如&#xff1a;</p> 
<h3>LSP initialize</h3> 
<p>编辑器问语言服务器&#xff1a;</p> 


```text
你支持 hover 吗？
你支持 completion 吗？
你支持 definition 吗？
```

 
<p>语言服务器返回&#xff1a;</p> 


```text
我支持 hover、completion、diagnostics。
```

 
<h3>MCP initialize</h3> 
<p>AI Client 问 MCP Server&#xff1a;</p> 


```text
你支持 tools 吗？
你支持 resources 吗？
你支持 prompts 吗？
```

 
<p>MCP Server 返回&#xff1a;</p> 


```text
我支持 tools/list 和 tools/call。
```

 
<h3>ACP initialize</h3> 
<p>IDE 问 Coding Agent&#xff1a;</p> 


```text
你支持文件编辑吗？
你支持任务取消吗？
你支持流式进度吗？
```

 
<p>Agent 返回&#xff1a;</p> 


```text
我支持 file edit、streaming update、cancel。
```

 
<p>这就是 capabilities 协商。</p> 
<p>它的好处是&#xff1a;</p> 
<ol><li>客户端不用提前写死服务端能力。</li><li>服务端可以按需暴露能力。</li><li>不同版本之间更容易兼容。</li><li>可以做渐进式增强。</li><li>可以支持实验性能力。</li></ol> 
<hr /> 
<h2>十、协议设计中的几个关键概念</h2> 
<h3>10.1 Request / Response</h3> 
<p>用于需要结果的调用。</p> 
<p>例如&#xff1a;</p> 


```text
tools/list
tools/call
session/prompt
textDocument/hover
```

 
<p>都有明确返回值。</p> 
<h3>10.2 Notification</h3> 
<p>用于不需要响应的事件。</p> 
<p>例如&#xff1a;</p> 


```text
initialized
progress update
file changed
diagnostics published
```

 
<p>适合“我告诉你一件事&#xff0c;但你不用回复我”。</p> 
<h3>10.3 Capability</h3> 
<p>表示一方支持什么能力。</p> 
<p>例如&#xff1a;</p> 


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"capabilities"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"tools"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"resources"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"prompts"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```

 
<h3>10.4 Session</h3> 
<p>表示一次上下文连续的任务。</p> 
<p>ACP 中 session 很重要&#xff0c;因为 coding agent 通常不是一次调用就结束&#xff0c;而是持续多轮工作。</p> 
<h3>10.5 Transport</h3> 
<p>Transport 是消息怎么传。</p> 
<p>常见有&#xff1a;</p> 


```text
stdio
HTTP
SSE
WebSocket
Streamable HTTP
```

 
<p>协议定义“消息长什么样”&#xff0c;Transport 定义“消息怎么发”。</p> 
<hr /> 
<h2>十一、自己实现一个最小 MCP Server</h2> 
<p>下面用 Python 写一个非常简化的 MCP 风格 Server。</p> 
<p>注意&#xff1a;这是教学示例&#xff0c;不是完整 MCP SDK 实现。</p> 


```python
<span class="token keyword">import</span> sys
<span class="token keyword">import</span> json

TOOLS <span class="token operator">=</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string">"name"</span><span class="token punctuation">:</span> <span class="token string">"search_company_wiki"</span><span class="token punctuation">,</span>
<span class="token string">"description"</span><span class="token punctuation">:</span> <span class="token string">"搜索公司内部 Wiki"</span><span class="token punctuation">,</span>
<span class="token string">"inputSchema"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"object"</span><span class="token punctuation">,</span>
<span class="token string">"properties"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"query"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"string"</span><span class="token punctuation">,</span>
<span class="token string">"description"</span><span class="token punctuation">:</span> <span class="token string">"搜索关键词"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string">"required"</span><span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token string">"query"</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>

<span class="token keyword">def</span> <span class="token function">send_response</span><span class="token punctuation">(</span>request_id<span class="token punctuation">,</span> result<span class="token punctuation">)</span><span class="token punctuation">:</span>
resp <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"jsonrpc"</span><span class="token punctuation">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string">"id"</span><span class="token punctuation">:</span> request_id<span class="token punctuation">,</span>
<span class="token string">"result"</span><span class="token punctuation">:</span> result
<span class="token punctuation">}</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>json<span class="token punctuation">.</span>dumps<span class="token punctuation">(</span>resp<span class="token punctuation">,</span> ensure_ascii<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span><span class="token punctuation">,</span> flush<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">send_error</span><span class="token punctuation">(</span>request_id<span class="token punctuation">,</span> code<span class="token punctuation">,</span> message<span class="token punctuation">)</span><span class="token punctuation">:</span>
resp <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"jsonrpc"</span><span class="token punctuation">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string">"id"</span><span class="token punctuation">:</span> request_id<span class="token punctuation">,</span>
<span class="token string">"error"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"code"</span><span class="token punctuation">:</span> code<span class="token punctuation">,</span>
<span class="token string">"message"</span><span class="token punctuation">:</span> message
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>json<span class="token punctuation">.</span>dumps<span class="token punctuation">(</span>resp<span class="token punctuation">,</span> ensure_ascii<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span><span class="token punctuation">,</span> flush<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">handle_initialize</span><span class="token punctuation">(</span>req<span class="token punctuation">)</span><span class="token punctuation">:</span>
send_response<span class="token punctuation">(</span>req<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"protocolVersion"</span><span class="token punctuation">:</span> <span class="token string">"2024-11-05"</span><span class="token punctuation">,</span>
<span class="token string">"capabilities"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"tools"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string">"serverInfo"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"name"</span><span class="token punctuation">:</span> <span class="token string">"demo-wiki-mcp"</span><span class="token punctuation">,</span>
<span class="token string">"version"</span><span class="token punctuation">:</span> <span class="token string">"1.0.0"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">handle_tools_list</span><span class="token punctuation">(</span>req<span class="token punctuation">)</span><span class="token punctuation">:</span>
send_response<span class="token punctuation">(</span>req<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"tools"</span><span class="token punctuation">:</span> TOOLS
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">handle_tools_call</span><span class="token punctuation">(</span>req<span class="token punctuation">)</span><span class="token punctuation">:</span>
params <span class="token operator">=</span> req<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"params"</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">)</span>
name <span class="token operator">=</span> params<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"name"</span><span class="token punctuation">)</span>
arguments <span class="token operator">=</span> params<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"arguments"</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">if</span> name <span class="token operator">!=</span> <span class="token string">"search_company_wiki"</span><span class="token punctuation">:</span>
send_error<span class="token punctuation">(</span>req<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token operator">-</span><span class="token number">32601</span><span class="token punctuation">,</span> <span class="token string-interpolation"><span class="token string">f"Unknown tool: </span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>name<span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span>
<span class="token keyword">return</span>

query <span class="token operator">=</span> arguments<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"query"</span><span class="token punctuation">,</span> <span class="token string">""</span><span class="token punctuation">)</span>

<span class="token comment"># 这里应该接真实的 Wiki / RAG / 数据库</span>
result <span class="token operator">=</span> <span class="token string-interpolation"><span class="token string">f"你查询的是：</span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>query<span class="token punctuation">}</span></span><span class="token string">。这里返回公司内部 Wiki 的模拟结果。"</span></span>

send_response<span class="token punctuation">(</span>req<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"content"</span><span class="token punctuation">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"text"</span><span class="token punctuation">,</span>
<span class="token string">"text"</span><span class="token punctuation">:</span> result
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">main</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">for</span> line <span class="token keyword">in</span> sys<span class="token punctuation">.</span>stdin<span class="token punctuation">:</span>
line <span class="token operator">=</span> line<span class="token punctuation">.</span>strip<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">if</span> <span class="token keyword">not</span> line<span class="token punctuation">:</span>
<span class="token keyword">continue</span>

req <span class="token operator">=</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>line<span class="token punctuation">)</span>
method <span class="token operator">=</span> req<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"method"</span><span class="token punctuation">)</span>

<span class="token keyword">if</span> method <span class="token operator">==</span> <span class="token string">"initialize"</span><span class="token punctuation">:</span>
handle_initialize<span class="token punctuation">(</span>req<span class="token punctuation">)</span>
<span class="token keyword">elif</span> method <span class="token operator">==</span> <span class="token string">"tools/list"</span><span class="token punctuation">:</span>
handle_tools_list<span class="token punctuation">(</span>req<span class="token punctuation">)</span>
<span class="token keyword">elif</span> method <span class="token operator">==</span> <span class="token string">"tools/call"</span><span class="token punctuation">:</span>
handle_tools_call<span class="token punctuation">(</span>req<span class="token punctuation">)</span>
<span class="token keyword">else</span><span class="token punctuation">:</span>
send_error<span class="token punctuation">(</span>req<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"id"</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token operator">-</span><span class="token number">32601</span><span class="token punctuation">,</span> <span class="token string-interpolation"><span class="token string">f"Method not found: </span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>method<span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span>

<span class="token keyword">if</span> __name__ <span class="token operator">==</span> <span class="token string">"__main__"</span><span class="token punctuation">:</span>
main<span class="token punctuation">(</span><span class="token punctuation">)</span>
```

 
<p>这个最小版本实现了&#xff1a;</p> 


```text
initialize
tools/list
tools/call
```

 
<p>也就是 MCP 最核心的工具调用闭环。</p> 
<hr /> 
<h2>十二、自己实现一个最小 ACP 风格 Agent</h2> 
<p>下面是一个简化版 ACP 风格 Agent。</p> 


```python
<span class="token keyword">import</span> sys
<span class="token keyword">import</span> json
<span class="token keyword">import</span> uuid

sessions <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>

<span class="token keyword">def</span> <span class="token function">send_response</span><span class="token punctuation">(</span>request_id<span class="token punctuation">,</span> result<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>json<span class="token punctuation">.</span>dumps<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"jsonrpc"</span><span class="token punctuation">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string">"id"</span><span class="token punctuation">:</span> request_id<span class="token punctuation">,</span>
<span class="token string">"result"</span><span class="token punctuation">:</span> result
<span class="token punctuation">}</span><span class="token punctuation">,</span> ensure_ascii<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span><span class="token punctuation">,</span> flush<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">send_notification</span><span class="token punctuation">(</span>method<span class="token punctuation">,</span> params<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>json<span class="token punctuation">.</span>dumps<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"jsonrpc"</span><span class="token punctuation">:</span> <span class="token string">"2.0"</span><span class="token punctuation">,</span>
<span class="token string">"method"</span><span class="token punctuation">:</span> method<span class="token punctuation">,</span>
<span class="token string">"params"</span><span class="token punctuation">:</span> params
<span class="token punctuation">}</span><span class="token punctuation">,</span> ensure_ascii<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span><span class="token punctuation">,</span> flush<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">handle_initialize</span><span class="token punctuation">(</span>req<span class="token punctuation">)</span><span class="token punctuation">:</span>
send_response<span class="token punctuation">(</span>req<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"agentInfo"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"name"</span><span class="token punctuation">:</span> <span class="token string">"demo-coding-agent"</span><span class="token punctuation">,</span>
<span class="token string">"version"</span><span class="token punctuation">:</span> <span class="token string">"1.0.0"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string">"capabilities"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"streaming"</span><span class="token punctuation">:</span> <span class="token boolean">True</span><span class="token punctuation">,</span>
<span class="token string">"cancellation"</span><span class="token punctuation">:</span> <span class="token boolean">True</span><span class="token punctuation">,</span>
<span class="token string">"fileEdit"</span><span class="token punctuation">:</span> <span class="token boolean">False</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">handle_session_new</span><span class="token punctuation">(</span>req<span class="token punctuation">)</span><span class="token punctuation">:</span>
session_id <span class="token operator">=</span> <span class="token builtin">str</span><span class="token punctuation">(</span>uuid<span class="token punctuation">.</span>uuid4<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
sessions<span class="token punctuation">[</span>session_id<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"messages"</span><span class="token punctuation">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"cancelled"</span><span class="token punctuation">:</span> <span class="token boolean">False</span>
<span class="token punctuation">}</span>

send_response<span class="token punctuation">(</span>req<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"sessionId"</span><span class="token punctuation">:</span> session_id
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">handle_session_prompt</span><span class="token punctuation">(</span>req<span class="token punctuation">)</span><span class="token punctuation">:</span>
params <span class="token operator">=</span> req<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"params"</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">)</span>
session_id <span class="token operator">=</span> params<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"sessionId"</span><span class="token punctuation">)</span>
prompt <span class="token operator">=</span> params<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"prompt"</span><span class="token punctuation">)</span>

<span class="token keyword">if</span> session_id <span class="token keyword">not</span> <span class="token keyword">in</span> sessions<span class="token punctuation">:</span>
send_response<span class="token punctuation">(</span>req<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"stopReason"</span><span class="token punctuation">:</span> <span class="token string">"error"</span><span class="token punctuation">,</span>
<span class="token string">"message"</span><span class="token punctuation">:</span> <span class="token string">"Session not found"</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>
<span class="token keyword">return</span>

sessions<span class="token punctuation">[</span>session_id<span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"messages"</span><span class="token punctuation">]</span><span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> prompt
<span class="token punctuation">}</span><span class="token punctuation">)</span>

send_notification<span class="token punctuation">(</span><span class="token string">"session/update"</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"sessionId"</span><span class="token punctuation">:</span> session_id<span class="token punctuation">,</span>
<span class="token string">"message"</span><span class="token punctuation">:</span> <span class="token string">"正在分析任务..."</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

send_notification<span class="token punctuation">(</span><span class="token string">"session/update"</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"sessionId"</span><span class="token punctuation">:</span> session_id<span class="token punctuation">,</span>
<span class="token string">"message"</span><span class="token punctuation">:</span> <span class="token string">"正在制定执行计划..."</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

send_response<span class="token punctuation">(</span>req<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"stopReason"</span><span class="token punctuation">:</span> <span class="token string">"completed"</span><span class="token punctuation">,</span>
<span class="token string">"summary"</span><span class="token punctuation">:</span> <span class="token string-interpolation"><span class="token string">f"已收到任务：</span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>prompt<span class="token punctuation">}</span></span><span class="token string">。这里是模拟执行结果。"</span></span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">handle_session_cancel</span><span class="token punctuation">(</span>req<span class="token punctuation">)</span><span class="token punctuation">:</span>
params <span class="token operator">=</span> req<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"params"</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">)</span>
session_id <span class="token operator">=</span> params<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"sessionId"</span><span class="token punctuation">)</span>

<span class="token keyword">if</span> session_id <span class="token keyword">in</span> sessions<span class="token punctuation">:</span>
sessions<span class="token punctuation">[</span>session_id<span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"cancelled"</span><span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token boolean">True</span>

send_response<span class="token punctuation">(</span>req<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"cancelled"</span><span class="token punctuation">:</span> <span class="token boolean">True</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">main</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">for</span> line <span class="token keyword">in</span> sys<span class="token punctuation">.</span>stdin<span class="token punctuation">:</span>
line <span class="token operator">=</span> line<span class="token punctuation">.</span>strip<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">if</span> <span class="token keyword">not</span> line<span class="token punctuation">:</span>
<span class="token keyword">continue</span>

req <span class="token operator">=</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>line<span class="token punctuation">)</span>
method <span class="token operator">=</span> req<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"method"</span><span class="token punctuation">)</span>

<span class="token keyword">if</span> method <span class="token operator">==</span> <span class="token string">"initialize"</span><span class="token punctuation">:</span>
handle_initialize<span class="token punctuation">(</span>req<span class="token punctuation">)</span>
<span class="token keyword">elif</span> method <span class="token operator">==</span> <span class="token string">"session/new"</span><span class="token punctuation">:</span>
handle_session_new<span class="token punctuation">(</span>req<span class="token punctuation">)</span>
<span class="token keyword">elif</span> method <span class="token operator">==</span> <span class="token string">"session/prompt"</span><span class="token punctuation">:</span>
handle_session_prompt<span class="token punctuation">(</span>req<span class="token punctuation">)</span>
<span class="token keyword">elif</span> method <span class="token operator">==</span> <span class="token string">"session/cancel"</span><span class="token punctuation">:</span>
handle_session_cancel<span class="token punctuation">(</span>req<span class="token punctuation">)</span>
<span class="token keyword">else</span><span class="token punctuation">:</span>
send_response<span class="token punctuation">(</span>req<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"id"</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"error"</span><span class="token punctuation">:</span> <span class="token string-interpolation"><span class="token string">f"Unknown method: </span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>method<span class="token punctuation">}</span></span><span class="token string">"</span></span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">if</span> __name__ <span class="token operator">==</span> <span class="token string">"__main__"</span><span class="token punctuation">:</span>
main<span class="token punctuation">(</span><span class="token punctuation">)</span>
```

 
<p>这个示例体现了 ACP 的核心思想&#xff1a;</p> 


```text
不是调用一个工具，而是创建一个 Agent 会话，让 Agent 持续处理任务。
```

 
<hr /> 
<h2>十三、Gateway Plugin 最小实现</h2> 
<p>下面是一个简单的插件抽象。</p> 


```python
<span class="token keyword">from</span> dataclasses <span class="token keyword">import</span> dataclass
<span class="token keyword">from</span> typing <span class="token keyword">import</span> Any<span class="token punctuation">,</span> Dict<span class="token punctuation">,</span> List

<span class="token decorator annotation punctuation">@dataclass</span>
<span class="token keyword">class</span> <span class="token class-name">Message</span><span class="token punctuation">:</span>
platform<span class="token punctuation">:</span> <span class="token builtin">str</span>
conversation_id<span class="token punctuation">:</span> <span class="token builtin">str</span>
user_id<span class="token punctuation">:</span> <span class="token builtin">str</span>
text<span class="token punctuation">:</span> <span class="token builtin">str</span>
attachments<span class="token punctuation">:</span> List<span class="token punctuation">[</span>Any<span class="token punctuation">]</span>
raw<span class="token punctuation">:</span> Dict<span class="token punctuation">[</span><span class="token builtin">str</span><span class="token punctuation">,</span> Any<span class="token punctuation">]</span>

<span class="token keyword">class</span> <span class="token class-name">PlatformPlugin</span><span class="token punctuation">:</span>
name<span class="token punctuation">:</span> <span class="token builtin">str</span>

<span class="token keyword">def</span> <span class="token function">authenticate</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token builtin">bool</span><span class="token punctuation">:</span>
<span class="token keyword">raise</span> NotImplementedError

<span class="token keyword">def</span> <span class="token function">receive_message</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> Message<span class="token punctuation">:</span>
<span class="token keyword">raise</span> NotImplementedError

<span class="token keyword">def</span> <span class="token function">send_message</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> msg<span class="token punctuation">:</span> Message<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token boolean">None</span><span class="token punctuation">:</span>
<span class="token keyword">raise</span> NotImplementedError
```

 
<p>Slack 插件示例&#xff1a;</p> 


```python
<span class="token keyword">class</span> <span class="token class-name">SlackPlugin</span><span class="token punctuation">(</span>PlatformPlugin<span class="token punctuation">)</span><span class="token punctuation">:</span>
name <span class="token operator">=</span> <span class="token string">"slack"</span>

<span class="token keyword">def</span> <span class="token function">authenticate</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token builtin">bool</span><span class="token punctuation">:</span>
<span class="token comment"># 校验 bot token</span>
<span class="token keyword">return</span> <span class="token boolean">True</span>

<span class="token keyword">def</span> <span class="token function">receive_message</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> Message<span class="token punctuation">:</span>
<span class="token comment"># 从 Slack Events API 接收事件</span>
event <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"channel"</span><span class="token punctuation">:</span> <span class="token string">"C123"</span><span class="token punctuation">,</span>
<span class="token string">"user"</span><span class="token punctuation">:</span> <span class="token string">"U123"</span><span class="token punctuation">,</span>
<span class="token string">"text"</span><span class="token punctuation">:</span> <span class="token string">"帮我总结一下今天的工作"</span>
<span class="token punctuation">}</span>

<span class="token keyword">return</span> Message<span class="token punctuation">(</span>
platform<span class="token operator">=</span><span class="token string">"slack"</span><span class="token punctuation">,</span>
conversation_id<span class="token operator">=</span>event<span class="token punctuation">[</span><span class="token string">"channel"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
user_id<span class="token operator">=</span>event<span class="token punctuation">[</span><span class="token string">"user"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
text<span class="token operator">=</span>event<span class="token punctuation">[</span><span class="token string">"text"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
attachments<span class="token operator">=</span><span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
raw<span class="token operator">=</span>event
<span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">send_message</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> msg<span class="token punctuation">:</span> Message<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token boolean">None</span><span class="token punctuation">:</span>
<span class="token comment"># 调用 Slack Web API 发送消息</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"[Slack] send to </span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>msg<span class="token punctuation">.</span>conversation_id<span class="token punctuation">}</span></span><span class="token string">: </span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>msg<span class="token punctuation">.</span>text<span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span>
```

 
<p>Telegram 插件示例&#xff1a;</p> 


```python
<span class="token keyword">class</span> <span class="token class-name">TelegramPlugin</span><span class="token punctuation">(</span>PlatformPlugin<span class="token punctuation">)</span><span class="token punctuation">:</span>
name <span class="token operator">=</span> <span class="token string">"telegram"</span>

<span class="token keyword">def</span> <span class="token function">authenticate</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token builtin">bool</span><span class="token punctuation">:</span>
<span class="token comment"># 校验 bot token</span>
<span class="token keyword">return</span> <span class="token boolean">True</span>

<span class="token keyword">def</span> <span class="token function">receive_message</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> Message<span class="token punctuation">:</span>
update <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"message"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"chat"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span><span class="token string">"id"</span><span class="token punctuation">:</span> <span class="token string">"T_CHAT_123"</span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string">"from"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span><span class="token string">"id"</span><span class="token punctuation">:</span> <span class="token string">"T_USER_456"</span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string">"text"</span><span class="token punctuation">:</span> <span class="token string">"帮我查一下年假制度"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>

msg <span class="token operator">=</span> update<span class="token punctuation">[</span><span class="token string">"message"</span><span class="token punctuation">]</span>

<span class="token keyword">return</span> Message<span class="token punctuation">(</span>
platform<span class="token operator">=</span><span class="token string">"telegram"</span><span class="token punctuation">,</span>
conversation_id<span class="token operator">=</span><span class="token builtin">str</span><span class="token punctuation">(</span>msg<span class="token punctuation">[</span><span class="token string">"chat"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
user_id<span class="token operator">=</span><span class="token builtin">str</span><span class="token punctuation">(</span>msg<span class="token punctuation">[</span><span class="token string">"from"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
text<span class="token operator">=</span>msg<span class="token punctuation">[</span><span class="token string">"text"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
attachments<span class="token operator">=</span><span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
raw<span class="token operator">=</span>update
<span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">send_message</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> msg<span class="token punctuation">:</span> Message<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token boolean">None</span><span class="token punctuation">:</span>
<span class="token comment"># 调用 Telegram sendMessage</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string-interpolation"><span class="token string">f"[Telegram] send to </span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>msg<span class="token punctuation">.</span>conversation_id<span class="token punctuation">}</span></span><span class="token string">: </span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>msg<span class="token punctuation">.</span>text<span class="token punctuation">}</span></span><span class="token string">"</span></span><span class="token punctuation">)</span>
```

 
<p>Gateway Core&#xff1a;</p> 


```python
<span class="token keyword">class</span> <span class="token class-name">Gateway</span><span class="token punctuation">:</span>
<span class="token keyword">def</span> <span class="token function">__init__</span><span class="token punctuation">(</span>self<span class="token punctuation">)</span><span class="token punctuation">:</span>
self<span class="token punctuation">.</span>plugins <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>

<span class="token keyword">def</span> <span class="token function">register</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> plugin<span class="token punctuation">:</span> PlatformPlugin<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">if</span> plugin<span class="token punctuation">.</span>authenticate<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
self<span class="token punctuation">.</span>plugins<span class="token punctuation">[</span>plugin<span class="token punctuation">.</span>name<span class="token punctuation">]</span> <span class="token operator">=</span> plugin

<span class="token keyword">def</span> <span class="token function">handle_message</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> plugin_name<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
plugin <span class="token operator">=</span> self<span class="token punctuation">.</span>plugins<span class="token punctuation">[</span>plugin_name<span class="token punctuation">]</span>

incoming <span class="token operator">=</span> plugin<span class="token punctuation">.</span>receive_message<span class="token punctuation">(</span><span class="token punctuation">)</span>

response_text <span class="token operator">=</span> self<span class="token punctuation">.</span>call_agent<span class="token punctuation">(</span>incoming<span class="token punctuation">)</span>

outgoing <span class="token operator">=</span> Message<span class="token punctuation">(</span>
platform<span class="token operator">=</span>incoming<span class="token punctuation">.</span>platform<span class="token punctuation">,</span>
conversation_id<span class="token operator">=</span>incoming<span class="token punctuation">.</span>conversation_id<span class="token punctuation">,</span>
user_id<span class="token operator">=</span>incoming<span class="token punctuation">.</span>user_id<span class="token punctuation">,</span>
text<span class="token operator">=</span>response_text<span class="token punctuation">,</span>
attachments<span class="token operator">=</span><span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
raw<span class="token operator">=</span><span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
<span class="token punctuation">)</span>

plugin<span class="token punctuation">.</span>send_message<span class="token punctuation">(</span>outgoing<span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">call_agent</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> msg<span class="token punctuation">:</span> Message<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token builtin">str</span><span class="token punctuation">:</span>
<span class="token comment"># 这里可以调用真实 Agent Runtime</span>
<span class="token keyword">return</span> <span class="token string-interpolation"><span class="token string">f"Agent 收到你的消息：</span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>msg<span class="token punctuation">.</span>text<span class="token punctuation">}</span></span><span class="token string">"</span></span>
```

 
<p>使用&#xff1a;</p> 


```python
gateway <span class="token operator">=</span> Gateway<span class="token punctuation">(</span><span class="token punctuation">)</span>
gateway<span class="token punctuation">.</span>register<span class="token punctuation">(</span>SlackPlugin<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
gateway<span class="token punctuation">.</span>register<span class="token punctuation">(</span>TelegramPlugin<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

gateway<span class="token punctuation">.</span>handle_message<span class="token punctuation">(</span><span class="token string">"slack"</span><span class="token punctuation">)</span>
gateway<span class="token punctuation">.</span>handle_message<span class="token punctuation">(</span><span class="token string">"telegram"</span><span class="token punctuation">)</span>
```

 
<hr /> 
<h2>十四、这几类协议应该怎么选&#xff1f;</h2> 
<h3>场景一&#xff1a;我想让 Claude / Cursor 调用我写的工具</h3> 
<p>用 MCP。</p> 
<p>例如&#xff1a;</p> 


```text
查数据库
查 Wiki
读文件
调用内部 API
查订单
查日志
```

 
<p>你应该写&#xff1a;</p> 


```text
xxx_mcp_server.py
```

 
<p>暴露&#xff1a;</p> 


```text
tools/list
tools/call
```

 
<h3>场景二&#xff1a;我想让 IDE 调用我的 coding agent</h3> 
<p>用 ACP。</p> 
<p>例如&#xff1a;</p> 


```text
帮我重构代码
帮我修 bug
帮我生成测试
帮我解释项目结构
帮我迁移接口
```

 
<p>你应该实现&#xff1a;</p> 


```text
initialize
session/new
session/prompt
session/update
session/cancel
```

 
<h3>场景三&#xff1a;我想让 Agent 接入微信、Slack、Telegram</h3> 
<p>用 Gateway Plugin。</p> 
<p>例如&#xff1a;</p> 


```text
企业微信 AI 助手
Slack 工作流助手
Telegram 私人助理
Discord 社区机器人
```

 
<p>你应该设计&#xff1a;</p> 


```text
PlatformPlugin
receive_message
send_message
authenticate
```

 
<h3>场景四&#xff1a;我想给 IDE 做代码补全、跳转、诊断</h3> 
<p>用 LSP。</p> 
<p>例如&#xff1a;</p> 


```text
自研 DSL 的语法检查
SQL 编辑器智能提示
配置文件 schema 校验
工作流 DSL 补全
```

 
<p>你应该实现&#xff1a;</p> 


```text
initialize
textDocument/didOpen
textDocument/didChange
textDocument/completion
textDocument/hover
textDocument/publishDiagnostics
```

 
<hr /> 
<h2>十五、总结</h2> 
<p>AI Agent 时代&#xff0c;协议会变得越来越重要。</p> 
<p>因为我们不再只是做一个“调用大模型的应用”&#xff0c;而是在构建一个复杂生态&#xff1a;</p> 


```text
AI Client
IDE
Agent
Tool
Resource
Prompt
File System
Chat Platform
Enterprise System
```

 
<p>这些组件之间如果没有统一协议&#xff0c;就会变成大量脆弱的胶水代码。</p> 
<p>本文介绍的几个协议和架构&#xff0c;可以这样理解&#xff1a;</p> 


```text
JSON-RPC：消息格式底座
stdio / HTTP / SSE：传输方式
LSP：IDE 调语言能力
MCP：AI 调外部工具
ACP：IDE 调 AI Agent
Gateway Plugin：聊天平台接入 Agent
```

 
<p>它们之间不是替代关系&#xff0c;而是分工关系。</p> 
<p>最终可以总结成一句话&#xff1a;</p> 
<blockquote> 
<p>LSP 解决“编辑器如何使用语言能力”&#xff0c;MCP 解决“模型如何使用外部工具”&#xff0c;ACP 解决“IDE 如何驱动 Agent”&#xff0c;Gateway Plugin 解决“不同聊天平台如何接入 Agent”。</p> 
</blockquote> 
<p>如果你正在做企业级 Agent、AI IDE、内部知识库助手、OA 智能助手、代码智能体平台&#xff0c;这几个协议和架构思想都非常值得深入理解。</p> 
<p>因为未来复杂 Agent 系统的竞争&#xff0c;不只是模型能力的竞争&#xff0c;更是&#xff1a;</p> 


```text
协议设计能力
工具生态能力
上下文组织能力
多端接入能力
工程化落地能力
```

 
<p>谁能把这些能力标准化、模块化、插件化&#xff0c;谁就更容易构建真正可扩展的 AI Agent 平台。</p>
