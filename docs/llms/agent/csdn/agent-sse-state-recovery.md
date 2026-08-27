---
title: "Agent 对话中的 SSE 状态恢复：切换页面、重新进入与刷新后如何继续流式输出"
description: "CSDN 原文全文镜像：本文探讨了AI Agent对话系统中SSE（Server-Sent Events）的设计挑战与解决方案。主要问题在于如何应对用户切换会话、页面刷新、网络中断等场景导致的状态丢失问题。作者提出\"快照+事件日志+实时流\"的三层架构：通过数据……"
pageType: article
module: agent
updated: '2026-08-04'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "人工智能"
  - "大模型"
  - "软件工程"
level: advanced
prerequisites:
  - "/llms/agent/"
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-04，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-04。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163478430](https://blog.csdn.net/m0_63309778/article/details/163478430)
- 站内分区：Agent / Agent SSE 状态恢复
:::

<p><img src="https://i-blog.csdnimg.cn/direct/9f876efb4d834a6ab2e8b069ae173bc8.png" alt="在这里插入图片描述" /></p>
<h3>前言</h3>
<p>在普通 AI 对话中&#xff0c;前端通常发起一次请求&#xff0c;后端调用大模型&#xff0c;并通过 SSE 将生成结果逐步推送给浏览器。</p>


```text
用户发送消息
↓
后端创建 Agent Run
↓
Agent 开始执行
↓
SSE 持续推送事件
↓
前端实时更新对话界面
```


<p>只要用户一直停留在当前页面&#xff0c;这套流程并不复杂。</p>
<p>真正棘手的问题出现在以下场景&#xff1a;</p>
<ul><li>Agent 正在输出时&#xff0c;用户切换到了另一个会话&#xff1b;</li><li>用户进入其他页面&#xff0c;稍后又返回&#xff1b;</li><li>浏览器刷新&#xff1b;</li><li>网络短暂断开&#xff1b;</li><li>前端重新部署导致页面重载&#xff1b;</li><li>同一个会话在多个标签页打开&#xff1b;</li><li>Agent 后台执行完了&#xff0c;但原来的 SSE 连接已经不存在。</li></ul>
<p>例如&#xff0c;Agent 已经执行到一半&#xff1a;</p>


```text
用户：分析这个项目的技术架构

Agent：
1. 正在读取项目文件
2. 正在分析服务依赖
3. 正在调用代码分析工具
4. 正在生成架构总结……
```


<p>这时用户刷新页面。</p>
<p>如果系统只依赖当前 SSE 连接&#xff0c;页面刷新后可能出现&#xff1a;</p>
<ul><li>已生成内容消失&#xff1b;</li><li>消息一直显示“生成中”&#xff1b;</li><li>工具调用卡在运行状态&#xff1b;</li><li>Agent 已经完成&#xff0c;但前端不知道&#xff1b;</li><li>重新连接后内容重复&#xff1b;</li><li>同一个回答出现两份&#xff1b;</li><li>无法恢复中间执行步骤。</li></ul>
<p>因此&#xff0c;Agent 对话中的 SSE 设计&#xff0c;不能只解决“如何流式输出”&#xff0c;还必须解决&#xff1a;</p>
<blockquote>
<p>当连接随时可能中断时&#xff0c;如何重建完整对话状态&#xff0c;并从中断位置继续接收事件。</p>
</blockquote>
<hr />
<h2>一、先理解 SSE 在 Agent 系统中的定位</h2>
<p>SSE&#xff0c;全称是 Server-Sent Events&#xff0c;是一种基于 HTTP 的服务器单向推送机制。</p>
<p>浏览器建立连接后&#xff0c;服务器持续返回&#xff1a;</p>


```text
Content-Type: text/event-stream
```


<p>服务器可以不断发送事件&#xff1a;</p>


```text
event: assistant.delta
id: 101
data: {"text":"你好"}

event: assistant.delta
id: 102
data: {"text":"，我是"}

event: assistant.delta
id: 103
data: {"text":"智能助手"}
```


<p>标准 SSE 事件可以包含 <code>event</code>、<code>data</code>、<code>id</code> 和 <code>retry</code> 等字段&#xff1b;浏览器的 <code>EventSource</code> 会保持长连接&#xff0c;并在连接异常时尝试重新连接。</p>
<p>但是必须认识到&#xff1a;</p>
<blockquote>
<p>SSE 只是事件传输通道&#xff0c;不应该成为 Agent 状态的唯一来源。</p>
</blockquote>
<p>如果所有状态只存在于当前连接中&#xff1a;</p>


```text
Agent 输出
↓
SSE 连接
↓
浏览器内存
```


<p>那么页面刷新后&#xff0c;浏览器内存和 SSE 连接都会消失。</p>
<p>更可靠的架构应该是&#xff1a;</p>


```text
持久化状态
+
可回放事件
+
实时 SSE
```


<p>也就是&#xff1a;</p>


```text
数据库快照
负责恢复完整页面

事件日志
负责补发断线期间的变化

SSE
负责推送最新实时事件
```


<hr />
<h2>二、核心设计&#xff1a;Snapshot &#43; Event Log &#43; Live Stream</h2>
<p>一个可恢复的 Agent 对话系统&#xff0c;建议采用三层结构。</p>


```text
第一层：Snapshot
当前会话和消息的完整快照

第二层：Event Log
Agent 执行过程中产生的有序事件

第三层：Live Stream
通过 SSE 推送最新事件
```


<h3>1. Snapshot&#xff1a;状态快照</h3>
<p>Snapshot 是当前会话已经确定的业务状态&#xff0c;例如&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"conversationId"</span><span class="token operator">:</span> <span class="token string">"conv_1001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"messages"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"msg_user_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"分析一下这个项目"</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"completed"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"msg_assistant_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"项目采用了微服务架构……"</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"streaming"</span><span class="token punctuation">,</span>
<span class="token string-property property">"runId"</span><span class="token operator">:</span> <span class="token string">"run_9001"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"activeRun"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"run_9001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"running"</span><span class="token punctuation">,</span>
<span class="token string-property property">"lastEventId"</span><span class="token operator">:</span> <span class="token string">"1058"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>它用于页面首次进入或刷新时恢复整个界面。</p>
<h3>2. Event Log&#xff1a;事件日志</h3>
<p>Event Log 记录 Agent Run 执行过程中产生的变化&#xff1a;</p>


```text
1051 run.started
1052 assistant.message.created
1053 assistant.delta
1054 assistant.delta
1055 tool.started
1056 tool.completed
1057 assistant.delta
1058 run.progress
```


<p>它用于补发用户离开期间错过的事件。</p>
<h3>3. Live Stream&#xff1a;实时流</h3>
<p>当历史事件补发完成后&#xff0c;SSE 连接继续等待新事件&#xff1a;</p>


```text
历史事件补发
↓
追上当前最新位置
↓
阻塞等待新事件
↓
实时推送
```


<p>因此&#xff0c;一个完整的恢复过程应该是&#xff1a;</p>


```text
读取 Snapshot
↓
获得当前 lastEventId
↓
补发 lastEventId 之后的事件
↓
进入实时监听
```


<hr />
<h2>三、Agent 对话中的几个核心数据对象</h2>
<p>建议至少明确区分以下四类对象。</p>
<h3>1. Conversation</h3>
<p>Conversation 表示一段完整会话。</p>


```text
conversation
├── id
├── user_id
├── title
├── status
├── created_at
└── updated_at
```


<h3>2. Message</h3>
<p>Message 表示用户或 Agent 的最终消息。</p>


```text
message
├── id
├── conversation_id
├── role
├── content
├── status
├── run_id
├── created_at
└── updated_at
```


<p>消息状态可以设计为&#xff1a;</p>


```text
pending
streaming
completed
failed
cancelled
```


<h3>3. Run</h3>
<p>Run 表示 Agent 的一次执行过程。</p>


```text
run
├── id
├── conversation_id
├── input_message_id
├── output_message_id
├── status
├── last_event_id
├── started_at
├── finished_at
└── error
```


<p>常见状态&#xff1a;</p>


```text
queued
running
waiting_tool
waiting_user
completed
failed
cancelled
```


<h3>4. Event</h3>
<p>Event 表示 Run 中发生的一次状态变化。</p>


```text
event
├── id
├── conversation_id
├── run_id
├── sequence
├── type
├── payload
└── created_at
```


<p>这四个概念不要混在一起&#xff1a;</p>


```text
Conversation：一段对话

Message：用户或 Agent 的消息

Run：Agent 的一次执行

Event：执行过程中发生的变化
```


<p>一段 Conversation 可以有很多 Message。</p>
<p>一次用户提问通常会创建一个 Run。</p>
<p>一个 Run 又会产生很多 Event。</p>
<hr />
<h2>四、不要只发送文本 Token&#xff0c;要发送结构化事件</h2>
<p>很多系统最初只发送一种 SSE 消息&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"text"</span><span class="token operator">:</span> <span class="token string">"你好"</span>
<span class="token punctuation">}</span>
```


<p>这种设计只能处理简单的大模型文本输出。</p>
<p>但 Agent 对话中通常还包含&#xff1a;</p>
<ul><li>思考状态&#xff1b;</li><li>计划&#xff1b;</li><li>工具调用&#xff1b;</li><li>子 Agent&#xff1b;</li><li>文件解析&#xff1b;</li><li>进度变化&#xff1b;</li><li>审批&#xff1b;</li><li>错误&#xff1b;</li><li>重试&#xff1b;</li><li>最终结果。</li></ul>
<p>因此应该设计结构化事件。</p>
<p>推荐事件类型包括&#xff1a;</p>


```text
run.created
run.started
run.progress
run.completed
run.failed
run.cancelled

message.created
assistant.delta
assistant.message.completed

tool.started
tool.progress
tool.completed
tool.failed

subagent.started
subagent.completed
subagent.failed

approval.required
approval.resolved

heartbeat
```


<p>统一事件结构可以设计为&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"eventId"</span><span class="token operator">:</span> <span class="token string">"1058"</span><span class="token punctuation">,</span>
<span class="token string-property property">"sequence"</span><span class="token operator">:</span> <span class="token number">1058</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool.completed"</span><span class="token punctuation">,</span>
<span class="token string-property property">"conversationId"</span><span class="token operator">:</span> <span class="token string">"conv_1001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"runId"</span><span class="token operator">:</span> <span class="token string">"run_9001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"blockId"</span><span class="token operator">:</span> <span class="token string">"block_tool_3"</span><span class="token punctuation">,</span>
<span class="token string-property property">"timestamp"</span><span class="token operator">:</span> <span class="token string">"2026-08-04T16:10:00+08:00"</span><span class="token punctuation">,</span>
<span class="token string-property property">"payload"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"toolName"</span><span class="token operator">:</span> <span class="token string">"search_documents"</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"completed"</span><span class="token punctuation">,</span>
<span class="token string-property property">"resultSummary"</span><span class="token operator">:</span> <span class="token string">"找到 12 个相关文档"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>对应 SSE&#xff1a;</p>


```text
event: tool.completed
id: 1058
data: {"eventId":"1058","sequence":1058,"type":"tool.completed","runId":"run_9001","blockId":"block_tool_3","payload":{"toolName":"search_documents","status":"completed"}}
```


<p>这里的 <code>id</code> 非常重要。</p>
<p>它不是为了展示&#xff0c;而是为了&#xff1a;</p>
<ul><li>事件排序&#xff1b;</li><li>前端去重&#xff1b;</li><li>断线补发&#xff1b;</li><li>判断缺失事件&#xff1b;</li><li>恢复消费位置。</li></ul>
<hr />
<h2>五、三种页面中断场景应该分别如何处理</h2>
<h3>场景一&#xff1a;切换到另一个会话</h3>
<p>例如用户从&#xff1a;</p>


```text
/conversations/1001
```


<p>切换到&#xff1a;</p>


```text
/conversations/1002
```


<p>此时前端应该主动关闭原会话的 SSE&#xff1a;</p>


```typescript
eventSource<span class="token punctuation">.</span><span class="token function">close</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
```


<p>关闭 SSE 只表示&#xff1a;</p>
<blockquote>
<p>当前页面不再实时接收这个 Run 的事件。</p>
</blockquote>
<p>它不应该取消后台 Agent Run。</p>
<p>正确关系是&#xff1a;</p>


```text
关闭页面订阅
≠
取消 Agent 执行
```


<p>Agent 仍然在服务器后台运行&#xff0c;继续&#xff1a;</p>
<ul><li>调用模型&#xff1b;</li><li>执行工具&#xff1b;</li><li>写入事件&#xff1b;</li><li>更新消息&#xff1b;</li><li>保存最终状态。</li></ul>
<p>当用户再次进入会话 1001 时&#xff0c;再重新恢复。</p>
<hr />
<h3>场景二&#xff1a;切换回来重新进入会话</h3>
<p>用户重新进入会话时&#xff0c;不要直接创建一个空页面然后只等待新 SSE。</p>
<p>正确顺序应该是&#xff1a;</p>


```text
1. 请求会话 Snapshot
2. 渲染已有消息
3. 检查是否存在 activeRun
4. 获取 lastEventId
5. 建立 SSE 连接
6. 补发遗漏事件
7. 继续实时接收
```


<p>例如&#xff1a;</p>


```http
GET /api/conversations/conv_1001
```


<p>返回&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"conversationId"</span><span class="token operator">:</span> <span class="token string">"conv_1001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"messages"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"activeRun"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"run_9001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"running"</span><span class="token punctuation">,</span>
<span class="token string-property property">"lastEventId"</span><span class="token operator">:</span> <span class="token string">"1058"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>然后前端建立&#xff1a;</p>


```http
GET /api/runs/run_9001/events?after=1058
```


<p>如果 Agent 在用户离开期间又产生了&#xff1a;</p>


```text
1059 assistant.delta
1060 tool.started
1061 tool.completed
1062 assistant.delta
```


<p>后端先补发这些事件&#xff0c;再继续等待 1063 之后的新事件。</p>
<hr />
<h3>场景三&#xff1a;浏览器刷新</h3>
<p>刷新与普通路由切换的区别是&#xff1a;</p>


```text
React/Vue 内存状态消失
SSE 对象消失
当前 lastEventId 可能消失
未完成消息的本地文本也可能消失
```


<p>因此页面刷新后的恢复不能依赖前端内存。</p>
<p>应该重新执行&#xff1a;</p>


```text
刷新页面
↓
查询 Conversation Snapshot
↓
恢复 Messages 和 Blocks
↓
识别 activeRun
↓
重新建立 SSE
↓
从服务端游标继续
```


<p>浏览器的 <code>EventSource</code> 在同一个对象发生临时断线时&#xff0c;可以自动重连&#xff0c;并在重新建立连接时使用最近收到的事件 ID&#xff1b;相关标准定义了 <code>Last-Event-ID</code> 请求头。</p>
<p>但是完整页面刷新后&#xff0c;原来的 <code>EventSource</code> 对象已经销毁。</p>
<p>新页面不能只依赖旧对象的自动恢复能力&#xff0c;因此仍然需要显式保存和恢复游标&#xff0c;例如&#xff1a;</p>


```text
服务端 Snapshot 中的 lastEventId
```


<p>或者&#xff1a;</p>


```text
GET /runs/{runId}/events?after={lastEventId}
```


<hr />
<h2>六、推荐的后端接口设计</h2>
<p>可以将“发送消息”和“订阅事件”拆成两个接口。</p>
<h3>1. 提交消息</h3>


```http
POST /api/conversations/{conversationId}/messages
```


<p>请求&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"请分析这个项目的技术架构"</span>
<span class="token punctuation">}</span>
```


<p>响应&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"messageId"</span><span class="token operator">:</span> <span class="token string">"msg_user_1001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"runId"</span><span class="token operator">:</span> <span class="token string">"run_9001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"queued"</span>
<span class="token punctuation">}</span>
```


<p>这个接口负责&#xff1a;</p>
<ol><li>保存用户消息&#xff1b;</li><li>创建 Assistant 占位消息&#xff1b;</li><li>创建 Agent Run&#xff1b;</li><li>启动后台 Agent&#xff1b;</li><li>立即返回 <code>runId</code>。</li></ol>
<p>不要让这个接口持续保持几十分钟的 HTTP 请求。</p>
<h3>2. 获取会话快照</h3>


```http
GET /api/conversations/{conversationId}
```


<p>负责返回&#xff1a;</p>
<ul><li>历史消息&#xff1b;</li><li>已完成工具调用&#xff1b;</li><li>当前运行状态&#xff1b;</li><li>Assistant 当前已生成文本&#xff1b;</li><li>最近事件游标。</li></ul>
<h3>3. 订阅 Run 事件</h3>


```http
GET /api/runs/{runId}/events?after={eventId}
```


<p>响应类型&#xff1a;</p>


```text
text/event-stream
```


<p>这个接口负责&#xff1a;</p>
<ol><li>校验用户是否有权访问 Run&#xff1b;</li><li>补发 <code>after</code> 之后的历史事件&#xff1b;</li><li>继续监听实时事件&#xff1b;</li><li>Run 结束后发送终态事件&#xff1b;</li><li>关闭连接。</li></ol>
<hr />
<h2>七、后端推荐执行流程</h2>
<p>整体流程可以设计成&#xff1a;</p>


```text
用户发送消息
↓
API 创建 Message 和 Run
↓
后台 Agent Runtime 开始执行
↓
每发生一个动作就写入 Event Log
↓
SSE Gateway 读取 Event Log
↓
推送给浏览器
```


<p>关键点是&#xff1a;</p>
<blockquote>
<p>Agent Runtime 不应该直接依赖某一条浏览器连接。</p>
</blockquote>
<p>错误架构&#xff1a;</p>


```text
Agent Runtime
↓
直接向当前 HTTP Response 写数据
```


<p>这种架构一旦连接断开&#xff0c;后续事件就无处可去。</p>
<p>推荐架构&#xff1a;</p>


```text
Agent Runtime
↓
写入统一事件总线
↓
Event Log
├── 数据库投影
├── SSE Gateway
├── 日志系统
└── 监控系统
```


<p>这样&#xff0c;即使当前没有用户在线&#xff0c;Agent 仍然可以继续运行并保存事件。</p>
<hr />
<h2>八、使用 Redis Streams 保存可回放事件</h2>
<p>中小型 Agent 系统可以使用 Redis Streams 作为短期事件日志。</p>
<p>Redis Stream 是一种有序、可追加的日志结构&#xff0c;支持时间有序 ID、历史读取、阻塞读取和保留策略。Redis 官方也将 <code>XREAD</code> 描述为适合 UI 实时流、调试器和类似 <code>tail -f</code> 的只读消费场景。</p>
<p>每个 Run 可以对应一个 Stream&#xff1a;</p>


```text
agent:run:run_9001:events
```


<p>写入事件&#xff1a;</p>


```text
XADD agent:run:run_9001:events *
type assistant.delta
data {"text":"项目采用"}
```


<p>Redis 返回事件 ID&#xff1a;</p>


```text
1722768000000-0
```


<p>SSE Gateway 可以从指定 ID 后开始读取&#xff1a;</p>


```text
XREAD BLOCK 15000
STREAMS agent:run:run_9001:events 1722768000000-0
```


<p>其行为是&#xff1a;</p>


```text
存在历史事件
↓
立即返回

没有新事件
↓
阻塞等待

有新事件写入
↓
立即返回并推送
```


<p>对于浏览器订阅&#xff0c;不建议使用一个共享 Consumer Group 将事件分给不同浏览器。</p>
<p>因为每个打开该会话的客户端通常都应该看到完整事件流&#xff0c;而不是多个浏览器共同瓜分事件。</p>
<p>因此&#xff1a;</p>


```text
任务 Worker 消费：
可以使用 XREADGROUP

浏览器 SSE 实时订阅：
通常使用 XREAD
```


<hr />
<h2>九、Snapshot 和 Redis Stream 应如何配合</h2>
<p>Redis Stream 不应该成为长期保存所有对话内容的唯一数据库。</p>
<p>推荐职责划分&#xff1a;</p>


```text
MySQL/PostgreSQL
保存 Conversation
保存最终 Message
保存 Run 状态
保存工具调用结果
保存当前 Assistant 内容

Redis Stream
保存短期实时事件
支持断线补发
支持 SSE 实时监听
```


<p>Agent 输出 Token 时&#xff0c;不需要每个 Token 都写一次数据库。</p>
<p>可以采用批量策略&#xff1a;</p>


```text
模型不断产生 delta
↓
每个 delta 写入 Redis Stream
↓
前端实时显示
↓
每 300～1000 毫秒批量更新数据库
↓
完成时强制写入最终完整内容
```


<p>例如&#xff1a;</p>


```text
assistant.delta：实时写事件流
assistant.message.completed：保存最终 Message
```


<p>这样同时兼顾&#xff1a;</p>
<ul><li>实时性&#xff1b;</li><li>数据库压力&#xff1b;</li><li>页面恢复&#xff1b;</li><li>最终一致性。</li></ul>
<hr />
<h2>十、SSE 服务端示例</h2>
<p>下面使用接近 FastAPI 的伪代码展示核心逻辑。</p>
<p>FastAPI 当前提供 SSE 响应支持&#xff0c;可以通过生成器持续 <code>yield</code> 事件&#xff0c;并设置 <code>event</code>、<code>id</code>、<code>data</code>、<code>retry</code> 等字段。</p>


```python
<span class="token keyword">import</span> asyncio
<span class="token keyword">import</span> json
<span class="token keyword">from</span> collections<span class="token punctuation">.</span>abc <span class="token keyword">import</span> AsyncGenerator

<span class="token keyword">from</span> fastapi <span class="token keyword">import</span> APIRouter<span class="token punctuation">,</span> Depends<span class="token punctuation">,</span> Request
<span class="token keyword">from</span> fastapi<span class="token punctuation">.</span>sse <span class="token keyword">import</span> EventSourceResponse<span class="token punctuation">,</span> ServerSentEvent

router <span class="token operator">=</span> APIRouter<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token decorator annotation punctuation">@router<span class="token punctuation">.</span>get</span><span class="token punctuation">(</span><span class="token string">"/api/runs/{run_id}/events"</span><span class="token punctuation">)</span>
<span class="token keyword">async</span> <span class="token keyword">def</span> <span class="token function">subscribe_run_events</span><span class="token punctuation">(</span>
run_id<span class="token punctuation">:</span> <span class="token builtin">str</span><span class="token punctuation">,</span>
request<span class="token punctuation">:</span> Request<span class="token punctuation">,</span>
after<span class="token punctuation">:</span> <span class="token builtin">str</span> <span class="token operator">|</span> <span class="token boolean">None</span> <span class="token operator">=</span> <span class="token boolean">None</span><span class="token punctuation">,</span>
current_user<span class="token operator">=</span>Depends<span class="token punctuation">(</span>get_current_user<span class="token punctuation">)</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> EventSourceResponse<span class="token punctuation">:</span>
<span class="token keyword">await</span> check_run_permission<span class="token punctuation">(</span>
run_id<span class="token operator">=</span>run_id<span class="token punctuation">,</span>
user_id<span class="token operator">=</span>current_user<span class="token punctuation">.</span><span class="token builtin">id</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>

<span class="token keyword">async</span> <span class="token keyword">def</span> <span class="token function">event_generator</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> AsyncGenerator<span class="token punctuation">[</span>ServerSentEvent<span class="token punctuation">,</span> <span class="token boolean">None</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
cursor <span class="token operator">=</span> after <span class="token keyword">or</span> <span class="token string">"0-0"</span>

<span class="token keyword">while</span> <span class="token boolean">True</span><span class="token punctuation">:</span>
<span class="token keyword">if</span> <span class="token keyword">await</span> request<span class="token punctuation">.</span>is_disconnected<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">break</span>

events <span class="token operator">=</span> <span class="token keyword">await</span> event_store<span class="token punctuation">.</span>read_after<span class="token punctuation">(</span>
run_id<span class="token operator">=</span>run_id<span class="token punctuation">,</span>
cursor<span class="token operator">=</span>cursor<span class="token punctuation">,</span>
block_ms<span class="token operator">=</span><span class="token number">15_000</span><span class="token punctuation">,</span>
count<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>

<span class="token keyword">if</span> <span class="token keyword">not</span> events<span class="token punctuation">:</span>
<span class="token keyword">yield</span> ServerSentEvent<span class="token punctuation">(</span>
comment<span class="token operator">=</span><span class="token string">"heartbeat"</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
<span class="token keyword">continue</span>

<span class="token keyword">for</span> event <span class="token keyword">in</span> events<span class="token punctuation">:</span>
cursor <span class="token operator">=</span> event<span class="token punctuation">.</span><span class="token builtin">id</span>

<span class="token keyword">yield</span> ServerSentEvent<span class="token punctuation">(</span>
event<span class="token operator">=</span>event<span class="token punctuation">.</span><span class="token builtin">type</span><span class="token punctuation">,</span>
<span class="token builtin">id</span><span class="token operator">=</span>event<span class="token punctuation">.</span><span class="token builtin">id</span><span class="token punctuation">,</span>
data<span class="token operator">=</span>json<span class="token punctuation">.</span>dumps<span class="token punctuation">(</span>
event<span class="token punctuation">.</span>to_dict<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
ensure_ascii<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token punctuation">,</span>
retry<span class="token operator">=</span><span class="token number">3000</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>

<span class="token keyword">if</span> event<span class="token punctuation">.</span><span class="token builtin">type</span> <span class="token keyword">in</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"run.completed"</span><span class="token punctuation">,</span>
<span class="token string">"run.failed"</span><span class="token punctuation">,</span>
<span class="token string">"run.cancelled"</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">:</span>
<span class="token keyword">return</span>

<span class="token keyword">return</span> EventSourceResponse<span class="token punctuation">(</span>
event_generator<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
headers<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"Cache-Control"</span><span class="token punctuation">:</span> <span class="token string">"no-cache"</span><span class="token punctuation">,</span>
<span class="token string">"X-Accel-Buffering"</span><span class="token punctuation">:</span> <span class="token string">"no"</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
```


<p>这里有几个重要设计。</p>
<h3>1. <code>after</code></h3>
<p>前端告诉服务端&#xff1a;</p>


```text
我最后处理到了哪个事件
```


<h3>2. 先补历史事件</h3>
<p>如果 Redis Stream 中已经有新事件&#xff0c;先返回历史事件。</p>
<h3>3. 再阻塞等待</h3>
<p>历史事件追平后&#xff0c;等待新事件。</p>
<h3>4. 心跳</h3>
<p>长时间没有事件时发送&#xff1a;</p>


```text
: heartbeat
```


<p>SSE 中以冒号开头的内容是注释&#xff0c;不会作为普通业务事件处理&#xff0c;但可以用于维持连接。</p>
<h3>5. 终态事件后结束</h3>
<p>收到以下事件后可以主动结束当前流&#xff1a;</p>


```text
run.completed
run.failed
run.cancelled
```


<hr />
<h2>十一、前端进入会话时的标准流程</h2>
<p>前端不要一进入页面就直接连接 SSE。</p>
<p>更合理的流程是&#xff1a;</p>


```typescript
<span class="token keyword">async</span> <span class="token keyword">function</span> <span class="token function">enterConversation</span><span class="token punctuation">(</span>conversationId<span class="token operator">:</span> <span class="token builtin">string</span><span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
<span class="token keyword">const</span> snapshot <span class="token operator">=</span> <span class="token keyword">await</span> conversationApi<span class="token punctuation">.</span><span class="token function">getConversation</span><span class="token punctuation">(</span>
conversationId<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token punctuation">;</span>

conversationStore<span class="token punctuation">.</span><span class="token function">replaceSnapshot</span><span class="token punctuation">(</span>snapshot<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">const</span> activeRun <span class="token operator">=</span> snapshot<span class="token punctuation">.</span>activeRun<span class="token punctuation">;</span>

<span class="token keyword">if</span> <span class="token punctuation">(</span><span class="token operator">!</span>activeRun<span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
<span class="token keyword">return</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token function">subscribeRun</span><span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
runId<span class="token operator">:</span> activeRun<span class="token punctuation">.</span>id<span class="token punctuation">,</span>
after<span class="token operator">:</span> activeRun<span class="token punctuation">.</span>lastEventId<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
```


<p>这里首先用 Snapshot 恢复页面&#xff0c;再订阅后续事件。</p>
<p>否则可能出现&#xff1a;</p>


```text
SSE 事件先到
↓
对应 Message 和 Block 还没有创建
↓
前端不知道把事件应用到哪里
```


<hr />
<h2>十二、前端事件必须使用 Reducer 统一处理</h2>
<p>不要在不同监听器中随意修改 UI&#xff1a;</p>


```typescript
source<span class="token punctuation">.</span><span class="token function">addEventListener</span><span class="token punctuation">(</span><span class="token string">"tool.started"</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">=></span> <span class="token punctuation">{<!-- --></span>
<span class="token comment">// 随意修改某个组件</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

source<span class="token punctuation">.</span><span class="token function">addEventListener</span><span class="token punctuation">(</span><span class="token string">"assistant.delta"</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">=></span> <span class="token punctuation">{<!-- --></span>
<span class="token comment">// 再修改另一份状态</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
```


<p>更推荐把全部事件交给统一 Reducer&#xff1a;</p>


```typescript
<span class="token keyword">function</span> <span class="token function">applyAgentEvent</span><span class="token punctuation">(</span>
state<span class="token operator">:</span> ConversationState<span class="token punctuation">,</span>
event<span class="token operator">:</span> AgentEvent<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token operator">:</span> ConversationState <span class="token punctuation">{<!-- --></span>
<span class="token keyword">if</span> <span class="token punctuation">(</span>state<span class="token punctuation">.</span>processedEventIds<span class="token punctuation">.</span><span class="token function">has</span><span class="token punctuation">(</span>event<span class="token punctuation">.</span>eventId<span class="token punctuation">)</span><span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
<span class="token keyword">return</span> state<span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">const</span> nextState <span class="token operator">=</span> <span class="token function">reduceAgentEvent</span><span class="token punctuation">(</span>state<span class="token punctuation">,</span> event<span class="token punctuation">)</span><span class="token punctuation">;</span>

nextState<span class="token punctuation">.</span>processedEventIds<span class="token punctuation">.</span><span class="token function">add</span><span class="token punctuation">(</span>event<span class="token punctuation">.</span>eventId<span class="token punctuation">)</span><span class="token punctuation">;</span>
nextState<span class="token punctuation">.</span>lastEventId <span class="token operator">=</span> event<span class="token punctuation">.</span>eventId<span class="token punctuation">;</span>

<span class="token keyword">return</span> nextState<span class="token punctuation">;</span>
<span class="token punctuation">}</span>
```


<p>事件处理逻辑示例&#xff1a;</p>


```typescript
<span class="token keyword">function</span> <span class="token function">reduceAgentEvent</span><span class="token punctuation">(</span>
state<span class="token operator">:</span> ConversationState<span class="token punctuation">,</span>
event<span class="token operator">:</span> AgentEvent<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token operator">:</span> ConversationState <span class="token punctuation">{<!-- --></span>
<span class="token keyword">switch</span> <span class="token punctuation">(</span>event<span class="token punctuation">.</span>type<span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
<span class="token keyword">case</span> <span class="token string">"assistant.delta"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">appendAssistantText</span><span class="token punctuation">(</span>
state<span class="token punctuation">,</span>
event<span class="token punctuation">.</span>blockId<span class="token punctuation">,</span>
event<span class="token punctuation">.</span>payload<span class="token punctuation">.</span>text<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"tool.started"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">createToolBlock</span><span class="token punctuation">(</span>
state<span class="token punctuation">,</span>
event<span class="token punctuation">.</span>blockId<span class="token punctuation">,</span>
event<span class="token punctuation">.</span>payload<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"tool.completed"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">completeToolBlock</span><span class="token punctuation">(</span>
state<span class="token punctuation">,</span>
event<span class="token punctuation">.</span>blockId<span class="token punctuation">,</span>
event<span class="token punctuation">.</span>payload<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"run.completed"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">completeRun</span><span class="token punctuation">(</span>
state<span class="token punctuation">,</span>
event<span class="token punctuation">.</span>runId<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"run.failed"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">failRun</span><span class="token punctuation">(</span>
state<span class="token punctuation">,</span>
event<span class="token punctuation">.</span>runId<span class="token punctuation">,</span>
event<span class="token punctuation">.</span>payload<span class="token punctuation">.</span>error<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">default</span><span class="token operator">:</span>
<span class="token keyword">return</span> state<span class="token punctuation">;</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>这样可以保证&#xff1a;</p>
<ul><li>实时事件和补发事件使用同一逻辑&#xff1b;</li><li>页面恢复逻辑一致&#xff1b;</li><li>事件可以测试&#xff1b;</li><li>重复事件不会重复渲染&#xff1b;</li><li>状态流转更加清晰。</li></ul>
<hr />
<h2>十三、EventSource 前端示例</h2>


```typescript
<span class="token keyword">let</span> currentEventSource<span class="token operator">:</span> EventSource <span class="token operator">|</span> <span class="token keyword">null</span> <span class="token operator">=</span> <span class="token keyword">null</span><span class="token punctuation">;</span>

<span class="token keyword">interface</span> <span class="token class-name">SubscribeRunOptions</span> <span class="token punctuation">{<!-- --></span>
runId<span class="token operator">:</span> <span class="token builtin">string</span><span class="token punctuation">;</span>
after<span class="token operator">?</span><span class="token operator">:</span> <span class="token builtin">string</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">function</span> <span class="token function">subscribeRun</span><span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
runId<span class="token punctuation">,</span>
after<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token operator">:</span> SubscribeRunOptions<span class="token punctuation">)</span><span class="token operator">:</span> <span class="token keyword">void</span> <span class="token punctuation">{<!-- --></span>
currentEventSource<span class="token operator">?.</span><span class="token function">close</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">const</span> params <span class="token operator">=</span> <span class="token keyword">new</span> <span class="token class-name">URLSearchParams</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">if</span> <span class="token punctuation">(</span>after<span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
params<span class="token punctuation">.</span><span class="token function">set</span><span class="token punctuation">(</span><span class="token string">"after"</span><span class="token punctuation">,</span> after<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">const</span> url <span class="token operator">=</span> <span class="token template-string"><span class="token template-punctuation string">`</span><span class="token string">/api/runs/</span><span class="token interpolation"><span class="token interpolation-punctuation punctuation">${<!-- --></span>runId<span class="token interpolation-punctuation punctuation">}</span></span><span class="token string">/events?</span><span class="token interpolation"><span class="token interpolation-punctuation punctuation">${<!-- --></span>params<span class="token punctuation">.</span><span class="token function">toString</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token interpolation-punctuation punctuation">}</span></span><span class="token template-punctuation string">`</span></span><span class="token punctuation">;</span>

<span class="token keyword">const</span> source <span class="token operator">=</span> <span class="token keyword">new</span> <span class="token class-name">EventSource</span><span class="token punctuation">(</span>url<span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
withCredentials<span class="token operator">:</span> <span class="token boolean">true</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

currentEventSource <span class="token operator">=</span> source<span class="token punctuation">;</span>

<span class="token keyword">const</span> eventTypes <span class="token operator">=</span> <span class="token punctuation">[</span>
<span class="token string">"run.started"</span><span class="token punctuation">,</span>
<span class="token string">"run.progress"</span><span class="token punctuation">,</span>
<span class="token string">"assistant.delta"</span><span class="token punctuation">,</span>
<span class="token string">"assistant.message.completed"</span><span class="token punctuation">,</span>
<span class="token string">"tool.started"</span><span class="token punctuation">,</span>
<span class="token string">"tool.completed"</span><span class="token punctuation">,</span>
<span class="token string">"tool.failed"</span><span class="token punctuation">,</span>
<span class="token string">"run.completed"</span><span class="token punctuation">,</span>
<span class="token string">"run.failed"</span><span class="token punctuation">,</span>
<span class="token string">"run.cancelled"</span><span class="token punctuation">,</span>
<span class="token punctuation">]</span><span class="token punctuation">;</span>

<span class="token keyword">for</span> <span class="token punctuation">(</span><span class="token keyword">const</span> eventType <span class="token keyword">of</span> eventTypes<span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
source<span class="token punctuation">.</span><span class="token function">addEventListener</span><span class="token punctuation">(</span>eventType<span class="token punctuation">,</span> <span class="token punctuation">(</span>rawEvent<span class="token punctuation">)</span> <span class="token operator">=></span> <span class="token punctuation">{<!-- --></span>
<span class="token keyword">const</span> messageEvent <span class="token operator">=</span> rawEvent <span class="token keyword">as</span> MessageEvent<span class="token punctuation">;</span>

<span class="token keyword">const</span> event <span class="token operator">=</span> <span class="token constant">JSON</span><span class="token punctuation">.</span><span class="token function">parse</span><span class="token punctuation">(</span>
messageEvent<span class="token punctuation">.</span>data<span class="token punctuation">,</span>
<span class="token punctuation">)</span> <span class="token keyword">as</span> AgentEvent<span class="token punctuation">;</span>

conversationStore<span class="token punctuation">.</span><span class="token function">applyEvent</span><span class="token punctuation">(</span>event<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">if</span> <span class="token punctuation">(</span>messageEvent<span class="token punctuation">.</span>lastEventId<span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
conversationStore<span class="token punctuation">.</span><span class="token function">setLastEventId</span><span class="token punctuation">(</span>
runId<span class="token punctuation">,</span>
messageEvent<span class="token punctuation">.</span>lastEventId<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

<span class="token keyword">if</span> <span class="token punctuation">(</span>
event<span class="token punctuation">.</span>type <span class="token operator">===</span> <span class="token string">"run.completed"</span> <span class="token operator">||</span>
event<span class="token punctuation">.</span>type <span class="token operator">===</span> <span class="token string">"run.failed"</span> <span class="token operator">||</span>
event<span class="token punctuation">.</span>type <span class="token operator">===</span> <span class="token string">"run.cancelled"</span>
<span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
source<span class="token punctuation">.</span><span class="token function">close</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>

source<span class="token punctuation">.</span><span class="token function-variable function">onerror</span> <span class="token operator">=</span> <span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">=></span> <span class="token punctuation">{<!-- --></span>
conversationStore<span class="token punctuation">.</span><span class="token function">markReconnecting</span><span class="token punctuation">(</span>runId<span class="token punctuation">)</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
```


<p>离开页面时&#xff1a;</p>


```typescript
<span class="token keyword">function</span> <span class="token function">leaveConversation</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token operator">:</span> <span class="token keyword">void</span> <span class="token punctuation">{<!-- --></span>
currentEventSource<span class="token operator">?.</span><span class="token function">close</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
currentEventSource <span class="token operator">=</span> <span class="token keyword">null</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
```


<p>这里关闭的只是当前页面连接&#xff0c;不是后台 Run。</p>
<hr />
<h2>十四、为什么必须去重</h2>
<p>SSE 和分布式事件系统一般需要按照“至少一次”的思路设计。</p>
<p>以下场景都可能导致事件重复&#xff1a;</p>


```text
事件已发送给浏览器
↓
连接突然中断
↓
浏览器没有来得及保存游标
↓
重新连接后再次补发
```


<p>或者&#xff1a;</p>


```text
前端已经处理事件 1058
↓
服务端恢复时又发送一次 1058
```


<p>因此前端必须根据 <code>eventId</code> 去重&#xff1a;</p>


```typescript
<span class="token keyword">if</span> <span class="token punctuation">(</span>processedEventIds<span class="token punctuation">.</span><span class="token function">has</span><span class="token punctuation">(</span>event<span class="token punctuation">.</span>eventId<span class="token punctuation">)</span><span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
<span class="token keyword">return</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
```


<p>仅仅根据文本内容去重是不可靠的。</p>
<p>例如大模型完全可能连续生成&#xff1a;</p>


```text
哈哈
哈哈
```


<p>两段文本内容相同&#xff0c;但它们可能是两个合法 delta。</p>
<p>正确去重依据应该是&#xff1a;</p>


```text
runId + eventId
```


<p>或者&#xff1a;</p>


```text
runId + sequence
```


<hr />
<h2>十五、顺序错乱如何处理</h2>
<p>每个 Run 的事件应该有严格递增的序号&#xff1a;</p>


```text
1051
1052
1053
1054
```


<p>前端收到 1054 时&#xff0c;如果当前只处理到了 1052&#xff0c;就能发现&#xff1a;</p>


```text
缺少 1053
```


<p>此时不应盲目继续应用&#xff0c;而应触发恢复&#xff1a;</p>


```text
检测到事件断档
↓
暂停当前流
↓
请求 after=1052
↓
补发 1053、1054
↓
继续处理
```


<p>事件结构中可以同时保留&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"eventId"</span><span class="token operator">:</span> <span class="token string">"1722768000000-0"</span><span class="token punctuation">,</span>
<span class="token string-property property">"sequence"</span><span class="token operator">:</span> <span class="token number">1054</span>
<span class="token punctuation">}</span>
```


<p>其中&#xff1a;</p>
<ul><li><code>eventId</code> 用于事件存储和读取&#xff1b;</li><li><code>sequence</code> 用于业务顺序检测。</li></ul>
<hr />
<h2>十六、页面恢复时以谁为准</h2>
<p>页面恢复时通常会同时存在&#xff1a;</p>
<ul><li>数据库 Snapshot&#xff1b;</li><li>Redis Event Log&#xff1b;</li><li>浏览器本地状态。</li></ul>
<p>推荐优先级&#xff1a;</p>


```text
数据库 Snapshot
作为业务基础状态

Event Log
补充 Snapshot 之后的新变化

浏览器本地缓存
只用于优化体验
```


<p>不要让 LocalStorage 成为最终状态来源。</p>
<p>因为它可能&#xff1a;</p>
<ul><li>被清理&#xff1b;</li><li>过期&#xff1b;</li><li>多标签页冲突&#xff1b;</li><li>与服务端不一致&#xff1b;</li><li>保存了错误的半成品状态。</li></ul>
<p>本地缓存可以保存&#xff1a;</p>


```text
当前会话 ID
最近事件游标
未发送输入框内容
滚动位置
折叠状态
```


<p>但消息和 Run 的最终状态仍然应以服务端为准。</p>
<hr />
<h2>十七、Snapshot 和 Event Log 之间的竞争问题</h2>
<p>一个典型问题是&#xff1a;</p>


```text
前端开始请求 Snapshot
↓
Snapshot 查询完成前，Agent 又产生两个事件
↓
前端随后连接 SSE
```


<p>如果处理不正确&#xff0c;中间两个事件可能丢失。</p>
<p>解决方式是让 Snapshot 返回一个一致的游标&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"messages"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"activeRun"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"run_9001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"lastEventId"</span><span class="token operator">:</span> <span class="token string">"1058"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>这个响应表达&#xff1a;</p>
<blockquote>
<p>当前 Snapshot 已经包含了截至事件 1058 的状态。</p>
</blockquote>
<p>前端随后请求&#xff1a;</p>


```text
/events?after=1058
```


<p>这样事件 1059 以后一定会被补发。</p>
<p>因此 Snapshot 中的 <code>lastEventId</code> 不能随便读取&#xff0c;它需要与 Snapshot 状态保持一致。</p>
<p>可以在同一个事务或投影更新过程中&#xff1a;</p>


```text
应用事件
↓
更新 Message Snapshot
↓
更新 Run.last_event_id
```


<hr />
<h2>十八、终态事件非常重要</h2>
<p>Agent 执行结束时&#xff0c;一定要发出明确的终态事件&#xff1a;</p>


```text
run.completed
run.failed
run.cancelled
```


<p>不要让前端根据“很久没收到 Token”猜测任务是否完成。</p>
<p>成功事件&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"eventId"</span><span class="token operator">:</span> <span class="token string">"1100"</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"run.completed"</span><span class="token punctuation">,</span>
<span class="token string-property property">"runId"</span><span class="token operator">:</span> <span class="token string">"run_9001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"payload"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"outputMessageId"</span><span class="token operator">:</span> <span class="token string">"msg_assistant_1001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"finishReason"</span><span class="token operator">:</span> <span class="token string">"stop"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>失败事件&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"eventId"</span><span class="token operator">:</span> <span class="token string">"1100"</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"run.failed"</span><span class="token punctuation">,</span>
<span class="token string-property property">"runId"</span><span class="token operator">:</span> <span class="token string">"run_9001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"payload"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"errorCode"</span><span class="token operator">:</span> <span class="token string">"TOOL_TIMEOUT"</span><span class="token punctuation">,</span>
<span class="token string-property property">"errorMessage"</span><span class="token operator">:</span> <span class="token string">"代码分析工具执行超时"</span><span class="token punctuation">,</span>
<span class="token string-property property">"retryable"</span><span class="token operator">:</span> <span class="token boolean">true</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>前端收到终态事件后&#xff1a;</p>
<ol><li>更新 Run 状态&#xff1b;</li><li>更新 Assistant Message 状态&#xff1b;</li><li>停止 loading&#xff1b;</li><li>关闭 SSE&#xff1b;</li><li>必要时重新拉取一次最终 Snapshot。</li></ol>
<hr />
<h2>十九、EventSource 还是 Fetch Streaming</h2>
<h3>使用 EventSource</h3>
<p>浏览器原生 <code>EventSource</code> 的优势是&#xff1a;</p>
<ul><li>API 简单&#xff1b;</li><li>原生支持 SSE 格式&#xff1b;</li><li>自动重连&#xff1b;</li><li>支持事件类型&#xff1b;</li><li>支持 <code>Last-Event-ID</code>&#xff1b;</li><li>适合 GET 长连接。</li></ul>
<p>但它也有一些限制&#xff1a;</p>
<ul><li>只能使用 GET&#xff1b;</li><li>不方便自定义任意请求头&#xff1b;</li><li>不能直接携带复杂请求体&#xff1b;</li><li>手动控制重连策略的能力较弱。</li></ul>
<h3>使用 Fetch Streaming</h3>
<p>如果需要&#xff1a;</p>
<ul><li>POST 请求&#xff1b;</li><li>Bearer Token Header&#xff1b;</li><li>请求体&#xff1b;</li><li>更灵活的取消&#xff1b;</li><li>自定义重连逻辑&#xff1b;</li></ul>
<p>可以使用&#xff1a;</p>


```typescript
<span class="token keyword">const</span> response <span class="token operator">=</span> <span class="token keyword">await</span> <span class="token function">fetch</span><span class="token punctuation">(</span>url<span class="token punctuation">,</span> <span class="token punctuation">{<!-- --></span>
method<span class="token operator">:</span> <span class="token string">"GET"</span><span class="token punctuation">,</span>
headers<span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
Authorization<span class="token operator">:</span> <span class="token template-string"><span class="token template-punctuation string">`</span><span class="token string">Bearer </span><span class="token interpolation"><span class="token interpolation-punctuation punctuation">${<!-- --></span>token<span class="token interpolation-punctuation punctuation">}</span></span><span class="token template-punctuation string">`</span></span><span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
signal<span class="token operator">:</span> abortController<span class="token punctuation">.</span>signal<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">const</span> reader <span class="token operator">=</span> response<span class="token punctuation">.</span>body<span class="token operator">?.</span><span class="token function">getReader</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
```


<p>Fetch 的响应体可以通过 <code>ReadableStream</code> 增量读取&#xff0c;而不必等待整个响应结束。</p>
<p>不过使用 Fetch Streaming 时&#xff0c;需要自己处理&#xff1a;</p>
<ul><li>SSE 文本解析&#xff1b;</li><li>断线重连&#xff1b;</li><li>游标保存&#xff1b;</li><li>AbortController&#xff1b;</li><li>错误重试&#xff1b;</li><li>心跳超时。</li></ul>
<p>对于普通 Cookie 鉴权的 Agent 对话&#xff0c;优先选择 EventSource 会更简单。</p>
<p>对于需要自定义 Header 的系统&#xff0c;可以使用 Fetch Streaming。</p>
<hr />
<h2>二十、Nginx 和网关配置</h2>
<p>SSE 在本地正常&#xff0c;部署后“每隔几十秒一次性出来”&#xff0c;通常不是前端问题&#xff0c;而是代理层缓冲。</p>
<p>Nginx 默认可能缓冲上游响应。关闭缓冲后&#xff0c;数据会在接收到时直接转发给客户端&#xff1b;也可以通过响应头 <code>X-Accel-Buffering: no</code> 控制。</p>
<p>配置示例&#xff1a;</p>


```nginx
location /api/runs/ {
proxy_pass http://agent_backend;

proxy_http_version 1.1;
proxy_set_header Connection "";

proxy_buffering off;
proxy_cache off;

proxy_read_timeout 3600s;
proxy_send_timeout 3600s;

gzip off;
}
```


<p>后端响应头建议包含&#xff1a;</p>


```text
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```


<p>同时建议每隔 15&#xff5e;30 秒发送心跳&#xff1a;</p>


```text
: heartbeat
```


<p>避免&#xff1a;</p>
<ul><li>Nginx 认为连接空闲&#xff1b;</li><li>网关关闭连接&#xff1b;</li><li>负载均衡器超时&#xff1b;</li><li>浏览器长时间无法感知断线。</li></ul>
<hr />
<h2>二十一、多实例部署时不能只使用进程内队列</h2>
<p>单实例开发阶段&#xff0c;可能这样实现&#xff1a;</p>


```python
run_queues<span class="token punctuation">:</span> <span class="token builtin">dict</span><span class="token punctuation">[</span><span class="token builtin">str</span><span class="token punctuation">,</span> asyncio<span class="token punctuation">.</span>Queue<span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
```


<p>Agent 将事件写入本地 <code>asyncio.Queue</code>&#xff0c;SSE 接口从中读取。</p>
<p>这在单进程中可以工作。</p>
<p>但扩容为多个实例后&#xff1a;</p>


```text
Agent Runtime 在实例 A
SSE 请求被负载均衡到实例 B
```


<p>实例 B 的内存中没有实例 A 的事件。</p>
<p>因此多实例环境不能只依赖&#xff1a;</p>
<ul><li>进程内 Queue&#xff1b;</li><li>本地 EventEmitter&#xff1b;</li><li>单机内存&#xff1b;</li><li>本地文件。</li></ul>
<p>应该引入共享事件基础设施&#xff1a;</p>


```text
Redis Streams
Kafka
NATS JetStream
数据库事件表
```


<p>对于大多数中小型 Agent 平台&#xff1a;</p>


```text
业务状态：PostgreSQL/MySQL
短期事件：Redis Streams
浏览器推送：SSE Gateway
```


<p>通常已经能够满足需求。</p>
<hr />
<h2>二十二、推荐的完整架构</h2>


```text
┌────────────────────┐
│      Web 前端       │
│ Snapshot + SSE恢复  │
└─────────┬──────────┘
│
├── GET Conversation Snapshot
│
└── GET Run Event Stream
│
┌─────────────────────▼─────────────────────┐
│               Agent API                   │
│                                           │
│  Conversation API     SSE Gateway         │
│         │                  │               │
└─────────┼──────────────────┼───────────────┘
│                  │
▼                  ▼
┌────────────────┐   ┌──────────────────────┐
│ MySQL/Postgres │   │    Redis Streams     │
│                │   │                      │
│ Conversation   │   │ Run Event Log        │
│ Message        │   │ Replay + Live Tail   │
│ Run Snapshot   │   │                      │
└───────▲────────┘   └──────────▲───────────┘
│                       │
└──────────┬────────────┘
│
┌────────▼────────┐
│  Agent Runtime  │
│                 │
│ LLM             │
│ Tool            │
│ Subagent        │
│ Memory          │
└─────────────────┘
```


<p>其中&#xff1a;</p>


```text
Agent Runtime
负责执行

Redis Streams
负责实时事件和短期回放

数据库
负责完整业务状态

SSE Gateway
负责向浏览器推送

前端 Reducer
负责重建界面
```


<hr />
<h2>二十三、常见错误设计</h2>
<h3>错误一&#xff1a;SSE 断开就取消 Agent</h3>
<p>用户切换页面并不代表用户要求取消任务。</p>
<p>应区分&#xff1a;</p>


```text
unsubscribe：停止订阅

cancel：取消后台执行
```


<h3>错误二&#xff1a;只保存最终回答</h3>
<p>如果只在 Agent 完成时保存回答&#xff0c;刷新时会丢失当前已经生成的内容。</p>
<p>应定期保存流式中的 Message Snapshot。</p>
<h3>错误三&#xff1a;只保存文本&#xff0c;不保存工具状态</h3>
<p>刷新后文本恢复了&#xff0c;但工具调用全部消失。</p>
<p>工具调用、子 Agent、计划和审批也应该是可持久化 Block。</p>
<h3>错误四&#xff1a;重新进入后只连接最新事件</h3>
<p>如果不补发历史事件&#xff0c;用户离开期间发生的工具调用和输出会丢失。</p>
<h3>错误五&#xff1a;不设置事件 ID</h3>
<p>没有事件 ID&#xff0c;就无法可靠去重、排序和恢复。</p>
<h3>错误六&#xff1a;一个 Token 对应一次数据库写入</h3>
<p>这会造成大量小事务和数据库压力。</p>
<p>应对 delta 做批量持久化。</p>
<h3>错误七&#xff1a;仅依赖 Redis Pub/Sub</h3>
<p>Pub/Sub 更适合在线广播。订阅者离线期间无法自然回放已经错过的消息。</p>
<p>需要恢复能力时&#xff0c;应使用可回放事件日志&#xff0c;例如 Redis Streams。</p>
<h3>错误八&#xff1a;页面刷新后直接创建新的 Run</h3>
<p>这可能让同一个用户问题执行两遍。</p>
<p>页面恢复应该连接已有 Run&#xff0c;而不是重新提交用户消息。</p>
<hr />
<h2>二十四、建议的恢复算法</h2>
<p>最终可以把前端恢复过程总结为以下算法。</p>


```text
进入 Conversation 页面
↓
关闭上一个页面的 SSE
↓
获取 Conversation Snapshot
↓
用 Snapshot 替换本地状态
↓
是否存在未完成 Run？
├── 否：结束
└── 是
↓
获取 Snapshot.lastEventId
↓
连接 /events?after=lastEventId
↓
服务端补发遗漏事件
↓
前端按 eventId 去重
↓
按 sequence 检查顺序
↓
Reducer 更新 Message、Tool 和 Run
↓
继续接收实时事件
↓
收到 completed/failed/cancelled
↓
关闭 SSE
```


<p>服务端恢复算法&#xff1a;</p>


```text
收到 SSE 请求
↓
校验 Run 访问权限
↓
读取 after / Last-Event-ID
↓
从 Event Log 查询之后的事件
↓
依次补发
↓
阻塞等待新事件
↓
持续推送
↓
发送心跳
↓
遇到终态事件后关闭
```


<hr />
<h2>二十五、测试清单</h2>
<p>这类功能不能只测试“正常生成”。</p>
<p>至少应该覆盖以下场景&#xff1a;</p>
<h4>页面操作</h4>
<ul><li>Agent 输出中切换会话&#xff1b;</li><li>切换回来&#xff1b;</li><li>Agent 输出中刷新页面&#xff1b;</li><li>Agent 完成后重新进入&#xff1b;</li><li>同一会话打开两个标签页&#xff1b;</li><li>快速来回切换会话。</li></ul>
<h4>网络异常</h4>
<ul><li>SSE 短暂断线&#xff1b;</li><li>断网后恢复&#xff1b;</li><li>Nginx 重启&#xff1b;</li><li>后端实例重启&#xff1b;</li><li>Redis 短暂不可用。</li></ul>
<h4>事件一致性</h4>
<ul><li>同一事件重复发送&#xff1b;</li><li>事件中间缺失&#xff1b;</li><li>事件乱序&#xff1b;</li><li>Event Log 已被清理&#xff1b;</li><li>Snapshot 比事件流更新&#xff1b;</li><li>Snapshot 比事件流落后。</li></ul>
<h4>Agent 状态</h4>
<ul><li>文本生成中刷新&#xff1b;</li><li>工具执行中刷新&#xff1b;</li><li>子 Agent 执行中刷新&#xff1b;</li><li>等待用户审批时刷新&#xff1b;</li><li>Agent 失败时刷新&#xff1b;</li><li>Agent 取消时刷新。</li></ul>
<p>最终验收标准应该是&#xff1a;</p>
<blockquote>
<p>无论用户何时离开、切换或刷新&#xff0c;再次进入时都能看到正确的完整状态&#xff0c;并且不会重复生成、不会丢事件、不会永远卡在运行中。</p>
</blockquote>
<hr />
<h2>总结</h2>
<p>Agent 对话中的 SSE 恢复问题&#xff0c;本质上不是一个简单的前端重连问题。</p>
<p>它是一个完整的分布式状态恢复问题。</p>
<p>可靠方案需要同时具备&#xff1a;</p>


```text
Conversation Snapshot
+
Run 状态机
+
有序 Event Log
+
事件游标
+
断线补发
+
前端幂等 Reducer
+
实时 SSE
```


<p>最重要的设计原则是&#xff1a;</p>
<blockquote>
<p>SSE 负责传输事件&#xff0c;但数据库和事件日志才负责保存事实。</p>
</blockquote>
<p>切换页面时&#xff0c;可以关闭 SSE&#xff0c;但不要停止 Agent。</p>
<p>重新进入时&#xff0c;先加载 Snapshot&#xff0c;再从事件游标之后补发事件。</p>
<p>刷新页面时&#xff0c;不依赖浏览器内存&#xff0c;而是通过服务端状态重新构建 UI。</p>
<p>收到重复事件时&#xff0c;根据 <code>eventId</code> 去重。</p>
<p>发现事件缺口时&#xff0c;根据 <code>sequence</code> 重新补发。</p>
<p>Agent 完成时&#xff0c;必须发送明确的终态事件&#xff0c;并保存最终消息。</p>
<p>最终形成的标准链路是&#xff1a;</p>


```text
提交消息
↓
创建 Run
↓
Agent 后台执行
↓
事件写入 Event Log
↓
状态投影到数据库
↓
SSE 实时推送
↓
页面断开
↓
Snapshot 恢复
↓
事件补发
↓
继续实时接收
```


<p>当这套机制建立以后&#xff0c;Agent 对话才真正具备生产级的可恢复性&#xff0c;而不只是一个“页面不刷新时能够流式输出”的演示系统。</p>
