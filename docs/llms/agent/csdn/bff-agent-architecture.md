---
title: "什么是 BFF？从前后端解耦到微服务与 Agent 架构中的实践"
description: "CSDN 原文全文镜像：BFF 并不是什么复杂的新技术。前端需要的数据，与后端提供的业务能力之间存在天然差异。Frontend↓BackendFrontend↓十几个 Microservices服务发现接口聚合数据转换异常处理权限拼装业务编排系统边界会越来越混……"
pageType: article
module: agent
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "架构"
  - "微服务"
  - "状态模式"
level: advanced
prerequisites:
  - "/llms/agent/"
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-26，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-26。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/164094876](https://blog.csdn.net/m0_63309778/article/details/164094876)
- 站内分区：Agent / BFF 与 Agent 架构
:::

<p><img src="https://i-blog.csdnimg.cn/direct/7b93bb8dbc6c41e7a2308922b1fc0da1.png" alt="在这里插入图片描述" /></p>
<p>在很多项目刚开始时&#xff0c;系统架构往往非常简单&#xff1a;</p>


```text
Frontend
|
v
Backend
|
v
Database
```


<p>前端调用后端提供的几个接口&#xff0c;后端查询数据库并返回数据。</p>
<p>这种架构没有什么问题。</p>
<p>但随着业务越来越复杂&#xff0c;后端开始拆成&#xff1a;</p>


```text
用户服务
订单服务
文件服务
权限服务
搜索服务
Agent 服务
RAG 服务
模型服务
观测服务
……
```


<p>前端也从一个简单页面逐渐发展成 Web、App、小程序甚至多个管理后台。</p>
<p>这时候&#xff0c;经常会出现一个问题&#xff1a;</p>
<blockquote>
<p><strong>前端为了展示一个页面&#xff0c;需要调用五六个甚至十几个后端服务。</strong></p>
</blockquote>
<p>例如一个 Agent 详情页可能需要&#xff1a;</p>


```text
Agent 基本信息
模型配置
知识库
Tools
最近运行记录
Token 消耗
用户权限
```


<p>前端开始不得不理解整个后端微服务体系。</p>
<p>这时候&#xff0c;BFF 就开始变得非常有价值。</p>
<p>BFF&#xff0c;全称&#xff1a;</p>
<blockquote>
<p><strong>Backend for Frontend</strong></p>
</blockquote>
<p>直译就是&#xff1a;</p>
<blockquote>
<p><strong>为前端服务的后端。</strong></p>
</blockquote>
<p>它不是一个新的编程框架&#xff0c;也不是某个具体产品&#xff0c;而是一种<strong>架构模式</strong>。</p>
<p>它真正想解决的问题是&#xff1a;</p>
<blockquote>
<p><strong>前端真正需要的数据模型&#xff0c;与后端按照业务领域设计的数据模型&#xff0c;往往不是一回事。</strong></p>
</blockquote>
<hr />
<h2>一、为什么会出现 BFF&#xff1f;</h2>
<p>假设我们有一个 Agent 平台。</p>
<p>后端已经拆成多个服务&#xff1a;</p>


```text
User Service
Agent Service
RAG Service
Knowledge Service
File Service
Metrics Service
Permission Service
```


<p>现在前端打开 Agent 详情页面。</p>
<p>需要请求&#xff1a;</p>


```http
GET /agents/123

GET /agents/123/model

GET /agents/123/knowledge-bases

GET /agents/123/tools

GET /agents/123/runs

GET /agents/123/metrics

GET /agents/123/permissions
```


<p>为了展示一个页面&#xff0c;浏览器发送了七个请求。</p>
<p>这还只是最简单的情况。</p>
<p>前端接下来还需要&#xff1a;</p>


```text
判断接口有没有失败
↓
处理超时
↓
处理鉴权
↓
合并返回结果
↓
转换字段
↓
转换状态
↓
处理部分服务不可用
↓
决定页面显示什么
```


<p>慢慢地&#xff0c;我们会发现&#xff1a;</p>
<blockquote>
<p>前端承担的已经不只是 UI 展示逻辑&#xff0c;而是在做后端服务编排。</p>
</blockquote>
<p>更麻烦的是&#xff0c;前端开始知道&#xff1a;</p>


```text
Agent Service 是什么

RAG Service 是什么

Metrics Service 是什么

Knowledge Service 在哪里

哪个接口先调用

哪个接口失败可以忽略

哪个接口失败整个页面不能显示
```


<p>这意味着&#xff1a;</p>
<blockquote>
<p><strong>后端微服务架构已经泄漏到了前端。</strong></p>
</blockquote>
<p>于是可以增加一层&#xff1a;</p>


```text
                Web
|
v
BFF
|
+---------+---------+
|         |         |
v         v         v
Agent      RAG       User
Service   Service   Service
|                   |
v                   v
Metrics             Permission
```


<p>前端只调用&#xff1a;</p>


```http
GET /bff/agents/123/detail
```


<p>BFF 内部完成&#xff1a;</p>


```text
查询 Agent
+
查询知识库
+
查询 Tools
+
查询运行记录
+
查询 Metrics
+
查询权限
↓
并发调用
↓
数据聚合
↓
字段转换
↓
返回页面需要的数据
```


<p>最终前端拿到&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"agent"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"model"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"knowledgeBases"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"tools"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"recentRuns"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"metrics"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"permissions"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span><span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>然后直接渲染页面。</p>
<p>这就是 BFF 最核心的价值。</p>
<hr />
<h2>二、为什么叫 Backend <strong>for Frontend</strong>&#xff1f;</h2>
<p>BFF 这个名字真正重要的不是 Backend。</p>
<p>而是&#xff1a;</p>
<blockquote>
<p><strong>for Frontend</strong></p>
</blockquote>
<p>传统后端 API 通常站在<strong>业务领域</strong>的角度设计。</p>
<p>例如用户服务&#xff1a;</p>


```text
User
```


<p>订单服务&#xff1a;</p>


```text
Order
```


<p>Agent 服务&#xff1a;</p>


```text
Agent
AgentVersion
Run
```


<p>知识库服务&#xff1a;</p>


```text
KnowledgeBase
Document
Chunk
```


<p>所以接口可能是&#xff1a;</p>


```http
GET /users/{id}

GET /orders/{id}

GET /agents/{id}

GET /runs/{id}

GET /knowledge-bases/{id}
```


<p>这些 API 在回答&#xff1a;</p>
<blockquote>
<p><strong>我的业务能力是什么&#xff1f;</strong></p>
</blockquote>
<p>而 BFF 站在前端页面的角度思考。</p>
<p>比如&#xff1a;</p>


```text
首页
Agent 详情页
Agent 编辑页
Run 详情页
Dashboard
```


<p>所以它可能提供&#xff1a;</p>


```http
GET /bff/home

GET /bff/agents/{id}/detail

GET /bff/agents/{id}/editor

GET /bff/runs/{id}/detail

GET /bff/dashboard
```


<p>注意&#xff1a;</p>


```text
Dashboard
```


<p>本身甚至不是一个真正的业务领域。</p>
<p>后端数据库里通常也不会存在&#xff1a;</p>


```text
dashboard_table
```


<p>Dashboard 只是把&#xff1a;</p>


```text
Agent 数据
+
用户数据
+
运行数据
+
Token 数据
+
费用数据
```


<p>组合成一个前端视图。</p>
<p>这正是 BFF 最擅长处理的问题。</p>
<hr />
<h2>三、BFF 本质上是在做一次“模型转换”</h2>
<p>前后端之间存在一个非常重要的问题&#xff1a;</p>


```text
Backend Domain Model
≠
Frontend View Model
```


<p>例如后端可能围绕这些对象设计&#xff1a;</p>


```text
User
Agent
AgentVersion
KnowledgeBase
Document
Run
RunStep
Metric
Tool
```


<p>但前端真正关心的对象可能是&#xff1a;</p>


```text
AgentListItem

AgentDetailPage

RunDetailPage

DashboardCard

ChatMessage
```


<p>后端设计关注的是&#xff1a;</p>
<blockquote>
<p>业务领域如何建模。</p>
</blockquote>
<p>前端设计关注的是&#xff1a;</p>
<blockquote>
<p>页面怎样展示。</p>
</blockquote>
<p>因此 BFF 可以理解成&#xff1a;</p>


```text
Domain Model
↓
BFF
↓
View Model
```


<p>它在两个世界之间建立了一层转换。</p>
<hr />
<h2>四、BFF 最核心的四种职责</h2>
<p>一个比较健康的 BFF&#xff0c;通常主要负责四件事情。</p>
<h3>1. Aggregation&#xff1a;接口聚合</h3>
<p>例如&#xff1a;</p>


```text
Frontend
|
v
BFF
|
+------> Agent Service
|
+------> User Service
|
+------> Knowledge Service
|
+------> Metrics Service
```


<p>然后 BFF 将结果聚合后一次返回。</p>
<p>而且这些服务通常可以并发请求&#xff1a;</p>


```text
Agent ──────────┐
Knowledge ──────┤
Metrics ────────┼──→ Merge → Response
Runs ───────────┘
```


<p>而不是串行&#xff1a;</p>


```text
Agent
↓
Knowledge
↓
Metrics
↓
Runs
```


<p>这既降低了前端复杂度&#xff0c;也可能减少页面整体加载时间。</p>
<hr />
<h2>五、Transformation&#xff1a;数据转换</h2>
<p>后端 API 往往更接近数据库和领域模型。</p>
<p>例如&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">123</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token number">2</span><span class="token punctuation">,</span>
<span class="token string-property property">"owner_id"</span><span class="token operator">:</span> <span class="token number">98</span><span class="token punctuation">,</span>
<span class="token string-property property">"created_at"</span><span class="token operator">:</span> <span class="token string">"2026-08-26T10:00:00Z"</span>
<span class="token punctuation">}</span>
```


<p>但前端真正希望得到的可能是&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"123"</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"code"</span><span class="token operator">:</span> <span class="token number">2</span><span class="token punctuation">,</span>
<span class="token string-property property">"label"</span><span class="token operator">:</span> <span class="token string">"运行中"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"owner"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token number">98</span><span class="token punctuation">,</span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"张三"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string-property property">"createdAt"</span><span class="token operator">:</span> <span class="token string">"2026-08-26 18:00"</span>
<span class="token punctuation">}</span>
```


<p>这里涉及&#xff1a;</p>


```text
字段重命名
状态转换
时间转换
对象拼装
字段裁剪
数据补充
```


<p>这些逻辑如果大量散落在 React/Vue Component 里面&#xff0c;前端代码会越来越难维护。</p>
<p>BFF 可以统一完成这层转换。</p>
<hr />
<h2>六、Orchestration&#xff1a;服务编排</h2>
<p>BFF 不一定只是简单地&#xff1a;</p>


```text
请求 A
请求 B
Merge
```


<p>有时候一个前端操作可能需要&#xff1a;</p>


```text
先查询用户权限
↓
查询 Agent
↓
获取 AgentVersion
↓
根据 Version 查询 Tools
↓
根据 KnowledgeBase 查询知识库
↓
拼装编辑页面
```


<p>甚至&#xff1a;</p>


```text
A 成功以后调用 B

B 失败以后降级到 C
```


<p>这属于轻量级的&#xff1a;</p>
<blockquote>
<p><strong>Orchestration</strong></p>
</blockquote>
<p>也就是服务编排。</p>
<p>不过这里要特别注意一个边界&#xff1a;</p>
<blockquote>
<p><strong>BFF 可以编排业务能力&#xff0c;但不应该成为业务能力本身。</strong></p>
</blockquote>
<p>这个问题后面还会重点讨论。</p>
<hr />
<h2>七、Frontend-specific Logic&#xff1a;前端特有逻辑</h2>
<p>还有一些逻辑天然属于某个客户端。</p>
<p>例如&#xff1a;</p>


```text
Web Dashboard 展示哪些字段

App 首页返回多少条数据

小程序是否需要分享信息

Web 是否需要 SEO Metadata

App 是否需要 Deep Link
```


<p>这些内容没有必要进入核心领域服务。</p>
<p>非常适合放到对应 BFF。</p>
<p>于是可以出现&#xff1a;</p>


```text
                 Backend Services
↑
+----------+----------+
|          |          |
Web BFF     App BFF    Mini BFF
↑          ↑          ↑
Web        App       小程序
```


<p>这其实也是 BFF 模式最初的思想之一&#xff1a;</p>
<blockquote>
<p><strong>不同前端&#xff0c;可以拥有自己的 Backend for Frontend。</strong></p>
</blockquote>
<hr />
<h2>八、BFF 和 API Gateway 到底有什么区别&#xff1f;</h2>
<p>这是讨论 BFF 时最容易混淆的问题。</p>
<p>很多人会问&#xff1a;</p>
<blockquote>
<p>“我已经有 API Gateway 了&#xff0c;为什么还需要 BFF&#xff1f;”</p>
</blockquote>
<p>因为它们解决的问题完全不同。</p>
<p>典型架构&#xff1a;</p>


```text
Client
|
v
API Gateway
|
v
BFF
|
+-------- User Service
|
+-------- Agent Service
|
+-------- RAG Service
|
+-------- File Service
```


<p>API Gateway 更多负责&#xff1a;</p>


```text
统一入口
路由
认证
限流
WAF
负载均衡
IP 黑白名单
TLS
流量治理
```


<p>而 BFF 负责&#xff1a;</p>


```text
数据聚合
接口编排
DTO 转换
前端 ViewModel
客户端特有逻辑
```


<p>可以用一句话区分&#xff1a;</p>
<blockquote>
<p><strong>Gateway 管“流量怎样进入系统”&#xff0c;BFF 管“前端需要什么数据”。</strong></p>
</blockquote>
<p>例如&#xff1a;</p>


```text
用户请求有没有 Token？
```


<p>更偏 Gateway。</p>


```text
这个 Agent 详情页应该返回哪些信息？
```


<p>更偏 BFF。</p>
<p>当然现实系统的边界并不会永远这么绝对&#xff0c;但从职责设计上应该尽量保持这个方向。</p>
<hr />
<h2>九、BFF 和 Controller 不是一回事</h2>
<p>另外一个容易混淆的概念是 Controller。</p>
<p>例如一个典型 Java 服务&#xff1a;</p>


```text
Controller
↓
Service
↓
Repository
↓
Database
```


<p>这里 Controller 只是&#xff1a;</p>
<blockquote>
<p><strong>单个服务内部的接口入口层。</strong></p>
</blockquote>
<p>而 BFF 是&#xff1a;</p>


```text
             BFF
|
+-------+-------+
|       |       |
v       v       v
User    Agent    RAG
Service  Service  Service
```


<p>它通常运行在多个 Backend Service 上层。</p>
<p>所以&#xff1a;</p>


```text
Controller ≠ BFF
```


<p>但是&#xff1a;</p>
<blockquote>
<p>BFF 自己当然也可以有 Controller。</p>
</blockquote>
<p>例如&#xff1a;</p>


```text
AgentBffController
↓
AgentDetailFacade
↓
+-------+--------+--------+
|       |        |        |
Agent  User     RAG     Metrics
API    API      API       API
```


<hr />
<h2>十、BFF 和 GraphQL 是什么关系&#xff1f;</h2>
<p>GraphQL 也经常和 BFF 一起出现。</p>
<p>但二者同样不是一回事。</p>
<p>GraphQL 是&#xff1a;</p>
<blockquote>
<p>API Query Language / Runtime</p>
</blockquote>
<p>而 BFF 是&#xff1a;</p>
<blockquote>
<p>Architecture Pattern</p>
</blockquote>
<p>因此完全可以&#xff1a;</p>


```text
Frontend
|
v
GraphQL BFF
|
+------ User Service
+------ Agent Service
+------ RAG Service
```


<p>前端请求&#xff1a;</p>


```graphql
query {
agent(id: "123") {
name
model {
name
}
knowledgeBases {
name
}
runs {
status
}
}
}
```


<p>GraphQL BFF 再负责从后端多个服务获取这些数据。</p>
<p>所以&#xff1a;</p>
<blockquote>
<p><strong>GraphQL 可以是实现 BFF 的一种技术方案&#xff0c;但 GraphQL 本身不是 BFF。</strong></p>
</blockquote>
<hr />
<h2>十一、Next.js 为什么天然适合承担 BFF&#xff1f;</h2>
<p>现在很多 React 项目使用 Next.js。</p>
<p>这时&#xff1a;</p>


```text
Route Handler
Server Action
Server Component
```


<p>本身就天然拥有服务器运行环境。</p>
<p>于是架构可以变成&#xff1a;</p>


```text
Browser
|
v
Next.js
|
+------ Java Backend
|
+------ Python Agent Service
|
+------ RAG Service
```


<p>此时 Next.js Server Layer 就可以承担一部分 BFF 职责。</p>
<p>例如&#xff1a;</p>


```text
Cookie / Session 读取
接口聚合
Token 转换
服务端鉴权
DTO 转换
内部服务地址隐藏
SSE 转发
错误转换
```


<p>浏览器完全不需要知道&#xff1a;</p>


```text
agent-service.internal:8000

rag-service.internal:8111

java-api.internal:8080
```


<p>浏览器只访问&#xff1a;</p>


```text
/api/*
```


<p>这是现代 Web 项目中非常常见的一种 BFF 实现方式。</p>
<hr />
<h2>十二、BFF 在 Agent 系统里尤其有价值</h2>
<p>随着 Agent 系统逐渐复杂&#xff0c;BFF 的价值会更加明显。</p>
<p>例如&#xff1a;</p>


```text
                     Web
|
v
Agent BFF
|
+-----------+-----------+
|           |           |
v           v           v
Agent       Conversation   File
Runtime        Service     Service
|
+-----+-----+
|           |
v           v
RAG        Tools
```


<p>用户发送&#xff1a;</p>


```http
POST /api/chat
```


<p>BFF 内部可能执行&#xff1a;</p>


```text
获取当前 User
↓
验证 Agent 权限
↓
查询 Conversation
↓
处理附件
↓
创建 Agent Run
↓
连接 Agent Runtime
↓
代理 SSE Stream
↓
转换事件
↓
返回浏览器
```


<p>这里有一个特别典型的 Agent 场景&#xff1a;</p>
<h3>SSE Event Adapter</h3>
<p>Agent Runtime 内部可能产生&#xff1a;</p>


```text
run.created

model.started

model.delta

retrieval.started

retrieval.completed

tool.started

tool.completed

subagent.started

subagent.completed

run.completed
```


<p>但是前端不一定想理解整个 Agent Runtime 协议。</p>
<p>前端真正关心的可能只是&#xff1a;</p>


```text
text_delta

thinking

tool_call

status

done
```


<p>于是&#xff1a;</p>


```text
Agent Runtime Events
↓
BFF
↓
Frontend Event Model
```


<p>这样即使以后底层 Agent Runtime 改了&#xff1a;</p>


```text
LangGraph
→ 自研 Runtime
```


<p>只要 BFF 对外协议不变&#xff0c;前端就不需要全部重写。</p>
<p>这其实也是 BFF 很重要的一层价值&#xff1a;</p>
<blockquote>
<p><strong>隔离前端与内部实现细节。</strong></p>
</blockquote>
<hr />
<h2>十三、BFF 最大的坑&#xff1a;慢慢变成“第二个业务后端”</h2>
<p>BFF 很方便。</p>
<p>也正因为方便&#xff0c;非常容易失控。</p>
<p>最开始&#xff1a;</p>


```text
BFF 调用订单服务
```


<p>后来&#xff1a;</p>


```text
顺手判断一下订单状态。
```


<p>再后来&#xff1a;</p>


```text
顺手判断退款条件。
```


<p>然后&#xff1a;</p>


```text
顺手算一下退款金额。
```


<p>最后甚至&#xff1a;</p>


```text
BFF 直接操作数据库。
```


<p>慢慢变成&#xff1a;</p>


```text
               BFF
|
大量业务规则
|
+-------+-------+
|               |
DB            Backend
```


<p>这时 BFF 实际已经变成&#xff1a;</p>
<blockquote>
<p>第二个业务后端。</p>
</blockquote>
<p>这是非常危险的架构演化。</p>
<hr />
<h2>十四、BFF 到底应该放什么&#xff0c;不应该放什么&#xff1f;</h2>
<p>我比较推荐用下面这个公式理解&#xff1a;</p>


```text
BFF
=
Aggregation
+
Transformation
+
Orchestration
+
Frontend-specific Logic
```


<p>而不是&#xff1a;</p>


```text
BFF
=
Domain Business Logic
```


<h3>适合放 BFF</h3>
<p>例如&#xff1a;</p>


```text
多个 API 聚合

并发请求多个服务

DTO 转换

ViewModel 拼装

Cookie → Token 转换

SSE 代理

Agent Event 转换

页面级数据裁剪

客户端特有字段

轻量接口降级
```


<h3>不应该放 BFF</h3>
<p>例如&#xff1a;</p>


```text
退款业务规则

订单金额计算

库存扣减规则

权限核心模型

RAG 检索算法

Agent Runtime 状态机

招投标业务规则

财务规则
```


<p>一个很好判断的方法是&#xff1a;</p>
<blockquote>
<p><strong>如果明天没有这个 Web 页面&#xff0c;这段逻辑还应该存在吗&#xff1f;</strong></p>
</blockquote>
<p>如果答案是&#xff1a;</p>
<blockquote>
<p>应该存在。</p>
</blockquote>
<p>那它很可能属于&#xff1a;</p>


```text
Domain Service
```


<p>而不是 BFF。</p>
<hr />
<h2>十五、BFF 什么时候值得引入&#xff1f;</h2>
<p>不是所有系统都需要 BFF。</p>
<p>如果你的架构只有&#xff1a;</p>


```text
一个 Web
+
一个 Backend
```


<p>而且&#xff1a;</p>


```text
接口简单
页面不复杂
前端请求数量不多
```


<p>那么&#xff1a;</p>


```text
Frontend → Backend
```


<p>完全够用。</p>
<p>没必要为了&#xff1a;</p>
<blockquote>
<p>“架构看起来高级”</p>
</blockquote>
<p>再增加一个 BFF。</p>
<p>因为每增加一个服务&#xff0c;就意味着增加&#xff1a;</p>


```text
部署
日志
监控
Trace
超时
重试
容灾
开发成本
维护成本
```


<p>BFF 真正适合下面这些场景。</p>
<hr />
<h3>场景一&#xff1a;前端开始调用很多微服务</h3>
<p>例如&#xff1a;</p>


```text
一个页面调用 5～10 个服务。
```


<p>这是非常典型的信号。</p>
<hr />
<h3>场景二&#xff1a;前端大量写数据拼装代码</h3>
<p>如果 React/Vue 项目里面开始大量出现&#xff1a;</p>


```text
Promise.all

map

filter

merge

normalize

transform

permissionCheck
```


<p>而且这些逻辑并不是 UI 本身&#xff0c;而是在适配后端数据。</p>
<p>就值得考虑 BFF。</p>
<hr />
<h3>场景三&#xff1a;有多个客户端</h3>
<p>例如&#xff1a;</p>


```text
Web
App
小程序
后台管理端
```


<p>而不同客户端的数据需求差异明显。</p>
<p>这正是 BFF 最经典的应用场景。</p>
<hr />
<h3>场景四&#xff1a;微服务内部结构不希望暴露给前端</h3>
<p>前端已经开始知道&#xff1a;</p>


```text
User Service
Agent Service
File Service
RAG Service
Metrics Service
```


<p>甚至保存多个&#xff1a;</p>


```text
BASE_URL
```


<p>通常意味着服务边界已经泄漏到客户端。</p>
<hr />
<h3>场景五&#xff1a;Agent / AI 应用</h3>
<p>尤其是&#xff1a;</p>


```text
Agent Runtime
RAG
File
Conversation
User
Permission
Observability
```


<p>多个系统共同组成一个 AI 产品时。</p>
<p>BFF 非常适合作为前端与 AI Backend 之间的适配层。</p>
<hr />
<h2>十六、一个比较完整的企业架构</h2>
<p>最终系统可能演进成&#xff1a;</p>


```text
                      Client Layer

Web          App        Mini Program
|            |              |
v            v              v
Web BFF      App BFF        Mini BFF
\            |             /
\           |            /
+----------+-----------+
|
API Gateway
|
+-------------+-------------+
|             |             |
v             v             v
User          Agent          File
Service       Service        Service
|
+-------+-------+
|               |
v               v
RAG            Model
Service         Service
```


<p>现实系统也非常常见&#xff1a;</p>


```text
Client
↓
Gateway
↓
BFF
↓
Microservices
```


<p>究竟 Gateway 在 BFF 前还是后&#xff0c;取决于&#xff1a;</p>


```text
网络拓扑
部署方式
Ingress
安全边界
服务治理方案
```


<p>并没有唯一答案。</p>
<hr />
<h2>十七、理解 BFF 最简单的比喻&#xff1a;餐厅服务员</h2>
<p>可以把整个系统想象成一家餐厅。</p>
<p>前端&#xff1a;</p>
<blockquote>
<p>顾客。</p>
</blockquote>
<p>不同微服务&#xff1a;</p>


```text
主食档口
饮料档口
甜品档口
凉菜档口
```


<p>如果没有 BFF&#xff1a;</p>


```text
顾客
↓
跑去主食档点菜
↓
跑去饮料档点饮料
↓
跑去甜品档点甜点
↓
自己拿回来拼成一顿饭
```


<p>这就类似&#xff1a;</p>


```text
Frontend
|
+------ User Service
+------ Agent Service
+------ RAG Service
+------ File Service
```


<p>有 BFF 后&#xff1a;</p>


```text
顾客
↓
服务员
↓
各个厨房档口
```


<p>服务员负责&#xff1a;</p>


```text
理解这一桌要什么

分别找对应档口下单

协调多个请求

把东西整理好

一次性交给顾客
```


<p>这个“服务员”就是&#xff1a;</p>
<blockquote>
<p>BFF。</p>
</blockquote>
<p>但这里还有一个非常重要的边界&#xff1a;</p>
<blockquote>
<p><strong>服务员不会自己去厨房炒菜。</strong></p>
</blockquote>
<p>这句话几乎可以帮助我们判断绝大多数 BFF 架构问题。</p>
<p>BFF 可以&#xff1a;</p>


```text
叫菜
协调
组合
整理
交付
```


<p>但是业务服务才真正负责&#xff1a;</p>


```text
做菜。
```


<hr />
<h2>十八、从 BFF 背后真正应该理解的架构思想</h2>
<p>BFF 本身并不是最重要的。</p>
<p>更重要的是它体现出的一个架构原则&#xff1a;</p>
<blockquote>
<p><strong>不同系统应该面向自己的消费者设计边界。</strong></p>
</blockquote>
<p>业务后端面向的是&#xff1a;</p>


```text
业务能力
```


<p>所以围绕&#xff1a;</p>


```text
用户
订单
支付
Agent
知识库
```


<p>设计。</p>
<p>而 BFF 面向的是&#xff1a;</p>


```text
前端体验
```


<p>所以围绕&#xff1a;</p>


```text
首页
详情页
工作台
会话页
Dashboard
```


<p>设计。</p>
<p>这两个视角本来就不应该被强行统一。</p>
<p>因此&#xff1a;</p>


```text
Backend Domain Model
```


<p>和&#xff1a;</p>


```text
Frontend View Model
```


<p>之间拥有一个清晰的适配层&#xff0c;往往反而能让整个系统的边界更加稳定。</p>
<hr />
<h2>结语</h2>
<p>BFF 并不是什么复杂的新技术。</p>
<p>它真正解决的是一个随着系统规模扩大几乎必然出现的问题&#xff1a;</p>
<blockquote>
<p><strong>前端需要的数据&#xff0c;与后端提供的业务能力之间存在天然差异。</strong></p>
</blockquote>
<p>当一个系统从&#xff1a;</p>


```text
Frontend
↓
Backend
```


<p>逐渐演变成&#xff1a;</p>


```text
Frontend
↓
十几个 Microservices
```


<p>如果仍然让浏览器直接理解所有后端服务&#xff0c;前端最终就会逐渐承担&#xff1a;</p>


```text
服务发现
接口聚合
数据转换
异常处理
权限拼装
业务编排
```


<p>系统边界会越来越混乱。</p>
<p>BFF 的意义&#xff0c;就是重新把这层复杂度收回来&#xff1a;</p>


```text
Frontend
↓
BFF
↓
Backend Services
```


<p>所以理解 BFF&#xff0c;可以记住两句话&#xff1a;</p>
<blockquote>
<p><strong>Gateway 解决“请求怎么进系统”&#xff0c;BFF 解决“前端需要什么”。</strong></p>
</blockquote>
<p>以及&#xff1a;</p>
<blockquote>
<p><strong>BFF 可以负责点菜、协调和拼盘&#xff0c;但不要让它自己下厨房炒菜。</strong></p>
</blockquote>
<p>对于现代微服务&#xff0c;以及越来越复杂的 Agent / RAG / AI 应用&#xff0c;这种架构思想会越来越有价值。</p>
