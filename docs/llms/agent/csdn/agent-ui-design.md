---
title: "从 Chat UI 到 Agent UI：如何设计一个真正可用的智能体交互界面"
description: "CSDN 原文全文镜像：文章摘要： 随着AI Agent的发展，传统聊天界面(Chat UI)已无法满足复杂任务需求。本文分析了Chat UI与Agent UI的本质差异：Chat UI围绕消息展开，适合简单问答；而Agent UI需处理持续运行的任务执行过程……"
pageType: article
module: agent
updated: '2026-08-05'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "ui"
  - "交互"
  - "prompt"
level: advanced
prerequisites:
  - "/llms/agent/"
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-05，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-05。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163493846](https://blog.csdn.net/m0_63309778/article/details/163493846)
- 站内分区：Agent / Agent UI
:::

<p><img src="https://i-blog.csdnimg.cn/direct/49aafe00eb0f4d6f9ec24d119c069617.png" alt="# 从 Chat UI 到 Agent UI&#xff1a;如何设计一个真正可用的智能体交互界面" /></p>
<h3>前言</h3>
<p>很多团队在开发 Agent 产品时&#xff0c;最先做出来的通常是一个聊天页面&#xff1a;</p>
<ul><li>左边是用户消息&#xff1b;</li><li>右边是 AI 回复&#xff1b;</li><li>底部是输入框&#xff1b;</li><li>后端通过 SSE 返回流式文本。</li></ul>
<p>从界面上看&#xff0c;它和 ChatGPT 很像&#xff0c;因此也很容易让人产生一种错觉&#xff1a;</p>
<blockquote>
<p>只要把聊天页面做得更漂亮&#xff0c;就能得到一个好的 Agent UI。</p>
</blockquote>
<p>但真正进入 Agent 场景后&#xff0c;很快就会发现&#xff0c;传统聊天界面的能力远远不够。</p>
<p>一个普通 Chatbot 的主要任务&#xff0c;是理解用户问题并返回一段回答。一个 Agent 则可能需要先制订计划&#xff0c;再调用工具&#xff0c;读取文件&#xff0c;查询数据库&#xff0c;启动子任务&#xff0c;等待审批&#xff0c;生成文档&#xff0c;并在几分钟后返回最终结果。</p>
<p>因此&#xff0c;Agent UI 面对的不是一次简单的问答&#xff0c;而是一个持续运行、不断产生状态变化的执行过程。</p>
<p>用户真正关心的&#xff0c;也不再只是“AI 回答了什么”&#xff0c;而是&#xff1a;</p>
<ul><li>Agent 当前正在做什么&#xff1b;</li><li>为什么还没有完成&#xff1b;</li><li>已经执行了哪些步骤&#xff1b;</li><li>调用了哪些工具&#xff1b;</li><li>工具调用是否成功&#xff1b;</li><li>是否需要自己确认&#xff1b;</li><li>能否暂停、取消或者重试&#xff1b;</li><li>刷新页面之后还能不能继续&#xff1b;</li><li>最终生成的文档、表格和图表在哪里。</li></ul>
<p>所以&#xff0c;Agent UI 的核心并不是聊天气泡&#xff0c;而是&#xff1a;</p>
<blockquote>
<p>如何让用户理解 Agent 的执行过程&#xff0c;参与关键决策&#xff0c;并最终获得可操作的业务结果。</p>
</blockquote>
<hr />
<h2>一、Chat UI 和 Agent UI 的本质区别</h2>
<p>传统 Chat UI 主要围绕消息展开。</p>
<p>它通常只需要处理三类状态&#xff1a;</p>


```text
用户消息
Assistant 消息
正在生成
```


<p>一次交互的过程也比较简单&#xff1a;</p>


```text
用户输入
↓
发送请求
↓
模型生成文本
↓
前端流式展示
↓
生成完成
```


<p>这类界面很适合&#xff1a;</p>
<ul><li>智能客服&#xff1b;</li><li>FAQ 问答&#xff1b;</li><li>知识库检索&#xff1b;</li><li>简单文本生成&#xff1b;</li><li>日常聊天。</li></ul>
<p>Alibaba ChatUI、Chatscope 等组件库&#xff0c;主要解决的就是这一类问题&#xff0c;包括消息列表、输入框、快捷回复、响应式布局和移动端适配。</p>
<p>但 Agent 的执行过程更接近一个后台任务系统&#xff1a;</p>


```text
用户提出目标
↓
Agent 创建 Run
↓
分析任务
↓
制订计划
↓
调用工具
↓
处理工具结果
↓
更新计划
↓
等待用户确认
↓
继续执行
↓
生成最终结果
```


<p>这个过程可能持续几十秒&#xff0c;也可能持续几十分钟。</p>
<p>页面可能关闭&#xff0c;用户可能切换会话&#xff0c;网络可能断开&#xff0c;工具可能失败&#xff0c;Agent 也可能暂停等待用户输入。</p>
<p>因此&#xff0c;Agent UI 需要围绕以下对象来设计&#xff1a;</p>


```text
Conversation
Message
Run
Step
Tool Call
Approval
Artifact
Event
```


<p>其中&#xff1a;</p>
<ul><li>Conversation 表示一整段会话&#xff1b;</li><li>Message 表示用户或 Agent 的消息&#xff1b;</li><li>Run 表示 Agent 的一次完整执行&#xff1b;</li><li>Step 表示执行计划中的某个步骤&#xff1b;</li><li>Tool Call 表示一次工具调用&#xff1b;</li><li>Approval 表示等待用户确认的操作&#xff1b;</li><li>Artifact 表示生成的文档、表格、图表等成果&#xff1b;</li><li>Event 表示执行过程中发生的一次状态变化。</li></ul>
<p>这意味着 Agent UI 不应该只是一个 Message List&#xff0c;而应该是一个围绕 Run 展开的工作界面。</p>
<hr />
<h2>二、Agent UI 应该围绕“执行过程”设计</h2>
<p>传统聊天页面的中心是消息。</p>
<p>Agent UI 的中心则应该是执行过程。</p>
<p>用户发送&#xff1a;</p>


```text
请分析这份招标文件，并生成投标建议。
```


<p>背后可能发生&#xff1a;</p>


```text
读取文件
↓
解析 PDF
↓
提取资格条件
↓
提取评分办法
↓
检索历史案例
↓
分析风险
↓
生成投标建议
```


<p>如果页面只显示一个“正在思考”的动画&#xff0c;用户很难判断&#xff1a;</p>
<ul><li>系统是否卡住了&#xff1b;</li><li>当前执行到了哪里&#xff1b;</li><li>是否真的读取了文件&#xff1b;</li><li>是否调用了企业知识库&#xff1b;</li><li>预计还需要多久&#xff1b;</li><li>某一步失败后会不会继续。</li></ul>
<p>更合理的页面应该把执行过程转换为用户可以理解的状态。</p>
<p>例如&#xff1a;</p>


```text
正在分析招标文件

✓ 文件读取完成
✓ 已提取资格条件
● 正在分析评分办法
○ 等待匹配历史案例
○ 等待生成投标建议
```


<p>这类展示不是为了暴露模型内部的完整思维过程&#xff0c;而是为了呈现可解释的执行状态。</p>
<p>用户不需要看到模型每一步隐含推理&#xff0c;但需要知道&#xff1a;</p>
<ul><li>Agent 当前处在哪个阶段&#xff1b;</li><li>已经做了什么&#xff1b;</li><li>下一步准备做什么&#xff1b;</li><li>是否出现错误&#xff1b;</li><li>是否需要用户参与。</li></ul>
<p>因此&#xff0c;Agent UI 的第一原则应该是&#xff1a;</p>
<blockquote>
<p>把 Agent 的内部执行状态转换为用户可以理解、可以判断、可以操作的界面状态。</p>
</blockquote>
<hr />
<h2>三、一个成熟的 Agent UI 应该是什么结构</h2>
<p>一个复杂 Agent 产品&#xff0c;通常不适合只使用单栏聊天页面。</p>
<p>更合理的是三栏或者两栏半布局&#xff1a;</p>


```text
┌──────────────────────────────────────────────────────────┐
│ Agent 名称 / 当前状态 / 模型 / 停止 / 重新执行           │
├──────────────┬──────────────────────────┬────────────────┤
│              │                          │                │
│ 会话列表     │       对话与执行区       │   工作区       │
│              │                          │                │
│ 历史会话     │ 用户消息                 │ 文档           │
│ 运行中会话   │ Agent 回复               │ 表格           │
│ 失败会话     │ 执行计划                 │ 图表           │
│ 等待确认     │ 工具调用                 │ 文件预览       │
│              │ 审批卡片                 │ 代码预览       │
│              │                          │                │
├──────────────┴──────────────────────────┴────────────────┤
│ 输入框 / 附件 / 技能 / Agent / 模型 / 发送 / 停止       │
└──────────────────────────────────────────────────────────┘
```


<p>左侧负责会话管理&#xff0c;中间负责交流和执行过程&#xff0c;右侧负责展示复杂成果。</p>
<p>这样的布局能够把“聊天”和“工作”分开。</p>
<p>聊天区域适合展示&#xff1a;</p>
<ul><li>用户提问&#xff1b;</li><li>Agent 的解释&#xff1b;</li><li>执行计划&#xff1b;</li><li>工具调用摘要&#xff1b;</li><li>审批请求&#xff1b;</li><li>最终结果摘要。</li></ul>
<p>工作区适合展示&#xff1a;</p>
<ul><li>长文档&#xff1b;</li><li>数据表格&#xff1b;</li><li>图表&#xff1b;</li><li>代码&#xff1b;</li><li>PDF&#xff1b;</li><li>HTML 页面&#xff1b;</li><li>可编辑报告&#xff1b;</li><li>数据分析结果。</li></ul>
<p>例如 Agent 生成了一份 5000 字的风险报告。</p>
<p>不应该把整份报告全部塞进消息气泡&#xff0c;而可以在聊天区显示&#xff1a;</p>


```text
已生成《项目风险分析报告》。

共识别 18 项风险，其中 4 项为高风险。

[打开报告] [导出 Word] [继续修改]
```


<p>用户点击后&#xff0c;在右侧工作区打开完整报告。</p>
<p>这样页面职责更加清晰&#xff1a;</p>


```text
聊天区负责沟通和解释
工作区负责承载实际成果
```


<hr />
<h2>四、Agent UI 的核心不是 Message&#xff0c;而是 Message Part</h2>
<p>很多聊天页面的数据模型是&#xff1a;</p>


```typescript
<span class="token keyword">interface</span> <span class="token class-name">Message</span> <span class="token punctuation">{<!-- --></span>
role<span class="token operator">:</span> <span class="token string">"user"</span> <span class="token operator">|</span> <span class="token string">"assistant"</span><span class="token punctuation">;</span>
content<span class="token operator">:</span> <span class="token builtin">string</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
```


<p>这对普通聊天足够&#xff0c;但对 Agent 来说过于简单。</p>
<p>因为 Agent 的一条回复中&#xff0c;可能同时包含&#xff1a;</p>
<ul><li>文本&#xff1b;</li><li>工具调用&#xff1b;</li><li>工具结果&#xff1b;</li><li>执行计划&#xff1b;</li><li>来源引用&#xff1b;</li><li>审批请求&#xff1b;</li><li>生成文件&#xff1b;</li><li>错误信息。</li></ul>
<p>因此更合理的数据结构是&#xff1a;</p>


```text
Message
↓
多个 Message Part
```


<p>例如&#xff1a;</p>


```typescript
<span class="token keyword">interface</span> <span class="token class-name">AgentMessage</span> <span class="token punctuation">{<!-- --></span>
id<span class="token operator">:</span> <span class="token builtin">string</span><span class="token punctuation">;</span>
role<span class="token operator">:</span> <span class="token string">"user"</span> <span class="token operator">|</span> <span class="token string">"assistant"</span><span class="token punctuation">;</span>
status<span class="token operator">:</span> <span class="token string">"pending"</span> <span class="token operator">|</span> <span class="token string">"streaming"</span> <span class="token operator">|</span> <span class="token string">"completed"</span> <span class="token operator">|</span> <span class="token string">"failed"</span><span class="token punctuation">;</span>
parts<span class="token operator">:</span> MessagePart<span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
```


<p>其中&#xff1a;</p>


```typescript
<span class="token keyword">type</span> <span class="token class-name">MessagePart</span> <span class="token operator">=</span>
<span class="token operator">|</span> TextPart
<span class="token operator">|</span> PlanPart
<span class="token operator">|</span> ToolPart
<span class="token operator">|</span> ApprovalPart
<span class="token operator">|</span> ArtifactPart
<span class="token operator">|</span> SourcePart
<span class="token operator">|</span> ErrorPart<span class="token punctuation">;</span>
```


<p>一条 Agent 消息可以是&#xff1a;</p>


```text
Assistant Message
├── Text Part
│   “我会先分析文件结构。”
├── Plan Part
│   5 个执行步骤
├── Tool Part
│   正在解析 PDF
├── Tool Part
│   正在查询案例库
├── Artifact Part
│   风险分析报告
└── Text Part
“已经完成分析。”
```


<p>这种设计带来的最大好处是&#xff1a;</p>
<blockquote>
<p>前端不再把所有内容都当成字符串&#xff0c;而是根据内容类型选择专门的组件。</p>
</blockquote>
<p>例如&#xff1a;</p>


```typescript
<span class="token keyword">function</span> <span class="token function">MessagePartRenderer</span><span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span> part <span class="token punctuation">}</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span> part<span class="token operator">:</span> MessagePart <span class="token punctuation">}</span><span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
<span class="token keyword">switch</span> <span class="token punctuation">(</span>part<span class="token punctuation">.</span>type<span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
<span class="token keyword">case</span> <span class="token string">"text"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token operator"><</span>MarkdownContent text<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>part<span class="token punctuation">.</span>text<span class="token punctuation">}</span> <span class="token operator">/</span><span class="token operator">></span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"plan"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token operator"><</span>PlanCard plan<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>part<span class="token punctuation">}</span> <span class="token operator">/</span><span class="token operator">></span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"tool"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token operator"><</span>ToolCard tool<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>part<span class="token punctuation">}</span> <span class="token operator">/</span><span class="token operator">></span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"approval"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token operator"><</span>ApprovalCard approval<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>part<span class="token punctuation">}</span> <span class="token operator">/</span><span class="token operator">></span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"artifact"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token operator"><</span>ArtifactCard artifact<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>part<span class="token punctuation">}</span> <span class="token operator">/</span><span class="token operator">></span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"error"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token operator"><</span>ErrorCard error<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>part<span class="token punctuation">}</span> <span class="token operator">/</span><span class="token operator">></span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>assistant-ui 和 Vercel AI SDK UI 都采用了类似思想&#xff1a;消息不是简单文本&#xff0c;而是由多个结构化 Part 组成。</p>
<p>这是从 Chat UI 走向 Agent UI 最重要的一次数据模型升级。</p>
<hr />
<h2>五、工具调用是 Agent UI 的核心内容</h2>
<p>Agent 和普通聊天机器人最大的区别之一&#xff0c;就是 Agent 会调用工具。</p>
<p>例如&#xff1a;</p>


```text
search_documents
query_database
parse_file
generate_chart
send_email
create_document
```


<p>最原始的实现通常把工具调用直接显示为 JSON&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"tool"</span><span class="token operator">:</span> <span class="token string">"search_documents"</span><span class="token punctuation">,</span>
<span class="token string-property property">"args"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"keyword"</span><span class="token operator">:</span> <span class="token string">"评分办法"</span><span class="token punctuation">,</span>
<span class="token string-property property">"top_k"</span><span class="token operator">:</span> <span class="token number">20</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>这种方式虽然方便开发&#xff0c;但对普通用户非常不友好。</p>
<p>用户并不关心内部工具名称&#xff0c;也不关心参数结构。</p>
<p>他真正关心的是&#xff1a;</p>


```text
正在搜索企业案例库

关键词：评分办法
搜索范围：当前企业
已找到 12 个相关文档
```


<p>因此工具 UI 应该分成两个层次。</p>
<p>默认层面展示业务含义&#xff1a;</p>


```text
✓ 已完成案例检索
找到 12 个相关项目
耗时 1.8 秒
```


<p>展开后再展示技术详情&#xff1a;</p>


```text
工具名称：search_documents
参数：{"keyword":"评分办法","top_k":20}
Trace ID：trace_10001
```


<p>同时&#xff0c;不同工具应该拥有不同的展示组件。</p>
<p>例如&#xff1a;</p>


```text
搜索工具
→ 搜索结果列表

数据库查询
→ 数据表格

图表生成
→ 图表组件

文件解析
→ 文件进度卡片

文档生成
→ Artifact 卡片

发送邮件
→ 邮件预览与确认按钮
```


<p>不应该所有工具都使用同一张 JSON 卡片。</p>
<p>更合理的方式是建立 Tool Renderer Registry&#xff1a;</p>


```typescript
<span class="token keyword">const</span> toolRenderers <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
search_documents<span class="token operator">:</span> SearchResultCard<span class="token punctuation">,</span>
query_database<span class="token operator">:</span> DataTableCard<span class="token punctuation">,</span>
generate_chart<span class="token operator">:</span> ChartCard<span class="token punctuation">,</span>
create_document<span class="token operator">:</span> DocumentArtifactCard<span class="token punctuation">,</span>
send_email<span class="token operator">:</span> EmailApprovalCard<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
```


<p>前端根据工具名称选择专用渲染器。</p>
<p>这种模式也是 Vercel AI SDK、assistant-ui 和 CopilotKit 中非常重要的一类设计。</p>
<hr />
<h2>六、执行计划应该成为用户理解 Agent 的入口</h2>
<p>复杂 Agent 往往会先生成执行计划。</p>
<p>例如&#xff1a;</p>


```text
分析招标文件
├── 读取文件
├── 提取资格要求
├── 提取评分办法
├── 匹配历史案例
└── 生成投标建议
```


<p>如果计划只是 Agent 内部数据&#xff0c;用户看不到&#xff0c;就失去了很大价值。</p>
<p>一个好的 Plan UI 应该显示&#xff1a;</p>


```text
执行计划                              3 / 5

✓ 读取招标文件
✓ 提取资格条件
● 分析评分办法
○ 匹配历史案例
○ 生成投标建议
```


<p>它可以让用户快速知道&#xff1a;</p>
<ul><li>任务被拆成了哪些步骤&#xff1b;</li><li>当前执行到哪一步&#xff1b;</li><li>哪些步骤已经完成&#xff1b;</li><li>哪一步失败&#xff1b;</li><li>是否还有后续步骤。</li></ul>
<p>Plan 的状态可以设计为&#xff1a;</p>


```typescript
<span class="token keyword">type</span> <span class="token class-name">StepStatus</span> <span class="token operator">=</span>
<span class="token operator">|</span> <span class="token string">"pending"</span>
<span class="token operator">|</span> <span class="token string">"running"</span>
<span class="token operator">|</span> <span class="token string">"completed"</span>
<span class="token operator">|</span> <span class="token string">"failed"</span>
<span class="token operator">|</span> <span class="token string">"skipped"</span><span class="token punctuation">;</span>
```


<p>需要特别注意&#xff0c;Plan 和 Todo 不应该被混为一谈。</p>
<p>Plan 是&#xff1a;</p>


```text
展示给用户看的执行计划
```


<p>Todo Tool 是&#xff1a;</p>


```text
Agent 自己可以调用和更新的任务管理工具
```


<p>Todo 的变化可以投影成 Plan UI&#xff0c;但二者在架构中仍然是两个概念。</p>
<hr />
<h2>七、Agent UI 必须支持 Human-in-the-loop</h2>
<p>Agent 的能力越强&#xff0c;越需要人工确认。</p>
<p>如果 Agent 可以&#xff1a;</p>
<ul><li>删除文件&#xff1b;</li><li>发送邮件&#xff1b;</li><li>修改数据库&#xff1b;</li><li>提交审批&#xff1b;</li><li>发布内容&#xff1b;</li><li>创建生产任务&#xff1b;</li><li>执行付款&#xff1b;</li></ul>
<p>就不能允许它在没有用户确认的情况下自动执行。</p>
<p>此时 Agent 应该进入等待状态&#xff1a;</p>


```text
running
↓
waiting_user
↓
用户确认
↓
running
```


<p>页面需要展示 Approval Card&#xff1a;</p>


```text
需要你的确认

Agent 准备向 126 位客户发送通知邮件。

主题：产品升级通知
接收人数：126
发送时间：立即发送

[查看邮件内容] [取消] [确认发送]
```


<p>这类交互的重点&#xff0c;不只是提供两个按钮&#xff0c;而是要让后台 Agent 真正暂停。</p>
<p>用户确认后&#xff0c;Agent 从暂停位置继续&#xff0c;而不是重新运行整个任务。</p>
<p>因此后端需要支持&#xff1a;</p>


```text
approval.required
approval.approved
approval.rejected
```


<p>前端收到 <code>approval.required</code> 后&#xff1a;</p>
<ol><li>显示审批卡片&#xff1b;</li><li>将 Run 状态设置为 <code>waiting_user</code>&#xff1b;</li><li>停止展示“正在执行”&#xff1b;</li><li>等待用户操作&#xff1b;</li><li>用户确认后发送结果&#xff1b;</li><li>后端恢复 Run。</li></ol>
<p>Human-in-the-loop 是 Agent 产品从“自动生成内容”走向“参与真实业务操作”的关键能力。</p>
<hr />
<h2>八、复杂结果应该进入 Artifact Workspace</h2>
<p>Agent 经常会生成复杂结果。</p>
<p>例如&#xff1a;</p>
<ul><li>一份报告&#xff1b;</li><li>一个表格&#xff1b;</li><li>一张图表&#xff1b;</li><li>一段代码&#xff1b;</li><li>一个 HTML 页面&#xff1b;</li><li>一份投标文件&#xff1b;</li><li>一个数据分析看板。</li></ul>
<p>这些内容不适合只作为聊天消息存在。</p>
<p>因此需要引入 Artifact 概念。</p>


```typescript
<span class="token keyword">interface</span> <span class="token class-name">Artifact</span> <span class="token punctuation">{<!-- --></span>
id<span class="token operator">:</span> <span class="token builtin">string</span><span class="token punctuation">;</span>
type<span class="token operator">:</span> <span class="token string">"document"</span> <span class="token operator">|</span> <span class="token string">"table"</span> <span class="token operator">|</span> <span class="token string">"chart"</span> <span class="token operator">|</span> <span class="token string">"code"</span> <span class="token operator">|</span> <span class="token string">"html"</span><span class="token punctuation">;</span>
title<span class="token operator">:</span> <span class="token builtin">string</span><span class="token punctuation">;</span>
status<span class="token operator">:</span> <span class="token string">"creating"</span> <span class="token operator">|</span> <span class="token string">"ready"</span> <span class="token operator">|</span> <span class="token string">"failed"</span><span class="token punctuation">;</span>
version<span class="token operator">:</span> <span class="token builtin">number</span><span class="token punctuation">;</span>
<span class="token punctuation">}</span>
```


<p>聊天区只负责说明&#xff1a;</p>


```text
已经为你生成销售分析报告。
```


<p>Artifact 区负责真正展示&#xff1a;</p>


```text
报告正文
可编辑内容
版本历史
导出按钮
分享按钮
继续修改
```


<p>这种设计会让 Agent 更像一个工作助手&#xff0c;而不是一个只能输出文字的聊天机器人。</p>
<p>Artifact 还可以支持版本迭代&#xff1a;</p>


```text
销售分析报告 v1
↓
用户要求补充区域对比
↓
销售分析报告 v2
↓
用户要求加入图表
↓
销售分析报告 v3
```


<p>对话负责描述修改意图&#xff0c;Artifact 负责保存真实成果。</p>
<hr />
<h2>九、Agent UI 的状态必须围绕 Run 管理</h2>
<p>普通聊天页面通常只有&#xff1a;</p>


```text
idle
streaming
completed
```


<p>但 Agent Run 的状态会复杂得多&#xff1a;</p>


```text
queued
running
waiting_tool
waiting_user
paused
completed
failed
cancelled
```


<p>顶部状态区域可以展示&#xff1a;</p>


```text
● 正在分析文件
已运行 01:32
已完成 3 / 6 个步骤

[停止] [在后台继续]
```


<p>当 Agent 正在执行时&#xff0c;用户切换到其他会话&#xff0c;当前 SSE 可以关闭&#xff0c;但 Run 不能自动停止。</p>
<p>必须区分&#xff1a;</p>


```text
取消页面订阅
≠
取消 Agent 执行
```


<p>用户再次进入时&#xff0c;页面应该重新恢复&#xff1a;</p>


```text
会话消息
Run 状态
Plan 状态
工具状态
Artifact 状态
```


<p>而不是重新执行用户的任务。</p>
<p>所以 Agent UI 和普通聊天 UI 的一个重要差别是&#xff1a;</p>
<blockquote>
<p>Agent Run 是独立于当前页面连接存在的后台执行对象。</p>
</blockquote>
<hr />
<h2>十、Agent UI 不能只依赖 SSE</h2>
<p>SSE 很适合实时推送 Agent 事件。</p>
<p>例如&#xff1a;</p>


```text
run.started
assistant.delta
tool.started
tool.completed
artifact.created
approval.required
run.completed
```


<p>但 SSE 只是传输通道&#xff0c;不应该是状态的唯一来源。</p>
<p>如果用户刷新页面&#xff1a;</p>


```text
SSE 连接消失
前端内存消失
本地流式文本消失
```


<p>如果后端没有保存状态&#xff0c;页面就无法恢复。</p>
<p>因此&#xff0c;一个可靠的 Agent UI 应该采用&#xff1a;</p>


```text
Snapshot
+
Event Log
+
Live Stream
```


<p>其中&#xff1a;</p>
<ul><li>Snapshot 用于恢复当前完整界面&#xff1b;</li><li>Event Log 用于补发离线期间的事件&#xff1b;</li><li>Live Stream 用于接收最新事件。</li></ul>
<p>进入会话时的标准流程是&#xff1a;</p>


```text
加载 Conversation Snapshot
↓
恢复消息、计划、工具和 Artifact
↓
识别 activeRun
↓
读取 lastEventId
↓
重新连接 SSE
↓
补发遗漏事件
↓
继续实时接收
```


<p>这也是 Agent UI 和普通 Chat UI 在架构上的关键差异。</p>
<p>普通聊天可能只需要流式文本。</p>
<p>Agent UI 必须考虑&#xff1a;</p>


```text
页面刷新
切换会话
网络断开
多标签页
后台执行
事件补发
事件去重
```


<hr />
<h2>十一、前端应该使用统一事件 Reducer</h2>
<p>很多项目会在 SSE 监听代码中直接修改页面状态&#xff1a;</p>


```typescript
source<span class="token punctuation">.</span><span class="token function">addEventListener</span><span class="token punctuation">(</span><span class="token string">"tool.started"</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">=></span> <span class="token punctuation">{<!-- --></span>
<span class="token comment">// 修改工具组件</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">;</span>

source<span class="token punctuation">.</span><span class="token function">addEventListener</span><span class="token punctuation">(</span><span class="token string">"assistant.delta"</span><span class="token punctuation">,</span> <span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">=></span> <span class="token punctuation">{<!-- --></span>
<span class="token comment">// 修改消息内容</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span><span class="token punctuation">;</span>
```


<p>随着事件越来越多&#xff0c;这种方式会变得非常混乱。</p>
<p>更合理的方式&#xff0c;是将所有事件交给统一 Reducer&#xff1a;</p>


```typescript
<span class="token keyword">function</span> <span class="token function">applyAgentEvent</span><span class="token punctuation">(</span>
state<span class="token operator">:</span> AgentState<span class="token punctuation">,</span>
event<span class="token operator">:</span> AgentEvent<span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token operator">:</span> AgentState <span class="token punctuation">{<!-- --></span>
<span class="token keyword">switch</span> <span class="token punctuation">(</span>event<span class="token punctuation">.</span>type<span class="token punctuation">)</span> <span class="token punctuation">{<!-- --></span>
<span class="token keyword">case</span> <span class="token string">"run.started"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">startRun</span><span class="token punctuation">(</span>state<span class="token punctuation">,</span> event<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"assistant.delta"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">appendText</span><span class="token punctuation">(</span>state<span class="token punctuation">,</span> event<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"plan.updated"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">updatePlan</span><span class="token punctuation">(</span>state<span class="token punctuation">,</span> event<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"tool.started"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">startTool</span><span class="token punctuation">(</span>state<span class="token punctuation">,</span> event<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"tool.completed"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">completeTool</span><span class="token punctuation">(</span>state<span class="token punctuation">,</span> event<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"approval.required"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">requireApproval</span><span class="token punctuation">(</span>state<span class="token punctuation">,</span> event<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"artifact.created"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">createArtifact</span><span class="token punctuation">(</span>state<span class="token punctuation">,</span> event<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">case</span> <span class="token string">"run.completed"</span><span class="token operator">:</span>
<span class="token keyword">return</span> <span class="token function">completeRun</span><span class="token punctuation">(</span>state<span class="token punctuation">,</span> event<span class="token punctuation">)</span><span class="token punctuation">;</span>

<span class="token keyword">default</span><span class="token operator">:</span>
<span class="token keyword">return</span> state<span class="token punctuation">;</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>这样做有几个好处&#xff1a;</p>
<ul><li>实时事件和补发事件使用同一套逻辑&#xff1b;</li><li>页面刷新后可以重新构建状态&#xff1b;</li><li>可以根据 eventId 去重&#xff1b;</li><li>可以检测事件顺序&#xff1b;</li><li>状态变化容易测试&#xff1b;</li><li>后端框架发生变化时&#xff0c;前端改动更小。</li></ul>
<p>AG-UI 的重要价值就在于&#xff0c;它尝试定义一套 Agent 和前端之间的统一事件协议。</p>
<p>前端不需要直接理解 LangGraph、OpenAI Agents 或自研 Runtime 的所有私有事件&#xff0c;而是先通过 Adapter 转换成统一 UI Event。</p>
<p>架构可以设计为&#xff1a;</p>


```text
LangGraph Events ─┐
OpenAI Events ────┤
自研 Runtime ─────┼→ UI Event Protocol → Reducer → Components
其他 Agent ───────┘
```


<hr />
<h2>十二、Generative UI 应该怎么做</h2>
<p>Generative UI 是 Agent UI 的重要方向。</p>
<p>它意味着 Agent 不只返回文字&#xff0c;还可以决定页面应该显示什么结构化组件。</p>
<p>例如用户说&#xff1a;</p>


```text
帮我比较这三种服务器配置。
```


<p>普通 Chat UI 可能返回一个 Markdown 表格。</p>
<p>Generative UI 可以直接生成&#xff1a;</p>


```text
服务器配置对比组件
├── CPU 对比
├── 内存对比
├── 价格对比
├── 推荐标签
└── 选择按钮
```


<p>但 Generative UI 不应该等于让模型随意生成和执行 HTML。</p>
<p>企业系统更适合两种方式。</p>
<p>第一种是受控组件&#xff1a;</p>


```typescript
<span class="token keyword">const</span> componentRegistry <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
risk_list<span class="token operator">:</span> RiskList<span class="token punctuation">,</span>
project_table<span class="token operator">:</span> ProjectTable<span class="token punctuation">,</span>
approval_form<span class="token operator">:</span> ApprovalForm<span class="token punctuation">,</span>
comparison_chart<span class="token operator">:</span> ComparisonChart<span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">;</span>
```


<p>Agent 只能选择已有组件&#xff0c;并传递参数&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"component"</span><span class="token operator">:</span> <span class="token string">"risk_list"</span><span class="token punctuation">,</span>
<span class="token string-property property">"props"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"items"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>第二种是声明式 UI Schema&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"card"</span><span class="token punctuation">,</span>
<span class="token string-property property">"children"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"heading"</span><span class="token punctuation">,</span>
<span class="token string-property property">"text"</span><span class="token operator">:</span> <span class="token string">"风险分析"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"list"</span><span class="token punctuation">,</span>
<span class="token string-property property">"items"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>前端根据 Schema 渲染已有设计系统中的组件。</p>
<p>不推荐在企业系统中直接执行模型生成的任意 HTML 或 React 代码&#xff0c;因为会带来&#xff1a;</p>
<ul><li>XSS 风险&#xff1b;</li><li>样式不可控&#xff1b;</li><li>性能不可控&#xff1b;</li><li>组件质量不稳定&#xff1b;</li><li>与设计系统不一致。</li></ul>
<hr />
<h2>十三、如何借鉴主流 Agent UI 项目</h2>
<p>目前主流方案各自解决的问题不同&#xff0c;不适合简单地互相替代。</p>
<h3>ChatUI</h3>
<p>ChatUI 更适合参考基础聊天体验&#xff1a;</p>
<ul><li>消息列表&#xff1b;</li><li>输入框&#xff1b;</li><li>快捷回复&#xff1b;</li><li>移动端布局&#xff1b;</li><li>会话基本组件。</li></ul>
<p>它适合作为 Agent UI 的基础层&#xff0c;但无法单独解决复杂 Agent Run、工具调用和状态恢复。</p>
<h3>assistant-ui</h3>
<p>assistant-ui 更适合参考组件化和 Runtime 设计。</p>
<p>它将页面拆成&#xff1a;</p>
<ul><li>Thread&#xff1b;</li><li>Message&#xff1b;</li><li>Composer&#xff1b;</li><li>Message Part&#xff1b;</li><li>Attachment&#xff1b;</li><li>Action Bar&#xff1b;</li><li>Branch Picker。</li></ul>
<p>这种 Primitive 化的思路非常值得借鉴。</p>
<p>不要只做一个巨大的 <code>&lt;AgentChat /&gt;</code> 组件&#xff0c;而是把行为和视觉拆成可组合组件。</p>
<h3>Vercel AI SDK UI</h3>
<p>Vercel AI SDK UI 更适合参考&#xff1a;</p>
<ul><li>UIMessage&#xff1b;</li><li>Message Parts&#xff1b;</li><li>工具调用流&#xff1b;</li><li>前后端消息协议&#xff1b;</li><li>Generative UI&#xff1b;</li><li>数据流处理。</li></ul>
<p>它对 Next.js 和 React 项目尤其友好。</p>
<h3>CopilotKit</h3>
<p>CopilotKit 更适合参考 Agent 如何深度嵌入业务页面。</p>
<p>重点能力包括&#xff1a;</p>
<ul><li>Generative UI&#xff1b;</li><li>Shared State&#xff1b;</li><li>Human-in-the-loop&#xff1b;</li><li>Agent 与 React 组件交互&#xff1b;</li><li>Agent 读取和修改页面状态。</li></ul>
<p>如果产品不是一个独立聊天页面&#xff0c;而是希望 Agent 操作现有业务系统&#xff0c;CopilotKit 的方向更值得参考。</p>
<h3>LangChain Agent Chat UI</h3>
<p>它更适合参考 LangGraph 场景中的&#xff1a;</p>
<ul><li>工具调用展示&#xff1b;</li><li>Run 中断&#xff1b;</li><li>状态恢复&#xff1b;</li><li>Thread&#xff1b;</li><li>Time Travel&#xff1b;</li><li>调试。</li></ul>
<h3>AG-UI</h3>
<p>AG-UI 更适合参考协议层设计。</p>
<p>它解决的问题是&#xff1a;</p>


```text
不同 Agent Runtime
↓
如何使用统一事件与前端通信
```


<p>如果团队计划长期建设自研 Agent 平台&#xff0c;统一 UI 协议会比单纯选择某一个组件库更重要。</p>
<hr />
<h2>十四、一个推荐的 Agent UI 前端架构</h2>
<p>完整前端可以划分为五层&#xff1a;</p>


```text
┌─────────────────────────────────────────┐
│              页面与组件层               │
│                                         │
│ Thread / Composer / Tool / Plan         │
│ Approval / Artifact / Sources           │
├─────────────────────────────────────────┤
│                状态层                   │
│                                         │
│ Conversation State                      │
│ Run State                               │
│ Artifact State                          │
├─────────────────────────────────────────┤
│              Event Reducer              │
├─────────────────────────────────────────┤
│               Transport                 │
│                                         │
│ SSE / Fetch Stream / WebSocket          │
├─────────────────────────────────────────┤
│           Agent Protocol Adapter        │
├─────────────────────────────────────────┤
│ LangGraph / OpenAI / 自研 Agent Runtime │
└─────────────────────────────────────────┘
```


<p>目录可以设计为&#xff1a;</p>


```text
agent-ui/
├── components/
│   ├── thread/
│   ├── message/
│   ├── composer/
│   ├── plan/
│   ├── tool/
│   ├── approval/
│   ├── artifact/
│   └── source/
├── runtime/
│   ├── reducer/
│   ├── transport/
│   ├── recovery/
│   └── adapters/
├── store/
│   ├── conversation-store.ts
│   ├── run-store.ts
│   └── artifact-store.ts
├── protocol/
│   ├── event.ts
│   ├── message.ts
│   └── tool.ts
└── renderers/
├── message-part-renderer.tsx
├── tool-renderer.tsx
└── artifact-renderer.tsx
```


<p>这样的结构能够避免&#xff1a;</p>
<ul><li>UI 组件直接依赖 SSE&#xff1b;</li><li>SSE 直接修改页面&#xff1b;</li><li>前端强绑定某一个 Agent 框架&#xff1b;</li><li>所有工具逻辑堆在 Message 组件中。</li></ul>
<hr />
<h2>十五、从现有 Chat UI 演进到 Agent UI</h2>
<p>如果当前已经有一个基本的聊天页面&#xff0c;不需要一次性全部重做。</p>
<p>可以分四个阶段建设。</p>
<h3>第一阶段&#xff1a;完善基础聊天体验</h3>
<p>先把基本能力做好&#xff1a;</p>
<ul><li>Message Part&#xff1b;</li><li>Markdown&#xff1b;</li><li>代码块&#xff1b;</li><li>附件&#xff1b;</li><li>停止生成&#xff1b;</li><li>重试&#xff1b;</li><li>历史会话&#xff1b;</li><li>自动滚动&#xff1b;</li><li>错误提示。</li></ul>
<p>这个阶段可以重点参考 ChatUI 和 assistant-ui。</p>
<h3>第二阶段&#xff1a;增加 Agent 执行展示</h3>
<p>开始展示&#xff1a;</p>
<ul><li>Run 状态&#xff1b;</li><li>Plan&#xff1b;</li><li>Tool Call&#xff1b;</li><li>Tool Result&#xff1b;</li><li>Progress&#xff1b;</li><li>Error&#xff1b;</li><li>Cancel&#xff1b;</li><li>Retry。</li></ul>
<p>这个阶段&#xff0c;产品就从普通 Chat UI 开始进入 Agent UI。</p>
<h3>第三阶段&#xff1a;实现可靠状态恢复</h3>
<p>加入&#xff1a;</p>
<ul><li>Snapshot&#xff1b;</li><li>Event Log&#xff1b;</li><li>eventId&#xff1b;</li><li>SSE 重连&#xff1b;</li><li>切换会话恢复&#xff1b;</li><li>页面刷新恢复&#xff1b;</li><li>后台 Run&#xff1b;</li><li>事件去重。</li></ul>
<p>这个阶段决定系统是否真正能够进入生产环境。</p>
<h3>第四阶段&#xff1a;建设 Agentic UI</h3>
<p>最后加入&#xff1a;</p>
<ul><li>Human-in-the-loop&#xff1b;</li><li>Artifact Workspace&#xff1b;</li><li>Subagent&#xff1b;</li><li>Shared State&#xff1b;</li><li>Generative UI&#xff1b;</li><li>UI Protocol Adapter。</li></ul>
<p>最终的演进路线可以概括为&#xff1a;</p>


```text
Chat UI
↓
Tool-aware Chat UI
↓
Agent Runtime UI
↓
Agentic Application UI
```


<hr />
<h2>结语</h2>
<p>传统 Chat UI 的目标&#xff0c;是让用户和模型进行自然对话。</p>
<p>Agent UI 的目标则更复杂&#xff1a;</p>
<blockquote>
<p>让用户能够理解、控制并参与一个持续运行的智能执行系统。</p>
</blockquote>
<p>因此&#xff0c;一个成熟的 Agent UI 不应该只有&#xff1a;</p>


```text
消息列表
+
输入框
+
流式输出
```


<p>它应该至少具备&#xff1a;</p>


```text
结构化 Message Part
Agent Run 状态
执行计划
工具调用展示
人工确认
复杂成果工作区
状态恢复
统一事件协议
```


<p>ChatUI 可以帮助我们解决基础聊天体验。</p>
<p>assistant-ui 可以帮助我们理解组件 Primitive 和 Runtime。</p>
<p>Vercel AI SDK UI 可以帮助我们设计 Message Part、工具流和 Generative UI。</p>
<p>CopilotKit 可以帮助我们理解 Shared State、Human-in-the-loop 和业务页面协同。</p>
<p>AG-UI 可以帮助我们建立 Agent Runtime 与前端之间的统一协议。</p>
<p>但真正决定 Agent UI 质量的&#xff0c;不是选择了哪个组件库&#xff0c;而是是否真正围绕 Agent 的执行过程设计。</p>
<p>最终&#xff0c;一个好的 Agent UI 应该让用户清楚地知道&#xff1a;</p>


```text
Agent 正在做什么
为什么这样做
已经完成了什么
哪里需要自己参与
最终成果在哪里
```


<p>只有做到这一点&#xff0c;Agent UI 才不再只是一个“更复杂的聊天窗口”&#xff0c;而是用户与智能体共同完成工作的操作界面。</p>
