---
title: "深入理解 Agent Runtime：智能体真正运行起来的核心执行引擎"
description: "CSDN 原文全文镜像：文章摘要： Agent Runtime 是大模型应用的核心执行系统，负责管理 Agent 任务的完整生命周期，而不仅仅是简单的“LLM + Tool”调用。生产级 Agent 任务可能涉及多工具调用、子任务协调、人工审批、上下文管理等复……"
pageType: article
module: agent
updated: '2026-08-10'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "大模型"
  - "agent runtime"
level: advanced
prerequisites:
  - "/llms/agent/"
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-10，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-10。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163638436](https://blog.csdn.net/m0_63309778/article/details/163638436)
- 站内分区：Agent / Agent Runtime
:::

<p><img src="https://i-blog.csdnimg.cn/direct/547db36d473d45148c6c626573f5072c.png" alt="在这里插入图片描述" /></p>
<h3>前言</h3>
<p>在开发大模型应用时&#xff0c;我们很容易把 Agent 理解成&#xff1a;</p>


```text
LLM + Tool
```


<p>模型负责思考&#xff0c;Tool 负责执行动作。</p>
<p>这种理解并没有错&#xff0c;但只适合解释一个非常简单的 Agent Demo。</p>
<p>当 Agent 真正进入生产系统之后&#xff0c;一个任务可能持续几十秒、几分钟&#xff0c;甚至更久。执行过程中&#xff0c;它可能需要读取历史对话、加载 Skill、检索 Memory、执行 RAG、调用多个工具、启动子 Agent、等待人工审批、生成文件&#xff0c;并把整个执行过程实时推送到前端。</p>
<p>此时真正需要解决的问题已经不是&#xff1a;</p>
<blockquote>
<p>怎么让模型调用一个工具&#xff1f;</p>
</blockquote>
<p>而是&#xff1a;</p>
<blockquote>
<p>怎么让一个 Agent 任务稳定地从开始运行到结束&#xff0c;并且能够管理它的上下文、状态、工具、记忆、事件、错误、暂停和恢复&#xff1f;</p>
</blockquote>
<p>负责这一切的&#xff0c;就是 <strong>Agent Runtime</strong>。</p>
<p>如果一定要给 Agent Runtime 一个直观的定义&#xff0c;可以把它理解成&#xff1a;</p>
<blockquote>
<p><strong>Agent Runtime 是 Agent 的运行时执行系统。它负责承载一次 Agent Run&#xff0c;并在整个任务生命周期中协调 LLM、Tool、Context、RAG、Memory、Skill、Subagent、Human-in-the-loop、Event 和持久化等能力。</strong></p>
</blockquote>
<p>从这个角度看&#xff0c;LLM 更像 Agent 的“大脑”&#xff0c;Tool 是 Agent 的“手和脚”&#xff0c;Memory 和 RAG 是它获取信息的能力&#xff0c;而 Runtime 更像&#xff1a;</p>
<blockquote>
<p><strong>Agent 的操作系统。</strong></p>
</blockquote>
<hr />
<h2>一、为什么 Agent 需要 Runtime</h2>
<p>最简单的大模型调用可能只有&#xff1a;</p>


```python
response <span class="token operator">=</span> llm<span class="token punctuation">.</span>chat<span class="token punctuation">(</span>messages<span class="token punctuation">)</span>
```


<p>调用过程非常短&#xff1a;</p>


```text
用户问题
↓
LLM
↓
回答
```


<p>如果应用只是做问答&#xff0c;这已经足够。</p>
<p>但是假设用户给 Agent 一个任务&#xff1a;</p>
<blockquote>
<p>帮我分析这份招标文件&#xff0c;找出资格条件和评分办法&#xff0c;再结合公司的历史项目生成一份投标策略。</p>
</blockquote>
<p>这个任务背后可能发生&#xff1a;</p>


```text
读取招标文件
↓
解析 PDF
↓
理解用户目标
↓
生成执行计划
↓
提取资格条件
↓
提取评分办法
↓
查询企业知识库
↓
匹配历史案例
↓
调用风险分析工具
↓
启动子 Agent
↓
汇总分析结果
↓
生成投标策略
↓
生成 Word 报告
```


<p>这里已经出现了很多新的问题。</p>
<p>例如&#xff0c;Agent 调用了一个工具以后&#xff0c;工具结果应该放在哪里&#xff1f;下一次调用 LLM 时&#xff0c;要不要把工具结果继续塞进去&#xff1f;</p>
<p>如果工具执行失败&#xff0c;是直接结束任务&#xff0c;还是重试&#xff1f;如果重试三次仍然失败怎么办&#xff1f;</p>
<p>如果用户刷新页面&#xff0c;Agent 是否继续执行&#xff1f;如果继续执行&#xff0c;用户重新进入页面后怎么恢复当前状态&#xff1f;</p>
<p>如果执行过程中需要用户确认&#xff0c;Agent 怎么暂停&#xff1f;用户十分钟以后确认&#xff0c;又怎么从原来的位置继续&#xff1f;</p>
<p>如果 Agent 创建了三个 Subagent&#xff0c;父 Agent 怎么知道它们什么时候完成&#xff1f;</p>
<p>如果一次任务调用十几次模型&#xff0c;Token、耗时和成本应该如何统计&#xff1f;</p>
<p>这些问题都不是 LLM 本身会解决的。</p>
<p>因此生产级 Agent 需要一个独立的运行时系统&#xff1a;</p>


```text
Agent Runtime
```


<p>它负责把整个 Agent 从“调用一次模型”升级成“运行一个持续存在的智能任务”。</p>
<hr />
<h2>二、Agent Runtime 的核心不是模型&#xff0c;而是 Run</h2>
<p>理解 Agent Runtime&#xff0c;第一个非常重要的概念不是 LLM&#xff0c;而是&#xff1a;</p>


```text
Run
```


<p>Run 可以理解为&#xff1a;</p>
<blockquote>
<p><strong>Agent 执行一次用户任务的实例。</strong></p>
</blockquote>
<p>例如一个 Conversation 中可能发生&#xff1a;</p>


```text
Conversation: conv_001

用户：
你好

Assistant：
你好，有什么可以帮助你的？

用户：
帮我分析这份文件

↓

Run: run_001
```


<p>从这一刻开始&#xff0c;“分析这份文件”不再只是一次 HTTP 请求&#xff0c;而是一个真正的后台运行任务。</p>
<p>Run 可以拥有自己的状态&#xff1a;</p>


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


<p>例如&#xff1a;</p>


```text
run_001

status = running
conversation_id = conv_001
input_message_id = msg_100
output_message_id = msg_101

started_at = ...
finished_at = ...
last_event_id = ...
```


<p>为什么一定要引入 Run&#xff1f;</p>
<p>因为 Agent 的生命周期通常比一次 HTTP 请求长得多。</p>
<p>用户发送消息以后&#xff1a;</p>


```text
POST /messages
```


<p>服务端可以快速返回&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"runId"</span><span class="token operator">:</span> <span class="token string">"run_001"</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"running"</span><span class="token punctuation">,</span>
<span class="token string-property property">"streamUrl"</span><span class="token operator">:</span> <span class="token string">"/api/runs/run_001/events"</span>
<span class="token punctuation">}</span>
```


<p>但 Agent 仍然在后台继续执行。</p>
<p>这就实现了&#xff1a;</p>


```text
HTTP Request 生命周期
≠
Agent Run 生命周期
```


<p>即使用户刷新页面&#xff0c;原来的 SSE 连接断开&#xff1a;</p>


```text
SSE disconnected
```


<p>Run 依然可以是&#xff1a;</p>


```text
run_001
status = running
```


<p>因此&#xff0c;<strong>Run 是 Runtime 中真正的执行主体。</strong></p>
<p>Runtime 的第一项核心工作就是&#xff1a;</p>
<blockquote>
<p>创建 Run、驱动 Run、保存 Run 状态&#xff0c;并最终让 Run 进入 completed、failed 或 cancelled 等终态。</p>
</blockquote>
<hr />
<h2>三、Runtime 本质上是一个持续运行的执行循环</h2>
<p>如果把 Agent Runtime 的复杂能力暂时全部拿掉&#xff0c;它最核心的机制其实是一个循环&#xff1a;</p>


```text
构建上下文
↓
调用 LLM
↓
理解 LLM 的决定
↓
执行动作
↓
拿到结果
↓
继续构建上下文
↓
再次调用 LLM
```


<p>伪代码可以简化成&#xff1a;</p>


```python
<span class="token keyword">while</span> <span class="token keyword">not</span> run<span class="token punctuation">.</span>finished<span class="token punctuation">:</span>

context <span class="token operator">=</span> build_context<span class="token punctuation">(</span>run<span class="token punctuation">)</span>

response <span class="token operator">=</span> llm<span class="token punctuation">.</span>generate<span class="token punctuation">(</span>
context<span class="token operator">=</span>context<span class="token punctuation">,</span>
tools<span class="token operator">=</span>available_tools<span class="token punctuation">,</span>
<span class="token punctuation">)</span>

<span class="token keyword">if</span> response<span class="token punctuation">.</span>has_tool_call<span class="token punctuation">:</span>
result <span class="token operator">=</span> execute_tool<span class="token punctuation">(</span>
response<span class="token punctuation">.</span>tool_call
<span class="token punctuation">)</span>

append_tool_result<span class="token punctuation">(</span>
run<span class="token punctuation">,</span>
result<span class="token punctuation">,</span>
<span class="token punctuation">)</span>

<span class="token keyword">continue</span>

save_assistant_message<span class="token punctuation">(</span>
response<span class="token punctuation">.</span>text
<span class="token punctuation">)</span>

run<span class="token punctuation">.</span>complete<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>这个过程就是典型的&#xff1a;</p>


```text
Think
↓
Act
↓
Observe
↓
Think
↓
Act
↓
Observe
```


<p>但生产环境里的 Runtime 会在这个循环外面增加大量控制能力&#xff1a;</p>


```text
状态机
上下文构建
Tool 调度
事件发布
权限控制
重试
超时
取消
暂停
持久化
恢复
Subagent
审批
监控
```


<p>因此可以认为&#xff1a;</p>
<blockquote>
<p>Agent Runtime 的真正核心是 <strong>Execution Loop &#43; State Machine</strong>。</p>
</blockquote>
<p>Execution Loop 决定 Agent 下一步继续执行什么。</p>
<p>State Machine 决定 Agent 当前到底处于什么状态&#xff0c;以及哪些状态之间可以发生转换。</p>
<hr />
<h2>四、Context Builder&#xff1a;Runtime 每一轮真正给模型什么</h2>
<p>Agent 每次调用模型&#xff0c;并不是简单发送用户最后一句话。</p>
<p>真正的 Prompt 可能包含&#xff1a;</p>


```text
System Prompt

Agent Instructions

用户当前问题

历史 Conversation

当前 Run 状态

当前 Plan

已经执行过的 Tool Call

Tool Result

RAG Context

Knowledge Evidence

Memory

Skill

用户身份

租户信息

当前 Artifact

当前时间
```


<p>所以 Runtime 中一个极其重要的模块通常叫&#xff1a;</p>


```text
Context Builder
```


<p>或者&#xff1a;</p>


```text
Context Manager
Context Assembly
Context Manifest
```


<p>它负责回答一个问题&#xff1a;</p>
<blockquote>
<p><strong>这一轮 LLM 推理&#xff0c;到底应该让模型看到什么&#xff1f;</strong></p>
</blockquote>
<p>例如&#xff1a;</p>


```text
┌────────────────────────┐
│ System Prompt          │
├────────────────────────┤
│ Agent Instructions     │
├────────────────────────┤
│ Loaded Skills          │
├────────────────────────┤
│ Retrieved Memory       │
├────────────────────────┤
│ Conversation History   │
├────────────────────────┤
│ RAG Context            │
├────────────────────────┤
│ Current Plan           │
├────────────────────────┤
│ Tool Results           │
├────────────────────────┤
│ Current User Message   │
└────────────────────────┘
↓
LLM
```


<p>这部分经常被低估。</p>
<p>实际上&#xff0c;很多所谓“Agent 能力不稳定”&#xff0c;并不是模型本身不够聪明&#xff0c;而是 Context 构造出现了问题。</p>
<p>比如历史消息无限增长&#xff1a;</p>


```text
Conversation History
越来越长
```


<p>最终可能导致&#xff1a;</p>
<ul><li>Token 成本不断增加&#xff1b;</li><li>关键信息被淹没&#xff1b;</li><li>上下文窗口不足&#xff1b;</li><li>模型关注错误的信息。</li></ul>
<p>又比如 RAG 一次召回大量文档&#xff1a;</p>


```text
20 个 Chunk
每个 2000 Token
```


<p>直接塞给 LLM&#xff0c;就可能产生几万 Token 的上下文。</p>
<p>再比如 Tool Result 返回了一大段 JSON&#xff0c;完整注入模型&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token string">"...几十万字符..."</span>
<span class="token punctuation">}</span>
```


<p>不仅浪费 Token&#xff0c;还可能让模型失去重点。</p>
<p>所以成熟 Runtime 的 Context Builder 通常还会负责&#xff1a;</p>


```text
裁剪
压缩
摘要
优先级
去重
Token Budget
```


<p>最终目的不是“给模型越多越好”&#xff0c;而是&#xff1a;</p>
<blockquote>
<p><strong>在有限的 Context Window 中&#xff0c;把当前决策真正需要的信息组织得足够清楚。</strong></p>
</blockquote>
<hr />
<h2>五、LLM Adapter&#xff1a;Runtime 不应该和某一家模型绑定</h2>
<p>Agent Runtime 中&#xff0c;LLM 是最核心的推理能力&#xff0c;但 Runtime 本身不应该直接依赖某个具体厂商 API。</p>
<p>否则代码很容易变成&#xff1a;</p>


```python
<span class="token keyword">if</span> model <span class="token operator">==</span> <span class="token string">"openai"</span><span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>

<span class="token keyword">elif</span> model <span class="token operator">==</span> <span class="token string">"qwen"</span><span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>

<span class="token keyword">elif</span> model <span class="token operator">==</span> <span class="token string">"deepseek"</span><span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>

<span class="token keyword">elif</span> model <span class="token operator">==</span> <span class="token string">"vllm"</span><span class="token punctuation">:</span>
<span class="token punctuation">.</span><span class="token punctuation">.</span><span class="token punctuation">.</span>
```


<p>最终所有模型差异都会污染 Runtime。</p>
<p>更合理的结构是&#xff1a;</p>


```text
Agent Runtime
↓
LLM Adapter / Provider
↓
┌──────────────┐
│ OpenAI       │
│ Qwen         │
│ Claude       │
│ Gemini       │
│ DeepSeek     │
│ Local vLLM   │
└──────────────┘
```


<p>Runtime 只使用统一接口&#xff1a;</p>


```python
response <span class="token operator">=</span> llm_provider<span class="token punctuation">.</span>generate<span class="token punctuation">(</span>
messages<span class="token operator">=</span>context<span class="token punctuation">.</span>messages<span class="token punctuation">,</span>
tools<span class="token operator">=</span>context<span class="token punctuation">.</span>tools<span class="token punctuation">,</span>
model<span class="token operator">=</span>model<span class="token punctuation">,</span>
<span class="token punctuation">)</span>
```


<p>Provider 再将不同厂商的格式统一成&#xff1a;</p>


```text
text
tool_calls
finish_reason
usage
reasoning_summary
provider_metadata
```


<p>这样 Runtime 不需要知道&#xff1a;</p>


```text
OpenAI Tool Call 长什么样
Qwen Function Call 长什么样
Claude Tool Use 长什么样
```


<p>这些差异全部由 Adapter 层解决。</p>
<p>这也是为什么生产 Agent 平台中&#xff0c;经常会看到&#xff1a;</p>


```text
Model Provider
LLM Gateway
Model Adapter
```


<p>这些抽象。</p>
<hr />
<h2>六、Tool Runtime&#xff1a;Agent 从“会说”到“会做”的关键</h2>
<p>如果没有 Tool&#xff0c;大多数 Agent 仍然只是一个聊天系统。</p>
<p>Tool 让 Agent 能够真正作用于外部系统。</p>
<p>例如&#xff1a;</p>


```text
search_documents
query_database
parse_document
browser
send_email
create_report
memory_search
execute_code
```


<p>当 LLM 返回&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"search_documents"</span><span class="token punctuation">,</span>
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"query"</span><span class="token operator">:</span> <span class="token string">"评分办法"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>Runtime 不应该简单地执行&#xff1a;</p>


```python
search_documents<span class="token punctuation">(</span><span class="token string">"评分办法"</span><span class="token punctuation">)</span>
```


<p>生产级 Tool Runtime 至少还要经历&#xff1a;</p>


```text
Tool Call
↓
查找 Tool Registry
↓
Schema 校验
↓
参数标准化
↓
权限校验
↓
租户校验
↓
风险等级判断
↓
执行
↓
超时控制
↓
结果标准化
↓
写入 Run Context
```


<p>因此真正的 Tool Runtime 通常是&#xff1a;</p>


```text
Tool Registry
↓
Tool Resolver
↓
Argument Validation
↓
Permission / Policy
↓
Tool Executor
↓
Result Normalizer
```


<p>一个 Tool 可能包含&#xff1a;</p>


```text
name
description
input_schema
output_schema
executor

timeout
retry_policy
risk_level
permissions
```


<p>LLM 看到的是&#xff1a;</p>
<blockquote>
<p>我有哪些工具&#xff0c;它们分别有什么能力&#xff0c;需要什么参数&#xff1f;</p>
</blockquote>
<p>Runtime 看到的是&#xff1a;</p>
<blockquote>
<p>这个 Tool 到底对应哪个执行器&#xff1f;是否允许执行&#xff1f;失败怎么办&#xff1f;</p>
</blockquote>
<p>这两个层面是不同的。</p>
<hr />
<h2>七、MCP 在 Runtime 中处于什么位置</h2>
<p>当 Agent 工具越来越多时&#xff0c;不可能每个 Tool 都在 Runtime 内部直接写实现。</p>
<p>于是会出现 MCP 一类协议。</p>
<p>架构可能变成&#xff1a;</p>


```text
Agent Runtime
↓
Tool Runtime
↓
┌─────────────────┐
│ Internal Tool   │
│ HTTP Tool       │
│ MCP Tool        │
│ Browser Tool    │
└─────────────────┘
```


<p>对于 Runtime 来说&#xff0c;MCP 更像一种&#xff1a;</p>
<blockquote>
<p>外部 Tool Provider 协议。</p>
</blockquote>
<p>Runtime 可以从 MCP Server 获取&#xff1a;</p>


```text
有哪些 Tools
Tool Schema
如何调用
调用结果
```


<p>这样 Agent Runtime 与真正业务能力进一步解耦。</p>
<hr />
<h2>八、RAG 在 Runtime 中有两种完全不同的位置</h2>
<p>很多系统都说自己的 Agent 集成了 RAG&#xff0c;但实际上有两种设计。</p>
<p>第一种是&#xff1a;</p>


```text
RAG as Tool
```


<p>也就是 Agent 自己决定什么时候检索。</p>


```text
LLM
↓
决定需要知识
↓
调用 search_knowledge
↓
返回结果
↓
继续推理
```


<p>这种模式的优点是 Agent 自主性更强。</p>
<p>第二种是&#xff1a;</p>


```text
Automatic RAG
```


<p>Runtime 在调用 LLM 之前自动执行&#xff1a;</p>


```text
用户问题
↓
Query Rewrite
↓
Embedding
↓
Vector Search
↓
Rerank
↓
Context Assembly
↓
LLM
```


<p>这种情况下&#xff0c;LLM 甚至不知道“搜索”是一项 Tool&#xff0c;而是直接看到 Runtime 注入的知识。</p>
<p>两者可以同时存在&#xff1a;</p>


```text
Automatic RAG
负责基础知识增强

RAG Tool
负责 Agent 主动进一步检索
```


<p>Runtime 需要明确负责&#xff1a;</p>


```text
什么时候自动检索
使用哪个知识库
租户过滤
召回多少 Chunk
是否 Rerank
上下文如何压缩
Evidence 如何记录
```


<p>因此 RAG 不只是一个外部服务&#xff0c;往往还是 Runtime Context 管理的一部分。</p>
<hr />
<h2>九、Memory&#xff1a;为什么 Runtime 不能简单把历史消息当记忆</h2>
<p>Conversation History 和 Memory 很容易被混淆。</p>
<p>Conversation History 是&#xff1a;</p>
<blockquote>
<p>当前会话之前发生了什么。</p>
</blockquote>
<p>Memory 更接近&#xff1a;</p>
<blockquote>
<p>从过去的信息中提取出来&#xff0c;并可以跨时间、跨 Run 被重新使用的信息。</p>
</blockquote>
<p>例如&#xff1a;</p>


```text
某个项目的背景
用户曾经作出的选择
之前 Run 的关键结论
长期业务上下文
```


<p>Runtime 一般会在执行前&#xff1a;</p>


```text
当前任务
↓
Memory Search
↓
召回相关 Memory
↓
注入 Context
```


<p>执行结束后&#xff1a;</p>


```text
Run 完成
↓
Memory Extraction
↓
判断哪些信息值得保存
↓
Memory Write
```


<p>因此完整的 Memory 能力通常不仅仅是&#xff1a;</p>


```text
search
```


<p>而是&#xff1a;</p>


```text
search
write
update
delete
```


<p>Runtime 负责决定什么时候读取 Memory&#xff0c;以及 Memory 应该如何影响当前 Run。</p>
<hr />
<h2>十、Skill&#xff1a;为什么 Skill 不是 Tool</h2>
<p>Tool 往往表示一个原子动作&#xff1a;</p>


```text
搜索数据库
下载文件
发送邮件
```


<p>Skill 更像&#xff1a;</p>
<blockquote>
<p>完成一类业务任务所需要的一整套方法。</p>
</blockquote>
<p>例如&#xff1a;</p>


```text
招投标分析 Skill

包含：

System Instructions
任务步骤
风险识别规则
输出结构
可以使用哪些 Tools
领域知识提示
```


<p>于是 Runtime 可能进行&#xff1a;</p>


```text
用户任务
↓
识别 Skill
↓
加载 Skill
↓
把 Skill Instructions 注入 Context
↓
启用对应 Tool Set
↓
执行 Agent
```


<p>所以可以简单理解&#xff1a;</p>


```text
Tool
=
Agent 能做什么

Skill
=
Agent 应该如何完成某一类任务
```


<p>Runtime 则负责 Skill 的&#xff1a;</p>


```text
发现
选择
加载
版本固定
使用记录
失败处理
```


<hr />
<h2>十一、Plan&#xff1a;复杂 Agent 为什么需要显式计划</h2>
<p>对于简单任务&#xff0c;Agent 可以不断&#xff1a;</p>


```text
LLM → Tool → LLM → Tool
```


<p>直到结束。</p>
<p>但复杂任务如果完全没有 Plan&#xff0c;很容易产生&#xff1a;</p>
<ul><li>重复执行&#xff1b;</li><li>忘记目标&#xff1b;</li><li>顺序混乱&#xff1b;</li><li>用户无法知道进度。</li></ul>
<p>于是 Runtime 会维护一个 Plan&#xff1a;</p>


```text
分析招标文件

✓ 读取文件
✓ 提取资格条件
● 分析评分办法
○ 匹配历史案例
○ 生成投标策略
```


<p>但需要特别注意&#xff1a;</p>
<blockquote>
<p>Plan 不是 Agent Runtime 的执行状态本身。</p>
</blockquote>
<p>Plan 表示&#xff1a;</p>


```text
Agent 认为应该完成哪些步骤
```


<p>Run State 表示&#xff1a;</p>


```text
系统当前实际上运行到了哪里
```


<p>例如 Plan 可能认为&#xff1a;</p>


```text
Step 3 = running
```


<p>但真实 Runtime 正在&#xff1a;</p>


```text
waiting_tool
```


<p>因为某个工具还没有完成。</p>
<p>因此 Runtime 需要同时管理&#xff1a;</p>


```text
Logical Plan
+
Execution State
```


<p>而前端看到的 Plan UI&#xff0c;一般是 Runtime 状态的一种投影。</p>
<hr />
<h2>十二、Subagent&#xff1a;Runtime 如何管理多个智能体</h2>
<p>复杂任务可能进一步拆给多个 Agent&#xff1a;</p>


```text
Main Agent
│
├── Document Agent
├── Research Agent
└── Risk Agent
```


<p>这里不能简单认为&#xff1a;</p>


```text
Subagent = 再调用一次 LLM
```


<p>更准确地说&#xff0c;Subagent 通常拥有自己的&#xff1a;</p>


```text
Context
Run
Tools
Model
状态
事件
结果
```


<p>所以 Runtime 中可能形成&#xff1a;</p>


```text
Parent Run
│
├── Child Run A
│      └── completed
│
├── Child Run B
│      └── running
│
└── Child Run C
└── failed
```


<p>主 Runtime 需要处理&#xff1a;</p>


```text
如何创建 Subagent
给它什么输入
允许使用什么工具
最大并发多少
什么时候取消
如何收集结果
子任务失败是否影响父任务
```


<p>因此 Subagent Runtime 本质上是一套&#xff1a;</p>
<blockquote>
<p>嵌套 Run 管理机制。</p>
</blockquote>
<hr />
<h2>十三、Human-in-the-loop&#xff1a;Agent Runtime 必须知道什么时候停</h2>
<p>生产中的 Agent 不应该所有事情都自动执行。</p>
<p>例如&#xff1a;</p>


```text
发送邮件
删除数据
提交审批
修改合同
发布内容
付款
```


<p>都属于可能产生真实业务影响的操作。</p>
<p>当 Runtime 识别到高风险 Tool 时&#xff0c;可以&#xff1a;</p>


```text
running
↓
approval.required
↓
waiting_user
```


<p>此时 Run 不应该结束。</p>
<p>它只是&#xff1a;</p>


```text
暂停
```


<p>Runtime 需要把当前状态完整保存下来。</p>
<p>十分钟甚至十小时后&#xff0c;用户点击&#xff1a;</p>


```text
批准
```


<p>Runtime 收到&#xff1a;</p>


```text
approval.approved
```


<p>然后&#xff1a;</p>


```text
恢复 Context
↓
继续 Tool
↓
继续 Execution Loop
```


<p>所以 Human-in-the-loop 的真正难点不在前端的&#xff1a;</p>


```text
[批准] [拒绝]
```


<p>而在 Runtime 能不能&#xff1a;</p>
<blockquote>
<p><strong>暂停、持久化&#xff0c;然后从原来的位置恢复。</strong></p>
</blockquote>
<p>这也是 Runtime 是否真正生产可用的重要判断标准。</p>
<hr />
<h2>十四、Event System&#xff1a;Runtime 如何把执行过程暴露出来</h2>
<p>一个 Agent Run 内部会不断发生变化&#xff1a;</p>


```text
Run 开始
LLM 开始生成
产生文本增量
Tool 开始
Tool 完成
Plan 更新
Subagent 创建
Artifact 创建
Run 完成
```


<p>这些变化如果只存在 Runtime 内存里&#xff0c;前端和监控系统都无法感知。</p>
<p>因此 Runtime 通常会将变化转换成统一 Event&#xff1a;</p>


```text
run.created
run.started

message.item.created
message.item.delta
message.item.completed

plan.updated

tool.started
tool.completed
tool.failed

subagent.started
subagent.completed

approval.required

artifact.created

run.completed
run.failed
```


<p>于是架构变成&#xff1a;</p>


```text
Agent Runtime
↓
Event Bus
│
├── Event Log
├── SSE Gateway
├── Timeline Projection
├── Trace
├── Metrics
└── Audit Log
```


<p>同一个 Runtime Event 可以同时驱动多个下游能力。</p>
<p>例如&#xff1a;</p>


```text
tool.completed
```


<p>可以同时&#xff1a;</p>
<ul><li>更新数据库中的 Tool 状态&#xff1b;</li><li>写入 <code>agent_stream_event</code>&#xff1b;</li><li>通过 SSE 推给前端&#xff1b;</li><li>出现在 Trace 页面&#xff1b;</li><li>进入运行指标&#xff1b;</li><li>参与最终 Timeline 构建。</li></ul>
<p>这也是为什么成熟 Runtime 往往围绕 Event 架构&#xff0c;而不是到处直接调用前端逻辑。</p>
<hr />
<h2>十五、SSE 和 streamUrl 不属于 Runtime 本身&#xff0c;但它们观察 Runtime</h2>
<p>这里需要明确边界。</p>
<p>Runtime 的职责是&#xff1a;</p>


```text
执行
```


<p>SSE Gateway 的职责是&#xff1a;</p>


```text
把 Runtime 产生的事件推给客户端
```


<p>因此&#xff1a;</p>


```text
Runtime
↓
Event Log / Event Bus
↓
SSE Gateway
↓
streamUrl
↓
Agent UI
```


<p>这意味着&#xff1a;</p>


```text
关闭 SSE
≠
停止 Runtime
```


<p>页面切换或者刷新时&#xff0c;客户端的 streamUrl 连接可能消失&#xff0c;但 Agent Run 仍然继续。</p>
<p>这就是&#xff1a;</p>


```text
Run 生命周期
和
网络连接生命周期
```


<p>解耦。</p>
<p>对于 Agent 系统&#xff0c;这是非常重要的设计。</p>
<hr />
<h2>十六、为什么 Runtime 必须有 Snapshot 和 Event Log</h2>
<p>如果 Runtime 的状态只存在内存&#xff1a;</p>


```text
process memory
```


<p>服务一重启&#xff1a;</p>


```text
Agent 全丢
```


<p>显然无法生产使用。</p>
<p>因此 Runtime 必须进行持久化。</p>
<p>通常至少有两种数据&#xff1a;</p>


```text
Snapshot
+
Event Log
```


<p>Snapshot 表示&#xff1a;</p>
<blockquote>
<p>当前 Run 现在是什么状态。</p>
</blockquote>
<p>例如&#xff1a;</p>


```text
run.status = waiting_user

message.content = "经过分析……"

tool_call.status = completed

plan.step_3 = running
```


<p>Event Log 表示&#xff1a;</p>
<blockquote>
<p>Run 是如何一步步变成现在这个状态的。</p>
</blockquote>
<p>例如&#xff1a;</p>


```text
1001 run.started
1002 message.item.created
1003 message.item.delta
1004 tool.started
1005 tool.completed
1006 approval.required
```


<p>两者结合&#xff0c;可以同时解决&#xff1a;</p>


```text
快速恢复
+
事件追踪
+
断线补发
```


<p>例如用户刷新页面&#xff1a;</p>


```text
读取 Snapshot
↓
立即恢复完整 UI
↓
拿到 lastEventId
↓
从 Event Log 补发遗漏事件
↓
重新进入 Live Stream
```


<p>这也是&#xff1a;</p>


```text
Snapshot + Event Log + Live Stream
```


<p>模型的来源。</p>
<hr />
<h2>十七、Artifact&#xff1a;Agent 的结果不应该全部塞进 Message</h2>
<p>Agent 最终产生的结果不一定是文本。</p>
<p>它可能创建&#xff1a;</p>


```text
Word
PDF
Excel
图表
HTML
代码
数据文件
```


<p>这些更适合作为 Artifact。</p>
<p>Runtime 可能执行&#xff1a;</p>


```text
Agent 决定创建报告
↓
Artifact 创建
↓
写入内容
↓
保存 MinIO / S3
↓
更新 Artifact Status
↓
artifact.created
↓
前端展示
```


<p>Artifact 可以拥有&#xff1a;</p>


```text
artifact_id
run_id
message_id
type
title
version
status
object_key
metadata
```


<p>于是 Conversation 中可以只展示&#xff1a;</p>


```text
已经生成《项目风险分析报告》

[打开] [下载]
```


<p>真正的报告则在 Artifact Workspace 中展示。</p>
<p>这说明 Runtime 不只是生成 Message&#xff0c;也负责管理任务产生的业务成果。</p>
<hr />
<h2>十八、Retry、Timeout 与 Cancellation&#xff1a;Runtime 如何处理失败</h2>
<p>Agent 的执行高度依赖外部系统&#xff1a;</p>


```text
LLM API
数据库
搜索系统
MCP
Browser
对象存储
第三方 API
```


<p>失败是必然发生的。</p>
<p>因此 Runtime 必须有明确的 Failure Policy。</p>
<p>例如&#xff1a;</p>


```text
Tool Timeout
↓
Retry 1
↓
Retry 2
↓
Retry 3
↓
仍然失败
```


<p>此时 Runtime 可以&#xff1a;</p>


```text
让 LLM 选择其他 Tool
```


<p>或者&#xff1a;</p>


```text
将当前 Step 标记失败
```


<p>或者&#xff1a;</p>


```text
直接 run.failed
```


<p>不同类型错误不能一律重试。</p>
<p>例如网络超时&#xff1a;</p>


```text
retryable = true
```


<p>但权限不足&#xff1a;</p>


```text
retryable = false
```


<p>因此 Runtime 往往需要&#xff1a;</p>


```text
Retry Policy
Exponential Backoff
Timeout
Circuit Breaker
Error Classification
Fallback
```


<p>取消同样不能只是杀掉 HTTP 请求。</p>
<p>用户点击停止时&#xff1a;</p>


```text
cancel_requested = true
```


<p>Runtime 应在安全位置检查&#xff1a;</p>


```text
Cancellation Point
```


<p>然后依次取消&#xff1a;</p>


```text
当前 Tool
Subagent
后续计划
```


<p>最终产生&#xff1a;</p>


```text
run.cancelled
```


<hr />
<h2>十九、Runtime 还需要做资源和成本治理</h2>
<p>Agent 与普通 Chat 最大的成本差异之一&#xff0c;就是&#xff1a;</p>
<blockquote>
<p>一个用户请求可能产生很多次 LLM 调用。</p>
</blockquote>
<p>例如一个 Run&#xff1a;</p>


```text
第一次：任务理解
第二次：生成 Plan
第三次：决定 Tool
第四次：处理 Tool Result
第五次：调用 Subagent
第六次：总结结果
```


<p>最终可能调用模型十几次。</p>
<p>因此 Runtime 通常需要记录&#xff1a;</p>


```text
provider
model
prompt_tokens
completion_tokens
cached_tokens
reasoning_tokens

latency
TTFT

tool_cost
model_cost
```


<p>最后汇总成&#xff1a;</p>


```text
Run Usage
Run Cost
```


<p>这些数据可以用于&#xff1a;</p>
<ul><li>租户计费&#xff1b;</li><li>用户限额&#xff1b;</li><li>成本分析&#xff1b;</li><li>模型路由&#xff1b;</li><li>性能优化&#xff1b;</li><li>运营监控。</li></ul>
<p>例如&#xff1a;</p>


```text
简单请求
→ flash 模型

复杂任务
→ plus 模型

超复杂规划
→ reasoning 模型
```


<p>这种 Model Routing 通常也是 Runtime 或其上层 Policy Engine 的能力。</p>
<hr />
<h2>二十、Trace 和 Observability 为什么是 Runtime 的必要组成部分</h2>
<p>Agent 一旦出现问题&#xff0c;仅仅看到&#xff1a;</p>


```text
run.failed
```


<p>几乎没有任何排查价值。</p>
<p>开发人员真正需要知道&#xff1a;</p>


```text
用户输入是什么
加载了哪个 Skill
检索了什么 Memory
RAG 召回了什么
调用了哪个模型
LLM 返回了什么 Tool Call
Tool 参数是什么
Tool 执行多久
Subagent 做了什么
在哪里失败
```


<p>所以生产 Runtime 通常会生成完整 Trace&#xff1a;</p>


```text
Run
├── LLM Call 1
├── RAG Retrieval
├── Tool Call 1
├── LLM Call 2
├── Subagent Run
│      ├── LLM
│      └── Tool
└── LLM Call 3
```


<p>可以建立&#xff1a;</p>


```text
run_id
trace_id
span_id
parent_span_id
```


<p>这样 Agent 执行就从一个“黑盒”变成可以观察和分析的执行图。</p>
<hr />
<h2>二十一、一个完整的 Agent Runtime 架构</h2>
<p>把前面的能力放到一起&#xff0c;一个比较完整的 Runtime 可以抽象为&#xff1a;</p>


```text
                       Agent API
│
▼
┌────────────────┐
│  Run Manager   │
└────────┬───────┘
│
▼
┌────────────────────────┐
│     Agent Runtime      │
│                        │
│   Execution Loop       │
│   State Machine        │
│   Scheduler            │
│   Policy Engine        │
└───────────┬────────────┘
│
┌───────────────┼────────────────┐
│               │                │
▼               ▼                ▼
Context Builder    LLM Adapter      Tool Runtime
│               │                │
┌─────┼──────┐        │         ┌──────┼────────┐
▼     ▼      ▼        ▼         ▼      ▼        ▼
RAG  Memory  Skill   Model      MCP   Browser   API
Providers

│
┌───────────────┼─────────────────┐
▼               ▼                 ▼
Subagent       Approval Runtime    Artifact
Runtime

│
▼
Event Bus
│
┌───────────────┼────────────────┐
▼               ▼                ▼
Event Log        Snapshot       Trace / Metrics
│
▼
SSE Gateway
│
▼
Agent UI
```


<p>这张图中最值得关注的不是模块数量&#xff0c;而是数据流。</p>
<p>Agent API 创建 Run。</p>
<p>Run Manager 把 Run 交给 Runtime。</p>
<p>Runtime 通过 Context Builder 组织信息&#xff0c;再调用 LLM。</p>
<p>LLM 可能返回 Tool Call&#xff0c;Tool Runtime 执行后把结果重新交给 Runtime。</p>
<p>复杂任务可以继续创建 Subagent&#xff0c;危险动作可以进入 Approval。</p>
<p>整个执行过程中不断产生 Event。</p>
<p>Event 一方面被持久化&#xff0c;另一方面通过 SSE 推送给 UI。</p>
<p>无论前端是否在线&#xff0c;Run 都可以继续存在。</p>
<p>这就是 Runtime 真正的运行模型。</p>
<hr />
<h2>二十二、Agent Runtime 和 Agent Framework 有什么区别</h2>
<p>经常有人会把&#xff1a;</p>


```text
LangGraph
Agent SDK
CrewAI
AutoGen
```


<p>直接称为 Agent Runtime。</p>
<p>这种说法在某些语境下没有问题&#xff0c;但严格来说二者还是有区别。</p>
<p>Agent Framework 更多提供&#xff1a;</p>


```text
Agent 定义
Graph
Tool
Workflow
Model 调用
```


<p>而完整生产 Runtime 还需要&#xff1a;</p>


```text
Run 生命周期
事件系统
持久化
恢复
权限
租户
并发控制
取消
审批
成本
Trace
SSE
版本管理
```


<p>所以企业实际建设时&#xff0c;经常会出现&#xff1a;</p>


```text
LangGraph / Agent SDK
↓
作为 Agent Execution Engine

自研 Agent Runtime
↓
负责生产运行能力
```


<p>也就是说&#xff0c;没有必要把整个 Runtime 都自己重新造一遍。</p>
<p>可以使用成熟 Agent Framework 解决“怎么执行 Agent”&#xff0c;再在外层建立自己的&#xff1a;</p>


```text
Run
Event
State
Tool Policy
Persistence
Streaming
Observability
```


<p>这些平台能力。</p>
<hr />
<h2>二十三、怎样判断一个 Agent Runtime 是否真正成熟</h2>
<p>一个 Agent Demo 可能只需要做到&#xff1a;</p>


```text
LLM 能调用 Tool
```


<p>但生产 Runtime 更应该回答下面这些问题&#xff1a;</p>
<p>当用户刷新页面以后&#xff0c;Run 会不会消失&#xff1f;</p>
<p>服务重启以后&#xff0c;正在运行的任务能不能恢复&#xff1f;</p>
<p>同一个 Tool 重复执行时会不会产生重复副作用&#xff1f;</p>
<p>高风险操作能不能暂停等待用户审批&#xff1f;</p>
<p>Subagent 能不能独立取消和重试&#xff1f;</p>
<p>模型 API 超时以后 Runtime 怎么处理&#xff1f;</p>
<p>Run 产生十万条流式事件以后怎么恢复&#xff1f;</p>
<p>一个用户最多允许启动多少个并发 Run&#xff1f;</p>
<p>不同租户能不能使用不同 Tool 和 Skill&#xff1f;</p>
<p>一次 Run 到底调用了多少 Token&#xff0c;花了多少钱&#xff1f;</p>
<p>出现问题以后能不能通过 Trace 定位&#xff1f;</p>
<p>如果这些问题都没有答案&#xff0c;那么它更多还是一个&#xff1a;</p>


```text
Agent Execution Demo
```


<p>而不是完整的&#xff1a;</p>


```text
Agent Runtime
```


<hr />
<h2>二十四、从 0 到 1 建设 Agent Runtime 的推荐路径</h2>
<p>如果自己建设 Agent Runtime&#xff0c;不建议一开始就同时实现 Multi-Agent、Memory、Skill 和复杂 Workflow。</p>
<p>更合理的路线是先把最核心的一条链路做稳定。</p>
<p>第一阶段先做到&#xff1a;</p>


```text
Run
+
LLM
+
Tool
+
Event
+
SSE
```


<p>确保&#xff1a;</p>


```text
用户发送消息
↓
创建 Run
↓
LLM
↓
Tool
↓
LLM
↓
完成
```


<p>整个生命周期能够稳定运行。</p>
<p>第二阶段解决生产可靠性&#xff1a;</p>


```text
Snapshot
Event Log
Retry
Timeout
Cancellation
SSE Recovery
Trace
```


<p>确保&#xff1a;</p>


```text
页面能刷新
网络能断
Worker 能重启
任务能失败
Run 还能正确恢复
```


<p>第三阶段再扩展智能能力&#xff1a;</p>


```text
RAG
Memory
Skill
Plan
Artifact
```


<p>第四阶段才适合增加&#xff1a;</p>


```text
Subagent
Human-in-the-loop
Generative UI
复杂 Workflow
```


<p>这样的建设顺序比一开始追求“Agent 能力很多”更容易真正落地。</p>
<hr />
<h2>结语</h2>
<p>Agent Runtime 最容易被误解成&#xff1a;</p>


```text
一个 while 循环
+
LLM
+
Tool
```


<p>但生产级 Agent Runtime 实际解决的是&#xff1a;</p>
<blockquote>
<p><strong>如何让一个智能任务稳定、持续、可控、可恢复地运行。</strong></p>
</blockquote>
<p>从用户的一条请求开始&#xff1a;</p>


```text
用户任务
↓
创建 Run
↓
Runtime 构建 Context
↓
LLM 做出决策
↓
调用 Tool / RAG / Memory / Skill
↓
可能创建 Subagent
↓
可能暂停等待用户审批
↓
不断产生 Event
↓
保存 Snapshot 和 Event Log
↓
通过 SSE 推送运行过程
↓
处理异常、重试和取消
↓
生成 Message / Artifact
↓
Run 完成
```


<p>整个过程真正的核心并不是某一次 LLM 调用&#xff0c;而是 Runtime 对整个执行生命周期的管理。</p>
<p>因此&#xff0c;可以用一句话总结 Agent Runtime&#xff1a;</p>
<blockquote>
<p><strong>LLM 负责决定下一步“想做什么”&#xff0c;Tool 负责提供“能做什么”&#xff0c;而 Agent Runtime 负责决定这些事情“什么时候做、怎么做、做到哪里了、失败怎么办、状态怎么保存、如何恢复&#xff0c;以及怎样把整个过程可靠地暴露给用户和系统”。</strong></p>
</blockquote>
<p>当 Agent 从简单问答逐渐走向长任务、工具调用、企业业务操作、Multi-Agent 和 Human-in-the-loop 时&#xff0c;Agent Runtime 就会从一个隐藏的技术模块&#xff0c;逐渐成为整个 Agent 平台最核心的基础设施。</p>
