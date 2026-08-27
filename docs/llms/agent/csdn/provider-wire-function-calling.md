---
title: "OpenAI-wire 与 Anthropic-wire 的差异：从 Function Calling 协议看 Agent 框架的 Provider 适配设计"
description: "CSDN 原文全文镜像：摘要 本文探讨了AI Agent框架中不同大模型厂商工具调用格式的差异问题。OpenAI和Anthropic采用完全不同的消息格式（分别称为OpenAI-wire和Anthropic-wire），主要差异体现在工具调用的放置位置、参数格……"
pageType: article
module: agent
updated: '2026-05-19'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "大模型"
  - "人工智能"
  - "软件工程"
level: advanced
prerequisites:
  - "/llms/agent/"
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-05-19，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-05-19。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/161193006](https://blog.csdn.net/m0_63309778/article/details/161193006)
- 站内分区：Agent / Provider 工具调用适配
:::

<p><img src="https://i-blog.csdnimg.cn/direct/516d8fa17d1f412c9ab06df8cdaad0aa.png" alt="在这里插入图片描述" /></p>
<h3>摘要</h3>
<p>在构建 AI Agent 框架时&#xff0c;我们经常会遇到一个工程问题&#xff1a;不同大模型厂商虽然都支持 Function Calling / Tool Use&#xff0c;但它们的消息格式并不一致。</p>
<p>例如 OpenAI、OpenRouter、vLLM 等生态通常采用一种接近 OpenAI Chat Completions 的工具调用格式&#xff0c;我们可以称之为 <strong>OpenAI-wire</strong>&#xff1b;而 Claude / Anthropic 则采用另一套基于 <code>content blocks</code> 的工具调用格式&#xff0c;我们可以称之为 <strong>Anthropic-wire</strong>。</p>
<p>这两种格式的核心差异包括&#xff1a;</p>
<ul><li>工具调用放置位置不同&#xff1b;</li><li>工具参数格式不同&#xff1b;</li><li>工具执行结果的消息 role 不同&#xff1b;</li><li>工具调用与工具结果的关联字段不同&#xff1b;</li><li>多工具并行调用的表达方式不同&#xff1b;</li><li>消息 content 的组织方式不同。</li></ul>
<p>对于 Agent 框架来说&#xff0c;一个非常重要的设计原则是&#xff1a;<strong>不要让核心 Agent 循环直接依赖某一个模型厂商的协议格式</strong>。</p>
<p>以 Hermes 这类 Agent 框架为例&#xff0c;它通常会在内部统一使用一种格式&#xff0c;比如 OpenAI-wire&#xff0c;然后在调用 Anthropic API 时&#xff0c;在适配层将 OpenAI-wire 转换成 Anthropic-wire。这样做可以让 Agent 主循环只维护一套逻辑&#xff0c;避免 provider 差异扩散到整个系统。</p>
<p>本文将详细介绍 OpenAI-wire 与 Anthropic-wire 的差异&#xff0c;并结合 Agent 框架的设计&#xff0c;讲清楚为什么需要做统一中间表示&#xff0c;以及如何实现格式转换。</p>
<hr />
<h2>1. 什么是 wire format&#xff1f;</h2>
<p>在讨论 OpenAI-wire 和 Anthropic-wire 之前&#xff0c;首先要理解一个概念&#xff1a;<strong>wire format</strong>。</p>
<p>所谓 wire format&#xff0c;可以理解为&#xff1a;</p>
<blockquote>
<p>LLM API 在网络传输时使用的 JSON 消息格式。</p>
</blockquote>
<p>也就是说&#xff0c;当我们的程序调用大模型时&#xff0c;最终都会把上下文消息、工具定义、工具调用结果等内容组织成一个 JSON 请求体&#xff0c;然后通过 HTTP API 发送给模型服务。</p>
<p>例如&#xff1a;</p>


```text
Agent 代码
↓
内部统一消息结构
↓
转换成不同模型厂商要求的 JSON
↓
发送给 OpenAI / Anthropic / OpenRouter / vLLM
↓
解析模型返回结果
↓
继续执行工具或返回最终答案
```


<p>不同模型厂商虽然都支持“让模型调用工具”&#xff0c;但它们对消息格式的要求不同。</p>
<p>同样是让模型调用一个 <code>read_file</code> 工具&#xff1a;</p>
<ul><li>OpenAI 风格会把工具调用放到 <code>assistant.tool_calls</code> 字段中&#xff1b;</li><li>Anthropic 风格会把工具调用放到 <code>assistant.content</code> 数组中的 <code>tool_use</code> block 里。</li></ul>
<p>所以&#xff0c;Agent 框架要解决的不是简单的“工具怎么执行”&#xff0c;而是&#xff1a;</p>
<blockquote>
<p>如何让同一个 Agent 工具调用循环&#xff0c;兼容不同 LLM provider 的通信协议。</p>
</blockquote>
<hr />
<h2>2. Function Calling 在 Agent 中的基本流程</h2>
<p>在 Agent 系统中&#xff0c;Function Calling / Tool Use 的基本流程通常是这样的&#xff1a;</p>


```text
用户提出任务
↓
模型判断是否需要调用工具
↓
模型返回工具调用请求
↓
程序解析工具名和参数
↓
程序执行本地工具 / 外部 API
↓
将工具结果返回给模型
↓
模型基于工具结果继续推理
↓
返回最终答案，或者继续调用工具
```


<p>例如用户说&#xff1a;</p>


```text
读取 README.md 文件
```


<p>模型可能不会直接回答&#xff0c;而是输出一个工具调用&#xff1a;</p>


```text
调用 read_file 工具，参数为 {"path": "README.md"}
```


<p>程序收到之后执行&#xff1a;</p>


```python
read_file<span class="token punctuation">(</span>path<span class="token operator">=</span><span class="token string">"README.md"</span><span class="token punctuation">)</span>
```


<p>然后把文件内容再发回给模型&#xff0c;让模型继续总结或回答。</p>
<p>这个过程听起来简单&#xff0c;但一旦接入多个模型厂商&#xff0c;就会遇到消息格式差异。</p>
<hr />
<h2>3. OpenAI-wire 的工具调用格式</h2>
<p>OpenAI-wire 是目前很多 Agent 框架和开源模型服务采用的格式。</p>
<p>OpenAI、OpenRouter&#xff0c;以及很多 OpenAI-compatible API&#xff0c;包括部分 vLLM 服务&#xff0c;都会采用类似结构。</p>
<p>一个典型请求如下&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"model"</span><span class="token operator">:</span> <span class="token string">"gpt-4"</span><span class="token punctuation">,</span>
<span class="token string-property property">"messages"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"读取 README.md"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token keyword">null</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_calls"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string-property property">"function"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token string">"{\"path\": \"README.md\"}"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"tool"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_call_id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"# Hermes Agent..."</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"tools"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string-property property">"function"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"description"</span><span class="token operator">:</span> <span class="token string">"Read a file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"parameters"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"object"</span><span class="token punctuation">,</span>
<span class="token string-property property">"properties"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"string"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<h3>3.1 OpenAI-wire 的核心特点</h3>
<p>OpenAI-wire 有几个非常重要的特点。</p>
<h4>第一&#xff0c;工具调用放在 <code>assistant.tool_calls</code> 中</h4>
<p>模型如果想调用工具&#xff0c;会返回一条 assistant 消息&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token keyword">null</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_calls"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string-property property">"function"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token string">"{\"path\": \"README.md\"}"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>也就是说&#xff0c;工具调用不是放在 <code>content</code> 文本里&#xff0c;而是放在结构化字段 <code>tool_calls</code> 中。</p>
<hr />
<h4>第二&#xff0c;<code>tool_calls</code> 是数组</h4>
<p>因为 <code>tool_calls</code> 是数组&#xff0c;所以模型一次可以返回多个工具调用。</p>
<p>例如&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token keyword">null</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_calls"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string-property property">"function"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token string">"{\"path\": \"README.md\"}"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"call_2"</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string-property property">"function"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token string">"{\"path\": \"package.json\"}"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>这意味着模型可以并行请求多个工具调用。</p>
<p>Agent 框架在处理时&#xff0c;需要遍历 <code>tool_calls</code>&#xff1a;</p>


```python
<span class="token keyword">for</span> tool_call <span class="token keyword">in</span> message<span class="token punctuation">[</span><span class="token string">"tool_calls"</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
name <span class="token operator">=</span> tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"name"</span><span class="token punctuation">]</span>
args <span class="token operator">=</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
result <span class="token operator">=</span> execute_tool<span class="token punctuation">(</span>name<span class="token punctuation">,</span> args<span class="token punctuation">)</span>
```


<hr />
<h4>第三&#xff0c;<code>arguments</code> 是 JSON 字符串</h4>
<p>这是 OpenAI-wire 中一个非常容易踩坑的地方。</p>
<p>工具参数不是对象&#xff0c;而是一个 JSON 字符串&#xff1a;</p>


```json
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token string">"{\"path\": \"README.md\"}"</span>
```


<p>所以程序执行工具前&#xff0c;需要先做反序列化&#xff1a;</p>


```python
<span class="token keyword">import</span> json

arguments_str <span class="token operator">=</span> tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span>
arguments <span class="token operator">=</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>arguments_str<span class="token punctuation">)</span>
```


<p>也就是说&#xff1a;</p>


```text
OpenAI-wire:
arguments 是字符串，需要 json.loads
```


<p>而不是&#xff1a;</p>


```python
arguments <span class="token operator">=</span> tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span>
```


<p>如果直接把字符串当对象使用&#xff0c;就会报错。</p>
<hr />
<h4>第四&#xff0c;工具执行结果使用独立的 <code>tool</code> role</h4>
<p>OpenAI-wire 中&#xff0c;工具执行结果会以一条独立消息追加回上下文&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"tool"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_call_id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"# Hermes Agent..."</span>
<span class="token punctuation">}</span>
```


<p>其中 <code>tool_call_id</code> 非常重要&#xff0c;它用于关联&#xff1a;</p>


```text
哪个工具调用请求
对应
哪个工具执行结果
```


<p>完整链路如下&#xff1a;</p>


```text
user:
读取 README.md

assistant:
tool_calls = [
read_file({"path": "README.md"})
]

程序执行工具:
read_file("README.md")

tool:
tool_call_id = call_1
content = "# Hermes Agent..."

assistant:
根据 README.md 的内容继续回答
```


<hr />
<h2>4. Anthropic-wire 的工具调用格式</h2>
<p>Anthropic-wire 是 Claude 使用的工具调用格式。</p>
<p>它和 OpenAI-wire 的最大不同在于&#xff1a;Claude 的消息内容采用 <code>content blocks</code> 结构。</p>
<p>Claude 不会把工具调用放到 <code>assistant.tool_calls</code> 字段&#xff0c;而是放在 <code>assistant.content</code> 数组中。</p>
<p>一个典型请求如下&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"model"</span><span class="token operator">:</span> <span class="token string">"claude-xxx"</span><span class="token punctuation">,</span>
<span class="token string-property property">"messages"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"读取 README.md"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_use"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"input"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token string">"README.md"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_result"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_use_id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"# Hermes Agent..."</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"tools"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"description"</span><span class="token operator">:</span> <span class="token string">"Read a file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"input_schema"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"object"</span><span class="token punctuation">,</span>
<span class="token string-property property">"properties"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"string"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<hr />
<h3>4.1 Anthropic-wire 的核心特点</h3>
<h4>第一&#xff0c;工具调用放在 <code>assistant.content</code> 数组中</h4>
<p>Claude 的 assistant 消息可能是这样的&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_use"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"input"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token string">"README.md"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>这里的工具调用是一个 <code>tool_use</code> block。</p>
<p>结构如下&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_use"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"input"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token string">"README.md"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<hr />
<h4>第二&#xff0c;<code>input</code> 是对象&#xff0c;不是 JSON 字符串</h4>
<p>这是 Anthropic-wire 和 OpenAI-wire 的关键区别之一。</p>
<p>Claude 工具调用参数是&#xff1a;</p>


```json
<span class="token string-property property">"input"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token string">"README.md"</span>
<span class="token punctuation">}</span>
```


<p>而不是&#xff1a;</p>


```json
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token string">"{\"path\": \"README.md\"}"</span>
```


<p>因此&#xff0c;解析 Claude 的工具参数时不需要 <code>json.loads</code>&#xff1a;</p>


```python
args <span class="token operator">=</span> block<span class="token punctuation">[</span><span class="token string">"input"</span><span class="token punctuation">]</span>
```


<p>对比一下&#xff1a;</p>


```python
<span class="token comment"># OpenAI-wire</span>
args <span class="token operator">=</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span><span class="token punctuation">)</span>

<span class="token comment"># Anthropic-wire</span>
args <span class="token operator">=</span> block<span class="token punctuation">[</span><span class="token string">"input"</span><span class="token punctuation">]</span>
```


<hr />
<h4>第三&#xff0c;工具执行结果放在 <code>user</code> role 中</h4>
<p>这也是一个容易让人疑惑的地方。</p>
<p>在 OpenAI-wire 中&#xff0c;工具结果是&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"tool"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_call_id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"工具执行结果"</span>
<span class="token punctuation">}</span>
```


<p>但是在 Anthropic-wire 中&#xff0c;工具结果是&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_result"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_use_id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"工具执行结果"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>也就是说&#xff1a;</p>


```text
OpenAI-wire: 工具结果使用 role = tool
Anthropic-wire: 工具结果使用 role = user
```


<p>这并不是说工具结果真的是用户输入&#xff0c;而是 Anthropic 的消息协议把“客户端提供给模型的工具结果”归在 user 消息中。</p>
<hr />
<h4>第四&#xff0c;<code>content</code> 可以混合文本和工具调用</h4>
<p>Claude 的 <code>content</code> 是一个数组&#xff0c;里面可以同时包含文本和工具调用。</p>
<p>例如&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"text"</span><span class="token punctuation">,</span>
<span class="token string-property property">"text"</span><span class="token operator">:</span> <span class="token string">"我先读取 README 文件。"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_use"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"input"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token string">"README.md"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>所以解析 Claude 返回结果时&#xff0c;不能简单认为 <code>content[0]</code> 一定是工具调用。</p>
<p>更稳妥的写法是&#xff1a;</p>


```python
<span class="token keyword">for</span> block <span class="token keyword">in</span> response<span class="token punctuation">[</span><span class="token string">"content"</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
<span class="token keyword">if</span> block<span class="token punctuation">[</span><span class="token string">"type"</span><span class="token punctuation">]</span> <span class="token operator">==</span> <span class="token string">"text"</span><span class="token punctuation">:</span>
collect_text<span class="token punctuation">(</span>block<span class="token punctuation">[</span><span class="token string">"text"</span><span class="token punctuation">]</span><span class="token punctuation">)</span>

<span class="token keyword">elif</span> block<span class="token punctuation">[</span><span class="token string">"type"</span><span class="token punctuation">]</span> <span class="token operator">==</span> <span class="token string">"tool_use"</span><span class="token punctuation">:</span>
collect_tool_call<span class="token punctuation">(</span>block<span class="token punctuation">)</span>
```


<hr />
<h2>5. OpenAI-wire 与 Anthropic-wire 对比表</h2>
<p>下面用一张表总结两种格式的差异&#xff1a;</p>

<table><thead><tr><th>对比项</th><th>OpenAI-wire</th><th>Anthropic-wire</th></tr></thead><tbody><tr><td>工具定义字段</td><td><code>tools[].type &#61; function</code>&#xff0c;工具信息放在 <code>function</code> 内</td><td>直接使用 <code>tools[].name</code>、<code>description</code>、<code>input_schema</code></td></tr><tr><td>工具参数 Schema</td><td><code>function.parameters</code></td><td><code>input_schema</code></td></tr><tr><td>工具调用位置</td><td><code>assistant.tool_calls</code></td><td><code>assistant.content[]</code> 中的 <code>tool_use</code> block</td></tr><tr><td>工具调用参数</td><td><code>function.arguments</code></td><td><code>input</code></td></tr><tr><td>参数格式</td><td>JSON 字符串</td><td>JSON 对象</td></tr><tr><td>是否需要 parse</td><td>需要 <code>json.loads(arguments)</code></td><td>不需要</td></tr><tr><td>工具结果 role</td><td><code>role: &#34;tool&#34;</code></td><td><code>role: &#34;user&#34;</code></td></tr><tr><td>工具结果关联字段</td><td><code>tool_call_id</code></td><td><code>tool_use_id</code></td></tr><tr><td>多工具调用</td><td><code>tool_calls</code> 数组</td><td>多个 <code>tool_use</code> block</td></tr><tr><td>文本和工具调用关系</td><td><code>content</code> 和 <code>tool_calls</code> 分开</td><td><code>content</code> 数组可混合 <code>text</code> 与 <code>tool_use</code></td></tr><tr><td>解析重点</td><td>解析 <code>tool_calls</code></td><td>遍历 <code>content blocks</code></td></tr></tbody></table><hr />
<h2>6. 为什么 Agent 框架需要统一内部格式&#xff1f;</h2>
<p>如果我们只接入一个模型厂商&#xff0c;其实可以直接按照该厂商的协议写代码。</p>
<p>但是 Agent 框架一般不会只支持一个模型。</p>
<p>常见的接入对象可能包括&#xff1a;</p>


```text
OpenAI
Anthropic Claude
OpenRouter
vLLM
Ollama
Gemini
Qwen
DeepSeek
本地私有化模型服务
```


<p>如果每个 provider 的格式都直接写进 Agent 主循环&#xff0c;代码会变得非常混乱。</p>
<p>例如&#xff1a;</p>


```python
<span class="token keyword">if</span> provider <span class="token operator">==</span> <span class="token string">"openai"</span><span class="token punctuation">:</span>
<span class="token comment"># 解析 assistant.tool_calls</span>
<span class="token comment"># 工具结果用 role=tool</span>
<span class="token keyword">elif</span> provider <span class="token operator">==</span> <span class="token string">"anthropic"</span><span class="token punctuation">:</span>
<span class="token comment"># 遍历 assistant.content</span>
<span class="token comment"># 工具结果用 role=user + tool_result</span>
<span class="token keyword">elif</span> provider <span class="token operator">==</span> <span class="token string">"gemini"</span><span class="token punctuation">:</span>
<span class="token comment"># 又是另一套格式</span>
<span class="token keyword">elif</span> provider <span class="token operator">==</span> <span class="token string">"qwen"</span><span class="token punctuation">:</span>
<span class="token comment"># 可能还有特殊格式</span>
```


<p>这样的问题是&#xff1a;</p>
<ol><li>Agent 主循环越来越复杂&#xff1b;</li><li>每新增一个模型 provider&#xff0c;都要修改核心逻辑&#xff1b;</li><li>工具执行逻辑和 provider 协议耦合&#xff1b;</li><li>多工具调用、错误处理、消息历史管理变得难维护&#xff1b;</li><li>后期调试困难。</li></ol>
<p>所以更好的设计是&#xff1a;</p>
<blockquote>
<p>Agent Core 内部统一使用一种标准格式&#xff0c;Provider Adapter 负责做格式转换。</p>
</blockquote>
<hr />
<h2>7. Hermes 的适配思路</h2>
<p>以 Hermes 这类 Agent 框架为例&#xff0c;它的设计思路可以理解为&#xff1a;</p>


```text
内部统一使用 OpenAI-wire
调用 OpenAI / OpenRouter / vLLM 时：基本原样发送
调用 Anthropic / Claude 时：转换为 Anthropic-wire
拿到 Anthropic 返回后：再转换回 OpenAI-wire
```


<p>也就是说&#xff1a;</p>


```text
Agent Core 只认识 OpenAI-wire
Provider Adapter 负责翻译不同模型厂商的格式
```


<p>架构可以抽象成&#xff1a;</p>


```text
┌──────────────────────────────┐
│ Agent Core                    │
│                              │
│ - 维护 messages               │
│ - 判断 tool_calls             │
│ - 执行工具                    │
│ - 回填工具结果                │
│                              │
│ 内部统一使用 OpenAI-wire       │
└───────────────┬──────────────┘
│
▼
┌──────────────────────────────┐
│ Provider Adapter              │
│                              │
│ - OpenAI: 原样发送             │
│ - Anthropic: 转换成 Claude 格式│
│ - Gemini: 转换成 Gemini 格式   │
│ - 其他模型: 转换成对应格式     │
└───────────────┬──────────────┘
│
▼
┌──────────────────────────────┐
│ LLM Provider                  │
│                              │
│ OpenAI / Claude / OpenRouter  │
│ vLLM / Ollama / Gemini 等      │
└──────────────────────────────┘
```


<p>这本质上是一个典型的 <strong>Adapter Pattern&#xff0c;适配器模式</strong>。</p>
<hr />
<h2>8. Agent Core 的核心循环</h2>
<p>Agent 的主循环大致可以写成这样&#xff1a;</p>


```python
<span class="token keyword">def</span> <span class="token function">run_agent</span><span class="token punctuation">(</span>user_input<span class="token punctuation">,</span> tools<span class="token punctuation">)</span><span class="token punctuation">:</span>
messages <span class="token operator">=</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> user_input
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>

<span class="token keyword">while</span> <span class="token boolean">True</span><span class="token punctuation">:</span>
response <span class="token operator">=</span> call_model<span class="token punctuation">(</span>messages<span class="token punctuation">,</span> tools<span class="token punctuation">)</span>

<span class="token comment"># 如果模型没有请求工具调用，说明可以直接返回最终答案</span>
<span class="token keyword">if</span> <span class="token keyword">not</span> has_tool_calls<span class="token punctuation">(</span>response<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">return</span> response<span class="token punctuation">[</span><span class="token string">"content"</span><span class="token punctuation">]</span>

<span class="token comment"># 如果模型请求了工具调用，则执行工具</span>
<span class="token keyword">for</span> tool_call <span class="token keyword">in</span> response<span class="token punctuation">[</span><span class="token string">"tool_calls"</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
tool_call_id <span class="token operator">=</span> tool_call<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span>
tool_name <span class="token operator">=</span> tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"name"</span><span class="token punctuation">]</span>
tool_args <span class="token operator">=</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span><span class="token punctuation">)</span>

result <span class="token operator">=</span> execute_tool<span class="token punctuation">(</span>tool_name<span class="token punctuation">,</span> tool_args<span class="token punctuation">)</span>

messages<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> <span class="token string">"tool"</span><span class="token punctuation">,</span>
<span class="token string">"tool_call_id"</span><span class="token punctuation">:</span> tool_call_id<span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> result
<span class="token punctuation">}</span><span class="token punctuation">)</span>
```


<p>注意&#xff0c;这段主循环假设所有模型返回的都是 OpenAI-wire&#xff1a;</p>


```python
response<span class="token punctuation">[</span><span class="token string">"tool_calls"</span><span class="token punctuation">]</span>
tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"name"</span><span class="token punctuation">]</span>
tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span>
role <span class="token operator">=</span> <span class="token string">"tool"</span>
tool_call_id <span class="token operator">=</span> xxx
```


<p>如果底层是 OpenAI-compatible provider&#xff0c;那么可以直接处理。</p>
<p>如果底层是 Anthropic&#xff0c;则需要在 <code>_call_anthropic()</code> 里做格式转换&#xff0c;让 Agent Core 仍然看到 OpenAI-wire。</p>
<hr />
<h2>9. OpenAI-wire 转 Anthropic-wire</h2>
<p>下面看一下核心转换逻辑。</p>
<h3>9.1 转换 assistant tool_calls</h3>
<p>Hermes 内部的 OpenAI-wire 消息可能是&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token keyword">null</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_calls"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string-property property">"function"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token string">"{\"path\":\"README.md\"}"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>发送给 Claude 前&#xff0c;需要转换成&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_use"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"input"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token string">"README.md"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>对应转换代码&#xff1a;</p>


```python
<span class="token keyword">import</span> json

<span class="token keyword">def</span> <span class="token function">openai_assistant_to_anthropic</span><span class="token punctuation">(</span>message<span class="token punctuation">)</span><span class="token punctuation">:</span>
content <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

<span class="token comment"># 如果原 assistant 消息里有普通文本，也需要保留</span>
<span class="token keyword">if</span> message<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"content"</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
content<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"text"</span><span class="token punctuation">,</span>
<span class="token string">"text"</span><span class="token punctuation">:</span> message<span class="token punctuation">[</span><span class="token string">"content"</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">for</span> tool_call <span class="token keyword">in</span> message<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"tool_calls"</span><span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
content<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"tool_use"</span><span class="token punctuation">,</span>
<span class="token string">"id"</span><span class="token punctuation">:</span> tool_call<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"name"</span><span class="token punctuation">:</span> tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"name"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"input"</span><span class="token punctuation">:</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">return</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> content
<span class="token punctuation">}</span>
```


<p>核心映射关系&#xff1a;</p>


```text
OpenAI tool_calls[].id
→ Anthropic content[].id

OpenAI tool_calls[].function.name
→ Anthropic content[].name

OpenAI tool_calls[].function.arguments
→ json.loads(...)
→ Anthropic content[].input
```


<hr />
<h3>9.2 转换 tool result</h3>
<p>Hermes 内部的 OpenAI-wire 工具结果是&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"tool"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_call_id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"# Hermes Agent..."</span>
<span class="token punctuation">}</span>
```


<p>发送给 Claude 前&#xff0c;需要转成&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_result"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_use_id"</span><span class="token operator">:</span> <span class="token string">"call_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"# Hermes Agent..."</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>对应转换代码&#xff1a;</p>


```python
<span class="token keyword">def</span> <span class="token function">openai_tool_result_to_anthropic</span><span class="token punctuation">(</span>message<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">return</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"tool_result"</span><span class="token punctuation">,</span>
<span class="token string">"tool_use_id"</span><span class="token punctuation">:</span> message<span class="token punctuation">[</span><span class="token string">"tool_call_id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> message<span class="token punctuation">[</span><span class="token string">"content"</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>核心映射关系&#xff1a;</p>


```text
OpenAI role = tool
→ Anthropic role = user

OpenAI tool_call_id
→ Anthropic tool_use_id

OpenAI content
→ Anthropic tool_result.content
```


<hr />
<h3>9.3 转换 tools 定义</h3>
<p>OpenAI-wire 的工具定义是&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string-property property">"function"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"description"</span><span class="token operator">:</span> <span class="token string">"Read a file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"parameters"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"object"</span><span class="token punctuation">,</span>
<span class="token string-property property">"properties"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"string"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>Anthropic-wire 的工具定义是&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"description"</span><span class="token operator">:</span> <span class="token string">"Read a file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"input_schema"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"object"</span><span class="token punctuation">,</span>
<span class="token string-property property">"properties"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"string"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>转换代码&#xff1a;</p>


```python
<span class="token keyword">def</span> <span class="token function">openai_tool_to_anthropic_tool</span><span class="token punctuation">(</span>tool<span class="token punctuation">)</span><span class="token punctuation">:</span>
function <span class="token operator">=</span> tool<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span>

<span class="token keyword">return</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"name"</span><span class="token punctuation">:</span> function<span class="token punctuation">[</span><span class="token string">"name"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"description"</span><span class="token punctuation">:</span> function<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"description"</span><span class="token punctuation">,</span> <span class="token string">""</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
<span class="token string">"input_schema"</span><span class="token punctuation">:</span> function<span class="token punctuation">[</span><span class="token string">"parameters"</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>映射关系&#xff1a;</p>


```text
OpenAI function.name
→ Anthropic name

OpenAI function.description
→ Anthropic description

OpenAI function.parameters
→ Anthropic input_schema
```


<hr />
<h2>10. Anthropic-wire 转 OpenAI-wire</h2>
<p>当 Claude 返回工具调用时&#xff0c;Hermes 还需要把 Anthropic-wire 转回内部 OpenAI-wire。</p>
<p>Claude 可能返回&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"text"</span><span class="token punctuation">,</span>
<span class="token string-property property">"text"</span><span class="token operator">:</span> <span class="token string">"我需要先读取 README 文件。"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_use"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"input"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token string">"README.md"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>Hermes 内部希望看到&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"我需要先读取 README 文件。"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_calls"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string-property property">"function"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"arguments"</span><span class="token operator">:</span> <span class="token string">"{\"path\":\"README.md\"}"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>转换代码如下&#xff1a;</p>


```python
<span class="token keyword">import</span> json

<span class="token keyword">def</span> <span class="token function">anthropic_assistant_to_openai</span><span class="token punctuation">(</span>message<span class="token punctuation">)</span><span class="token punctuation">:</span>
texts <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
tool_calls <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

<span class="token keyword">for</span> block <span class="token keyword">in</span> message<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"content"</span><span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">if</span> block<span class="token punctuation">[</span><span class="token string">"type"</span><span class="token punctuation">]</span> <span class="token operator">==</span> <span class="token string">"text"</span><span class="token punctuation">:</span>
texts<span class="token punctuation">.</span>append<span class="token punctuation">(</span>block<span class="token punctuation">[</span><span class="token string">"text"</span><span class="token punctuation">]</span><span class="token punctuation">)</span>

<span class="token keyword">elif</span> block<span class="token punctuation">[</span><span class="token string">"type"</span><span class="token punctuation">]</span> <span class="token operator">==</span> <span class="token string">"tool_use"</span><span class="token punctuation">:</span>
tool_calls<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"id"</span><span class="token punctuation">:</span> block<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string">"function"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"name"</span><span class="token punctuation">:</span> block<span class="token punctuation">[</span><span class="token string">"name"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"arguments"</span><span class="token punctuation">:</span> json<span class="token punctuation">.</span>dumps<span class="token punctuation">(</span>block<span class="token punctuation">[</span><span class="token string">"input"</span><span class="token punctuation">]</span><span class="token punctuation">,</span> ensure_ascii<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

openai_message <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> <span class="token string">"\n"</span><span class="token punctuation">.</span>join<span class="token punctuation">(</span>texts<span class="token punctuation">)</span> <span class="token keyword">if</span> texts <span class="token keyword">else</span> <span class="token boolean">None</span>
<span class="token punctuation">}</span>

<span class="token keyword">if</span> tool_calls<span class="token punctuation">:</span>
openai_message<span class="token punctuation">[</span><span class="token string">"tool_calls"</span><span class="token punctuation">]</span> <span class="token operator">=</span> tool_calls

<span class="token keyword">return</span> openai_message
```


<p>这里的关键点是&#xff1a;</p>


```text
Anthropic input 对象
→ json.dumps(...)
→ OpenAI arguments 字符串
```


<hr />
<h2>11. 一个完整的 Provider Adapter 示例</h2>
<p>下面给出一个简化版的适配器设计。</p>


```python
<span class="token keyword">class</span> <span class="token class-name">ProviderAdapter</span><span class="token punctuation">:</span>
<span class="token keyword">def</span> <span class="token function">call</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> messages<span class="token punctuation">,</span> tools<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">raise</span> NotImplementedError

<span class="token keyword">class</span> <span class="token class-name">OpenAIAdapter</span><span class="token punctuation">(</span>ProviderAdapter<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">def</span> <span class="token function">call</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> messages<span class="token punctuation">,</span> tools<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token comment"># OpenAI-wire 内部格式可以基本原样发送</span>
<span class="token keyword">return</span> openai_client<span class="token punctuation">.</span>chat<span class="token punctuation">.</span>completions<span class="token punctuation">.</span>create<span class="token punctuation">(</span>
model<span class="token operator">=</span><span class="token string">"gpt-4"</span><span class="token punctuation">,</span>
messages<span class="token operator">=</span>messages<span class="token punctuation">,</span>
tools<span class="token operator">=</span>tools
<span class="token punctuation">)</span>

<span class="token keyword">class</span> <span class="token class-name">AnthropicAdapter</span><span class="token punctuation">(</span>ProviderAdapter<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">def</span> <span class="token function">call</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> messages<span class="token punctuation">,</span> tools<span class="token punctuation">)</span><span class="token punctuation">:</span>
anthropic_messages <span class="token operator">=</span> self<span class="token punctuation">.</span>convert_messages_to_anthropic<span class="token punctuation">(</span>messages<span class="token punctuation">)</span>
anthropic_tools <span class="token operator">=</span> self<span class="token punctuation">.</span>convert_tools_to_anthropic<span class="token punctuation">(</span>tools<span class="token punctuation">)</span>

response <span class="token operator">=</span> anthropic_client<span class="token punctuation">.</span>messages<span class="token punctuation">.</span>create<span class="token punctuation">(</span>
model<span class="token operator">=</span><span class="token string">"claude-xxx"</span><span class="token punctuation">,</span>
messages<span class="token operator">=</span>anthropic_messages<span class="token punctuation">,</span>
tools<span class="token operator">=</span>anthropic_tools
<span class="token punctuation">)</span>

<span class="token keyword">return</span> self<span class="token punctuation">.</span>convert_response_to_openai<span class="token punctuation">(</span>response<span class="token punctuation">)</span>

<span class="token keyword">def</span> <span class="token function">convert_messages_to_anthropic</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> messages<span class="token punctuation">)</span><span class="token punctuation">:</span>
result <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

<span class="token keyword">for</span> message <span class="token keyword">in</span> messages<span class="token punctuation">:</span>
role <span class="token operator">=</span> message<span class="token punctuation">[</span><span class="token string">"role"</span><span class="token punctuation">]</span>

<span class="token keyword">if</span> role <span class="token operator">==</span> <span class="token string">"tool"</span><span class="token punctuation">:</span>
result<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"tool_result"</span><span class="token punctuation">,</span>
<span class="token string">"tool_use_id"</span><span class="token punctuation">:</span> message<span class="token punctuation">[</span><span class="token string">"tool_call_id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> message<span class="token punctuation">[</span><span class="token string">"content"</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">elif</span> role <span class="token operator">==</span> <span class="token string">"assistant"</span> <span class="token keyword">and</span> message<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"tool_calls"</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
content <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

<span class="token keyword">if</span> message<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"content"</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
content<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"text"</span><span class="token punctuation">,</span>
<span class="token string">"text"</span><span class="token punctuation">:</span> message<span class="token punctuation">[</span><span class="token string">"content"</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">for</span> tool_call <span class="token keyword">in</span> message<span class="token punctuation">[</span><span class="token string">"tool_calls"</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
content<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"tool_use"</span><span class="token punctuation">,</span>
<span class="token string">"id"</span><span class="token punctuation">:</span> tool_call<span class="token punctuation">[</span><span class="token string">"id"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"name"</span><span class="token punctuation">:</span> tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"name"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"input"</span><span class="token punctuation">:</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

result<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> content
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">else</span><span class="token punctuation">:</span>
result<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> role<span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> message<span class="token punctuation">[</span><span class="token string">"content"</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">return</span> result

<span class="token keyword">def</span> <span class="token function">convert_tools_to_anthropic</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> tools<span class="token punctuation">)</span><span class="token punctuation">:</span>
result <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

<span class="token keyword">for</span> tool <span class="token keyword">in</span> tools<span class="token punctuation">:</span>
function <span class="token operator">=</span> tool<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span>
result<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"name"</span><span class="token punctuation">:</span> function<span class="token punctuation">[</span><span class="token string">"name"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string">"description"</span><span class="token punctuation">:</span> function<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"description"</span><span class="token punctuation">,</span> <span class="token string">""</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
<span class="token string">"input_schema"</span><span class="token punctuation">:</span> function<span class="token punctuation">[</span><span class="token string">"parameters"</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

<span class="token keyword">return</span> result

<span class="token keyword">def</span> <span class="token function">convert_response_to_openai</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> response<span class="token punctuation">)</span><span class="token punctuation">:</span>
texts <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
tool_calls <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

<span class="token keyword">for</span> block <span class="token keyword">in</span> response<span class="token punctuation">.</span>content<span class="token punctuation">:</span>
<span class="token keyword">if</span> block<span class="token punctuation">.</span><span class="token builtin">type</span> <span class="token operator">==</span> <span class="token string">"text"</span><span class="token punctuation">:</span>
texts<span class="token punctuation">.</span>append<span class="token punctuation">(</span>block<span class="token punctuation">.</span>text<span class="token punctuation">)</span>

<span class="token keyword">elif</span> block<span class="token punctuation">.</span><span class="token builtin">type</span> <span class="token operator">==</span> <span class="token string">"tool_use"</span><span class="token punctuation">:</span>
tool_calls<span class="token punctuation">.</span>append<span class="token punctuation">(</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"id"</span><span class="token punctuation">:</span> block<span class="token punctuation">.</span><span class="token builtin">id</span><span class="token punctuation">,</span>
<span class="token string">"type"</span><span class="token punctuation">:</span> <span class="token string">"function"</span><span class="token punctuation">,</span>
<span class="token string">"function"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"name"</span><span class="token punctuation">:</span> block<span class="token punctuation">.</span>name<span class="token punctuation">,</span>
<span class="token string">"arguments"</span><span class="token punctuation">:</span> json<span class="token punctuation">.</span>dumps<span class="token punctuation">(</span>block<span class="token punctuation">.</span><span class="token builtin">input</span><span class="token punctuation">,</span> ensure_ascii<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">)</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span><span class="token punctuation">)</span>

message <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"role"</span><span class="token punctuation">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string">"content"</span><span class="token punctuation">:</span> <span class="token string">"\n"</span><span class="token punctuation">.</span>join<span class="token punctuation">(</span>texts<span class="token punctuation">)</span> <span class="token keyword">if</span> texts <span class="token keyword">else</span> <span class="token boolean">None</span>
<span class="token punctuation">}</span>

<span class="token keyword">if</span> tool_calls<span class="token punctuation">:</span>
message<span class="token punctuation">[</span><span class="token string">"tool_calls"</span><span class="token punctuation">]</span> <span class="token operator">=</span> tool_calls

<span class="token keyword">return</span> message
```


<p>这个例子只是用于说明思路&#xff0c;真实项目中还需要处理&#xff1a;</p>
<ul><li>streaming&#xff1b;</li><li>error handling&#xff1b;</li><li>tool result 顺序&#xff1b;</li><li>多工具并行调用&#xff1b;</li><li>content block 中的图片、文件等多模态内容&#xff1b;</li><li>token usage&#xff1b;</li><li>stop reason&#xff1b;</li><li>provider-specific 参数&#xff1b;</li><li>工具调用失败后的错误消息&#xff1b;</li><li>JSON 参数解析失败的 fallback 策略。</li></ul>
<hr />
<h2>12. 为什么选择 OpenAI-wire 作为内部统一格式&#xff1f;</h2>
<p>Hermes 选择 OpenAI-wire 作为内部统一格式&#xff0c;通常有几个原因。</p>
<h3>12.1 生态兼容性更强</h3>
<p>很多模型服务都提供 OpenAI-compatible API&#xff0c;例如&#xff1a;</p>


```text
OpenRouter
vLLM
Ollama 的部分 OpenAI-compatible 接口
DeepSeek API
Qwen API 的兼容接口
一些企业内部私有化模型网关
```


<p>因此&#xff0c;如果内部使用 OpenAI-wire&#xff0c;接入大量模型服务会更方便。</p>
<hr />
<h3>12.2 Agent 主循环更简单</h3>
<p>Agent Core 只需要处理一种格式&#xff1a;</p>


```python
assistant_message<span class="token punctuation">[</span><span class="token string">"tool_calls"</span><span class="token punctuation">]</span>
tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"name"</span><span class="token punctuation">]</span>
tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span>
role <span class="token operator">=</span> <span class="token string">"tool"</span>
tool_call_id <span class="token operator">=</span> xxx
```


<p>不用在主循环里写大量 provider 分支。</p>
<hr />
<h3>12.3 后续扩展更容易</h3>
<p>如果将来要支持 Gemini&#xff0c;只需要增加&#xff1a;</p>


```text
OpenAI-wire ↔ Gemini-wire
```


<p>如果要支持 Qwen 原生格式&#xff0c;只需要增加&#xff1a;</p>


```text
OpenAI-wire ↔ Qwen-wire
```


<p>如果要支持其他私有模型网关&#xff0c;只需要增加&#xff1a;</p>


```text
OpenAI-wire ↔ PrivateModel-wire
```


<p>Agent Core 不需要改。</p>
<p>这就是统一中间表示的价值。</p>
<hr />
<h2>13. 典型踩坑点</h2>
<h3>13.1 OpenAI 的 arguments 是字符串</h3>
<p>错误写法&#xff1a;</p>


```python
args <span class="token operator">=</span> tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span>
execute_tool<span class="token punctuation">(</span>name<span class="token punctuation">,</span> args<span class="token punctuation">)</span>
```


<p>这样拿到的是字符串&#xff1a;</p>


```python
<span class="token string">"{\"path\": \"README.md\"}"</span>
```


<p>正确写法&#xff1a;</p>


```python
args <span class="token operator">=</span> json<span class="token punctuation">.</span>loads<span class="token punctuation">(</span>tool_call<span class="token punctuation">[</span><span class="token string">"function"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token string">"arguments"</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
execute_tool<span class="token punctuation">(</span>name<span class="token punctuation">,</span> args<span class="token punctuation">)</span>
```


<hr />
<h3>13.2 Anthropic 的工具结果不是 role&#61;tool</h3>
<p>很多人第一次接 Claude tool use 的时候&#xff0c;会下意识这么写&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"tool"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_call_id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"工具结果"</span>
<span class="token punctuation">}</span>
```


<p>这是 OpenAI-wire 的写法&#xff0c;不适用于 Anthropic-wire。</p>
<p>Anthropic-wire 应该写成&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_result"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_use_id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"工具结果"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<hr />
<h3>13.3 Claude 的 content 可能同时包含 text 和 tool_use</h3>
<p>不能只取第一个 content block&#xff1a;</p>


```python
block <span class="token operator">=</span> response<span class="token punctuation">[</span><span class="token string">"content"</span><span class="token punctuation">]</span><span class="token punctuation">[</span><span class="token number">0</span><span class="token punctuation">]</span>
```


<p>因为第一个 block 可能是文本&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"text"</span><span class="token punctuation">,</span>
<span class="token string-property property">"text"</span><span class="token operator">:</span> <span class="token string">"我需要先读取文件。"</span>
<span class="token punctuation">}</span>
```


<p>正确方式是遍历&#xff1a;</p>


```python
<span class="token keyword">for</span> block <span class="token keyword">in</span> response<span class="token punctuation">[</span><span class="token string">"content"</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
<span class="token keyword">if</span> block<span class="token punctuation">[</span><span class="token string">"type"</span><span class="token punctuation">]</span> <span class="token operator">==</span> <span class="token string">"text"</span><span class="token punctuation">:</span>
handle_text<span class="token punctuation">(</span>block<span class="token punctuation">)</span>

<span class="token keyword">elif</span> block<span class="token punctuation">[</span><span class="token string">"type"</span><span class="token punctuation">]</span> <span class="token operator">==</span> <span class="token string">"tool_use"</span><span class="token punctuation">:</span>
handle_tool_use<span class="token punctuation">(</span>block<span class="token punctuation">)</span>
```


<hr />
<h3>13.4 多工具调用必须用 id 关联结果</h3>
<p>不要用工具名关联工具结果。</p>
<p>错误思路&#xff1a;</p>


```text
read_file → 工具结果
```


<p>因为同一个工具可能被调用多次&#xff1a;</p>


```text
read_file("README.md")
read_file("package.json")
read_file("src/main.py")
```


<p>必须使用 id&#xff1a;</p>


```text
OpenAI: tool_call_id
Anthropic: tool_use_id
```


<p>例如&#xff1a;</p>


```text
call_1 → read_file("README.md")
call_2 → read_file("package.json")
call_3 → read_file("src/main.py")
```


<p>工具结果必须分别对应&#xff1a;</p>


```text
tool_call_id = call_1
tool_call_id = call_2
tool_call_id = call_3
```


<hr />
<h3>13.5 Anthropic 的 tool_result 顺序要注意</h3>
<p>Claude 对工具结果的顺序比较敏感。</p>
<p>一般来说&#xff0c;模型返回 <code>tool_use</code> 后&#xff0c;下一条用户消息中应该尽快提供对应的 <code>tool_result</code>。</p>
<p>推荐结构&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"assistant"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_use"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"name"</span><span class="token operator">:</span> <span class="token string">"read_file"</span><span class="token punctuation">,</span>
<span class="token string-property property">"input"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"path"</span><span class="token operator">:</span> <span class="token string">"README.md"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"role"</span><span class="token operator">:</span> <span class="token string">"user"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"tool_result"</span><span class="token punctuation">,</span>
<span class="token string-property property">"tool_use_id"</span><span class="token operator">:</span> <span class="token string">"toolu_1"</span><span class="token punctuation">,</span>
<span class="token string-property property">"content"</span><span class="token operator">:</span> <span class="token string">"文件内容"</span>
<span class="token punctuation">}</span>
<span class="token punctuation">]</span>
<span class="token punctuation">}</span>
```


<p>不要在工具调用和工具结果之间插入无关消息。</p>
<hr />
<h2>14. 对 Agent 框架设计的启发</h2>
<p>OpenAI-wire 和 Anthropic-wire 的差异&#xff0c;本质上反映了一个 Agent 框架设计问题&#xff1a;</p>
<blockquote>
<p>框架内部到底应该直接使用某个 provider 的格式&#xff0c;还是定义自己的统一中间格式&#xff1f;</p>
</blockquote>
<p>对于一个长期维护的 Agent 框架来说&#xff0c;推荐做法是&#xff1a;</p>


```text
内部统一格式
外部适配转换
核心逻辑不依赖 provider
```


<p>也就是&#xff1a;</p>


```text
Agent Core
↓
Canonical Message Format
↓
Provider Adapter
↓
LLM Provider
```


<p>这种设计的好处是&#xff1a;</p>
<ol><li>主循环简单&#xff1b;</li><li>工具执行逻辑统一&#xff1b;</li><li>消息历史管理统一&#xff1b;</li><li>provider 扩展方便&#xff1b;</li><li>测试更容易&#xff1b;</li><li>不同模型之间切换成本低&#xff1b;</li><li>适合构建企业级 Agent 平台。</li></ol>
<hr />
<h2>15. 总结</h2>
<p>OpenAI-wire 和 Anthropic-wire 都是在表达同一件事&#xff1a;</p>
<blockquote>
<p>模型需要调用工具&#xff0c;程序执行工具&#xff0c;然后把工具结果返回给模型。</p>
</blockquote>
<p>但是它们的表达方式不同。</p>
<p>OpenAI-wire 的特点是&#xff1a;</p>


```text
assistant.tool_calls
function.arguments 是 JSON 字符串
工具结果使用 role=tool
通过 tool_call_id 关联工具结果
```


<p>Anthropic-wire 的特点是&#xff1a;</p>


```text
assistant.content 中包含 tool_use block
input 是 JSON 对象
工具结果使用 role=user
通过 tool_use_id 关联工具结果
content 可以混合 text 和 tool_use
```


<p>对于 Hermes 这类 Agent 框架来说&#xff0c;最合理的设计是&#xff1a;</p>


```text
内部统一使用 OpenAI-wire
调用 Anthropic 时转换成 Anthropic-wire
拿到 Anthropic 返回后再转换回 OpenAI-wire
```


<p>这样 Agent Core 只需要维护一套逻辑。</p>
<p>最终可以把整个设计总结成一句话&#xff1a;</p>
<blockquote>
<p><strong>OpenAI-wire 是内部统一语言&#xff0c;Anthropic-wire 是 Claude 的外部方言&#xff0c;Provider Adapter 是两者之间的翻译器。</strong></p>
</blockquote>
<p>这个设计看起来只是消息格式转换&#xff0c;但对 Agent 框架的可维护性、可扩展性和多模型兼容能力非常关键。对于需要支持多模型、多工具、多轮工具调用的企业级 Agent 系统来说&#xff0c;统一 wire format 几乎是必须要考虑的基础架构设计。</p>
