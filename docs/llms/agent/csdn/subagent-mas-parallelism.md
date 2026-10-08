---
title: "从 Subagent 到 MAS：深入理解多 Agent 协作、任务分工与并行执行"
description: "CSDN 原文全文镜像：本文讨论了Agent系统中的三个关键概念：Subagent（子智能体）、MAS（多智能体系统）和Multi-Agent Parallelism（多智能体并行）。Subagent指由主Agent委派执行边界清晰的子任务，并返回结果的智能体……"
pageType: article
module: agent
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "人工智能"
  - "大模型"
  - "软件工程"
  - "subagent"
  - "mas"
level: advanced
prerequisites:
  - "/llms/agent/"
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-12，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-12。本站补充导读与相关主线链接，并修复代码展示；原文观点、来源与发布时间保留。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163701731](https://blog.csdn.net/m0_63309778/article/details/163701731)
- 站内分区：Agent / Subagent 与 MAS
:::

::: tip 站内导读与实践边界
本文有助于区分组织关系与执行并发：多个 Agent 可以串行，一个 Agent 也可以并行调用工具。Subagent 与 MAS 的划分是分析视角，并非互斥标准。引入团队前，先明确写入范围、交付证据和唯一整合者，并用同预算的单 Agent 方案比较收益。

继续阅读：[多智能体协作](/llms/agent/multi-agent)、[并行化](/llms/agent/parallelization)。
:::

<p><img src="https://i-blog.csdnimg.cn/direct/3330ce68dba1442fa57e003dc70b1767.png" alt="在这里插入图片描述" /></p>
<h3>前言</h3>
<p>随着 Agent 系统能力越来越复杂&#xff0c;一个很自然的问题会出现&#xff1a;</p>
<blockquote>
<p>一个 Agent 是否应该负责所有事情&#xff1f;</p>
</blockquote>
<p>假设我们正在构建一个 Research Agent&#xff0c;用户要求&#xff1a;</p>
<blockquote>
<p>调研全球主要 AI Agent 平台&#xff0c;比较它们的技术路线、产品能力、商业模式和未来趋势&#xff0c;并最终形成一份研究报告。</p>
</blockquote>
<p>如果只使用一个 Agent&#xff0c;它可能按照这样的方式顺序工作&#xff1a;</p>


```text
理解研究目标
↓
搜索 OpenAI
↓
阅读和整理资料
↓
搜索 Anthropic
↓
阅读和整理资料
↓
搜索 Google
↓
阅读和整理资料
↓
搜索 Microsoft
↓
综合所有结果
↓
生成报告
```


<p>这种模式当然可以工作&#xff0c;但问题也很明显。</p>
<p>大量搜索结果、网页内容、Tool Result 和中间分析会不断进入同一个 Context Window&#xff1b;与此同时&#xff0c;很多研究方向实际上彼此独立&#xff0c;例如研究 OpenAI 和研究 Google 并不需要严格按照先后顺序执行。</p>
<p>于是很自然会演化成&#xff1a;</p>


```text
                         Lead Agent
│
┌───────────────┼───────────────┐
↓               ↓               ↓
OpenAI Research   Google Research   Anthropic Research
Agent             Agent             Agent
│               │               │
└───────────────┼───────────────┘
↓
综合结果
```


<p>到这里&#xff0c;我们就会接触到三个经常被混在一起的概念&#xff1a;</p>
<p><strong>Subagent、MAS&#xff08;Multi-Agent System&#xff09;和 Multi-Agent Parallelism。</strong></p>
<p>它们看起来都意味着“有多个 Agent”&#xff0c;但实际上描述的是三个不同的问题。</p>
<p>最简单的理解是&#xff1a;</p>
<blockquote>
<p><strong>Subagent 讨论的是任务委派关系&#xff0c;MAS 讨论的是整个多 Agent 系统如何组织&#xff0c;而多 Agent 并行讨论的是这些 Agent 在时间上如何执行。</strong></p>
</blockquote>
<p>理解了这层关系&#xff0c;很多多 Agent 架构问题就会清楚很多。</p>
<hr />
<h2>一、Subagent&#xff1a;把一个明确的子任务交给另一个 Agent</h2>
<p>Subagent&#xff0c;中文通常可以理解为“子智能体”。</p>
<p>它最典型的特征并不是模型更小&#xff0c;也不是能力更弱&#xff0c;而是&#xff1a;</p>
<blockquote>
<p><strong>它接受 Parent Agent 或 Lead Agent 的委派&#xff0c;完成一个边界相对清晰的子任务&#xff0c;然后把结果返回给上层 Agent。</strong></p>
</blockquote>
<p>例如一个 Coding Agent 正在进行大型项目分析。</p>
<p>主 Agent 当前已经需要保存&#xff1a;</p>


```text
用户需求
项目架构
当前计划
已经修改的文件
工具调用结果
历史消息
```


<p>这时候主 Agent 又需要运行完整测试套件。</p>
<p>测试可能产生几万行日志。</p>
<p>如果全部进入主 Agent Context&#xff0c;不仅消耗大量 Token&#xff0c;还可能让真正重要的信息被淹没。</p>
<p>因此可以把测试任务委派出去&#xff1a;</p>


```text
Main Coding Agent
│
│
│  “运行所有测试，
│   分析失败原因，
│   只返回重要结论。”
↓
Test Subagent
│
├── 执行测试
├── 阅读大量日志
├── 分析失败用例
└── 压缩结果
│
↓
返回 Main Agent：

“共发现 6 个失败测试，
其中 4 个与 Refresh Token 有关，
核心问题位于 AuthService。”
```


<p>这里 Test Subagent 完成自己的工作以后&#xff0c;责任基本就结束了。</p>
<p>它不需要决定&#xff1a;</p>
<ul><li>整个 Coding 任务下一步做什么&#xff1b;</li><li>是否修改 AuthService&#xff1b;</li><li>是否重新设计 JWT&#xff1b;</li><li>最终任务什么时候结束。</li></ul>
<p>这些责任仍然在 Main Agent。</p>
<p>因此&#xff0c;可以把 Subagent 的任务关系概括成&#xff1a;</p>
<blockquote>
<p><strong>“这块事情你帮我做一下&#xff0c;做好以后把结果告诉我。”</strong></p>
</blockquote>
<hr />
<h2>二、Subagent 真正重要的价值&#xff1a;Context Isolation</h2>
<p>很多人第一次接触 Subagent 时&#xff0c;会认为它的价值是&#xff1a;</p>
<blockquote>
<p>多创建一个 Agent&#xff0c;增加一点“智能”。</p>
</blockquote>
<p>实际上&#xff0c;从工程角度看&#xff0c;Subagent 最有价值的能力之一往往是&#xff1a;</p>
<p><strong>Context Isolation。</strong></p>
<p>每个 Subagent 可以拥有自己的&#xff1a;</p>


```text
System Prompt
Context Window
Tool Set
Permission
Memory
Execution State
```


<p>这意味着&#xff0c;大量只与当前子任务相关的信息不需要进入 Main Agent。</p>
<p>例如 Research 场景&#xff1a;</p>


```text
                        Lead Agent
│
┌───────────────┼───────────────┐
↓               ↓               ↓
Research A      Research B       Research C
Context A       Context B        Context C
80K             70K              90K
│               │               │
↓               ↓               ↓
Summary         Summary          Summary
└───────────────┼───────────────┘
↓
Lead Agent Context
```


<p>三个 Research Agent 可能总共阅读几十万 Token 的资料。</p>
<p>但 Lead Agent 最终只接收&#xff1a;</p>


```text
3K Summary
+
4K Summary
+
3K Summary
```


<p>相当于 Subagent 帮主 Agent完成了一次上下文压缩。</p>
<p>Anthropic 在 Claude Research 多 Agent 系统的实践中&#xff0c;也把这种架构描述为一种有效的“压缩”机制&#xff1a;Subagent 使用独立 Context Window 进行大量探索&#xff0c;再把最有价值的信息返回 Lead Agent。</p>
<p>这也是为什么 Research、日志分析、测试、代码搜索、文档阅读等场景特别适合 Subagent。</p>
<hr />
<h2>三、Subagent 做的通常是什么任务&#xff1f;</h2>
<p>Subagent 最适合的是&#xff1a;</p>
<blockquote>
<p><strong>目标明确、边界清晰、完成后能够产生一个结果的子任务。</strong></p>
</blockquote>
<p>例如 Coding Agent 可以委派&#xff1a;</p>


```text
搜索整个仓库中 JWT 相关代码

分析这一批失败测试

检查某个模块安全问题

阅读 Spring Boot 迁移文档

总结最近一次 Git Diff

检查数据库 Schema

分析某一个接口的调用链
```


<p>这些任务可能并不简单。</p>
<p>一个 Security Subagent 甚至可以运行十分钟、调用几十次 Tool。</p>
<p>但是从整个系统角度看&#xff0c;它仍然只负责&#xff1a;</p>


```text
Security Analysis
```


<p>完成后返回&#xff1a;</p>


```text
SecurityReport
```


<p>然后由 Parent Agent 决定接下来怎么办。</p>
<p>因此 Subagent 的核心不是&#xff1a;</p>


```text
任务简单
```


<p>而是&#xff1a;</p>


```text
责任边界清晰
```


<hr />
<h2>四、MAS&#xff1a;从“派一个助手”升级成“组织一个 Agent 团队”</h2>
<p>MAS 全称&#xff1a;</p>
<blockquote>
<p><strong>Multi-Agent System&#xff0c;多智能体系统。</strong></p>
</blockquote>
<p>它关注的问题比 Subagent 更高一层。</p>
<p>Subagent讨论&#xff1a;</p>
<blockquote>
<p>某块工作交给谁&#xff1f;</p>
</blockquote>
<p>MAS 讨论&#xff1a;</p>
<blockquote>
<p>多个能够自主推理和行动的 Agent&#xff0c;应该如何组成一个系统&#xff0c;共同完成一个复杂目标&#xff1f;</p>
</blockquote>
<p>假设我们的目标是&#xff1a;</p>
<blockquote>
<p>完成一次大型支付系统升级。</p>
</blockquote>
<p>MAS 可能设计成&#xff1a;</p>


```text
                   Payment Migration MAS
│
┌─────────────────┼─────────────────┐
↓                 ↓                 ↓
Architecture Agent    Backend Agent    Security Agent
│                 │                 │
│                 ↓                 │
│             Test Agent            │
│                 │                 │
└─────────────────┼─────────────────┘
↓
Review Agent
↓
最终结果
```


<p>这里 Backend Agent 不只是&#xff1a;</p>
<blockquote>
<p>帮 Main Agent 修改一个文件。</p>
</blockquote>
<p>它的职责可能是&#xff1a;</p>
<blockquote>
<p><strong>负责整个 Backend Migration&#xff0c;直到满足升级目标。</strong></p>
</blockquote>
<p>Security Agent 也不是简单&#xff1a;</p>
<blockquote>
<p>扫描一下安全问题。</p>
</blockquote>
<p>而可能是&#xff1a;</p>
<blockquote>
<p><strong>持续负责整个升级过程中的安全风险&#xff0c;必要时要求 Backend Agent 修改方案。</strong></p>
</blockquote>
<p>这种任务责任已经比普通 Subagent 更完整。</p>
<p>所以可以用一句非常直白的话区分&#xff1a;</p>
<blockquote>
<p><strong>Subagent 更像“帮我完成这块工作”。</strong></p>
</blockquote>
<blockquote>
<p><strong>MAS Agent 更像“你负责这个领域&#xff0c;我们一起把整个项目做完”。</strong></p>
</blockquote>
<hr />
<h2>五、两者在任务上的最大区别&#xff1a;Task-oriented 和 Role-oriented</h2>
<p>这是 Subagent 与 MAS 最值得讲清楚的地方。</p>
<p>假设主任务是&#xff1a;</p>
<blockquote>
<p>完成一次 Java 17 → Java 21 的大型项目升级。</p>
</blockquote>
<p>如果采用 Subagent 模式&#xff1a;</p>


```text
                   Main Coding Agent
│
┌────────────┼────────────┐
↓            ↓            ↓
Dependency       Test        Documentation
Subagent      Subagent        Subagent
```


<p>Main Agent 仍然承担核心任务。</p>
<p>它会&#xff1a;</p>


```text
分析架构
制定升级方案
修改核心代码
决定执行顺序
处理冲突
判断是否完成
```


<p>Subagent 则负责一些明确工作&#xff1a;</p>


```text
Dependency Subagent
→ 分析依赖兼容性

Test Subagent
→ 执行测试并总结错误

Documentation Subagent
→ 查询迁移文档
```


<p>它们完成以后&#xff0c;把结果交给 Main Agent。</p>
<p>这种模式本质是&#xff1a;</p>


```text
Main Agent
=
Project Owner

Subagent
=
Task Worker
```


<hr />
<p>如果使用 MAS&#xff0c;情况会变成&#xff1a;</p>


```text
                    Java Migration MAS
│
┌───────────────────┼───────────────────┐
↓                   ↓                   ↓
Architecture Agent   Backend Agent      Dependency Agent
│                   │                   │
│              Test Agent               │
│                   │                   │
└───────────────────┼───────────────────┘
↓
Review Agent
```


<p>这里不同 Agent 对不同领域负责。</p>
<p>Backend Agent 的 Goal 可能是&#xff1a;</p>
<blockquote>
<p>确保所有 Backend 代码能够在 Java 21 上正确运行。</p>
</blockquote>
<p>Test Agent 的 Goal 是&#xff1a;</p>
<blockquote>
<p>确保升级后的系统满足测试和回归要求。</p>
</blockquote>
<p>Dependency Agent 的 Goal 是&#xff1a;</p>
<blockquote>
<p>确保所有依赖版本兼容&#xff0c;并解决 Breaking Changes。</p>
</blockquote>
<p>也就是说&#xff0c;MAS 中 Agent 的任务往往更加&#xff1a;</p>


```text
Role-oriented
Goal-oriented
```


<p>而 Subagent 的任务更像&#xff1a;</p>


```text
Task-oriented
```


<p>这并不是绝对规则&#xff0c;但非常适合作为工程上的判断方式。</p>
<hr />
<h2>六、判断 Subagent 和 MAS 最实用的问题&#xff1a;谁对最终目标负责&#xff1f;</h2>
<p>这是区分两者最好用的方法。</p>
<p>假设有一个 Agent 负责&#xff1a;</p>
<blockquote>
<p>找出仓库中所有 JWT 相关代码。</p>
</blockquote>
<p>输入&#xff1a;</p>


```text
Repository
```


<p>输出&#xff1a;</p>


```text
17 个相关文件
8 个主要调用链
3 个风险点
```


<p>结果返回以后&#xff0c;它的任务就结束了。</p>
<p>这非常像&#xff1a;</p>


```text
Subagent
```


<p>至于&#xff1a;</p>


```text
JWT 应该怎么改？

应该先改 Backend 还是 Frontend？

需要修改数据库吗？

什么时候算迁移完成？
```


<p>都由上层 Agent 决定。</p>
<hr />
<p>而 MAS 中一个 Security Agent 可能承担&#xff1a;</p>
<blockquote>
<p>确保整个 JWT 改造不存在明显安全问题。</p>
</blockquote>
<p>这个 Agent 可能需要&#xff1a;</p>


```text
分析方案
↓
检查 Backend 修改
↓
发现风险
↓
要求 Backend Agent 调整
↓
重新审核
↓
运行安全测试
↓
确认满足目标
```


<p>它不是完成一次“扫描”就结束。</p>
<p>它需要持续对一个职责目标负责。</p>
<p>所以可以把二者概括成&#xff1a;</p>


```text
Subagent

Input
↓
Task
↓
Result
↓
Return to Parent
```


<p>而 MAS Agent 更接近&#xff1a;</p>


```text
Role
↓
Goal
↓
Observe system
↓
Act / Collaborate
↓
Evaluate
↓
Continue
↓
Goal satisfied
```


<hr />
<h2>七、但 Subagent 和 MAS 并不是互斥关系</h2>
<p>这里很容易产生另一个误区&#xff1a;</p>


```text
Subagent 架构
VS
MAS 架构
```


<p>实际上 Subagent 完全可以是 MAS 中的一种组织形式。</p>
<p>例如&#xff1a;</p>


```text
                       MAS
│
Lead Agent
│
┌──────────┼──────────┐
↓          ↓          ↓
Subagent   Subagent   Subagent
```


<p>这是&#xff1a;</p>
<blockquote>
<p><strong>Hierarchical MAS。</strong></p>
</blockquote>
<p>但 MAS 还可以有其他形态。</p>
<p>例如 Handoff&#xff1a;</p>


```text
Triage Agent
↓
Handoff
↓
Refund Agent
```


<p>或者 Peer Collaboration&#xff1a;</p>


```text
Agent A ↔ Agent B ↔ Agent C
```


<p>或者 Graph&#xff1a;</p>


```text
              Agent A
/       \
Agent B      Agent C
\       /
Aggregator
```


<p>所以&#xff1a;</p>
<blockquote>
<p><strong>Subagent 是 MAS 中非常常见的一种 Agent 关系&#xff0c;但 MAS 不等于 Subagent。</strong></p>
</blockquote>
<hr />
<h2>八、Manager 和 Handoff 是两种非常典型的 MAS</h2>
<p>OpenAI Agents SDK 对这一点给出了非常清晰的区分。</p>
<p>第一种是&#xff1a;</p>
<h3>Manager Pattern</h3>


```text
                      Manager Agent
│
┌─────────────┼─────────────┐
↓             ↓             ↓
Search Agent   Finance Agent  Writing Agent
```


<p>Manager 一直掌握用户会话。</p>
<p>其他 Agent 更像专业能力。</p>
<p>它们执行完以后&#xff1a;</p>


```text
Result
↓
Manager
```


<p>最终回答仍然由 Manager 生成。</p>
<p>OpenAI 将这种方式称为&#xff1a;</p>


```text
Agents as Tools
```


<p>这与 Subagent 思路非常接近。</p>
<hr />
<p>第二种是&#xff1a;</p>
<h3>Handoff Pattern</h3>
<p>例如&#xff1a;</p>


```text
用户
↓
Triage Agent
↓
判断：退款问题
↓
Handoff
↓
Refund Agent
↓
直接和用户继续交互
```


<p>这时候 Refund Agent 不再只是&#xff1a;</p>
<blockquote>
<p>给 Triage Agent 帮个忙。</p>
</blockquote>
<p>它直接接管整个后续流程。</p>
<p>所以&#xff1a;</p>


```text
Manager / Subagent
```


<p>强调集中控制。</p>


```text
Handoff
```


<p>则把控制权交给其他 Agent。</p>
<p>这也是 MAS 需要解决的问题&#xff1a;</p>
<blockquote>
<p><strong>控制权到底在哪里&#xff1f;</strong></p>
</blockquote>
<hr />
<h2>九、Multi-Agent Parallelism&#xff1a;这是执行方式&#xff0c;不是组织方式</h2>
<p>接下来是第三个容易被混淆的概念&#xff1a;</p>
<blockquote>
<p>多 Agent 并行。</p>
</blockquote>
<p>假设一个 Research MAS 中有&#xff1a;</p>


```text
Market Agent
Technology Agent
Policy Agent
```


<p>可以串行&#xff1a;</p>


```text
Market
↓
Technology
↓
Policy
```


<p>也可以并行&#xff1a;</p>


```text
           ┌── Market Agent
│
Research ──┼── Technology Agent
│
└── Policy Agent
```


<p>所以&#xff1a;</p>
<blockquote>
<p><strong>MAS 回答“谁和谁协作”&#xff0c;Parallelism 回答“谁和谁同时执行”。</strong></p>
</blockquote>
<p>Google ADK 的 ParallelAgent 就是一种非常典型的 Workflow Agent&#xff1a;它会同时启动多个 Subagent&#xff0c;用于彼此独立的任务。</p>
<p>需要特别注意&#xff1a;</p>
<blockquote>
<p><strong>只有相对独立的任务才真正适合并行。</strong></p>
</blockquote>
<p>例如&#xff1a;</p>


```text
研究 OpenAI
研究 Anthropic
研究 Google
```


<p>非常适合。</p>
<p>但&#xff1a;</p>


```text
设计数据库
↓
根据数据库设计 API
↓
根据 API 编写 Frontend
```


<p>显然存在依赖。</p>
<p>不能为了“多 Agent”强行全部并行。</p>
<hr />
<h2>十、多 Agent 并行最大的难题&#xff1a;共享状态和冲突</h2>
<p>真正创建几个并行 Agent并不困难。</p>
<p>困难的是&#xff1a;</p>
<blockquote>
<p>它们是否会互相干扰&#xff1f;</p>
</blockquote>
<p>Research Agent 通常比较安全&#xff1a;</p>


```text
Agent A
读资料

Agent B
读另一批资料
```


<p>因为大部分操作都是 Read。</p>
<p>但 Coding Agent 就复杂很多&#xff1a;</p>


```text
Agent A
修改 AuthService

Agent B
同时也修改 AuthService
```


<p>很容易出现&#xff1a;</p>


```text
代码覆盖
Merge Conflict
状态不一致
```


<p>因此 Coding MAS 往往需要&#xff1a;</p>


```text
独立 Workspace
Git Branch
Task Lock
File Ownership
Merge Strategy
```


<p>而 Research MAS 则更多需要&#xff1a;</p>


```text
Context Isolation
Search Boundary
Result Aggregation
```


<p>这也说明&#xff1a;</p>
<blockquote>
<p>多 Agent 系统并没有一个通用的最佳架构。</p>
</blockquote>
<p>系统的 Domain 会决定最难解决的问题是什么。</p>
<hr />
<h2>十一、真实案例&#xff1a;Anthropic Claude Research</h2>
<p>Anthropic 公开介绍的 Claude Research&#xff0c;是理解 Subagent、MAS 和并行 Agent 非常好的真实案例。</p>
<p>其核心结构可以简化为&#xff1a;</p>


```text
User Research Query
↓
Lead Research Agent
↓
理解问题
制定 Research Plan
↓
┌──────────────┬──────────────┐
↓              ↓              ↓
Research Subagent A  Subagent B    Subagent C
│              │              │
Web Search      Web Search      Web Search
│              │              │
Tool Calls      Tool Calls      Tool Calls
│              │              │
└──────────────┼──────────────┘
↓
Lead Research Agent
↓
综合
↓
Citation Agent
↓
Final Report
```


<p>这里正好能够对应前面三个概念。</p>
<h4>Subagent</h4>
<p>Lead Researcher 把不同研究方向委派给不同 Research Subagent。</p>
<p>每个 Subagent 有自己的 Context Window&#xff0c;并独立搜索、阅读、分析&#xff0c;再把结果压缩给 Lead Agent。</p>
<h4>MAS</h4>
<p>Lead Agent、多个 Research Subagent 和 Citation Agent 共同构成整个 Research Multi-Agent System。</p>
<h4>Parallelism</h4>
<p>多个 Research Subagent 可以同时搜索不同研究方向。</p>
<p>而每个 Subagent 内部甚至还可以同时发起多个 Web Search。</p>
<p>因此形成&#xff1a;</p>


```text
Agent Parallelism
×
Tool Parallelism
```


<hr />
<h2>十二、Anthropic 案例为什么能获得明显收益</h2>
<p>Research 是非常适合多 Agent 的场景。</p>
<p>因为它天然存在大量独立探索路径。</p>
<p>例如&#xff1a;</p>
<blockquote>
<p>调研全球主要 AI Agent Framework。</p>
</blockquote>
<p>可以拆成&#xff1a;</p>


```text
OpenAI
Anthropic
Google
Microsoft
开源生态
商业产品
```


<p>这些方向并不需要互相等待。</p>
<p>所以并行 Agent 可以直接增加&#xff1a;</p>


```text
搜索宽度
Tool Call Budget
Context Capacity
探索路径数量
```


<p>Anthropic 在内部 Research Evaluation 中报告&#xff0c;多 Agent Research 系统相较单独 Lead Agent 获得了明显质量提升&#xff0c;同时通过两层并行显著降低复杂研究任务的总体耗时。</p>
<p>这里很重要的一点是&#xff1a;</p>
<blockquote>
<p>Multi-Agent 并不是凭空让单个模型变聪明了。</p>
</blockquote>
<p>它更像是给整个系统增加&#xff1a;</p>


```text
更多 Token Budget
+
更多 Context Window
+
更多 Tool Calls
+
更多并行探索路径
```


<p>然后通过 Lead Agent 把这些探索结果组织起来。</p>
<hr />
<h2>十三、这个案例真正值得学习的是 Delegation</h2>
<p>Anthropic 实践中一个非常典型的问题是&#xff1a;</p>
<blockquote>
<p>Lead Agent 创建了多个 Subagent&#xff0c;但几个 Agent 做了几乎一样的事情。</p>
</blockquote>
<p>如果 Parent Agent 只告诉三个 Agent&#xff1a;</p>
<blockquote>
<p>调研 AI Agent 产品。</p>
</blockquote>
<p>三个 Agent 很可能都去搜索&#xff1a;</p>


```text
OpenAI
Anthropic
Google
```


<p>产生大量重复劳动。</p>
<p>所以优秀的 Task Delegation 应该明确&#xff1a;</p>


```text
Objective

Scope / Boundary

Expected Output

Allowed Sources / Tools

Stop Condition
```


<p>例如&#xff1a;</p>


```text
Subagent A
只负责 OpenAI 和微软生态。

Subagent B
只负责 Anthropic 和 Google。

Subagent C
只调查开源 Agent Framework。

最终分别返回：
产品能力、技术路线、商业模式、核心来源。
```


<p>这时候每个 Agent 的探索空间才真正互补。</p>
<p>因此 MAS 的难点从来不是&#xff1a;</p>


```text
spawn 5 agents
```


<p>而是&#xff1a;</p>
<blockquote>
<p><strong>如何正确拆任务&#xff0c;让五个 Agent 做五件真正不同而有价值的事情。</strong></p>
</blockquote>
<hr />
<h2>十四、为什么 Agent 越多不一定越好</h2>
<p>Multi-Agent 有一个非常现实的问题&#xff1a;</p>
<blockquote>
<p>贵。</p>
</blockquote>
<p>一个普通 Chat 可能只调用一次模型。</p>
<p>一个 Agent 可能调用&#xff1a;</p>


```text
10 次模型
+
20 次 Tool
```


<p>一个 MAS 又可能同时拥有&#xff1a;</p>


```text
1 Lead Agent
+
5 Subagents
+
1 Citation Agent
```


<p>整个 Token 消耗会迅速增长。</p>
<p>Anthropic 的生产实践也指出&#xff0c;Multi-Agent 的 Token 消耗明显高于普通对话和单 Agent。</p>
<p>所以不能因为&#xff1a;</p>


```text
Multi-Agent 更先进
```


<p>就所有请求都创建五个 Agent。</p>
<p>更合理的是根据任务复杂度分配 Agent Budget。</p>
<p>例如&#xff1a;</p>


```text
简单事实问题
→ 单 Agent

简单子任务
→ Main Agent + 1 Subagent

多方向比较
→ 2~3 个并行 Subagent

复杂开放式 Research
→ 完整 MAS
```


<p>因此真正的生产问题是&#xff1a;</p>
<blockquote>
<p><strong>这次任务值得花多少 Agent Budget&#xff1f;</strong></p>
</blockquote>
<hr />
<h2>十五、什么时候应该用 Subagent&#xff0c;什么时候升级 MAS</h2>
<p>可以用责任边界来判断。</p>
<p>如果主 Agent 能明确说&#xff1a;</p>
<blockquote>
<p>我只需要另一个 Agent帮我完成一件事情&#xff0c;然后把结果给我。</p>
</blockquote>
<p>优先考虑&#xff1a;</p>


```text
Subagent
```


<p>例如&#xff1a;</p>


```text
搜索代码
分析测试
查询资料
做一次安全 Review
```


<p>如果问题变成&#xff1a;</p>
<blockquote>
<p>这个目标需要多个不同角色长期负责各自领域&#xff0c;并持续互相协作。</p>
</blockquote>
<p>更适合&#xff1a;</p>


```text
MAS
```


<p>例如&#xff1a;</p>


```text
完整软件项目开发

大型 Research

企业尽调

复杂数据分析

自动化运营
```


<p>而是否并行&#xff0c;则继续问&#xff1a;</p>
<blockquote>
<p>这些任务之间有没有强依赖&#xff1f;</p>
</blockquote>
<p>没有&#xff1a;</p>


```text
Parallel
```


<p>有&#xff1a;</p>


```text
Sequential / DAG / Workflow
```


<p>于是可以形成一个很实用的决策过程&#xff1a;</p>


```text
一个 Agent 能做好吗？
│
┌───┴────┐
│        │
能       不能
│        │
单 Agent   ↓
能否拆成明确子任务？
│
┌───┴────┐
│        │
能       不能
│        │
Subagent   重新设计
│
↓
是否需要多个角色长期协作？
│
┌───┴────┐
│        │
否       是
│        │
Subagent     MAS
│
↓
子任务是否相互独立？
│
┌───┴────┐
│        │
是       否
│        │
Parallel   Workflow
```


<hr />
<h2>十六、大厂实践可以总结成什么</h2>
<p>结合 Anthropic、OpenAI、Google 和 Microsoft 当前的 Agent 体系&#xff0c;可以得到一套比较一致的工程原则。</p>
<p>第一&#xff0c;<strong>能用单 Agent 解决&#xff0c;就不要急着上 MAS。</strong></p>
<p>多 Agent 引入的不只是更多能力&#xff0c;还有更多 Token、状态、调度、Trace 和失败模式。</p>
<p>第二&#xff0c;<strong>Subagent 应该小而专。</strong></p>
<p>一个 Subagent 最好有明确职责、Prompt、Tool 和 Permission。</p>
<p>第三&#xff0c;<strong>Parent → Subagent 的委派必须有清晰 Contract。</strong></p>
<p>否则多 Agent 很容易产生重复工作。</p>
<p>第四&#xff0c;<strong>只有独立任务才真正适合 Parallel。</strong></p>
<p>任务之间存在依赖时&#xff0c;更适合 Workflow / DAG。</p>
<p>第五&#xff0c;<strong>MAS 中必须明确最终控制权。</strong></p>
<p>可以是 Manager&#xff0c;也可以通过 Handoff 转移。</p>
<p>第六&#xff0c;<strong>Context Isolation 是多 Agent 最重要的收益之一。</strong></p>
<p>不要让所有 Subagent 原始 Tool Log 都重新进入 Lead Context。</p>
<p>第七&#xff0c;<strong>Shared State 必须显式设计。</strong></p>
<p>尤其在 Coding、数据修改、业务操作场景中&#xff0c;要处理并发写入和 Conflict。</p>
<p>第八&#xff0c;<strong>Agent 数量和 Token Budget 必须受控。</strong></p>
<p>不要允许 Orchestrator 无限 spawn Agent。</p>
<p>第九&#xff0c;<strong>Multi-Agent 必须配套 Trace、Metrics、Cost 和 Eval。</strong></p>
<p>因为系统真正出现问题以后&#xff0c;你需要知道&#xff1a;</p>


```text
哪个 Agent
为什么被创建
拿到了什么任务
调用了什么 Tool
消耗了多少 Token
返回了什么
为什么最终结果错误
```


<hr />
<h2>结语</h2>
<p>Subagent、MAS 和多 Agent 并行虽然经常同时出现&#xff0c;但其实分别描述三个不同的问题。</p>
<p><strong>Subagent 关注任务委派&#xff1a;</strong></p>
<blockquote>
<p>这个明确的子任务交给谁做&#xff1f;</p>
</blockquote>
<p><strong>MAS 关注系统组织&#xff1a;</strong></p>
<blockquote>
<p>多个拥有独立目标、Context 和能力的 Agent&#xff0c;应该如何共同完成一个复杂目标&#xff1f;</p>
</blockquote>
<p><strong>Multi-Agent Parallelism 关注执行调度&#xff1a;</strong></p>
<blockquote>
<p>这些彼此独立的 Agent 工作是否可以同时执行&#xff1f;</p>
</blockquote>
<p>而 Subagent 与 MAS 在“做什么任务”上的真正区别&#xff0c;可以用一句非常简单的话概括&#xff1a;</p>
<blockquote>
<p><strong>Subagent 更像“这块事情你帮我做完&#xff0c;然后把结果给我”。</strong></p>
</blockquote>
<blockquote>
<p><strong>MAS 更像“这个项目我们几个角色分工合作&#xff0c;一起把最终目标完成”。</strong></p>
</blockquote>
<p>因此&#xff0c;一个典型的 Subagent 任务具有清晰的输入、输出和任务边界。</p>
<p>一个 MAS Agent 则更可能承担一个持续性的职责目标&#xff0c;需要不断观察系统、采取行动&#xff0c;并与其他 Agent 协作直到目标达成。</p>
<p>Anthropic Claude Research 是一个很典型的例子&#xff1a;</p>


```text
Lead Agent
+
Specialized Subagents
+
Parallel Research
+
Result Aggregation
+
Citation Validation
```


<p>这里 Subagent 解决专业任务拆分&#xff0c;MAS 解决整个系统的多 Agent 协作&#xff0c;而 Parallelism 让多个独立研究方向能够同时推进。</p>
<p>所以真正成熟的 Multi-Agent Architecture&#xff0c;从来不是&#xff1a;</p>


```text
多创建几个 Agent
```


<p>而是&#xff1a;</p>


```text
正确拆分任务
+
清晰定义责任
+
隔离 Context
+
合理进行 Delegation
+
只并行独立工作
+
显式管理 Shared State
+
控制 Agent Budget
+
正确汇总结果
+
完整 Observability
```


<p>只有这些能力同时存在时&#xff0c;多 Agent 才真正从&#xff1a;</p>
<blockquote>
<p>“多个模型同时跑”</p>
</blockquote>
<p>变成&#xff1a;</p>
<blockquote>
<p><strong>“多个智能体作为一个有组织的团队共同完成复杂目标”。</strong></p>
</blockquote>
<h3>参考资料</h3>
<p>Anthropic Engineering — <em>How we built our multi-agent research system</em><br />
重点介绍 Claude Research 的 Lead Agent、Subagent、并行 Research、Delegation、Context Compression、成本与评测实践。</p>
<p>Anthropic Claude Code Documentation — <em>Subagents</em><br />
介绍 Subagent 的独立 Context Window、System Prompt、Tool、Permission&#xff0c;以及串行和并行使用方式。</p>
<p>OpenAI Agents SDK — <em>Multi-agent orchestration</em><br />
介绍 Manager / Agents-as-Tools、Handoff、Code Orchestration 与 LLM Orchestration。</p>
<p>Google Agent Development Kit — <em>Parallel workflow agents</em><br />
介绍 ParallelAgent、独立执行分支以及 Shared State、并发访问等问题。</p>
<p>Microsoft AutoGen AgentChat<br />
提供 Selector Group Chat、Swarm、GraphFlow 等不同 MAS Coordination Pattern。</p>
<p>Anthropic Engineering — <em>Building effective agents</em><br />
强调从简单 Agent 和可组合 Workflow 开始&#xff0c;只有在复杂度确实需要时才升级到 Multi-Agent。</p>
<p>Anthropic Engineering — <em>Building a C compiler with a team of parallel Claudes</em><br />
展示 Coding 多 Agent 场景中 Task Lock、独立 Workspace、并行修改和共享代码库协调问题。</p>
