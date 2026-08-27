---
title: "从 Agent Run 到 Agent LongTask：以 Coding Agent 为例理解长时复杂任务的执行架构"
description: "CSDN 原文全文镜像：摘要：Agent LongTask 与普通 Agent Run 的本质区别 本文深入探讨了 Agent 系统中 LongTask 与普通 Run 的核心差异。关键结论是：LongTask 并非简单的时间延长版 Agent Run，而是系……"
pageType: article
module: agent
updated: '2026-08-12'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "架构"
  - "java"
  - "前端"
  - "大模型"
  - "人工智能"
  - "软件工程"
level: advanced
prerequisites:
  - "/llms/agent/"
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-12，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-12。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163697628](https://blog.csdn.net/m0_63309778/article/details/163697628)
- 站内分区：Agent / Agent LongTask
:::

<p><img src="https://i-blog.csdnimg.cn/direct/2fe438692d9b42e1970b13946ebbf7ed.png" alt="" /></p>
<h3>前言</h3>
<p>在设计 Agent 系统时&#xff0c;一个非常容易产生疑问的问题是&#xff1a;</p>
<blockquote>
<p>普通 Agent Run 本身也可以后台运行&#xff0c;也可以增加 Checkpoint、重试和状态恢复&#xff0c;为什么还需要专门提出 LongTask 这个概念&#xff1f;</p>
</blockquote>
<p>这个疑问本身是成立的。</p>
<p>LongTask 并不是一种全新的 Agent&#xff0c;也不是某种普通 Agent 无法实现的特殊算法。一个普通 Agent Run 如果不断加入 Persistence、Checkpoint、暂停恢复、Step、SubTask、Scheduler、Human-in-the-loop 等能力&#xff0c;它最终确实可以演化成长时任务系统。</p>
<p>所以真正的区别并不在于&#xff1a;</p>


```text
普通 Run 没有 Checkpoint
LongTask 有 Checkpoint
```


<p>而在于系统的核心执行抽象发生了变化。</p>
<p>普通 Agent Run 关注的是&#xff1a;</p>
<blockquote>
<p><strong>如何把一次 Agent 执行跑完。</strong></p>
</blockquote>
<p>而 LongTask 关注的是&#xff1a;</p>
<blockquote>
<p><strong>如何把一个由多个 Agent Run、确定性任务、等待节点、人工审批和外部系统共同组成的复杂目标可靠地做完。</strong></p>
</blockquote>
<p>因此更准确的关系应该是&#xff1a;</p>


```text
LongTask
↓
Stage / Step / Dependency
↓
Agent Run / Worker Task / Approval / Waiting
```


<p>而不是&#xff1a;</p>


```text
Agent Run VS LongTask Run
```


<p>也就是说&#xff1a;</p>
<blockquote>
<p><strong>LongTask 编排 Run&#xff0c;而不是替代 Run。</strong></p>
</blockquote>
<p>这也是本文理解 Agent LongTask 的核心。</p>
<hr />
<h2>一、先从最普通的 Coding Agent Run 说起</h2>
<p>假设用户给 Coding Agent 一个任务&#xff1a;</p>
<blockquote>
<p>修复登录接口偶发出现的 NullPointerException。</p>
</blockquote>
<p>Agent 接到任务后&#xff0c;通常会经历一个连续的执行过程&#xff1a;</p>


```text
理解问题
↓
搜索相关代码
↓
阅读 Controller / Service
↓
分析异常原因
↓
修改代码
↓
执行测试
↓
修复完成
```


<p>Agent Runtime 内部实际上一直在执行一个循环&#xff1a;</p>


```text
构建 Context
↓
调用 LLM
↓
LLM 决定下一步动作
↓
调用 Tool
↓
观察 Tool Result
↓
重新构建 Context
↓
继续调用 LLM
```


<p>OpenAI Agents SDK 对 Runner 的描述也是类似的 Agent Loop&#xff1a;Runner 调用模型&#xff0c;处理工具调用或 handoff&#xff0c;再继续下一轮&#xff0c;直到产生最终输出或满足停止条件。</p>
<p>对于“修复一个 Bug”这样的任务&#xff0c;这种 Run 模型非常自然。</p>
<p>即使它执行两三分钟&#xff0c;甚至更久&#xff0c;也没有必要因为时间长就把它定义成 LongTask。</p>
<p>此时用户交给系统的业务目标和 Agent Run 基本是一一对应的&#xff1a;</p>


```text
用户目标
“修复登录 Bug”

↓

Agent Run
run_10001
```


<p>Run 完成&#xff0c;用户目标也完成。</p>
<hr />
<h2>二、给普通 Run 加上 Persistence&#xff0c;它仍然可以只是 Run</h2>
<p>现在假设 Coding Agent 已经完成&#xff1a;</p>


```text
✓ 找到异常位置
✓ 确定问题原因
✓ 修改代码
● 正在运行测试
```


<p>这时 Worker 突然崩溃。</p>
<p>如果所有状态只保存在当前 Python 进程内存中&#xff0c;那么新的 Worker 很难知道&#xff1a;</p>
<blockquote>
<p>Agent 之前已经做了什么&#xff1f;</p>
</blockquote>
<p>于是我们开始引入 <strong>Persistence</strong>。</p>
<p>Persistence 的意思并不复杂&#xff1a;</p>
<blockquote>
<p>将 Agent 当前执行状态保存到进程之外&#xff0c;使状态可以跨请求、跨 Worker、跨进程继续存在。</p>
</blockquote>
<p>LangGraph 官方把 Persistence 明确拆成两类&#xff1a;Checkpointer 保存 thread 范围内的 Graph State Snapshot&#xff0c;而 Store 用于保存跨 thread 的长期应用数据。Checkpointer 可以支撑 conversation continuity、fault tolerance、human-in-the-loop 和 time travel 等能力。</p>
<p>于是 Agent 可以在执行过程中保存&#xff1a;</p>


```text
Checkpoint #1
完成代码搜索

Checkpoint #2
完成问题分析

Checkpoint #3
完成代码修改

当前：
准备执行测试
```


<p>Worker 崩溃以后&#xff1a;</p>


```text
Worker A Crash
↓
读取 Checkpoint
↓
Worker B
↓
继续执行测试
```


<p>LangGraph 的 Checkpointer 会在 Graph 执行过程中保存状态快照&#xff0c;这也是 fault-tolerant execution 和恢复能力的重要基础。</p>
<p>但请注意&#xff1a;</p>
<blockquote>
<p><strong>拥有 Checkpoint 并不意味着它一定应该变成 LongTask。</strong></p>
</blockquote>
<p>这个 Run 依然可能只是&#xff1a;</p>


```text
修复一个 Bug
```


<p>只不过它现在变成了一个更加可靠的、可以恢复的 Agent Run。</p>
<p>这也是理解 Run 和 LongTask 最容易被忽略的一点。</p>
<hr />
<h2>三、Persistence、Checkpoint 和 Durable Execution 到底是什么关系</h2>
<p>这几个概念非常容易被混在一起。</p>
<p>可以把它们理解成三个层次。</p>
<h4>Persistence&#xff1a;状态能不能保存下来</h4>
<p>Persistence 解决&#xff1a;</p>


```text
Agent 当前状态
不要只存在内存
```


<p>例如保存&#xff1a;</p>


```text
Messages
Tool Results
Plan
当前 Node
Run Status
Artifact Reference
```


<h4>Checkpoint&#xff1a;具体保存到哪个恢复点</h4>
<p>Checkpoint 是 Persistence 的一种具体表现。</p>
<p>例如&#xff1a;</p>


```text
Checkpoint 1
仓库分析完成

Checkpoint 2
方案设计完成

Checkpoint 3
后端修改完成
```


<p>Checkpoint 表示&#xff1a;</p>
<blockquote>
<p>如果执行中断&#xff0c;系统可以从哪个已经确认的状态继续。</p>
</blockquote>
<h4>Durable Execution&#xff1a;整个任务能不能跨故障继续</h4>
<p>Durable Execution 是最终获得的运行能力。</p>
<p>Temporal 对 Durable Execution 的定义非常直接&#xff1a;Workflow Execution 的状态和进度能够在 failure、crash 或 server outage 等情况下继续保持&#xff0c;并恢复执行。Temporal 的 Workflow 本身可以运行数秒&#xff0c;也可以持续多年&#xff0c;而不要求原来的 Worker 进程一直存在。</p>
<p>因此关系可以理解为&#xff1a;</p>


```text
Persistence
↓
保存状态

Checkpoint
↓
建立恢复边界

Replay / Resume
↓
重建执行

Durable Execution
↓
任务获得长期可靠运行能力
```


<p>所以&#xff1a;</p>
<blockquote>
<p><strong>Persistence 是机制&#xff0c;Checkpoint 是恢复点&#xff0c;而 Durable Execution 是最终获得的能力。</strong></p>
</blockquote>
<hr />
<h2>四、LongTask 的真正分界出现在“一个 Run 已经表达不了整个任务”时</h2>
<p>继续把 Coding Agent 的任务放大。</p>
<p>现在用户要求&#xff1a;</p>
<blockquote>
<p>把系统认证体系从 Session 改造成 JWT&#xff0c;同时增加 Refresh Token、权限校验、自动化测试、前端适配、数据库 Migration 和迁移文档。</p>
</blockquote>
<p>这已经不再像一个简单 Bug Fix。</p>
<p>它可能需要&#xff1a;</p>


```text
仓库分析
↓
架构设计
↓
后端认证改造
↓
Refresh Token
↓
权限体系调整
↓
前端认证改造
↓
数据库 Migration
↓
单元测试
↓
集成测试
↓
修复失败测试
↓
等待 CI
↓
人工确认
↓
生成迁移文档
```


<p>其中甚至可能同时存在&#xff1a;</p>


```text
Backend Agent

Frontend Agent

Test Agent

Documentation Agent
```


<p>这时候如果整个任务仍然只使用&#xff1a;</p>


```text
run_10001
```


<p>去表达&#xff0c;就会越来越别扭。</p>
<p>因为可能出现&#xff1a;</p>


```text
Backend Agent Run      completed
Frontend Agent Run     completed
Test Agent Run         failed
Migration Task         completed
CI                     waiting
Documentation Agent    pending
```


<p>那么问题来了&#xff1a;</p>
<blockquote>
<p>“JWT 认证体系改造”这个整体任务现在到底是什么状态&#xff1f;</p>
</blockquote>
<p>显然不能通过某一个 Run 的状态回答。</p>
<p>于是系统需要一个更高层的业务执行对象&#xff1a;</p>


```text
LongTask

JWT 认证体系改造
status = running
progress = 68%
current_stage = testing
```


<p>这就是 LongTask 真正出现的地方。</p>
<hr />
<h2>五、LongTask 的核心不是“执行更久”&#xff0c;而是“推进一个持久化任务状态机”</h2>
<p>普通 Agent Run 的执行思维通常是&#xff1a;</p>


```text
启动 run_agent()
↓
一直执行
↓
直到返回
```


<p>而 LongTask 的运行思想发生了变化&#xff1a;</p>


```text
读取 LongTask 当前状态
↓
判断哪些 Step 可以执行
↓
调度对应执行单元
↓
保存结果
↓
更新任务状态
↓
决定下一步
```


<p>当前 Worker 并不需要把整个任务执行到底。</p>
<p>它甚至可能只负责&#xff1a;</p>


```text
执行一个 Step
```


<p>执行完成以后释放。</p>
<p>下一步可能由另一台机器上的 Worker 继续。</p>
<p>所以&#xff1a;</p>


```text
Agent Run
```


<p>更接近&#xff1a;</p>
<blockquote>
<p>执行一次 Agent Loop。</p>
</blockquote>
<p>而&#xff1a;</p>


```text
LongTask
```


<p>更接近&#xff1a;</p>
<blockquote>
<p>推进一个 Persistent Task State Machine。</p>
</blockquote>
<p>这就是二者在代码和架构层面真正重要的分界。</p>
<hr />
<h2>六、为什么 Durable Execution 对 Coding Agent 特别重要</h2>
<p>大型 Coding Agent 的执行天然具有很多长时任务特征。</p>
<p>例如一次大型仓库升级&#xff1a;</p>
<blockquote>
<p>Java 17 升级 Java 21&#xff0c;同时升级 Spring Boot&#xff0c;修复 Breaking Changes&#xff0c;修改 Docker、CI 和测试&#xff0c;并提交 PR。</p>
</blockquote>
<p>任务可能持续几十分钟。</p>
<p>但这里“长”的并不只是计算时间。</p>
<p>更重要的是&#xff0c;它中间会不断发生&#xff1a;</p>


```text
执行代码修改
↓
等待测试
↓
修复失败
↓
等待 CI
↓
等待人工确认
↓
继续执行
```


<p>这意味着任务生命周期可能持续数小时&#xff0c;但是实际占用 Agent Worker 的时间只有其中一部分。</p>
<p>一个 Durable LongTask 应该允许&#xff1a;</p>


```text
10:00
Agent 完成代码修改

10:10
进入 waiting_ci

此时没有 Worker 持续执行

10:30
CI Callback

10:31
新的 Worker 恢复任务

10:40
进入 waiting_user

14:00
用户批准

14:01
再次恢复任务
```


<p>整个任务存在四个小时。</p>
<p>但没有任何线程需要连续运行四个小时。</p>
<p>这就是&#xff1a;</p>
<blockquote>
<p><strong>Long-running 不等于 Continuous Running。</strong></p>
</blockquote>
<p>真正需要的是&#xff1a;</p>
<blockquote>
<p><strong>Durable Execution。</strong></p>
</blockquote>
<hr />
<h2>七、Human-in-the-loop&#xff1a;LongTask 为什么能够“睡着以后再醒来”</h2>
<p>LangGraph 当前将 Human-in-the-loop 与 Durable Execution、Persistence、Streaming 一起视为 Agent orchestration runtime 的核心能力。</p>
<p>Human-in-the-loop 在 Coding Agent 中非常典型。</p>
<p>假设 Agent 分析以后发现&#xff1a;</p>
<blockquote>
<p>必须修改 users 表结构。</p>
</blockquote>
<p>这种操作风险较高&#xff0c;于是 LongTask 进入&#xff1a;</p>


```text
waiting_user
```


<p>前端展示&#xff1a;</p>


```text
Agent 准备执行数据库 Migration。

影响：
users
sessions

[查看 Migration]
[批准]
[拒绝]
```


<p>这时系统不应该让一个 Worker&#xff1a;</p>


```text
await approval...
```


<p>等几个小时。</p>
<p>而应该&#xff1a;</p>


```text
保存状态
↓
创建 Checkpoint
↓
LongTask = waiting_user
↓
释放 Worker
```


<p>LangGraph 的 <code>interrupt()</code> 就是类似机制&#xff1a;执行到 interrupt 时&#xff0c;当前 Graph State 会通过 persistence layer 保存&#xff0c;随后执行可以无限期暂停&#xff0c;直到外部再次传入 Command 恢复。</p>
<p>甚至用户第二天才点击批准&#xff0c;也没有问题。</p>
<p>OpenAI Agents SDK 的 <code>RunState</code> 同样支持将待审批 Run 序列化到数据库或 Queue&#xff0c;之后再恢复&#xff0c;这正是长时间 Human-in-the-loop 场景需要的能力。</p>
<p>所以 Human-in-the-loop 真正难的并不是 UI 上的两个按钮。</p>
<p>而是&#xff1a;</p>
<blockquote>
<p><strong>任务能不能暂停几个小时甚至几天以后&#xff0c;从正确的位置继续。</strong></p>
</blockquote>
<hr />
<h2>八、Replay / Resume&#xff1a;系统是怎么“恢复”的</h2>
<p>Durable Execution 还有一个非常重要的概念&#xff1a;</p>


```text
Replay
```


<p>或者&#xff1a;</p>


```text
Resume
```


<p>不同系统的实现方式不同。</p>
<p>LangGraph 更强调&#xff1a;</p>


```text
Checkpoint
+
Graph State
+
Resume
```


<p>而 Temporal 更强调&#xff1a;</p>


```text
Event History
+
Replay
```


<p>Temporal 会持久化 Workflow Event History。当 Worker 重新获得 Workflow Task 时&#xff0c;可以根据历史事件重新执行 Workflow Definition&#xff0c;并检查新的 Commands 是否与历史一致&#xff0c;从而重建 Workflow 当前状态。</p>
<p>可以简单理解为&#xff1a;</p>


```text
过去发生过：

仓库分析完成
后端修改完成
前端修改完成
CI Started
```


<p>这些事实已经持久化。</p>
<p>Worker 崩溃以后&#xff1a;</p>


```text
新的 Worker
↓
Replay Event History
↓
恢复任务状态
↓
继续等待 CI
```


<p>而不是&#xff1a;</p>


```text
重新分析仓库
重新修改后端
重新修改前端
```


<p>因此一个真正 Durable 的 LongTask&#xff0c;真正保存的不是&#xff1a;</p>
<blockquote>
<p>“一个 Python 函数现在运行到了第几行。”</p>
</blockquote>
<p>而是&#xff1a;</p>
<blockquote>
<p>“这个业务执行过程中&#xff0c;哪些事实已经发生并被确认。”</p>
</blockquote>
<hr />
<h2>九、Streaming&#xff1a;长任务为什么必须让用户看到“现在做到哪里了”</h2>
<p>LongTask 还有一个经常被忽略的能力&#xff1a;</p>


```text
Streaming
```


<p>它和 Durable Execution 解决的是完全不同的问题。</p>
<p>Durable Execution 解决&#xff1a;</p>
<blockquote>
<p>后台任务怎么可靠运行&#xff1f;</p>
</blockquote>
<p>Streaming 解决&#xff1a;</p>
<blockquote>
<p>用户怎么知道后台正在发生什么&#xff1f;</p>
</blockquote>
<p>大型 Coding Agent 如果只显示&#xff1a;</p>


```text
Agent 正在工作……
```


<p>然后持续 30 分钟&#xff0c;用户体验会非常差。</p>
<p>更合理的是&#xff1a;</p>


```text
JWT 认证体系改造                        Running

✓ 仓库分析
✓ 方案设计

代码改造
✓ Backend Agent
✓ Frontend Agent
✓ Database Migration

测试验证
● Integration Test
已执行 128 / 183

○ CI
○ 人工确认
○ 生成迁移文档
```


<p>LangGraph 官方将 Streaming 和 Durable Execution、Human-in-the-loop、Persistence 一同列为 orchestration runtime 的核心能力。</p>
<p>所以 LongTask 通常会持续产生事件&#xff1a;</p>


```text
longtask.started

stage.started

step.started
step.progress
step.completed

agent.run.started
agent.run.completed

ci.waiting
ci.completed

approval.required
approval.resolved

artifact.created

longtask.completed
```


<p>这些事件可以写入 Event Log&#xff0c;然后通过&#xff1a;</p>


```text
SSE / streamUrl
```


<p>推给 Agent UI。</p>
<hr />
<h2>十、Persistence &#43; Durable Execution &#43; HITL &#43; Streaming 是如何串起来的</h2>
<p>现在可以把这些概念全部连起来。</p>
<p>假设 Coding Agent 正在执行一次框架升级&#xff1a;</p>


```text
LongTask
Spring Boot 升级
```


<p>Runtime 执行&#xff1a;</p>


```text
Step 1
仓库分析
```


<p>完成后&#xff1a;</p>


```text
Persistence
保存 Task State

Checkpoint
记录 Step 1 completed
```


<p>然后&#xff1a;</p>


```text
Step 2
修改后端
```


<p>执行过程中&#xff1a;</p>


```text
Streaming
实时把修改进度推给 UI
```


<p>修改完成&#xff1a;</p>


```text
Checkpoint
Step 2 completed
```


<p>随后进入&#xff1a;</p>


```text
Human-in-the-loop

等待用户确认 Migration
```


<p>系统暂停。</p>
<p>几个小时以后用户批准&#xff1a;</p>


```text
Resume
```


<p>如果中间 Worker 重启&#xff1a;</p>


```text
Replay / State Recovery
```


<p>恢复执行。</p>
<p>最终&#xff1a;</p>


```text
LongTask completed
```


<p>所以它们之间并不是互相独立的功能。</p>
<p>而是一条完整的可靠执行链&#xff1a;</p>


```text
Persistence
↓
Checkpoint

↓

Durable Execution

↓

Interrupt / Human-in-the-loop

↓

Resume / Replay

↓

Streaming

↓

用户持续观察整个任务
```


<hr />
<h2>十一、为什么大型 Coding Agent 最终会变成 Agent &#43; Workflow</h2>
<p>普通 Coding Agent 最自然的结构是&#xff1a;</p>


```text
LLM
↓
Tool
↓
LLM
↓
Tool
```


<p>但一个大型仓库迁移中&#xff0c;有很多工作实际上并不需要 LLM 决策。</p>
<p>例如&#xff1a;</p>


```text
npm install
mvn test
pytest
docker build
git diff
运行 lint
等待 CI
```


<p>这些都是确定性任务。</p>
<p>因此成熟 LongTask 往往不会让 LLM 控制所有流程。</p>
<p>更好的结构是&#xff1a;</p>


```text
Workflow
负责整体执行骨架

Agent
负责需要智能判断的 Step
```


<p>例如&#xff1a;</p>


```text
分析 Repository
↓
Agent Run

修改 Backend
↓
Agent Run

修改 Frontend
↓
Agent Run

执行 Unit Test
↓
Worker Task

测试失败？
├── 否 → 下一步
└── 是
↓
Coding Agent Run
↓
再运行测试

等待 CI
↓
Waiting

人工批准
↓
Interrupt / Resume

创建 PR
↓
Artifact
```


<p>这就是为什么现代 Agent orchestration framework 会特别强调&#xff1a;</p>
<blockquote>
<p>deterministic workflow steps 和 agentic steps 可以组合。</p>
</blockquote>
<p>LangGraph 官方也明确把“deterministic steps 与 LLM-driven agentic steps 混合在同一 Graph”作为其核心 orchestration 能力之一。</p>
<hr />
<h2>十二、LongTask 为什么需要 Step 级恢复</h2>
<p>假设 Spring Boot 升级任务执行到了&#xff1a;</p>


```text
✓ Repository Analysis

✓ Dependency Upgrade

✓ Backend Migration

✓ Frontend Migration

✗ Integration Test

○ Documentation

○ PR
```


<p>如果整个 LongTask 重新执行&#xff1a;</p>


```text
重新分析 Repository

重新升级依赖

重新修改 Backend

重新修改 Frontend
```


<p>既浪费资源&#xff0c;也可能引入新的代码变化。</p>
<p>正确做法应该是&#xff1a;</p>


```text
Integration Test failed
↓
创建 Fix Test Step
↓
Coding Agent Run
↓
再次 Integration Test
```


<p>已经完成的 Step 不重新执行。</p>
<p>这就是&#xff1a;</p>


```text
Step-level Recovery
```


<p>而不是&#xff1a;</p>


```text
Run-level Retry Only
```


<p>LangGraph 的 checkpoint 机制本身就围绕状态恢复和 fault tolerance 构建&#xff1b;其生产文档也指出&#xff0c;在 failure、timeout 或 human-in-the-loop pause 后&#xff0c;可以从最近记录的状态恢复&#xff0c;而无需重新执行此前已经完成的工作。</p>
<hr />
<h2>十三、LongTask 里为什么还需要普通 Agent Run</h2>
<p>讲到这里&#xff0c;很容易又产生一个误解&#xff1a;</p>
<blockquote>
<p>那是不是有 LongTask 以后就不需要 Run 了&#xff1f;</p>
</blockquote>
<p>恰恰相反。</p>
<p>LongTask 中大量智能 Step 本身就是普通 Run。</p>
<p>例如&#xff1a;</p>


```text
LongTask
“升级认证体系”

├── Step
│   Repository Analysis
│      ↓
│   Agent Run
│
├── Step
│   Backend Migration
│      ↓
│   Coding Agent Run
│
├── Step
│   Frontend Migration
│      ↓
│   Coding Agent Run
│
├── Step
│   Unit Test
│      ↓
│   Worker Task
│
├── Step
│   Fix Test
│      ↓
│   Agent Run
│
├── Waiting
│   CI
│
├── Approval
│
└── Artifact
Pull Request
```


<p>因此更准确的技术关系是&#xff1a;</p>


```text
LongTask Runtime
↓
Step / Stage / Dependency / Scheduler
↓
Agent Run / Worker Task / Approval / Waiting
↓
Agent Runtime
↓
LLM / Tool / RAG / Memory / Skill
```


<p>这里各层职责非常清晰。</p>
<p><strong>Agent Runtime</strong> 解决&#xff1a;</p>
<blockquote>
<p>这一次 Agent 应该怎么思考和执行&#xff1f;</p>
</blockquote>
<p><strong>LongTask Runtime</strong> 解决&#xff1a;</p>
<blockquote>
<p>这么多执行单元应该怎么编排、暂停、恢复、重试和最终完成&#xff1f;</p>
</blockquote>
<hr />
<h2>十四、什么时候其实不需要 LongTask</h2>
<p>LongTask 不是越多越好。</p>
<p>如果你的 Coding Agent 大部分需求都是&#xff1a;</p>


```text
修一个 Bug
解释一段代码
增加一个接口
修改一个函数
补几个测试
```


<p>那么&#xff1a;</p>


```text
Agent Run
+
Checkpoint
+
Event Log
+
SSE
```


<p>已经完全足够。</p>
<p>即使 Run 偶尔执行五分钟&#xff0c;也没有必要额外建立复杂的 LongTask Domain Model。</p>
<p>真正判断是否需要 LongTask 的标准不是&#xff1a;</p>
<blockquote>
<p>任务运行超过多少分钟&#xff1f;</p>
</blockquote>
<p>而是&#xff1a;</p>
<blockquote>
<p><strong>一次 Agent Run 还能不能完整表达这个用户目标&#xff1f;</strong></p>
</blockquote>
<p>如果一个目标已经开始包含&#xff1a;</p>


```text
多个 Agent Run
多个 Worker Task
并行 SubTask
等待外部事件
Human Approval
Step Dependency
部分失败恢复
```


<p>那么这个目标已经明显高于 Run 层级。</p>
<p>此时 LongTask 才成为一个非常有价值的业务抽象。</p>
<hr />
<h2>十五、最终推荐的 Coding Agent 架构</h2>
<p>对于复杂 Coding Agent&#xff0c;可以采用这样的层次&#xff1a;</p>


```text
Repository / Workspace
↓
Conversation
↓
User Request
↓
LongTask
↓
Stage
↓
Step
↓
┌─────────────────────────────┐
│ Agent Run                   │
│ Worker Task                 │
│ Subagent Run                │
│ Approval                    │
│ Waiting                     │
└─────────────────────────────┘
↓
Agent Runtime
↓
LLM / Tool / RAG / Memory / Skill
```


<p>LongTask 周围再建立&#xff1a;</p>


```text
Persistence
Checkpoint Store
Event Log
Scheduler
Recovery Engine
Artifact Store
Trace / Metrics
SSE Gateway
```


<p>最终形成&#xff1a;</p>


```text
             LongTask State
│
┌──────────┼──────────┐
↓          ↓          ↓
Persistence   Scheduler   Event Log
│          │          │
Checkpoint       Step       SSE
│          │          │
└──── Recovery ───────┘
│
Durable Execution
```


<hr />
<h2>结语</h2>
<p>普通 Agent Run 和 Agent LongTask 之间&#xff0c;并不存在一道绝对的技术分界线。</p>
<p>普通 Run 完全可以增加&#xff1a;</p>


```text
Persistence
Checkpoint
后台执行
Retry
SSE
```


<p>而且对于大量 Agent 场景&#xff0c;这已经足够。</p>
<p>但随着任务复杂度不断提高&#xff0c;系统会逐渐出现&#xff1a;</p>


```text
多个 Run
多个 Step
并行 SubTask
外部 Waiting
Human-in-the-loop
复杂 Dependency
Step 级恢复
```


<p>这时候真正需要升级的就不再是 Agent Loop&#xff0c;而是任务编排层。</p>
<p>于是&#xff1a;</p>


```text
Agent Run
```


<p>从“整个任务”&#xff0c;变成&#xff1a;</p>


```text
LongTask 中的一个执行单元
```


<p>而 LongTask 成为&#xff1a;</p>
<blockquote>
<p><strong>代表用户完整工程目标&#xff0c;并管理它从创建、执行、暂停、失败、恢复到最终完成的持久化业务执行对象。</strong></p>
</blockquote>
<p>如果用 Coding Agent 来概括&#xff1a;</p>
<blockquote>
<p><strong>“修一个 Bug”通常是一次 Agent Run。</strong></p>
</blockquote>
<p>而&#xff1a;</p>
<blockquote>
<p><strong>“完成一次大型代码库迁移”通常是一个 LongTask。</strong></p>
</blockquote>
<p>大型迁移任务里面可以包含很多普通 Run。</p>
<p>这也是理解两者最重要的一句话&#xff1a;</p>
<blockquote>
<p><strong>LongTask 不是更长的 Run&#xff0c;而是 Run 上层的 Durable Task Orchestration。</strong></p>
</blockquote>
<p>其中 Persistence 让状态能够保存&#xff0c;Checkpoint 建立恢复边界&#xff0c;Durable Execution 让任务能够跨故障继续&#xff0c;Human-in-the-loop 让任务能够安全暂停&#xff0c;Replay / Resume 让执行重新恢复&#xff0c;而 Streaming 则让用户持续看到整个长任务正在发生什么。</p>
<p>这些能力组合起来以后&#xff0c;Coding Agent 才真正从&#xff1a;</p>


```text
“会修改代码的 Agent”
```


<p>逐渐走向&#xff1a;</p>


```text
“能够可靠完成完整软件工程任务的智能执行系统”
```


<h3>参考资料</h3>
<p>LangGraph 官方将自身定位为 Agent orchestration runtime&#xff0c;并把 Durable Execution、Streaming、Human-in-the-loop 和 Persistence 作为核心能力。</p>
<p>LangGraph Persistence 文档详细说明了 Checkpointer、Checkpoint、Thread State&#xff0c;以及它们如何支持 fault tolerance、conversation continuity、human-in-the-loop 和 time travel。</p>
<p>LangGraph Interrupts 文档说明了 Agent 如何保存状态、无限期暂停&#xff0c;并在外部输入到达后 Resume&#xff0c;非常适合人工审批等长时等待场景。</p>
<p>OpenAI Agents SDK 的 Human-in-the-loop 文档说明 <code>RunState</code> 可以序列化保存到数据库或 Queue&#xff0c;并在之后重新恢复&#xff1b;SDK 也提供 Temporal、Dapr、Restate、DBOS 等 durable orchestration 集成&#xff0c;用于 long-running agents、process restart 和长时间等待场景。</p>
<p>Temporal 官方将 Workflow Execution 定义为 durable、reliable、scalable 的函数执行&#xff0c;并通过持久化 Event History 与 Replay 在 Worker 或基础设施故障后恢复工作流。</p>
