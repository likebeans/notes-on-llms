---
title: "Agent 到底该怎么评？从 RAG 指标到科学的 Agent Evaluation 体系"
description: "CSDN 原文全文镜像：这个 Agent 到底怎样才算完成任务？case:input:messages:content: \"帮我退掉订单 A123\"order:id: A123expected:outcome:required:forbidden:respon……"
pageType: article
module: agent
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "数据库"
  - "服务器"
  - "网络"
  - "大模型"
  - "人工智能"
  - "软件工程"
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

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/164094411](https://blog.csdn.net/m0_63309778/article/details/164094411)
- 站内分区：Agent / Agent Evaluation
:::

<p><img src="https://i-blog.csdnimg.cn/direct/48fad3bd08fc42c7ae3471264a02ecd6.png" alt="" /></p>
<h3>前言&#xff1a;我们真的会评测 Agent 吗&#xff1f;</h3>
<p>过去做 RAG 时&#xff0c;我们谈评测&#xff0c;通常会想到这些指标&#xff1a;</p>
<ul><li>Context Precision</li><li>Context Recall</li><li>Faithfulness</li><li>Answer Relevance</li><li>Answer Correctness</li></ul>
<p>这套方法在 RAG 系统中已经比较成熟。</p>
<p>但进入 Agent 时代后&#xff0c;一个很明显的问题出现了&#xff1a;</p>
<blockquote>
<p><strong>Agent 还能继续用“输入—输出”这一套方式来评吗&#xff1f;</strong></p>
</blockquote>
<p>答案显然是否定的。</p>
<p>一个真实 Agent 的执行过程可能是&#xff1a;</p>


```text
User Task
↓
Intent Recognition
↓
Planning / Routing
↓
LLM
↓
Tool Selection
↓
Tool Arguments
↓
Tool Execution
↓
Environment State Change
↓
Observation
↓
Memory / Context Update
↓
Retry / Handoff / Guardrail
↓
Final Response
```


<p>这里的任何一步都可能出错。</p>
<p>例如用户说&#xff1a;</p>
<blockquote>
<p>帮我把订单 A123 取消掉。</p>
</blockquote>
<p>Agent 最后回复&#xff1a;</p>
<blockquote>
<p>已经帮你取消成功了。</p>
</blockquote>
<p>从“最终回答”看&#xff0c;这似乎是一个完美答案。</p>
<p>但真实情况可能是&#xff1a;</p>


```text
数据库订单状态：SHIPPED
取消接口：调用失败
退款记录：不存在
```


<p>Agent 只是“说自己完成了”。</p>
<p>这时候&#xff0c;如果我们只用 LLM-as-Judge 评价最终回答&#xff0c;很可能得到一个非常高的分数。</p>
<p>但从真实系统角度看&#xff1a;</p>
<blockquote>
<p><strong>这个 Agent 是彻底失败的。</strong></p>
</blockquote>
<p>因此&#xff0c;Agent Evaluation 和传统 LLM Evaluation 最大的区别并不是“多几个 Tool 指标”。</p>
<p>而是&#xff1a;</p>
<blockquote>
<p><strong>评测对象已经从模型输出&#xff0c;变成了一个具有状态、工具、环境和随机性的动态软件系统。</strong></p>
</blockquote>
<p>这篇文章想讨论的就是这个问题&#xff1a;</p>
<p><strong>怎样建立一套真正科学、可解释、可复现的 Agent 与 RAG Evaluation 体系&#xff1f;</strong></p>
<hr />
<h2>一、先搞清楚&#xff1a;我们到底在评什么&#xff1f;</h2>
<p>传统 LLM 应用可以近似抽象成&#xff1a;</p>


```text
Input → Model → Output
```


<p>RAG 增加了一个 Retrieval Pipeline&#xff1a;</p>


```text
Query
↓
Retriever
↓
Reranker
↓
Context
↓
LLM
↓
Answer
```


<p>而 Agent 更接近&#xff1a;</p>


```text
Task
↓
Reasoning
↓
Actions
↓
Environment
↓
New State
↓
Observation
↓
Next Action
↓
...
↓
Outcome
```


<p>这三个系统对应的评测对象其实完全不同。</p>

<table><thead><tr><th>系统</th><th>主要评测对象</th></tr></thead><tbody><tr><td>LLM</td><td>Output</td></tr><tr><td>RAG</td><td>Retrieval &#43; Generation</td></tr><tr><td>Agent</td><td>Outcome &#43; Trajectory &#43; Tool &#43; State &#43; Output</td></tr></tbody></table><p>因此&#xff0c;第一个非常重要的结论是&#xff1a;</p>
<blockquote>
<p><strong>Agent Evaluation 不能只是 LLM Evaluation 的指标扩展&#xff0c;而应该被视为一种系统级测试。</strong></p>
</blockquote>
<p>现在很多 Agent 评测框架也已经明显朝 trace-based evaluation 演进。</p>
<p>例如 DeepEval 当前把 Agent 评测拆为 Reasoning、Action 和 Execution 三层&#xff0c;分别评价 Plan Quality、Tool Correctness、Argument Correctness、Task Completion 和 Step Efficiency&#xff0c;并且这些指标依赖完整执行 Trace。</p>
<p>Phoenix 则更进一步&#xff0c;把 Tracing、Dataset、Experiment、Evaluation 和 Human Annotation 放在同一个闭环里&#xff0c;并基于 OpenTelemetry/OpenInference 做 Trace 基础设施。</p>
<p>这说明 Agent Evaluation 的基础对象正在逐渐从&#xff1a;</p>


```text
prompt + answer
```


<p>转变成&#xff1a;</p>


```text
task
+ trace
+ environment
+ outcome
```


<hr />
<h2>二、Agent 评测最重要的原则&#xff1a;Outcome First</h2>
<p>很多 Agent Evaluation Framework 首先想到的是&#xff1a;</p>


```text
Tool Correctness
Trajectory
Task Completion Judge
```


<p>这些当然重要。</p>
<p>但我认为真正的第一指标应该是&#xff1a;</p>
<h2>Outcome</h2>
<p>也就是&#xff1a;</p>
<blockquote>
<p><strong>Agent 最终有没有真的把事情办成。</strong></p>
</blockquote>
<p>例如退款 Agent&#xff1a;</p>


```text
用户目标：
退款订单 A123
```


<p>最终真正应该验证的不是&#xff1a;</p>


```text
assistant_response ==
"已经帮你退款成功"
```


<p>而是&#xff1a;</p>


```text
return_request.status == CREATED
order.id == A123
refund.amount == expected_amount
```


<p>对于 Coding Agent&#xff1a;</p>


```text
不是：
“代码已经帮你修改好了”
```


<p>而是&#xff1a;</p>


```text
unit_test_pass == true
integration_test_pass == true
build_success == true
```


<p>对于 Browser Agent&#xff1a;</p>


```text
不是：
“会议已经预约成功”
```


<p>而是&#xff1a;</p>


```text
calendar_event.exists == true
calendar_event.time == expected_time
```


<p>对于文件 Agent&#xff1a;</p>


```text
file.exists == true
file.content satisfies requirements
```


<p>这也是 τ-bench / 当前 τ³-bench 这一类 benchmark 最值得借鉴的地方。</p>
<p>τ³-bench 不只是给模型一个问题然后看最终回复&#xff0c;而是提供&#xff1a;</p>
<ul><li>Domain Policy&#xff1b;</li><li>Tools&#xff1b;</li><li>Tasks&#xff1b;</li><li>Environment&#xff1b;</li><li>User Simulator&#xff1b;</li></ul>
<p>然后真正运行 Agent 与环境之间的交互。当前版本还已经增加 Knowledge Retrieval 和 Voice 等场景。</p>
<p>因此&#xff1a;</p>
<blockquote>
<p><strong>Agent 的 Task Success 应尽可能来自 Environment Verification&#xff0c;而不是 Agent 自己的语言输出。</strong></p>
</blockquote>
<hr />
<h2>三、Trajectory 很重要&#xff0c;但不要迷信“标准路径”</h2>
<p>既然最终状态重要&#xff0c;是不是只看 Outcome 就够了&#xff1f;</p>
<p>也不是。</p>
<p>假设两个 Agent 都成功退款&#xff1a;</p>


```text
Agent A

get_order
→ check_policy
→ create_return
```


<p>另一个 Agent&#xff1a;</p>


```text
Agent B

get_order
→ get_user
→ search_order
→ get_order
→ check_policy
→ retry
→ get_order
→ create_return
```


<p>两者最终 Outcome 都是成功。</p>
<p>但显然 A 更优秀。</p>
<p>所以 Agent 还需要评价&#xff1a;</p>


```text
Trajectory
```


<p>包括&#xff1a;</p>
<ul><li>是否选择正确工具&#xff1b;</li><li>Tool 参数是否正确&#xff1b;</li><li>是否遗漏关键步骤&#xff1b;</li><li>是否违反执行顺序&#xff1b;</li><li>是否调用禁止工具&#xff1b;</li><li>是否出现无意义循环&#xff1b;</li><li>是否发生重复调用&#xff1b;</li><li>是否存在不必要步骤。</li></ul>
<p>DeepEval 当前已经提供 Tool Correctness、Argument Correctness、Plan Adherence 和 Step Efficiency 等指标&#xff0c;这正是在尝试解决这个问题。</p>
<p>但是这里容易出现另外一个误区&#xff1a;</p>
<blockquote>
<p><strong>要求 Agent 完全走 Ground Truth Trajectory。</strong></p>
</blockquote>
<p>例如&#xff1a;</p>


```text
Expected:

A → B → C
```


<p>实际&#xff1a;</p>


```text
A → D → B → C
```


<p>如果 D 是一个合理的辅助查询&#xff0c;这个 Agent 不应该因为 Exact Match 失败而被判错。</p>
<p>因此&#xff0c;更合理的方式不是&#xff1a;</p>


```text
trajectory == expected_trajectory
```


<p>而是定义&#xff1a;</p>
<h2>Trajectory Constraints</h2>
<p>例如&#xff1a;</p>


```yaml
<span class="token key atrule">trajectory</span><span class="token punctuation">:</span>

<span class="token key atrule">required</span><span class="token punctuation">:</span>
<span class="token punctuation">-</span> get_order
<span class="token punctuation">-</span> check_refund_policy

<span class="token key atrule">partial_order</span><span class="token punctuation">:</span>
<span class="token punctuation">-</span> get_order < check_refund_policy
<span class="token punctuation">-</span> check_refund_policy < create_return_request

<span class="token key atrule">forbidden</span><span class="token punctuation">:</span>
<span class="token punctuation">-</span> force_refund
<span class="token punctuation">-</span> modify_database_directly

<span class="token key atrule">max_calls</span><span class="token punctuation">:</span>
<span class="token key atrule">get_order</span><span class="token punctuation">:</span> <span class="token number">2</span>
```


<p>我们真正要评价的是&#xff1a;</p>
<blockquote>
<p>Agent 有没有遵守任务执行中的关键不变量。</p>
</blockquote>
<p>而不是&#xff1a;</p>
<blockquote>
<p>Agent 有没有背出我们提前写好的标准答案。</p>
</blockquote>
<hr />
<h2>四、Agent Evaluation 最适合使用“五层评测模型”</h2>
<p>为了避免所有东西最后压缩成一个所谓的“Agent 智能分”&#xff0c;我更倾向于将评测对象拆成五层。</p>
<h3>第一层&#xff1a;Output</h3>
<p>评价最终回复&#xff1a;</p>


```text
Correctness
Completeness
Helpfulness
Groundedness
Format
Safety
```


<p>这是传统 LLM Evaluation 最擅长的部分。</p>
<hr />
<h3>第二层&#xff1a;Tool Call</h3>
<p>评价单次 Action&#xff1a;</p>


```text
Tool Selection
Argument Correctness
Schema Validity
Execution Success
Tool Output Utilization
```


<p>其中大量指标应该直接使用代码完成。</p>
<p>例如&#xff1a;</p>


```text
tool_name == expected
```


<p>根本没必要调用 LLM Judge。</p>
<p>参数类型&#xff1a;</p>


```text
order_id: string
```


<p>也应该使用 JSON Schema Validation。</p>
<p>原则应该是&#xff1a;</p>
<blockquote>
<p><strong>能够 deterministic evaluation 的问题&#xff0c;不要交给 LLM。</strong></p>
</blockquote>
<hr />
<h3>第三层&#xff1a;Trace</h3>
<p>评价整个执行路径&#xff1a;</p>


```text
Required Steps
Forbidden Steps
Partial Order
Redundant Calls
Loop
Retry
Recovery
Efficiency
```


<p>这回答的是&#xff1a;</p>
<blockquote>
<p>Agent 为什么成功&#xff0c;或者为什么失败&#xff1f;</p>
</blockquote>
<hr />
<h3>第四层&#xff1a;Session</h3>
<p>真实 Agent 很少只是一次请求。</p>
<p>Session 层关注&#xff1a;</p>


```text
Multi-turn Goal Completion
Context Consistency
Intent Resolution
Clarification Quality
User Simulation
Long-horizon Reliability
```


<p>例如&#xff1a;</p>


```text
User:
我想退货

Agent:
订单号是多少？

User:
A123

Agent:
查询订单……

User:
算了，我想换货。
```


<p>这种场景已经不能用简单 Single-turn Metric 表达。</p>
<hr />
<h3>第五层&#xff1a;System</h3>
<p>最后才是整个 Agent 产品是否真的可以上线&#xff1a;</p>


```text
Task Success Rate
Safety Violation Rate
Pass^k
Latency
Cost
Timeout Rate
Retry Rate
Regression Rate
Cost Per Success
```


<p>这五层的关系可以理解成&#xff1a;</p>


```text
System
↑
Session
↑
Trace
↑
Tool
↑
Output
```


<p>下层负责诊断。</p>
<p>上层负责决策。</p>
<hr />
<h2>五、不要让一个“总分”毁掉整个 Evaluation</h2>
<p>这是很多 AI Evaluation 产品很容易犯的问题。</p>
<p>比如&#xff1a;</p>


```text
Agent Score = 87.6
```


<p>看起来非常直观。</p>
<p>但实际上&#xff1a;</p>
<blockquote>
<p>87.6 到底是什么意思&#xff1f;</p>
</blockquote>
<p>假设&#xff1a;</p>


```text
Agent A

Task Completion        96%
Tool Accuracy          82%
Safety                  91%
```


<p>另一个&#xff1a;</p>


```text
Agent B

Task Completion        92%
Tool Accuracy          97%
Safety                 100%
```


<p>哪个更好&#xff1f;</p>
<p>如果这是&#xff1a;</p>


```text
调研 Agent
```


<p>也许 A 可以接受。</p>
<p>如果这是&#xff1a;</p>


```text
银行转账 Agent
```


<p>A 可能根本不能上线。</p>
<p>因此&#xff0c;Evaluation Platform 最终应该提供&#xff1a;</p>


```text
Metrics
+
Quality Gates
```


<p>而不是只有&#xff1a;</p>


```text
Overall Score
```


<p>例如&#xff1a;</p>


```yaml
<span class="token key atrule">quality_gate</span><span class="token punctuation">:</span>

<span class="token key atrule">hard</span><span class="token punctuation">:</span>
<span class="token key atrule">forbidden_tool_call_rate</span><span class="token punctuation">:</span> <span class="token number">0</span>
<span class="token key atrule">permission_violation_rate</span><span class="token punctuation">:</span> <span class="token number">0</span>
<span class="token key atrule">pii_leakage_rate</span><span class="token punctuation">:</span> <span class="token number">0</span>

<span class="token key atrule">outcome</span><span class="token punctuation">:</span>
<span class="token key atrule">task_success_rate</span><span class="token punctuation">:</span>
<span class="token key atrule">min</span><span class="token punctuation">:</span> <span class="token number">0.90</span>

<span class="token key atrule">reliability</span><span class="token punctuation">:</span>
<span class="token key atrule">pass_power_3</span><span class="token punctuation">:</span>
<span class="token key atrule">min</span><span class="token punctuation">:</span> <span class="token number">0.80</span>

<span class="token key atrule">performance</span><span class="token punctuation">:</span>
<span class="token key atrule">p95_latency</span><span class="token punctuation">:</span>
<span class="token key atrule">max</span><span class="token punctuation">:</span> <span class="token number">8000</span>

<span class="token key atrule">cost</span><span class="token punctuation">:</span>
<span class="token key atrule">cost_per_success</span><span class="token punctuation">:</span>
<span class="token key atrule">max</span><span class="token punctuation">:</span> <span class="token number">0.05</span>
```


<p>如果&#xff1a;</p>


```text
Task Success ↑
但是 PII Leakage ↑
```


<p>这个版本依然不能上线。</p>
<hr />
<h2>六、真正科学的 Agent Evaluation 必须面对“随机性”</h2>
<p>Agent 是非确定性系统。</p>
<p>相同的&#xff1a;</p>


```text
Prompt
Model
Tools
Task
```


<p>连续运行十次&#xff0c;可能得到&#xff1a;</p>


```text
Success
Success
Failure
Success
Failure
Success
Success
Success
Success
Failure
```


<p>如果只跑第一次&#xff1a;</p>


```text
Success Rate = 100%
```


<p>如果跑十次&#xff1a;</p>


```text
Success Rate = 70%
```


<p>所以&#xff1a;</p>
<h2>Single Run Eval 本身就不够科学。</h2>
<p>τ-bench 一直非常强调 <code>Pass^k</code> 这类可靠性指标&#xff0c;其公开 benchmark 也会报告随着重复执行次数增加的成功稳定性。</p>
<p>因此一个真正面向 Agent 的 Evaluation Framework 应原生支持&#xff1a;</p>


```yaml
<span class="token key atrule">run</span><span class="token punctuation">:</span>
<span class="token key atrule">repetitions</span><span class="token punctuation">:</span> <span class="token number">5</span>
```


<p>然后输出&#xff1a;</p>


```text
Success Rate
Pass@K
Pass^K
Variance
Flaky Rate
Latency Distribution
Cost Distribution
```


<p>这里&#xff1a;</p>
<h4>Pass&#64;K</h4>
<p>回答&#xff1a;</p>
<blockquote>
<p>允许执行 K 次&#xff0c;至少成功一次的能力怎么样&#xff1f;</p>
</blockquote>
<p>更偏 Capability。</p>
<h4>Pass^K</h4>
<p>回答&#xff1a;</p>
<blockquote>
<p>连续 K 次都成功的概率怎么样&#xff1f;</p>
</blockquote>
<p>更偏 Reliability。</p>
<p>对于 Coding Benchmark&#xff0c;Pass&#64;K 很重要。</p>
<p>对于&#xff1a;</p>


```text
支付
退款
审批
企业 Agent
```


<p>Pass^K 往往更加重要。</p>
<p>因为真实用户并不关心&#xff1a;</p>
<blockquote>
<p>“这个 Agent 多试几次总有一次能成功。”</p>
</blockquote>
<p>他们更关心&#xff1a;</p>
<blockquote>
<p><strong>每次使用它是不是都可靠。</strong></p>
</blockquote>
<hr />
<h2>七、除了重复执行&#xff0c;还需要真正的实验设计</h2>
<p>假设&#xff1a;</p>


```text
Agent V1 = 86.8%
Agent V2 = 88.1%
```


<p>我们能不能说&#xff1a;</p>
<blockquote>
<p>V2 提升了 1.3%&#xff0c;效果更好&#xff1f;</p>
</blockquote>
<p>未必。</p>
<p>因为差异可能来自&#xff1a;</p>


```text
随机采样
模型抖动
网络延迟
工具异常
数据污染
Judge 波动
运行环境变化
```


<p>因此真正科学的 Agent Evaluation 至少应该支持&#xff1a;</p>


```text
Repeated Trials
Paired Experiment
Confidence Interval
Variance Analysis
Bootstrap
Regression Detection
```


<p>例如比较两个版本时&#xff1a;</p>


```text
Same Dataset
Same Case
Same Environment Snapshot
Same Corpus Snapshot
Same Resource Limit

↓

Baseline vs Candidate
```


<p>而不是&#xff1a;</p>


```text
V1 今天跑一次
V2 明天换了一批数据再跑一次
```


<p>然后直接比较平均分。</p>
<p>这也是为什么一个新的开源 Evaluation 项目真正值得做的部分&#xff0c;不应该只是再增加几个 LLM Judge Metric。</p>
<p>更重要的是&#xff1a;</p>
<blockquote>
<p><strong>建立 Experiment Runtime。</strong></p>
</blockquote>
<hr />
<h2>八、Evaluator 本身也必须被评测</h2>
<p>还有一个 Evaluation 系统经常被忽略的问题&#xff1a;</p>
<blockquote>
<p>谁来评价 Evaluator&#xff1f;</p>
</blockquote>
<p>比如我们写&#xff1a;</p>


```text
TaskCompletionJudge
```


<p>让某个 LLM 判断 Agent 是否完成任务。</p>
<p>然后得到&#xff1a;</p>


```text
Task Completion = 91%
```


<p>问题是&#xff1a;</p>
<blockquote>
<p>这个 Judge 自己有多准&#xff1f;</p>
</blockquote>
<p>如果 Judge 对 100 条人工标注数据只有&#xff1a;</p>


```text
Accuracy = 76%
```


<p>那么&#xff1a;</p>


```text
Agent Score = 91%
```


<p>实际上并没有想象中那么可信。</p>
<p>因此一个真正成熟的平台应该增加&#xff1a;</p>
<h2>Judge Calibration</h2>
<p>流程&#xff1a;</p>


```text
Human Label
↓
Gold Label
↓
Judge Prediction
↓
Comparison
```


<p>至少计算&#xff1a;</p>


```text
Accuracy
Precision
Recall
F1
Cohen's Kappa
Confusion Matrix
```


<p>同时 Judge 必须版本化&#xff1a;</p>


```text
Judge Model
Prompt
Rubric
Few-shot
Temperature
Output Schema
Version
```


<p>因为&#xff1a;</p>


```text
GPT-X + rubric-v1
```


<p>和&#xff1a;</p>


```text
GPT-X + rubric-v2
```


<p>根本不是同一个 Evaluator。</p>
<hr />
<h2>九、再看 RAG&#xff1a;RAG Evaluation 也不能只有 Faithfulness</h2>
<p>讲完 Agent&#xff0c;再看 RAG。</p>
<p>目前 RAG Evaluation 中 Ragas 是最有代表性的项目之一。</p>
<p>Ragas 当前已经不仅仅提供 Faithfulness 这类 RAG 指标&#xff0c;它正在向 experiments-first 的系统化评测方式扩展&#xff0c;并同时支持 LLM-based 和 deterministic metric。</p>
<p>但如果我们的目标是&#xff1a;</p>
<blockquote>
<p><strong>真正判断一个 RAG 系统到底哪里好、哪里不好。</strong></p>
</blockquote>
<p>我认为至少应该分成四层。</p>
<hr />
<h2>十、RAG 第一层&#xff1a;Retrieval Evaluation</h2>
<p>Retriever 首先应该回到传统 Information Retrieval。</p>
<p>如果我们有&#xff1a;</p>


```text
Query
Relevant Document IDs
```


<p>那么优先使用&#xff1a;</p>


```text
Recall@K
Precision@K
MRR
nDCG@K
Hit Rate
```


<p>而不是直接问 LLM&#xff1a;</p>
<blockquote>
<p>“这些 Chunk 看起来相关吗&#xff1f;”</p>
</blockquote>
<p>例如&#xff1a;</p>


```text
Query:
员工差旅酒店报销标准是多少？

Ground Truth:
doc_124
doc_189
```


<p>Retriever 返回&#xff1a;</p>


```text
doc_124
doc_300
doc_422
doc_189
```


<p>这是一个非常明确的 Ranking 问题。</p>
<p>传统 IR Metric 比 LLM Judge 更&#xff1a;</p>


```text
稳定
便宜
可复现
```


<hr />
<h2>十一、RAG 第二层&#xff1a;Claim-level Diagnosis</h2>
<p>传统 Retriever Metric 又存在一个问题。</p>
<p>一个 Answer 可能需要多个事实&#xff1a;</p>


```text
Claim A
Claim B
Claim C
```


<p>Document 1 支持 A。</p>
<p>Document 2 支持 B。</p>
<p>Document 3 支持 C。</p>
<p>因此单纯 Document Recall 有时候不够。</p>
<p>更进一步应该做到&#xff1a;</p>


```text
Claim → Supporting Evidence
```


<p>然后计算&#xff1a;</p>


```text
Claim Recall
Claim Precision
Unsupported Claim
Missing Claim
```


<p>这样才能回答&#xff1a;</p>
<blockquote>
<p>RAG 到底漏掉了哪个事实&#xff1f;</p>
</blockquote>
<p>而不是&#xff1a;</p>


```text
Faithfulness = 0.81
```


<p>然后工程师不知道下一步应该改什么。</p>
<hr />
<h2>十二、RAG 第三层&#xff1a;Generation Evaluation</h2>
<p>Retriever 找到信息以后&#xff0c;还需要判断 Generator 有没有正确使用。</p>
<p>至少应该包括&#xff1a;</p>


```text
Answer Correctness
Answer Completeness
Faithfulness
Citation Precision
Citation Recall
Citation Entailment
Abstention Correctness
```


<p>这里一个很重要的区别是&#xff1a;</p>
<h4>Correctness</h4>
<p>回答是不是事实正确。</p>
<h4>Faithfulness</h4>
<p>回答是不是来源于提供的 Context。</p>
<p>例如&#xff1a;</p>
<p>Context 没有答案。</p>
<p>模型凭参数记忆回答正确。</p>
<p>那么&#xff1a;</p>


```text
Correctness = High
Faithfulness = Low
```


<p>两者绝对不能混成一个分数。</p>
<hr />
<h2>十三、RAG 第四层&#xff1a;做 Ablation&#xff0c;而不是只看总分</h2>
<p>这是我认为 RAG 科学评测里非常关键&#xff0c;但实际项目中很容易被忽略的一步。</p>
<p>对于同一套 Dataset&#xff0c;可以跑四组实验&#xff1a;</p>


```text
A. Model Only
B. Oracle Context
C. Current RAG
D. Candidate RAG
```


<p>它们分别回答完全不同的问题。</p>
<h3>Model Only</h3>


```text
LLM 不检索时能答多少？
```


<hr />
<h3>Oracle Context</h3>
<p>人工提供正确 Context&#xff1a;</p>


```text
Retriever 完美时，Generator 的能力上限是多少？
```


<hr />
<h3>Current RAG</h3>


```text
现在的完整生产 Pipeline 有多好？
```


<hr />
<h3>Candidate RAG</h3>
<p>例如&#xff1a;</p>


```text
New Embedding
New Chunk Strategy
New Reranker
New Retrieval Strategy
```


<p>到底有没有提升。</p>
<p>通过这四组实验&#xff0c;我们才能区分&#xff1a;</p>


```text
Retriever 问题
Generator 问题
Model Knowledge 问题
Context Organization 问题
```


<p>否则&#xff1a;</p>


```text
Answer Correctness ↓
```


<p>之后大家很容易直接开始&#xff1a;</p>
<blockquote>
<p>换模型。</p>
</blockquote>
<p>但真正的问题也许只是 Retriever 没找到正确文档。</p>
<hr />
<h2>十四、RAG 和 Agent 最终其实应该进入同一套 Evaluation Runtime</h2>
<p>Agentic RAG 出现以后&#xff0c;再分别维护&#xff1a;</p>


```text
RAG Eval Platform
Agent Eval Platform
```


<p>其实会越来越奇怪。</p>
<p>因为一个 Agent Trace 完全可能是&#xff1a;</p>


```text
Agent
|
├─ model
|
├─ retrieval
|
├─ rerank
|
├─ model
|
├─ tool
|
└─ final_response
```


<p>这时 RAG 只是&#xff1a;</p>


```text
Agent Trace 中的一部分。
```


<p>因此一个更加合理的设计是&#xff1a;</p>


```text
Evaluation Core

├── Agent Evaluators
├── RAG Evaluators
├── Tool Evaluators
├── Safety Evaluators
├── Browser Evaluators
├── Code Evaluators
└── Multimodal Evaluators
```


<p>底层全部共享&#xff1a;</p>


```text
Dataset
Case
Trial
Trace
Evaluator
Experiment
```


<hr />
<h2>十五、现在的开源项目已经做到什么程度&#xff1f;</h2>
<p>理解这一点很重要&#xff0c;因为如果我们真的准备做一个新的开源项目&#xff0c;首先必须回答&#xff1a;</p>
<blockquote>
<p>为什么不是直接用现有项目&#xff1f;</p>
</blockquote>
<p>目前整个生态大概可以分成五类。</p>

<table><thead><tr><th>类型</th><th>代表项目</th><th>最值得借鉴的能力</th></tr></thead><tbody><tr><td>Metric / Eval SDK</td><td>Ragas、DeepEval</td><td>指标体系、开发体验</td></tr><tr><td>Eval / Observability Platform</td><td>Phoenix</td><td>Trace、Dataset、Experiment</td></tr><tr><td>CLI / CI / Red Team</td><td>Promptfoo</td><td>YAML、CI、安全测试</td></tr><tr><td>General Eval Harness</td><td>Inspect AI</td><td>Task、Scorer、Agent、Sandbox</td></tr><tr><td>Stateful Benchmark</td><td>τ³-bench、AgentDojo</td><td>Environment、Simulation、真实状态</td></tr></tbody></table><hr />
<h3>Ragas</h3>
<p>更擅长&#xff1a;</p>


```text
Metrics
Dataset
Experiment
RAG Evaluation
```


<p>而且已经支持自定义离散、数值和 Ranking Metric&#xff0c;并区分 LLM-based 与 deterministic metric。</p>
<p>值得学习&#xff1a;</p>
<blockquote>
<p>Metric API 与 RAG Evaluation。</p>
</blockquote>
<hr />
<h3>DeepEval</h3>
<p>更像&#xff1a;</p>
<blockquote>
<p>Pytest for LLM / Agent。</p>
</blockquote>
<p>当前 Agent Evaluation 已经覆盖&#xff1a;</p>


```text
Task Completion
Plan Quality
Plan Adherence
Tool Correctness
Argument Correctness
Step Efficiency
```


<p>而且能够直接利用完整 execution trace。</p>
<p>值得学习&#xff1a;</p>
<blockquote>
<p>Developer Experience。</p>
</blockquote>
<hr />
<h3>Phoenix</h3>
<p>Phoenix 已经非常接近一个完整的 AI Engineering Platform&#xff1a;</p>


```text
Tracing
Evaluation
Dataset
Experiment
Human Annotation
Prompt
```


<p>并基于 OpenTelemetry 与 OpenInference。</p>
<p>值得学习&#xff1a;</p>
<blockquote>
<p>Trace 与 Experiment。</p>
</blockquote>
<p>但也正因为如此&#xff0c;新项目不应该第一天就再造一个 Phoenix。</p>
<hr />
<h3>Promptfoo</h3>
<p>Promptfoo 的优势非常清晰&#xff1a;</p>


```text
CLI
YAML Config
Provider Abstraction
CI/CD
Red Team
```


<p>其 Red Team 模型把测试拆成&#xff1a;</p>


```text
Target
Plugin
Strategy
Context
```


<p>这种插件化攻击生成方式非常适合 Agent Security Evaluation。</p>
<p>值得学习&#xff1a;</p>
<blockquote>
<p>声明式 Case &#43; CI &#43; Security。</p>
</blockquote>
<hr />
<h3>Inspect AI</h3>
<p>Inspect 是目前非常值得研究的 Evaluation Harness。</p>
<p>它本身提供&#xff1a;</p>


```text
Dataset
Task
Solver
Scorer
Tool
Agent
Sandbox
```


<p>并已经支持自定义 Tool、MCP、多 Agent、外部 Coding Agent&#xff0c;以及 Docker、Kubernetes 等 Sandbox。</p>
<p>值得学习&#xff1a;</p>
<blockquote>
<p>Evaluation Runtime 的抽象。</p>
</blockquote>
<hr />
<h3>τ³-bench</h3>
<p>τ³-bench 更重要的不是某几个指标。</p>
<p>而是&#xff1a;</p>
<blockquote>
<p><strong>把 Agent 放进一个真正可以交互和改变状态的模拟业务环境中。</strong></p>
</blockquote>
<p>它提供 Domain、Policy、Tool、Task、User Simulator 和 Environment&#xff0c;并通过多次 Trial 评价 Agent。</p>
<p>值得学习&#xff1a;</p>
<blockquote>
<p>Environment &#43; Scenario &#43; Outcome。</p>
</blockquote>
<hr />
<h2>十六、所以新的开源项目到底应该做什么&#xff1f;</h2>
<p>如果重新做一个&#xff1a;</p>


```text
Ragas + 一些 Agent Metrics
```


<p>意义并不大。</p>
<p>如果重新做&#xff1a;</p>


```text
LLM-as-Judge SDK
```


<p>同样很难形成差异化。</p>
<p>如果重新做&#xff1a;</p>


```text
Phoenix
```


<p>工程量巨大&#xff0c;而且并没有明显优势。</p>
<p>我认为真正值得做的方向是&#xff1a;</p>
<h2>Scientific Evaluation Harness for Production Agents &amp; RAG</h2>
<p>核心不是&#xff1a;</p>
<blockquote>
<p>指标多。</p>
</blockquote>
<p>而是&#xff1a;</p>
<blockquote>
<p><strong>评测方法可信。</strong></p>
</blockquote>
<p>它需要解决四件事。</p>
<hr />
<h2>十七、第一件事&#xff1a;定义统一的 Case Specification</h2>
<p>首先需要一种语言描述&#xff1a;</p>
<blockquote>
<p><strong>这个 Agent 到底怎样才算完成任务&#xff1f;</strong></p>
</blockquote>
<p>例如&#xff1a;</p>


```yaml
<span class="token key atrule">spec_version</span><span class="token punctuation">:</span> <span class="token string">"1"</span>

<span class="token key atrule">case</span><span class="token punctuation">:</span>

<span class="token key atrule">id</span><span class="token punctuation">:</span> refund_shipped_order

<span class="token key atrule">input</span><span class="token punctuation">:</span>
<span class="token key atrule">messages</span><span class="token punctuation">:</span>
<span class="token punctuation">-</span> <span class="token key atrule">role</span><span class="token punctuation">:</span> user
<span class="token key atrule">content</span><span class="token punctuation">:</span> <span class="token string">"帮我退掉订单 A123"</span>

<span class="token key atrule">environment</span><span class="token punctuation">:</span>

<span class="token key atrule">initial_state</span><span class="token punctuation">:</span>
<span class="token key atrule">order</span><span class="token punctuation">:</span>
<span class="token key atrule">id</span><span class="token punctuation">:</span> A123
<span class="token key atrule">status</span><span class="token punctuation">:</span> shipped

<span class="token key atrule">expected</span><span class="token punctuation">:</span>

<span class="token key atrule">outcome</span><span class="token punctuation">:</span>

<span class="token punctuation">-</span> <span class="token key atrule">type</span><span class="token punctuation">:</span> database
<span class="token key atrule">path</span><span class="token punctuation">:</span> return_request.order_id
<span class="token key atrule">equals</span><span class="token punctuation">:</span> A123

<span class="token punctuation">-</span> <span class="token key atrule">type</span><span class="token punctuation">:</span> database
<span class="token key atrule">path</span><span class="token punctuation">:</span> return_request.status
<span class="token key atrule">equals</span><span class="token punctuation">:</span> created

<span class="token key atrule">trajectory</span><span class="token punctuation">:</span>

<span class="token key atrule">required</span><span class="token punctuation">:</span>
<span class="token punctuation">-</span> get_order
<span class="token punctuation">-</span> check_refund_policy

<span class="token key atrule">partial_order</span><span class="token punctuation">:</span>
<span class="token punctuation">-</span> get_order < check_refund_policy

<span class="token key atrule">forbidden</span><span class="token punctuation">:</span>
<span class="token punctuation">-</span> force_refund

<span class="token key atrule">response</span><span class="token punctuation">:</span>

<span class="token key atrule">assertions</span><span class="token punctuation">:</span>
<span class="token punctuation">-</span> 应告知用户需要退货
<span class="token punctuation">-</span> 不得承诺立即退款

<span class="token key atrule">run</span><span class="token punctuation">:</span>

<span class="token key atrule">repetitions</span><span class="token punctuation">:</span> <span class="token number">5</span>
```


<p>它不是&#xff1a;</p>


```text
Question + Expected Answer
```


<p>而是&#xff1a;</p>


```text
Task Specification。
```


<p>这会成为整个开源项目最重要的核心抽象之一。</p>
<hr />
<h2>十八、第二件事&#xff1a;Environment 必须成为一等公民</h2>
<p>这是我认为这个项目与普通 LLM Eval Framework 最需要拉开差异的地方。</p>
<p>应该支持&#xff1a;</p>


```text
EnvironmentAdapter
```


<p>例如&#xff1a;</p>


```text
SQLite
PostgreSQL
Redis
REST API
Filesystem
Docker
Browser
Custom Python
```


<p>每一个 Trial&#xff1a;</p>


```text
Environment Reset
↓
Setup Initial State
↓
Run Agent
↓
Freeze Final State
↓
Verify Assertions
↓
Destroy / Reset
```


<p>例如&#xff1a;</p>


```yaml
<span class="token key atrule">assertions</span><span class="token punctuation">:</span>

<span class="token punctuation">-</span> <span class="token key atrule">sql</span><span class="token punctuation">:</span>
<span class="token key atrule">query</span><span class="token punctuation">:</span> <span class="token punctuation">></span><span class="token scalar string">
SELECT status
FROM refund
WHERE order_id = 'A123'</span>

<span class="token key atrule">equals</span><span class="token punctuation">:</span> CREATED
```


<p>这样 Task Completion 就变成一个&#xff1a;</p>


```text
确定性测试。
```


<p>而不是&#xff1a;</p>


```text
让另一个 LLM 猜 Agent 做没做完。
```


<hr />
<h2>十九、第三件事&#xff1a;Trace Native&#xff0c;但不要重新造 Tracing</h2>
<p>Trace Schema 是整个项目的地基。</p>
<p>但没必要再发明&#xff1a;</p>


```text
MyAgentEvalTraceProtocol。
```


<p>现在 Phoenix/OpenInference 等项目已经明显围绕 OpenTelemetry 做 AI Trace 标准化。</p>
<p>内部只需要建立一个 Normalize Layer&#xff1a;</p>


```text
OpenTelemetry
OpenInference
LangGraph
OpenAI Agents
Custom Trace
↓
Normalized Trace
```


<p>例如统一 Span 类型&#xff1a;</p>


```text
agent
model
tool
retrieval
rerank
memory
guardrail
handoff
environment
human
```


<p>然后所有 Evaluator 都只针对&#xff1a;</p>


```text
Normalized Trace
```


<p>编写。</p>
<p>这样 Agent Framework 和 Evaluation Framework 完全解耦。</p>
<hr />
<h2>二十、第四件事&#xff1a;把 Statistics 做成一等能力</h2>
<p>我认为这可能是这个项目真正能够做出特色的一点。</p>
<p>传统 LLM Evaluation 通常是&#xff1a;</p>


```text
run
→ score
→ average
```


<p>新的项目应该是&#xff1a;</p>


```text
Experiment
↓
Repeated Trials
↓
Paired Comparison
↓
Statistical Analysis
↓
Regression Decision
```


<p>结果页面应该输出&#xff1a;</p>


```text
Agent V1

Task Success       87.4%
95% CI             [84.1%, 90.3%]
Pass^3             66.8%
Flaky Rate          9.3%
Cost / Success     $0.041
```


<p>对比&#xff1a;</p>


```text
Agent V2

Task Success       90.2%
95% CI             [87.5%, 92.6%]
Pass^3             73.4%
Flaky Rate          5.1%
Cost / Success     $0.039
```


<p>然后告诉开发者&#xff1a;</p>


```text
Task Success       +2.8%
Reliability        +6.6%
Flaky Rate         -4.2%
Cost / Success     -4.9%
```


<p>这比&#xff1a;</p>


```text
Agent Score:
87 → 89
```


<p>有意义得多。</p>
<hr />
<h2>二十一、整个系统最终应该长这样</h2>


```text
                         Dataset
|
CaseSpec
|
Experiment Runner
|
+---------------+---------------+
|                               |
Target Adapter                 Environment
|                               |
Agent/RAG                      Sandbox
|                               |
+---------------+---------------+
|
Trial
|
Trace Normalizer
|
+---------------------+----------------------+
|                     |                      |
Deterministic         LLM Judge              Human
Evaluator             Evaluator             Review
|                     |                      |
+---------------------+----------------------+
|
Statistical Engine
|
+-----------------+----------------+
|                 |                |
Report          Quality Gate      Failure Mining
```


<p>这个架构里真正核心的不是 UI。</p>
<p>而是八个对象&#xff1a;</p>


```text
Case
Dataset
Target
Environment
Trial
Trace
Evaluator
Experiment
```


<hr />
<h2>二十二、RAG 不需要单独建一套系统</h2>
<p>RAG 完全可以作为 Evaluator Plugin&#xff1a;</p>


```text
evaluators/

├── deterministic/
├── agent/
├── tool/
├── trajectory/
├── rag/
├── safety/
├── reliability/
└── judge/
```


<p>RAG Plugin 里面&#xff1a;</p>


```text
retrieval_recall
precision_at_k
mrr
ndcg
claim_recall
faithfulness
citation_correctness
noise_sensitivity
```


<p>第一阶段甚至可以&#xff1a;</p>


```text
Adapter Ragas
Adapter DeepEval
```


<p>而不是重新实现所有指标。</p>
<p>你的价值不应该是&#xff1a;</p>
<blockquote>
<p>我也实现了 Faithfulness。</p>
</blockquote>
<p>真正的价值应该是&#xff1a;</p>
<blockquote>
<p><strong>我能够告诉你某次 Agent/RAG 改动是否真的提升了生产系统&#xff0c;而且这个结论是可复现、可解释、有统计可信度的。</strong></p>
</blockquote>
<hr />
<h2>二十三、真正应该形成的是 Evaluation Flywheel</h2>
<p>最终整个系统应该闭环成&#xff1a;</p>


```text
Production
↓
Trace
↓
Failure Mining
↓
Dataset
↓
Evaluation
↓
Experiment
↓
Regression Analysis
↓
Quality Gate
↓
New Version
↓
Production
```


<p>例如线上发现&#xff1a;</p>


```text
“订单取消后偶尔仍然调用退款 API”
```


<p>不是只修 Bug。</p>
<p>而是&#xff1a;</p>


```text
生产失败
↓
转为 Regression Case
↓
加入 Dataset
↓
以后所有版本自动验证
```


<p>Dataset 会随着生产系统持续增长。</p>
<p>这样 Evaluation 才真正从&#xff1a;</p>


```text
上线前跑一次测试
```


<p>变成&#xff1a;</p>


```text
Agent 质量基础设施。
```


<hr />
<h2>二十四、最终我们到底想做什么&#xff1f;</h2>
<p>这个项目的定位最终可以浓缩成一句话&#xff1a;</p>
<blockquote>
<p><strong>Evaluate what your agent actually does, not only what it says.</strong></p>
</blockquote>
<p>它不是&#xff1a;</p>


```text
RAG Score Library
```


<p>也不是&#xff1a;</p>


```text
LLM Judge Wrapper
```


<p>更不是&#xff1a;</p>


```text
Another Observability Platform
```


<p>而应该是一套&#xff1a;</p>


```text
Case Specification
+
Resettable Environment
+
Outcome Verification
+
Normalized Trace
+
Deterministic / LLM / Human Evaluation
+
Repeated Trials
+
Statistical Experiment
+
CI Quality Gate
```


<p>组成的&#xff1a;</p>
<h2>Agent &amp; RAG Quality Engineering Framework</h2>
<hr />
<h2>结语</h2>
<p>LLM 时代&#xff0c;我们问的是&#xff1a;</p>
<blockquote>
<p>模型回答得对不对&#xff1f;</p>
</blockquote>
<p>RAG 时代&#xff0c;我们进一步问&#xff1a;</p>
<blockquote>
<p>模型回答的内容有没有证据&#xff1f;</p>
</blockquote>
<p>而 Agent 时代&#xff0c;我们必须继续往前走一步&#xff1a;</p>
<blockquote>
<p><strong>这个 AI 到底做了什么&#xff1f;</strong></p>
</blockquote>
<blockquote>
<p><strong>事情真的办成了吗&#xff1f;</strong></p>
</blockquote>
<blockquote>
<p><strong>用了哪些工具&#xff1f;</strong></p>
</blockquote>
<blockquote>
<p><strong>有没有越权&#xff1f;</strong></p>
</blockquote>
<blockquote>
<p><strong>为什么失败&#xff1f;</strong></p>
</blockquote>
<blockquote>
<p><strong>每次运行都能成功吗&#xff1f;</strong></p>
</blockquote>
<blockquote>
<p><strong>新版本是真的提升&#xff0c;还是随机波动&#xff1f;</strong></p>
</blockquote>
<blockquote>
<p><strong>Retriever、Generator、Tool、Prompt、Model&#xff0c;到底是谁出了问题&#xff1f;</strong></p>
</blockquote>
<p>这时 Evaluation 已经不再只是几个 Metric。</p>
<p>它更像传统软件工程中的&#xff1a;</p>


```text
Unit Test
+
Integration Test
+
End-to-End Test
+
Observability
+
Experimentation
+
Statistics
```


<p>只是它面对的是一个概率性的 AI 系统。</p>
<p>所以我越来越倾向于认为&#xff1a;</p>
<blockquote>
<p><strong>Agent Evaluation 的终局&#xff0c;不是给 Agent 打一个分。</strong></p>
</blockquote>
<p>而是建立一套可以回答&#xff1a;</p>
<blockquote>
<p><strong>“这个 Agent 是否值得被信任和上线&#xff1f;”</strong></p>
</blockquote>
<p>的质量工程体系。</p>
<p>而如果我们要做一个新的开源项目&#xff0c;我认为这才是最值得做的方向。</p>
<hr />
<h3>参考项目与资料</h3>
<p>本文对开源生态的分析主要参考&#xff1a;</p>
<ul><li>Ragas&#xff1a;Metrics、Dataset 与 Experiment 体系。</li><li>DeepEval&#xff1a;Trace-based Agent Evaluation&#xff0c;以及 Task Completion、Tool Correctness、Plan、Efficiency 等 Agent Metric。</li><li>Arize Phoenix&#xff1a;OpenTelemetry/OpenInference、Tracing、Evaluation、Dataset 与 Experiment。</li><li>Promptfoo&#xff1a;声明式 Eval、CI/CD 与 Red Team Plugin/Strategy。</li><li>Inspect AI&#xff1a;Task、Solver、Scorer、Tool、Agent 与 Sandbox Evaluation Harness。</li><li>τ³-bench&#xff1a;Domain、Policy、Tool、Environment、User Simulator 与多 Trial Stateful Agent Evaluation。</li></ul>
