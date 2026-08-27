---
title: "从一次请求到整套推理集群：彻底理解大模型的 QPS、TPM、并发与性能指标"
description: "CSDN 原文全文镜像：摘要：大模型性能指标解析 本文系统阐述了大模型服务中的关键性能指标及其相互关系。核心内容包括： 请求生命周期：大模型请求经历Prefill（处理输入）和Decode（生成输出）两个关键阶段，分别影响首字延迟和输出流畅度。 业务压力指标：……"
pageType: article
module: training
updated: '2026-08-05'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "training"
  - "java"
  - "服务器"
  - "开发语言"
  - "prompt"
  - "大模型"
level: advanced
prerequisites:
  - "/llms/training/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-05，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-05。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163510719](https://blog.csdn.net/m0_63309778/article/details/163510719)
- 站内分区：Training / 大模型推理指标
:::

<p><img src="https://i-blog.csdnimg.cn/direct/3aea12ea8ced49709eeb3e3d04994ab8.png" alt="在这里插入图片描述" /></p>
<h3>前言</h3>
<p>在大模型项目中&#xff0c;我们经常需要和模型服务商、基础设施团队或者业务方讨论下面这些问题&#xff1a;</p>
<ul><li>模型接口需要申请多少 QPS&#xff1f;</li><li>TPM 配置 100 万够不够&#xff1f;</li><li>并发设置成 32 还是 64&#xff1f;</li><li>为什么 GPU 利用率很高&#xff0c;用户还是觉得慢&#xff1f;</li><li>为什么提高并发后&#xff0c;吞吐没有明显增加&#xff0c;首字延迟却变得很高&#xff1f;</li><li>一个 RAG、Agent、文档解析平台&#xff0c;到底应该如何估算模型容量&#xff1f;</li></ul>
<p>这些问题看起来涉及很多独立指标&#xff0c;但它们其实都在描述同一件事&#xff1a;</p>
<blockquote>
<p>一批大模型请求进入系统后&#xff0c;系统需要用多少时间、多少计算资源和多少显存&#xff0c;才能完成这些请求。</p>
</blockquote>
<p>要真正理解 QPS、TPM、并发、TTFT、TPOT 和 Token 吞吐&#xff0c;不能从术语本身出发&#xff0c;而应该先观察一条请求在推理服务中的完整生命周期。</p>
<hr />
<h2>一、一条大模型请求在系统中经历了什么</h2>
<p>假设用户向一个企业知识库助手提问&#xff1a;</p>
<blockquote>
<p>请根据公司制度和历史案例&#xff0c;分析这个项目可能存在的风险&#xff0c;并给出改进建议。</p>
</blockquote>
<p>业务服务在调用模型之前&#xff0c;可能会组装出一段很长的输入&#xff1a;</p>


```text
系统提示词：1,500 Token
Agent 规则和工具说明：2,000 Token
历史对话：3,000 Token
RAG 检索结果：8,000 Token
用户当前问题：200 Token

总输入长度：14,700 Token
```


<p>模型最终生成了一段 1,000 Token 的回答。</p>
<p>从业务接口发出请求&#xff0c;到回答全部生成完成&#xff0c;通常需要经历以下过程&#xff1a;</p>


```text
请求到达网关
↓
鉴权与限流
↓
进入推理服务等待队列
↓
输入文本 Tokenize
↓
Prefill：处理全部输入 Token
↓
生成第一个输出 Token
↓
Decode：逐个生成后续 Token
↓
输出结束
```


<p>这条链路中存在两个非常关键的推理阶段&#xff1a;Prefill 和 Decode。</p>
<p>Prefill 是模型理解输入的过程。模型需要一次性读取系统提示词、历史消息、RAG 上下文和用户问题&#xff0c;并为这些输入 Token 生成中间计算结果和 KV Cache。输入越长&#xff0c;Prefill 的工作量通常越大&#xff0c;用户等待第一个字的时间也越长。</p>
<p>Decode 是模型生成回答的过程。大模型是自回归模型&#xff0c;它不能一次直接生成完整回答&#xff0c;而是一个 Token 接着一个 Token 地生成&#xff1a;</p>


```text
第 1 个 Token
第 2 个 Token
第 3 个 Token
……
第 1000 个 Token
```


<p>Prefill 决定用户需要等多久才能看到模型开始回答&#xff0c;Decode 决定回答开始后输出得是否流畅。</p>
<p>因此&#xff0c;大模型所谓的“响应速度”&#xff0c;实际上至少包含三部分&#xff1a;</p>


```text
排队时间 + Prefill 时间 + Decode 时间
```


<p>后面所有性能指标&#xff0c;基本都可以放回这条链路中理解。</p>
<hr />
<h2>二、QPS、RPM 和 TPM 描述的是进入系统的业务压力</h2>
<h3>1. QPS 描述请求数量&#xff0c;但不能代表真实计算量</h3>
<p>QPS 是 Queries Per Second&#xff0c;表示系统平均每秒处理多少个请求。</p>
<p>假设一分钟内成功完成了 600 个推理请求&#xff0c;那么平均 QPS 是&#xff1a;</p>


```text
600 ÷ 60 = 10 QPS
```


<p>RPM 是 Requests Per Minute&#xff0c;也就是每分钟请求数。600 RPM 平均约等于 10 QPS。</p>
<p>在普通业务接口中&#xff0c;QPS 是一个非常重要的指标。查询一个用户和查询另一个用户&#xff0c;通常消耗的资源比较接近&#xff0c;因此每秒能处理多少次请求&#xff0c;可以大致代表系统能力。</p>
<p>但在大模型服务中&#xff0c;请求之间的差异可能非常大。</p>
<p>请求 A 可能只有&#xff1a;</p>


```text
输入：100 Token
输出：20 Token
```


<p>请求 B 可能是&#xff1a;</p>


```text
输入：50,000 Token
输出：5,000 Token
```


<p>这两个请求在 QPS 统计中都只算一次&#xff0c;但请求 B 的计算成本、显存占用和执行时间可能远远超过请求 A。</p>
<p>所以&#xff0c;在大模型场景中&#xff0c;只说“需要 10 QPS”是不完整的。还必须说明&#xff1a;</p>


```text
10 QPS 对应多长的平均输入？
每个请求平均生成多少 Token？
请求长度的 P95 是多少？
```


<p>同样的 10 QPS&#xff0c;可能只是一个很轻的聊天业务&#xff0c;也可能足以压垮一套推理集群。</p>
<hr />
<h3>2. TPM 用 Token 数描述真实流量</h3>
<p>TPM 是 Tokens Per Minute&#xff0c;表示系统每分钟处理多少 Token。</p>
<p>通常需要区分&#xff1a;</p>


```text
Input TPM：每分钟处理的输入 Token
Output TPM：每分钟生成的输出 Token
Total TPM：输入和输出 Token 之和
```


<p>假设系统平均每个请求&#xff1a;</p>


```text
输入 4,000 Token
输出 1,000 Token
```


<p>每分钟有 300 个请求&#xff0c;那么&#xff1a;</p>


```text
Input TPM = 4,000 × 300 = 1,200,000

Output TPM = 1,000 × 300 = 300,000

Total TPM = 1,500,000
```


<p>这意味着&#xff0c;即使业务请求量只有 300 RPM&#xff0c;也需要至少约 150 万的总 TPM 配额。</p>
<p>RPM 和 TPM 共同决定了模型接口容量。</p>
<p>如果平台提供&#xff1a;</p>


```text
RPM 上限：1,000
TPM 上限：1,000,000
```


<p>而每个请求平均消耗 5,000 Token&#xff0c;那么 TPM 实际只允许&#xff1a;</p>


```text
1,000,000 ÷ 5,000 = 200 RPM
```


<p>虽然名义上的 RPM 是 1,000&#xff0c;但真正先触发的是 TPM 限制。</p>
<p>反过来&#xff0c;如果每个请求只消耗 500 Token&#xff0c;那么 1,000 RPM 只需要&#xff1a;</p>


```text
500 × 1,000 = 500,000 TPM
```


<p>此时 RPM 会先成为瓶颈。</p>
<p>因此&#xff0c;申请模型配额时不能分别拍脑袋填写 QPS、RPM 和 TPM&#xff0c;而应该从业务请求长度反推&#xff1a;</p>


```text
TPM ≈ RPM × 每个请求平均 Token 数
```


<p>更严谨一些&#xff0c;还应分别计算 Input TPM 和 Output TPM&#xff0c;因为输入和输出在推理系统中消耗的资源特征并不相同。</p>
<hr />
<h2>三、并发描述系统中同时存在多少请求</h2>
<p>QPS 描述的是单位时间完成多少请求&#xff0c;而并发描述的是某个时刻同时有多少请求处于系统中。</p>
<p>假设某一时刻&#xff1a;</p>


```text
20 个请求正在 GPU 上执行
30 个请求正在等待队列中排队
```


<p>那么可以分别描述为&#xff1a;</p>


```text
运行并发：20
等待请求：30
系统内请求总数：50
```


<p>并发与 QPS 并不是同一个概念。</p>
<p>一家餐厅当前有 100 位客人&#xff0c;描述的是并发&#xff1b;餐厅每分钟完成多少桌服务&#xff0c;描述的是吞吐。</p>
<p>大模型服务也是一样。系统中同时存在很多请求&#xff0c;不代表这些请求处理得快。它们可能只是在等待。</p>
<p>在稳定状态下&#xff0c;并发、QPS 和平均响应时间之间可以使用一个非常实用的近似关系&#xff1a;</p>


```text
平均并发 ≈ QPS × 平均响应时间
```


<p>假设系统稳定处理 5 QPS&#xff0c;每个请求从进入到结束平均需要 12 秒&#xff0c;那么平均并发大约是&#xff1a;</p>


```text
5 × 12 = 60
```


<p>也就是说&#xff0c;为了稳定支撑 5 QPS&#xff0c;系统中平均会同时存在约 60 个请求。</p>
<p>这个公式解释了为什么大模型服务的并发通常明显高于 QPS。因为大模型请求不是几十毫秒就结束&#xff0c;而是可能持续数秒甚至数分钟。</p>
<p>同样&#xff0c;它也说明了一个很重要的事实&#xff1a;</p>
<blockquote>
<p>提高并发上限&#xff0c;并不等于提高系统处理能力。</p>
</blockquote>
<p>如果 GPU 已经饱和&#xff0c;把最大并发从 64 调到 128&#xff0c;并不会让 GPU 计算能力翻倍&#xff0c;只会允许更多请求进入等待队列。最终可能出现&#xff1a;</p>


```text
完成 QPS 基本不变
等待队列越来越长
TTFT 持续升高
超时请求越来越多
```


<p>因此&#xff0c;并发配置本质上是一个容量和保护参数&#xff0c;而不是越高越好的性能参数。</p>
<hr />
<h2>四、TTFT、TPOT 和总延迟描述用户到底感觉快不快</h2>
<h3>1. TTFT 决定用户多久看到第一个字</h3>
<p>TTFT 是 Time To First Token&#xff0c;即从请求发出&#xff0c;到用户收到第一个输出 Token 的时间。</p>
<p>例如用户在 10:00:00 发出请求&#xff0c;10:00:02 收到模型输出的第一个字&#xff0c;那么&#xff1a;</p>


```text
TTFT = 2 秒
```


<p>TTFT 通常包含&#xff1a;</p>


```text
网络与网关时间
+ 请求排队时间
+ Tokenize 时间
+ Prefill 时间
+ 生成并传输第一个 Token 的时间
```


<p>输入上下文越长&#xff0c;Prefill 越重&#xff1b;并发越高&#xff0c;排队时间可能越长&#xff0c;因此 TTFT 也会随之上升。</p>
<p>对于聊天和 Agent 产品&#xff0c;TTFT 是非常重要的用户体验指标。用户点击发送后&#xff0c;如果 300 毫秒就开始看到输出&#xff0c;通常会觉得系统反应很快&#xff1b;如果等待 8 秒还没有任何内容&#xff0c;即使后续输出速度很快&#xff0c;用户也容易认为系统卡住了。</p>
<p>这也是为什么有些系统的总生成时间不短&#xff0c;但用户体验仍然不错——它们能很快返回第一个 Token&#xff0c;然后持续流式输出。</p>
<hr />
<h3>2. TPOT 决定回答开始后输出是否流畅</h3>
<p>TPOT 是 Time Per Output Token&#xff0c;即生成一个输出 Token 平均需要多少时间。</p>
<p>假设 TPOT 是 20 毫秒&#xff0c;那么单请求的平均输出速度约为&#xff1a;</p>


```text
1 ÷ 0.02 = 50 Token/s
```


<p>也可以使用&#xff1a;</p>


```text
Token/s ≈ 1000 ÷ TPOT(ms)
```


<p>如果 TPOT 为 50 毫秒&#xff0c;那么输出速度约为 20 Token/s。</p>
<p>用户可能不会直接感受到“20 毫秒”这样的数字&#xff0c;但会明显感受到文字是一段段快速出现&#xff0c;还是断断续续地蹦出来。</p>
<p>TTFT 和 TPOT 分别描述了两种不同的慢&#xff1a;</p>


```text
TTFT 高：
很久没有开始回答

TPOT 高：
已经开始回答，但输出很慢
```


<p>这两种情况背后的原因也不同。</p>
<p>如果 TTFT 很高但 TPOT 正常&#xff0c;通常说明请求在排队&#xff0c;或者 Prefill 很慢&#xff0c;例如输入上下文过长。</p>
<p>如果 TTFT 正常但 TPOT 很高&#xff0c;通常说明 Decode 阶段压力较大&#xff0c;可能是同时生成的序列太多、显存带宽不足&#xff0c;或者多卡通信开销较高。</p>
<hr />
<h3>3. 端到端延迟是完整回答需要多久</h3>
<p>端到端延迟&#xff0c;也可以称为 E2E Latency&#xff0c;表示从发出请求到完整回答生成结束所花费的时间。</p>
<p>对于流式生成&#xff0c;可以使用一个简化公式估算&#xff1a;</p>


```text
总延迟
≈
TTFT +（输出 Token 数 - 1）× TPOT
```


<p>例如&#xff1a;</p>


```text
TTFT：1 秒
输出长度：501 Token
TPOT：20 毫秒
```


<p>那么总延迟大约是&#xff1a;</p>


```text
1 + 500 × 0.02 = 11 秒
```


<p>这三个指标共同描述用户体验&#xff1a;</p>

<table><thead><tr><th>指标</th><th>用户感受</th></tr></thead><tbody><tr><td>TTFT</td><td>点击发送后多久开始回答</td></tr><tr><td>TPOT</td><td>回答开始后文字输出是否流畅</td></tr><tr><td>E2E Latency</td><td>完整任务什么时候结束</td></tr></tbody></table><p>普通聊天通常更关注 TTFT 和 TPOT&#xff1b;长文生成、文档解析和 Agent 任务则还需要重点关注 E2E Latency。</p>
<hr />
<h2>五、系统吞吐和单个用户的输出速度不是一回事</h2>
<p>假设一个请求的输出速度是 40 Token/s&#xff0c;这只是单个用户看到的生成速度。</p>
<p>如果系统同时为 100 个请求生成 Token&#xff0c;那么整个服务的输出吞吐可能达到&#xff1a;</p>


```text
100 × 40 = 4,000 Token/s
```


<p>这里需要区分两个指标&#xff1a;</p>


```text
Per-user Token/s：
单个请求的生成速度

Output Token Throughput：
整套服务每秒生成的总 Token 数
```


<p>提高并发后&#xff0c;系统总 Token 吞吐通常会上升&#xff0c;因为 GPU 可以在一次计算中同时处理更多请求。但单个请求的生成速度可能下降&#xff0c;因为所有请求在竞争同一套计算和显存资源。</p>
<p>系统性能通常会随着并发增加经历三个阶段。</p>
<p>第一阶段&#xff0c;GPU 尚未被充分利用。增加并发可以让 Batch 更充实&#xff0c;系统总吞吐明显提高&#xff0c;TTFT 和 TPOT 变化不大。</p>
<p>第二阶段&#xff0c;系统接近饱和。吞吐仍然增长&#xff0c;但增速开始下降&#xff0c;TTFT 和 TPOT逐渐升高。</p>
<p>第三阶段&#xff0c;系统已经过载。继续增加并发后&#xff0c;总吞吐几乎不再增长&#xff0c;但等待队列和 TTFT 快速上升。</p>
<p>可以把它理解为&#xff1a;</p>


```text
并发较低：
GPU 没吃饱

并发适中：
GPU 利用充分，吞吐较高

并发过高：
GPU 已经吃不下，新请求只能排队
```


<p>容量测试真正要找到的&#xff0c;不是服务能够接受多少并发连接&#xff0c;而是&#xff1a;</p>
<blockquote>
<p>在满足 TTFT、TPOT 和错误率要求的前提下&#xff0c;能够稳定维持的最大请求速率。</p>
</blockquote>
<hr />
<h2>六、连续批处理为什么能提高大模型吞吐</h2>
<p>大模型请求的输入长度和输出长度都不相同。</p>
<p>假设有三个请求&#xff1a;</p>


```text
请求 A：生成 20 Token
请求 B：生成 200 Token
请求 C：生成 2,000 Token
```


<p>如果使用传统固定批处理&#xff0c;三个请求组成一个 Batch 后&#xff0c;短请求即使已经完成&#xff0c;也可能因为长请求仍未结束而无法有效释放位置。</p>
<p>现代推理框架通常使用连续批处理。某个请求完成后&#xff0c;可以立即从 Batch 中移除&#xff0c;再把新的等待请求加入进来&#xff1a;</p>


```text
请求 A 完成
↓
释放它占用的位置
↓
请求 D 立即加入当前推理过程
```


<p>这样可以让 GPU 持续保持较高利用率&#xff0c;提高系统总吞吐。</p>
<p>但 Batch 也存在吞吐与延迟的权衡。</p>
<p>Batch 越大&#xff0c;一次 GPU 计算处理的 Token 越多&#xff0c;整体吞吐往往越高&#xff1b;但请求可能需要等待调度&#xff0c;单个用户的 TTFT 和 TPOT 可能变差。</p>
<p>因此&#xff1a;</p>


```text
在线聊天：
更重视低 TTFT、稳定 TPOT

离线批量生成：
更重视总 Token 吞吐和 GPU 利用率
```


<p>这两种业务不应该直接使用完全相同的调度参数和性能目标。</p>
<hr />
<h2>七、KV Cache 为什么决定长上下文和并发能力</h2>
<p>大模型每生成一个新 Token&#xff0c;都需要关注之前的上下文。</p>
<p>如果每次生成 Token 时都重新计算全部历史内容&#xff0c;性能会非常差。因此&#xff0c;推理服务会保存历史 Token 对应的 Attention Key 和 Value&#xff0c;这部分显存就是 KV Cache。</p>
<p>KV Cache 的占用与以下因素相关&#xff1a;</p>


```text
并发请求数
每个请求的上下文长度
模型层数
KV Head 数量
数据精度
```


<p>可以用一个简化关系理解&#xff1a;</p>


```text
KV Cache 占用
∝
并发数 × 每请求 Token 数
```


<p>假设一套服务的 KV Cache 最多可以容纳 100 万个 Token。</p>
<p>如果每个请求平均占用 10,000 Token&#xff0c;理论上可以同时容纳大约 100 个请求。</p>
<p>如果每个请求平均占用 100,000 Token&#xff0c;那么理论上只能容纳大约 10 个请求。</p>
<p>因此&#xff1a;</p>
<blockquote>
<p>模型支持 128K 或 1M 上下文&#xff0c;不代表在最大上下文下仍然可以保持高并发。</p>
</blockquote>
<p>这也是长上下文业务经常出现并发能力明显下降的原因。</p>
<p>当 KV Cache 使用率接近上限时&#xff0c;新请求可能无法立即进入运行状态&#xff0c;只能进入等待队列&#xff0c;进而导致 TTFT 上升。严重时还可能发生 Cache 换出、请求抢占甚至显存不足。</p>
<p>所以线上监控不能只看 GPU Utilization&#xff0c;还需要同时看&#xff1a;</p>


```text
KV Cache 使用率
运行请求数
等待请求数
平均上下文长度
Cache 命中率
```


<hr />
<h2>八、GPU 利用率高不代表服务一定健康</h2>
<p>很多团队判断推理服务性能时&#xff0c;首先看 GPU 利用率。</p>
<p>GPU 利用率当然重要&#xff0c;但它只能说明 GPU 在忙&#xff0c;不能说明 GPU 忙得是否高效&#xff0c;也不能说明用户体验是否良好。</p>
<p>例如 GPU 利用率长期是 100%&#xff0c;可能存在两种完全不同的情况。</p>
<p>第一种情况&#xff0c;系统处于高效状态&#xff1a;</p>


```text
GPU 利用率高
Token 吞吐高
TTFT 稳定
TPOT 稳定
队列较短
```


<p>第二种情况&#xff0c;系统已经过载&#xff1a;</p>


```text
GPU 利用率高
Token 吞吐没有继续增长
等待队列持续增加
TTFT P99 非常高
超时和取消请求增多
```


<p>两种情况下 GPU 都是 100%&#xff0c;但前者是高效利用&#xff0c;后者是拥塞。</p>
<p>因此&#xff0c;GPU 指标必须与业务性能指标放在一起看&#xff1a;</p>


```text
GPU 利用率
+
输入和输出 Token 吞吐
+
TTFT
+
TPOT
+
等待队列
+
KV Cache 使用率
```


<p>只有这些指标同时健康&#xff0c;才能说明推理服务运行良好。</p>
<hr />
<h2>九、为什么性能报告必须看 P50、P95 和 P99</h2>
<p>只看平均值很容易掩盖真实问题。</p>
<p>假设 100 个请求中&#xff1a;</p>


```text
99 个请求 TTFT 为 1 秒
1 个请求 TTFT 为 101 秒
```


<p>平均 TTFT 是&#xff1a;</p>


```text
(99 × 1 + 101) ÷ 100 = 2 秒
```


<p>从平均值看&#xff0c;系统似乎只需要等待 2 秒。但实际上有一个用户等了 101 秒。</p>
<p>因此生产系统通常使用分位数描述用户体验&#xff1a;</p>


```text
P50：
一半请求低于该数值，代表典型用户体验

P95：
95% 的请求低于该数值，代表大部分用户体验

P99：
99% 的请求低于该数值，代表尾部用户体验
```


<p>例如&#xff1a;</p>


```text
TTFT P50：500ms
TTFT P95：2s
TTFT P99：12s
```


<p>说明典型请求很快&#xff0c;但仍有约 1% 的请求等待非常久。</p>
<p>大模型服务容易出现尾延迟&#xff0c;因为请求长度和生成长度差异很大。少量超长上下文、超长输出或者 Cache 换出&#xff0c;都可能显著拉高 P99。</p>
<p>因此&#xff0c;一个生产级服务至少应该关注&#xff1a;</p>


```text
TTFT P50 / P95 / P99
TPOT P50 / P95 / P99
E2E Latency P50 / P95 / P99
Queue Time P50 / P95 / P99
```


<hr />
<h2>十、如何从业务流量反推并发、TPM 和模型配额</h2>
<p>下面通过一个完整例子&#xff0c;把这些指标连接起来。</p>
<p>假设某个 Agent 平台预计有以下负载&#xff1a;</p>


```text
平均请求速率：4 QPS
平均输入长度：6,000 Token
平均输出长度：1,000 Token
平均端到端时长：15 秒
```


<p>首先计算 RPM&#xff1a;</p>


```text
4 × 60 = 240 RPM
```


<p>然后计算 Input TPM&#xff1a;</p>


```text
6,000 × 240 = 1,440,000
```


<p>Output TPM&#xff1a;</p>


```text
1,000 × 240 = 240,000
```


<p>Total TPM&#xff1a;</p>


```text
1,440,000 + 240,000 = 1,680,000
```


<p>接着估算平均并发&#xff1a;</p>


```text
并发 ≈ QPS × 平均响应时间
≈ 4 × 15
≈ 60
```


<p>这意味着&#xff0c;如果业务稳定运行在 4 QPS&#xff0c;系统平均会同时存在约 60 个请求。</p>
<p>但生产配置不能只按照平均值申请。还需要考虑流量波动、长请求比例、失败重试和多个业务共享模型等情况。</p>
<p>例如增加 50% 的容量余量&#xff1a;</p>


```text
目标 QPS：6
目标并发：90
目标 Total TPM：约 252 万
```


<p>如果 RAG、Agent、文档解析增强和普通聊天共用同一模型&#xff0c;还要分别估算每种业务的请求特征。</p>

<table><thead><tr><th>业务</th><th align="right">请求量</th><th>输入特点</th><th>输出特点</th></tr></thead><tbody><tr><td>普通聊天</td><td align="right">高</td><td>输入较短</td><td>输出中等</td></tr><tr><td>RAG 问答</td><td align="right">中高</td><td>Context 较长</td><td>输出中等</td></tr><tr><td>Agent</td><td align="right">中</td><td>多轮调用、多次模型请求</td><td>输出和耗时不稳定</td></tr><tr><td>文档解析增强</td><td align="right">低到中</td><td>单次输入可能很长</td><td>结构化输出</td></tr><tr><td>批量任务</td><td align="right">波动大</td><td>可削峰</td><td>通常不要求低 TTFT</td></tr></tbody></table><p>最终配额应该按共享资源池汇总&#xff0c;而不是只计算某一个 RAG 接口。</p>
<hr />
<h2>十一、性能测试应该如何设计</h2>
<p>大模型性能测试不能只写&#xff1a;</p>


```text
并发 32，QPS 8，测试成功。
```


<p>这样的结果缺少输入和输出条件&#xff0c;几乎无法比较。</p>
<p>一次有效的性能测试&#xff0c;至少需要固定或记录以下信息&#xff1a;</p>


```text
模型名称和精度
GPU 型号和数量
推理框架
输入 Token 分布
输出 Token 分布
并发或请求到达速率
测试持续时间
```


<p>然后重点观察四类结果。</p>
<p>第一类是请求流量&#xff1a;</p>


```text
成功 QPS
失败 QPS
Input TPM
Output TPM
```


<p>第二类是用户体验&#xff1a;</p>


```text
TTFT P50 / P95 / P99
TPOT P50 / P95 / P99
E2E Latency P50 / P95 / P99
```


<p>第三类是调度状态&#xff1a;</p>


```text
运行请求数
等待请求数
队列等待时间
平均 Batch Token 数
```


<p>第四类是硬件资源&#xff1a;</p>


```text
GPU 利用率
GPU 显存
KV Cache 使用率
CPU 和内存
网络和多卡通信
```


<p>测试时应该逐级增加负载&#xff0c;例如&#xff1a;</p>


```text
并发 1
并发 4
并发 8
并发 16
并发 32
并发 64
并发 96
```


<p>观察每个阶段的总吞吐和延迟变化。</p>
<p>当出现以下现象时&#xff0c;通常意味着已经接近或超过稳定容量&#xff1a;</p>


```text
总 Token 吞吐几乎不再增加
TTFT P95 快速上升
等待队列持续增长
错误率或超时率增加
KV Cache 长期接近上限
```


<p>最终选定的生产并发&#xff0c;不应该是系统不崩溃的最高并发&#xff0c;而应该是满足业务 SLO 的最高并发。</p>
<hr />
<h2>十二、如何为线上服务定义合理的 SLO</h2>
<p>一个完整的 SLO 不应该只规定 QPS。</p>
<p>例如可以定义&#xff1a;</p>


```text
在输入 Token P95 不超过 12,000、
输出 Token P95 不超过 1,500、
稳定负载 6 QPS 的情况下：

成功率不低于 99.9%
TTFT P95 不超过 2 秒
TPOT P95 不超过 40ms
E2E Latency P95 不超过 60 秒
Queue Time P95 不超过 500ms
```


<p>这样的目标同时规定了&#xff1a;</p>


```text
请求有多重
系统要处理多少
用户最多等待多久
服务需要多稳定
```


<p>如果只说“系统支持并发 64”&#xff0c;无法判断这个并发是在什么上下文长度、什么生成长度和什么延迟下实现的。</p>
<p>同理&#xff0c;如果模型服务商告诉你“支持 TPM 200 万”&#xff0c;你仍然需要确认&#xff1a;</p>
<ul><li>是输入 TPM、输出 TPM&#xff0c;还是总 TPM&#xff1f;</li><li>是否所有模型共享&#xff1f;</li><li>是否多个 API Key 共享&#xff1f;</li><li>是否允许短时间突发&#xff1f;</li><li>Reasoning Token 是否计入&#xff1f;</li><li>缓存命中的输入是否计入&#xff1f;</li><li>并发限制是否独立存在&#xff1f;</li></ul>
<hr />
<h2>十三、遇到性能问题时如何定位</h2>
<p>理解了整条链路后&#xff0c;性能问题就可以按照指标组合判断。</p>
<p>如果 TTFT 很高&#xff0c;但 TPOT 正常&#xff0c;通常意味着请求开始前等待太久&#xff0c;或者 Prefill 太慢。应重点检查等待队列、输入长度、Prefill 吞吐和 Tokenize 开销。</p>
<p>如果 TTFT 正常&#xff0c;但 TPOT 很高&#xff0c;说明请求能够快速开始&#xff0c;但 Decode 过程较慢。应检查 Decode 并发、Batch 规模、显存带宽、KV Cache 和多卡通信。</p>
<p>如果 TTFT 和 TPOT 都在恶化&#xff0c;同时等待队列不断增长&#xff0c;通常表示系统整体过载&#xff0c;需要限流、扩容或者拆分业务流量。</p>
<p>如果平均指标正常&#xff0c;但 P99 很高&#xff0c;应重点检查少量超长上下文、超长输出、实例性能不均以及 KV Cache 换出。</p>
<p>如果 GPU 利用率不高&#xff0c;但等待队列很多&#xff0c;则不一定是 GPU 计算不足&#xff0c;还可能是 CPU Tokenize、请求调度、网络、预处理或者配置过于保守造成的瓶颈。</p>
<hr />
<h2>结语</h2>
<p>QPS、TPM、并发、TTFT 和 TPOT 并不是一组互相独立的术语。</p>
<p>它们共同描述了一条大模型请求从进入系统到生成完成的全过程&#xff1a;</p>


```text
请求进入
↓
QPS / RPM / TPM 描述流量
↓
进入等待和运行状态
↓
并发数与 Queue Time 描述系统压力
↓
Prefill
↓
TTFT 描述多久开始回答
↓
Decode
↓
TPOT 描述回答输出速度
↓
请求完成
↓
E2E Latency 描述完整耗时
```


<p>与此同时&#xff1a;</p>


```text
Token Throughput
描述整套系统单位时间完成多少计算

KV Cache
决定长上下文和并发容量

GPU 指标
描述硬件资源状态

P95 和 P99
描述大部分用户与尾部用户的真实体验
```


<p>因此&#xff0c;在评估一个大模型推理服务时&#xff0c;最准确的问题不是&#xff1a;</p>
<blockquote>
<p>这个模型支持多少 QPS&#xff1f;</p>
</blockquote>
<p>而应该是&#xff1a;</p>
<blockquote>
<p>在指定的输入长度、输出长度、并发规模和延迟目标下&#xff0c;这套服务能够稳定提供多少请求吞吐和 Token 吞吐&#xff1f;</p>
</blockquote>
<p>一个完整的容量结论至少应该同时包含&#xff1a;</p>


```text
模型与硬件配置
输入和输出 Token 分布
稳定 QPS
Input / Output TPM
稳定并发
TTFT P95
TPOT P95
E2E Latency P95
KV Cache 使用率
错误率
```


<p>只有把请求规模、Token 规模、用户延迟和硬件容量放在同一套体系中&#xff0c;大模型推理指标才真正具有指导架构设计、资源申请和生产扩容的价值。</p>
