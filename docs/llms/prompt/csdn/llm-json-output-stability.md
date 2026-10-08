---
title: "大模型输出 JSON 不完整怎么办？从 Prompt 问题到工程稳定性治理"
description: "CSDN 原文全文镜像：文章摘要：大模型输出JSON不完整是一个常见的工程稳定性问题，表现为语法不完整或内容缺失。原因包括输出长度限制、输入上下文过长、Schema设计复杂等。解决方案需从多层面治理：优先使用结构化输出能力，避免一次性生成大JSON，建立JSO……"
pageType: article
module: prompt
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "prompt"
  - "json"
  - "java"
level: intermediate
prerequisites:
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-07-09，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-07-09。本站保留原文主体与发布时间，补充主题导读，并修复代码块中残留的语法高亮标签；技术结论仍需结合原文时点与当前文档判断。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/162718633](https://blog.csdn.net/m0_63309778/article/details/162718633)
- 站内分区：Prompt / 结构化输出稳定性
:::

::: tip 站内阅读提示
阅读时先区分四类失败：响应被截断、JSON 语法不合法、Schema 不匹配、业务内容错误。结构化输出不能免除拒答/完成状态处理；修复 JSON 后仍需重新校验，不可把补出的字段当作真实事实。接入 SDK 与参数以所选版本官方文档为准。

主线关联：[结构化输出与校验](/llms/prompt/advanced) · [上下文预算](/llms/prompt/context)
:::

<p><img src="https://i-blog.csdnimg.cn/direct/cfc3f78d89a74ffb91ec559c12d1451a.png" alt="在这里插入图片描述" /></p>
<h2>大模型输出 JSON 不完整怎么办&#xff1f;从 Prompt 问题到工程稳定性治理</h2>
<h3>前言</h3>
<p>在大模型应用落地过程中&#xff0c;很多系统都会遇到一个非常典型的问题&#xff1a;</p>
<blockquote>
<p>明明已经在 Prompt 里要求模型“严格输出 JSON”&#xff0c;但模型还是经常输出不完整、格式错误、字段缺失&#xff0c;甚至输出到一半就停了。</p>
</blockquote>
<p>例如&#xff1a;</p>


```json
{<!-- -->
"projectName": "某某项目",
"budget": "120万元",
"requirements": [
"投标人须具备相关资质",
"项目负责人须具备相关经验"
```


<p>这个 JSON 明显没有闭合&#xff0c;程序无法直接 <code>JSON.parse</code>。</p>
<p>更麻烦的是&#xff0c;有些 JSON 虽然语法合法&#xff0c;但内容并不完整&#xff1a;</p>


```json
{<!-- -->
"projectName": "某某项目",
"budget": null,
"requirements": []
}
```


<p>它可以被正常解析&#xff0c;却丢失了关键业务信息。</p>
<p>很多人第一反应是优化 Prompt&#xff0c;比如反复强调&#xff1a;</p>


```text
请严格输出 JSON，不要输出解释，不要输出 markdown，不要遗漏字段。
```


<p>这些提示当然有用&#xff0c;但它们只能降低错误概率&#xff0c;不能从根本上解决问题。</p>
<p>因为大模型输出 JSON 不完整&#xff0c;本质上不是一个单纯的 Prompt 问题&#xff0c;而是一个工程稳定性问题。</p>
<p>要真正解决它&#xff0c;需要从模型调用、输出协议、Schema 设计、校验修复、重试补偿、任务拆分等多个层面共同治理。</p>
<hr />
<h3>一、问题本质&#xff1a;大模型不是 JSON 序列化器</h3>
<p>在传统程序里&#xff0c;JSON 通常是由代码序列化生成的。</p>
<p>例如&#xff1a;</p>


```ts
JSON.stringify(data)
```


<p>这种 JSON 输出天然具备结构完整性。</p>
<p>但大模型不是在执行 <code>JSON.stringify</code>&#xff0c;而是在逐 token 生成文本。</p>
<p>这意味着它可能出现&#xff1a;</p>
<ul><li>生成到一半达到输出长度上限&#xff1b;</li><li>生成过程中偏离结构&#xff1b;</li><li>忘记闭合数组或对象&#xff1b;</li><li>字符串引号不完整&#xff1b;</li><li>输出了额外解释性文字&#xff1b;</li><li>字段漏掉&#xff1b;</li><li>数组内容少了一部分&#xff1b;</li><li>内容看似合理但不符合业务约束。</li></ul>
<p>所以&#xff0c;大模型输出 JSON 的正确理解应该是&#xff1a;</p>
<blockquote>
<p>模型负责“生成候选结构化内容”&#xff0c;工程系统负责“验证、修复、重试、合并和入库”。</p>
</blockquote>
<p>不能把大模型直接当成稳定的 JSON 序列化器。</p>
<hr />
<p><img src="https://i-blog.csdnimg.cn/direct/c6642342643247e6aea86e7e67efd1a3.png" alt="在这里插入图片描述" /></p>
<h3>二、典型问题分类</h3>
<p>大模型输出 JSON 不完整&#xff0c;通常可以分为两类。</p>
<h4>1. 语法不完整</h4>
<p>这类问题比较明显&#xff0c;程序一解析就会报错。</p>
<p>例如&#xff1a;</p>


```json
{<!-- -->
"title": "文件上传一致性治理",
"tags": ["大模型", "工程化", "JSON"
```


<p>常见表现包括&#xff1a;</p>
<ul><li>少了右大括号 <code>}</code>&#xff1b;</li><li>少了右中括号 <code>]</code>&#xff1b;</li><li>字符串没有闭合&#xff1b;</li><li>JSON 中夹杂了 markdown&#xff1b;</li><li>输出了多余的自然语言&#xff1b;</li><li>多了尾逗号&#xff1b;</li><li>使用了非法注释&#xff1b;</li><li>使用了单引号&#xff1b;</li><li>key 没有加双引号。</li></ul>
<p>这类问题一般可以通过 JSON Repair、格式清洗、模型重试来解决。</p>
<h4>2. 内容不完整</h4>
<p>这类问题更隐蔽。</p>
<p>例如要求模型输出&#xff1a;</p>


```json
{<!-- -->
"basicInfo": {<!-- -->},
"timeline": [],
"qualificationRequirements": [],
"scoringRules": [],
"riskPoints": []
}
```


<p>模型返回&#xff1a;</p>


```json
{<!-- -->
"basicInfo": {<!-- -->
"projectName": "某某项目"
},
"timeline": [],
"qualificationRequirements": [],
"scoringRules": [],
"riskPoints": []
}
```


<p>这个 JSON 是合法的&#xff0c;但业务上可能是错误的。</p>
<p>因为原文里明明有大量资质要求、评分规则和风险点&#xff0c;模型却没有抽取出来。</p>
<p>所以&#xff0c;JSON 完整性不能只看语法&#xff0c;还要看业务字段是否完整。</p>
<hr />
<h3>三、为什么大模型会输出不完整 JSON</h3>
<h4>1. 输出内容太长&#xff0c;被截断</h4>
<p>这是最常见原因。</p>
<p>当你要求模型一次性输出一个很大的 JSON&#xff0c;例如&#xff1a;</p>


```json
{<!-- -->
"documents": [
{<!-- -->
"title": "...",
"sections": [
{<!-- -->
"heading": "...",
"clauses": [
"...",
"...",
"..."
]
}
]
}
]
}
```


<p>如果数组很长、字段很多、内容很大&#xff0c;模型很容易在输出中途停止。</p>
<p>这时即使 Prompt 写得再严格&#xff0c;也挡不住输出被截断。</p>
<p>本质原因是&#xff1a;单次模型响应有输出长度限制。</p>
<h4>2. 输入上下文太长&#xff0c;压缩了输出空间</h4>
<p>很多业务场景会把大量内容塞给模型&#xff0c;例如&#xff1a;</p>
<ul><li>招投标公告&#xff1b;</li><li>合同文本&#xff1b;</li><li>法律条款&#xff1b;</li><li>网页正文&#xff1b;</li><li>OCR 结果&#xff1b;</li><li>日志文件&#xff1b;</li><li>多轮对话记录&#xff1b;</li><li>RAG 检索上下文。</li></ul>
<p>输入越长&#xff0c;留给输出的空间就越少。</p>
<p>如果你把十几页文档塞进去&#xff0c;又要求模型输出一个完整复杂 JSON&#xff0c;就很容易出现后半截缺失。</p>
<h4>3. Schema 设计过于复杂</h4>
<p>有些 JSON Schema 嵌套太深&#xff1a;</p>


```json
{<!-- -->
"project": {<!-- -->
"basic": {<!-- -->},
"buyer": {<!-- -->},
"supplier": {<!-- -->},
"timeline": [],
"qualification": {<!-- -->
"company": [],
"personnel": [],
"performance": [],
"financial": []
},
"scoring": {<!-- -->
"business": [],
"technical": [],
"price": []
},
"risks": []
}
}
```


<p>这类结构虽然看起来规范&#xff0c;但对模型输出稳定性并不友好。</p>
<p>Schema 越复杂&#xff0c;模型越容易&#xff1a;</p>
<ul><li>漏字段&#xff1b;</li><li>放错层级&#xff1b;</li><li>数组对象结构不一致&#xff1b;</li><li>部分字段为空&#xff1b;</li><li>嵌套括号没有闭合。</li></ul>
<h4>4. 只靠 Prompt&#xff0c;没有程序校验</h4>
<p>很多系统只做了这一步&#xff1a;</p>


```text
请严格输出 JSON。
```


<p>然后就直接把模型输出传给业务系统。</p>
<p>这是非常危险的。</p>
<p>因为 Prompt 只是一种软约束&#xff0c;不能提供工程级保证。</p>
<p>只要模型输出进入业务系统&#xff0c;就必须经过程序化校验。</p>
<hr />
<h3>四、整体治理思路</h3>
<p>大模型 JSON 输出不稳定&#xff0c;不能靠单点优化解决&#xff0c;而应该设计一条完整的输出治理链路。</p>
<h4>图 1&#xff1a;大模型 JSON 输出治理链路</h4>
<div class="mermaid mermaid-newversion mermaid-flowchart"></div>
<p>这条链路的核心思想是&#xff1a;</p>


```text
模型输出不是终点，而是候选结果。
候选结果必须经过解析、校验、修复、重试和业务验收。
```


<p>推荐的治理顺序是&#xff1a;</p>


```text
结构化输出 > 拆分任务 > Schema 校验 > 自动修复 > 重试补偿 > 人工兜底
```


<hr />
<h3>五、第一层治理&#xff1a;优先使用结构化输出能力</h3>
<p>如果使用的模型平台支持结构化输出、函数调用、工具调用或 JSON Schema 约束&#xff0c;应该优先使用这些能力。</p>
<p>不要只在 Prompt 里写&#xff1a;</p>


```text
请输出 JSON。
```


<p>而应该显式定义输出结构。</p>
<p>例如抽取文章信息&#xff0c;可以定义为&#xff1a;</p>


```json
{<!-- -->
"title": "string",
"summary": "string",
"keywords": ["string"],
"sections": [
{<!-- -->
"heading": "string",
"summary": "string"
}
]
}
```


<p>结构化输出的优势是&#xff1a;</p>
<ul><li>减少非法 JSON&#xff1b;</li><li>限制模型输出范围&#xff1b;</li><li>降低额外解释文字&#xff1b;</li><li>提高字段稳定性&#xff1b;</li><li>方便后端做 Schema 校验。</li></ul>
<p>但要注意&#xff1a;</p>
<blockquote>
<p>结构化输出只能提升格式稳定性&#xff0c;不能保证业务内容一定完整。</p>
</blockquote>
<p>例如模型可以稳定输出&#xff1a;</p>


```json
{<!-- -->
"qualificationRequirements": []
}
```


<p>但这不代表原文里真的没有资质要求。</p>
<p>所以结构化输出后&#xff0c;仍然需要业务完整性校验。</p>
<hr />
<h3>六、第二层治理&#xff1a;不要一次性生成大 JSON</h3>
<p>这是最重要的一条工程经验。</p>
<p>很多 JSON 不完整问题&#xff0c;根源都是一次性输出太多。</p>
<p>错误做法&#xff1a;</p>


```text
请从这篇长文档中抽取所有信息，并输出完整 JSON。
```


<p>推荐做法&#xff1a;</p>


```text
第一步：只抽取基础信息
第二步：只抽取时间节点
第三步：只抽取资质要求
第四步：只抽取评分规则
第五步：只抽取风险点
第六步：由程序合并最终 JSON
```


<h4>图 2&#xff1a;从“大 JSON”改成“小 JSON 合并”</h4>
<div class="mermaid mermaid-newversion mermaid-flowchart"></div>
<p>例如&#xff0c;不要让模型一次性输出&#xff1a;</p>


```json
{<!-- -->
"basicInfo": {<!-- -->},
"timeline": [],
"qualificationRequirements": [],
"scoringRules": [],
"riskPoints": []
}
```


<p>而是拆成多个小 JSON&#xff1a;</p>


```json
{<!-- -->
"basicInfo": {<!-- -->
"projectName": "xxx",
"budget": "xxx",
"buyer": "xxx"
}
}
```




```json
{<!-- -->
"timeline": [
{<!-- -->
"name": "报名截止时间",
"date": "2026-07-10"
}
]
}
```




```json
{<!-- -->
"qualificationRequirements": [
"要求一",
"要求二"
]
}
```


<p>最后由后端程序合并&#xff0c;而不是让模型一次性输出一个巨大结构。</p>
<p>这样做有几个好处&#xff1a;</p>
<ul><li>单次输出更短&#xff1b;</li><li>JSON 更容易闭合&#xff1b;</li><li>单块失败可以单独重试&#xff1b;</li><li>不同字段可以使用不同 Prompt&#xff1b;</li><li>后端合并过程更可控&#xff1b;</li><li>更适合生产环境排查问题。</li></ul>
<hr />
<h3>七、第三层治理&#xff1a;数组内容分页输出</h3>
<p>如果某个字段本身就是长数组&#xff0c;例如&#xff1a;</p>
<ul><li>合同条款&#xff1b;</li><li>招标要求&#xff1b;</li><li>评分规则&#xff1b;</li><li>风险点&#xff1b;</li><li>表格行&#xff1b;</li><li>网页列表&#xff1b;</li><li>商品信息&#xff1b;</li><li>日志事件&#xff1b;</li><li>审计问题。</li></ul>
<p>不建议一次性让模型输出全部数组。</p>
<p>推荐使用分页协议。</p>
<p>例如&#xff1a;</p>


```json
{<!-- -->
"page": 1,
"pageSize": 20,
"hasMore": true,
"items": [
{<!-- -->
"index": 1,
"content": "..."
}
]
}
```


<p>下一次请求&#xff1a;</p>


```text
请继续输出第 2 页，只输出 JSON，不要重复第 1 页内容。
```


<p>返回&#xff1a;</p>


```json
{<!-- -->
"page": 2,
"pageSize": 20,
"hasMore": false,
"items": [
{<!-- -->
"index": 21,
"content": "..."
}
]
}
```


<h4>图 3&#xff1a;长数组分页抽取</h4>
<div class="mermaid mermaid-newversion mermaid-sequence"></div>
<p>分页抽取的本质是&#xff1a;</p>
<blockquote>
<p>让模型每次只完成一个小而确定的输出任务。</p>
</blockquote>
<p>这比单次生成超大 JSON 稳定得多。</p>
<hr />
<h3>八、第四层治理&#xff1a;后端必须做 Schema 校验</h3>
<p>无论 Prompt 多严格&#xff0c;后端都必须校验模型输出。</p>
<p>一个基本的处理流程应该是&#xff1a;</p>


```text
1. 接收模型原始输出
2. 清理 markdown 代码块
3. 尝试 JSON.parse
4. 使用 Schema 校验字段
5. 校验业务必填项
6. 失败则修复或重试
7. 通过后再入库
```


<p>以 TypeScript 为例&#xff0c;可以使用 Zod&#xff1a;</p>


```ts
import {<!-- --> z } from "zod";

const ExtractResultSchema = z.object({<!-- -->
title: z.string().nullable(),
summary: z.string().nullable(),
keywords: z.array(z.string()),
risks: z.array(z.string()),
});

function parseAndValidate(raw: string) {<!-- -->
const cleaned = cleanModelOutput(raw);
const parsed = JSON.parse(cleaned);
return ExtractResultSchema.parse(parsed);
}
```


<p>其中 <code>cleanModelOutput</code> 可以处理模型常见输出问题&#xff1a;</p>


````ts
function cleanModelOutput(raw: string): string {<!-- -->
return raw
.trim()
.replace(/^```json\s*/i, "")
.replace(/^```\s*/i, "")
.replace(/```$/i, "")
.trim();
}
````


<p>Schema 校验至少要解决三类问题：</p>

<table><thead><tr><th>校验类型</th><th>目标</th></tr></thead><tbody><tr><td>语法校验</td><td>JSON 是否能 parse</td></tr><tr><td>结构校验</td><td>字段类型是否正确</td></tr><tr><td>业务校验</td><td>字段是否满足业务完整性</td></tr></tbody></table><p>只做 <code>JSON.parse</code> 是不够的。</p>
<p>因为下面这个 JSON 可以 parse，但业务上可能不可接受：</p>


```json
{<!-- -->
"title": null,
"summary": null,
"keywords": [],
"risks": []
}
```


<hr />
<h3>九、第五层治理&#xff1a;JSON Repair 只能修语法&#xff0c;不能修业务</h3>
<p>当模型输出 JSON 语法错误时&#xff0c;可以使用 JSON Repair 类工具尝试修复。</p>
<p>例如模型输出&#xff1a;</p>


```json
{<!-- -->
"title": "大模型工程化",
"tags": ["LLM", "JSON", "结构化输出"
```


<p>Repair 之后可能变成&#xff1a;</p>


```json
{<!-- -->
"title": "大模型工程化",
"tags": ["LLM", "JSON", "结构化输出"]
}
```


<p>这对语法错误很有帮助。</p>
<p>但是要注意&#xff1a;</p>
<blockquote>
<p>JSON Repair 只能修语法&#xff0c;不能保证内容完整。</p>
</blockquote>
<p>如果模型本来只输出了一半&#xff0c;Repair 工具只是帮你补上括号&#xff0c;让它变成合法 JSON。</p>
<p>但业务内容仍然可能缺失。</p>
<p>所以修复后必须继续做&#xff1a;</p>
<ul><li>Schema 校验&#xff1b;</li><li>必填字段校验&#xff1b;</li><li>数组长度校验&#xff1b;</li><li>内容覆盖率校验&#xff1b;</li><li>是否疑似截断判断&#xff1b;</li><li>与原文的引用关系校验。</li></ul>
<p>一个常见策略是&#xff1a;</p>


```text
JSON.parse 失败
↓
JSON Repair
↓
再次 JSON.parse
↓
Schema 校验
↓
业务完整性校验
↓
仍失败则重试
```


<h4>图 4&#xff1a;JSON 修复不是终点</h4>
<div class="mermaid mermaid-newversion mermaid-flowchart"></div>
<hr />
<h3>十、第六层治理&#xff1a;重试要有策略&#xff0c;不能盲目重试</h3>
<p>很多系统遇到模型输出不完整后&#xff0c;会简单重试一次。</p>
<p>但如果不分析失败原因&#xff0c;盲目重试可能效果很差。</p>
<p>推荐按错误类型设计不同重试策略。</p>

<table><thead><tr><th>错误类型</th><th>重试策略</th></tr></thead><tbody><tr><td>JSON 语法错误</td><td>降低输出复杂度&#xff0c;要求只输出修复后的 JSON</td></tr><tr><td>字段缺失</td><td>针对缺失字段单独补充抽取</td></tr><tr><td>数组截断</td><td>使用分页或 continuation 协议</td></tr><tr><td>内容为空</td><td>换 Prompt 或提供更小上下文</td></tr><tr><td>Schema 不匹配</td><td>强化字段类型要求</td></tr><tr><td>多次失败</td><td>进入人工审核或降级流程</td></tr></tbody></table><p>例如字段缺失时&#xff0c;不要重新抽取全部内容&#xff0c;而是只抽缺失字段&#xff1a;</p>


```text
上一次抽取结果中 qualificationRequirements 为空。
请只从原文中抽取资质要求。
只输出如下 JSON：
{
"qualificationRequirements": []
}
```


<p>这样比重新生成整个大 JSON 更稳定。</p>
<hr />
<h3>十一、第七层治理&#xff1a;设计 continuation 断点续写协议</h3>
<p>如果 JSON 已经被截断&#xff0c;也可以设计续写机制。</p>
<p>但不要简单说&#xff1a;</p>


```text
继续。
```


<p>因为模型可能会&#xff1a;</p>
<ul><li>从头开始输出&#xff1b;</li><li>输出重复内容&#xff1b;</li><li>忘记之前的结构&#xff1b;</li><li>输出解释性文字&#xff1b;</li><li>继续生成非法 JSON。</li></ul>
<p>更好的方式是设计明确的 continuation 协议。</p>
<p>例如&#xff1a;</p>


```json
{<!-- -->
"continueFrom": 21,
"items": []
}
```


<p>请求模型&#xff1a;</p>


```text
上一次输出在 items 第 20 项后被截断。
请从第 21 项开始继续。
不要重复前 20 项。
只输出 continuation JSON：
{
"continueFrom": 21,
"items": []
}
```


<p>后端拿到结果后&#xff0c;不是直接拼接字符串&#xff0c;而是解析 JSON 后把 <code>items</code> 追加到已有数组。</p>
<p>注意&#xff1a;</p>
<blockquote>
<p>不推荐直接拼接两段 JSON 字符串&#xff0c;推荐用结构化 continuation 结果做程序合并。</p>
</blockquote>
<hr />
<h3>十二、第八层治理&#xff1a;将“大模型抽取”变成可观测任务</h3>
<p>在生产系统里&#xff0c;模型 JSON 输出失败不能只体现在日志里。</p>
<p>建议把每次抽取任务都记录下来。</p>
<p>至少记录&#xff1a;</p>
<ul><li>taskId&#xff1b;</li><li>model&#xff1b;</li><li>promptVersion&#xff1b;</li><li>inputHash&#xff1b;</li><li>schemaVersion&#xff1b;</li><li>rawOutput&#xff1b;</li><li>cleanedOutput&#xff1b;</li><li>parseStatus&#xff1b;</li><li>validationStatus&#xff1b;</li><li>retryCount&#xff1b;</li><li>errorType&#xff1b;</li><li>errorMessage&#xff1b;</li><li>finalStatus&#xff1b;</li><li>createdAt&#xff1b;</li><li>updatedAt。</li></ul>
<p>这样可以回答几个关键问题&#xff1a;</p>
<ul><li>哪类文档最容易失败&#xff1f;</li><li>哪个字段最容易缺失&#xff1f;</li><li>哪个 Prompt 版本质量更好&#xff1f;</li><li>哪个模型输出 JSON 更稳定&#xff1f;</li><li>是输入太长导致失败&#xff0c;还是 Schema 太复杂&#xff1f;</li><li>重试成功率是多少&#xff1f;</li><li>Repair 成功率是多少&#xff1f;</li></ul>
<h4>图 5&#xff1a;结构化抽取任务的可观测闭环</h4>
<div class="mermaid mermaid-newversion mermaid-flowchart"></div>
<p>没有可观测性&#xff0c;就很难持续提升结构化输出稳定性。</p>
<hr />
<h3>十三、推荐的工程架构</h3>
<p>在真实系统中&#xff0c;可以把大模型结构化输出设计成一个独立的 Extractor Pipeline。</p>
<h4>图 6&#xff1a;LLM 结构化抽取 Pipeline 架构</h4>
<div class="mermaid mermaid-newversion mermaid-flowchart"></div>
<p>每个模块职责如下&#xff1a;</p>

<table><thead><tr><th>模块</th><th>作用</th></tr></thead><tbody><tr><td>Input Normalizer</td><td>清洗 HTML、OCR 噪声、特殊字符</td></tr><tr><td>Chunker</td><td>将长文本拆成可控片段</td></tr><tr><td>Prompt Builder</td><td>根据任务类型和 Schema 构造 Prompt</td></tr><tr><td>LLM Client</td><td>调用模型并控制参数</td></tr><tr><td>Output Cleaner</td><td>去除 markdown、无关解释</td></tr><tr><td>JSON Parser</td><td>解析 JSON</td></tr><tr><td>Schema Validator</td><td>校验类型和结构</td></tr><tr><td>Business Validator</td><td>校验字段业务完整性</td></tr><tr><td>Retry Or Repair</td><td>修复、重试、补充抽取</td></tr><tr><td>Result Merger</td><td>合并多个小 JSON</td></tr><tr><td>Storage</td><td>保存最终结构化结果</td></tr></tbody></table><p>这套架构的关键不是“某个 Prompt 写得多好”&#xff0c;而是把模型输出当成一个不稳定外部依赖来治理。</p>
<hr />
<h3>十四、业务完整性校验怎么做</h3>
<p>业务完整性校验通常比 JSON 语法校验更重要。</p>
<p>以文档信息抽取为例&#xff0c;可以设计以下规则&#xff1a;</p>
<h4>1. 必填字段校验</h4>
<p>例如&#xff1a;</p>


```text
title 不能为空
source 不能为空
publishDate 不能为空
```


<p>如果为空&#xff0c;就标记为字段缺失。</p>
<h4>2. 数组最小长度校验</h4>
<p>例如&#xff1a;</p>


```text
如果原文中出现“资格要求”“投标人须具备”等关键词，
qualificationRequirements 不应该为空。
```


<h4>3. 内容引用校验</h4>
<p>要求模型输出每个结论时附带来源片段&#xff1a;</p>


```json
{<!-- -->
"value": "投标人须具备建筑工程施工总承包三级及以上资质",
"evidence": "原文引用片段..."
}
```


<p>这样可以降低幻觉&#xff0c;也方便人工审核。</p>
<h4>4. 覆盖率校验</h4>
<p>对于分块抽取&#xff0c;可以检查&#xff1a;</p>


```text
每个 chunk 是否都被处理
每个 chunk 是否都有结果
是否存在异常空结果
```


<h4>5. 异常值校验</h4>
<p>例如&#xff1a;</p>
<ul><li>日期格式不合法&#xff1b;</li><li>金额单位异常&#xff1b;</li><li>电话号码格式异常&#xff1b;</li><li>数组项重复率过高&#xff1b;</li><li>输出内容和原文语言不一致&#xff1b;</li><li>字段内容明显跑题。</li></ul>
<p>这些业务校验可以帮助系统发现“合法但不可用”的 JSON。</p>
<hr />
<h3>十五、Prompt 设计建议</h3>
<p>虽然 Prompt 不能解决所有问题&#xff0c;但好的 Prompt 仍然很重要。</p>
<p>一个比较稳的 Prompt 可以这样设计&#xff1a;</p>


````text
你是一个结构化信息抽取器。

任务：
从给定文本中抽取指定字段，并严格输出 JSON。

要求：
1. 只输出 JSON，不要输出 markdown。
2. 不要使用 ```json 代码块。
3. 所有字段必须出现。
4. 不确定的字段填 null。
5. 数组没有内容时返回 []。
6. 不要省略字段。
7. 不要输出解释性文字。
8. 不要编造原文不存在的信息。
9. 每个抽取结果尽量保留原文表达。
10. 输出必须符合指定 Schema。

Schema:
{
"title": "string | null",
"summary": "string | null",
"keywords": "string[]",
"risks": "string[]"
}

待抽取文本：
{<!-- -->{input}}
````


<p>对于复杂任务&#xff0c;还可以进一步要求&#xff1a;</p>


```text
如果字段无法从原文找到，请填 null 或 []，不要猜测。
如果数组内容超过 20 条，只返回前 20 条，并设置 hasMore=true。
```


<p>但仍然要记住&#xff1a;</p>
<blockquote>
<p>Prompt 是第一层防线&#xff0c;程序校验才是最后一道防线。</p>
</blockquote>
<hr />
<h3>十六、参数设置也会影响 JSON 稳定性</h3>
<p>模型调用参数也会影响结构化输出质量。</p>
<p>一般来说&#xff0c;结构化抽取任务不需要太强创造性&#xff0c;建议&#xff1a;</p>
<ul><li>降低 temperature&#xff1b;</li><li>控制 max output tokens&#xff1b;</li><li>对长文本先分块&#xff1b;</li><li>尽量使用结构化输出模式&#xff1b;</li><li>避免在同一次调用里要求模型做太多任务&#xff1b;</li><li>不要把“抽取、总结、判断、改写、生成建议”混在一个 JSON 里。</li></ul>
<p>一个典型错误是&#xff1a;</p>


```text
请抽取信息、总结文章、分析风险、生成建议、输出完整 JSON。
```


<p>这会显著增加失败概率。</p>
<p>更好的做法是拆成多个任务&#xff1a;</p>


```text
抽取信息
总结内容
分析风险
生成建议
```


<p>每个任务输出一个小 JSON。</p>
<hr />
<h3>十七、生产环境推荐处理流程</h3>
<p>综合来看&#xff0c;一个比较成熟的生产级处理流程如下&#xff1a;</p>


```text
1. 输入清洗
2. 文本分块
3. 每个分块独立抽取小 JSON
4. 模型使用结构化输出或严格 Schema
5. 清洗模型输出
6. JSON.parse
7. JSON Repair
8. Schema 校验
9. 业务完整性校验
10. 失败按错误类型重试
11. 多个小 JSON 合并
12. 合并后去重、排序、归一化
13. 最终结果二次校验
14. 入库
15. 记录任务日志和质量指标
```


<p>可以概括为&#xff1a;</p>


```text
不要追求一次模型调用解决所有问题，
而要把结构化输出设计成一条可恢复、可重试、可观测的工程链路。
```


<hr />
<h3>十八、一个完整的 TypeScript 示例</h3>
<p>下面是一个简化版的工程处理流程。</p>


````ts
import {<!-- --> z } from "zod";

const ResultSchema = z.object({<!-- -->
title: z.string().nullable(),
summary: z.string().nullable(),
keywords: z.array(z.string()),
risks: z.array(z.string()),
});

type ExtractResult = z.infer<typeof ResultSchema>;

function cleanOutput(raw: string): string {<!-- -->
return raw
.trim()
.replace(/^```json\s*/i, "")
.replace(/^```\s*/i, "")
.replace(/```$/i, "")
.trim();
}

async function repairJson(raw: string): Promise<string> {<!-- -->
// 实际项目中可以接入 json repair 库，或者调用模型做修复
return raw;
}

async function callModel(input: string): Promise<string> {<!-- -->
// 调用大模型
return "";
}

function businessValidate(result: ExtractResult) {<!-- -->
const errors: string[] = [];

if (!result.title) {<!-- -->
errors.push("title is empty");
}

if (!result.summary) {<!-- -->
errors.push("summary is empty");
}

return {<!-- -->
ok: errors.length === 0,
errors,
};
}

async function extractWithRetry(input: string, maxRetry = 2) {<!-- -->
let lastError: unknown;

for (let i = 0; i <= maxRetry; i++) {<!-- -->
try {<!-- -->
const raw = await callModel(input);
const cleaned = cleanOutput(raw);

let parsed: unknown;

try {<!-- -->
parsed = JSON.parse(cleaned);
} catch {<!-- -->
const repaired = await repairJson(cleaned);
parsed = JSON.parse(repaired);
}

const result = ResultSchema.parse(parsed);
const businessCheck = businessValidate(result);

if (!businessCheck.ok) {<!-- -->
throw new Error(
`Business validation failed: ${<!-- -->businessCheck.errors.join(", ")}`
);
}

return result;
} catch (error) {<!-- -->
lastError = error;
}
}

throw lastError;
}
````


<p>这个示例体现了几个关键点：</p>
<ul><li>不直接信任模型输出；</li><li>先清洗；</li><li>再解析；</li><li>解析失败尝试修复；</li><li>修复后继续 Schema 校验；</li><li>Schema 通过后还要做业务校验；</li><li>失败可以重试；</li><li>重试仍失败要抛出明确错误。</li></ul>
<hr /></hr>
<h3>十九、最佳实践总结</h3>
<h4>1. 不要让模型一次性输出巨大 JSON</h4>
<p>只要 JSON 大到你担心它会不会完整，就说明它应该被拆分。</p>
<h4>2. 复杂任务分阶段完成</h4>
<p>抽基础信息、抽列表、抽风险、做总结，最好分开调用。</p>
<h4>3. 长数组分页输出</h4>
<p>不要让模型一次输出几百条数组项。</p>
<h4>4. 使用结构化输出能力</h4>
<p>有 JSON Schema、Function Calling、Tool Calling 能力时优先使用。</p>
<h4>5. 后端必须做 Schema 校验</h4>
<p>模型输出不能直接入库。</p>
<h4>6. JSON Repair 只是语法兜底</h4>
<p>它不能保证业务内容完整。</p>
<h4>7. 重试要按错误类型设计</h4>
<p>字段缺失就补字段，数组截断就分页，不要无脑全量重试。</p>
<h4>8. 关键字段要带 evidence</h4>
<p>结构化抽取最好要求模型返回原文证据，方便校验和人工审核。</p>
<h4>9. 抽取任务要可观测</h4>
<p>记录 raw output、parse status、schema version、retry count、error type。</p>
<h4>10. 把模型当成不稳定外部依赖</h4>
<p>像治理第三方接口一样治理模型输出。</p>
<hr /></hr>
<h3>二十、结语</h3>
<p>大模型输出 JSON 不完整，是 AI 应用工程化中非常典型的问题。</p>
<p>它表面上看是模型“不听话”或者 Prompt“不够严格”，但本质上是系统没有为不稳定输出建立足够的工程防线。</p>
<p>真正可靠的方案不是反复强调“请严格输出 JSON”，而是建立一套完整机制：</p>


```text
结构化输出
任务拆分
分页协议
Schema 校验
业务校验
JSON Repair
错误分类
自动重试
断点续写
结果合并
任务观测
人工兜底
```


<p>当这套机制建立起来后&#xff0c;大模型就不再是一个不可控的文本生成器&#xff0c;而会变成一个可以被工程系统约束、校验、修复和治理的结构化信息抽取组件。</p>
<p>一句话总结&#xff1a;</p>
<blockquote>
<p>不要把大模型当 JSON 序列化器&#xff0c;要把它当信息抽取器&#xff1b;JSON 的完整性、合法性和可靠性&#xff0c;必须由工程系统兜底。</p>
</blockquote>
