---
title: "RAG 优化实践：别让分块毁掉你的知识库"
description: "CSDN 原文全文镜像：RAG优化实践：保护Markdown文档结构的三个关键点 本文针对RAG系统中Markdown文档处理面临的三大问题提出解决方案： 表格截断问题：提出\"原子语义块\"概念，建议小表格整体保留，大表格按行分组并重复表头，同时生成语义摘要辅助……"
pageType: article
module: rag
updated: '2026-06-10'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "rag"
  - "人工智能"
  - "大模型"
  - "软件工程"
  - "RAG"
level: intermediate
prerequisites:
  - "/llms/rag/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-06-10，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-06-10。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/161868156](https://blog.csdn.net/m0_63309778/article/details/161868156)
- 站内分区：RAG / RAG 文档分块
:::

<p><img src="https://i-blog.csdnimg.cn/direct/ccb442a6b66e4e0f90429df36238f5e3.png" alt="在这里插入图片描述" /></p>
<h2>RAG 优化实践&#xff1a;如何解决 Markdown 表格截断、代码块丢失和图片语义缺失问题</h2>
<p>很多人做 RAG 优化时&#xff0c;第一反应是调参数&#xff1a;<code>chunk_size</code> 要不要大一点&#xff1f;<code>top_k</code> 要不要多召回几个&#xff1f;embedding 模型要不要换&#xff1f;rerank 要不要加&#xff1f;</p>
<p>这些当然重要&#xff0c;但在真实项目里&#xff0c;我越来越感觉到一个问题&#xff1a;<strong>很多 RAG 效果差&#xff0c;并不是模型不够强&#xff0c;也不是向量库不够好&#xff0c;而是文档在进入向量库之前&#xff0c;结构已经被破坏了。</strong></p>
<p>尤其是 Markdown、PDF 转 Markdown、Word 转 Markdown 这类复杂文档&#xff0c;经常会出现几个典型问题&#xff1a;</p>
<ul><li>Markdown 表格被截断&#xff0c;表头和数据行分离&#xff1b;</li><li>代码块被切成两半&#xff0c;函数逻辑不完整&#xff1b;</li><li>图片只剩一个 OSS 地址&#xff0c;没有任何语义&#xff1b;</li><li>标题、正文、表格、图片之间的上下文关系丢失&#xff1b;</li><li>检索命中了片段&#xff0c;但真正回答时缺少完整证据。</li></ul>
<p>这篇文章想聊的不是某一个具体框架&#xff0c;而是一个更工程化的问题&#xff1a;<strong>复杂文档进入 RAG 系统之前&#xff0c;应该如何做结构保护、语义增强和多表示检索。</strong></p>
<hr />
<h3>一、为什么普通分块会破坏 Markdown 文档&#xff1f;</h3>
<p>很多 RAG 系统一开始都会使用固定长度分块&#xff0c;比如按照 500、800、1000 tokens 切分。这种方式对普通段落文本还算可用&#xff0c;但对 Markdown 文档来说非常危险。</p>
<p>比如下面这个表格&#xff1a;</p>


```markdown
| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | string | 用户唯一标识 |
| created_at | datetime | 创建时间 |
| status | int | 用户状态 |
```


<p>如果分块器按照字符数一刀切&#xff0c;很可能切成这样&#xff1a;</p>


```markdown
| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | string |
```


<p>另一个 chunk 里是&#xff1a;</p>


```markdown
用户唯一标识 |
| created_at | datetime | 创建时间 |
| status | int | 用户状态 |
```


<p>这样一来&#xff0c;问题就很明显了&#xff1a;</p>
<p>第一&#xff0c;表头和数据行可能分离。<br />
第二&#xff0c;某些单元格语义不完整。<br />
第三&#xff0c;embedding 时表格语义变弱。<br />
第四&#xff0c;检索命中了片段&#xff0c;但 LLM 看不到完整上下文。<br />
第五&#xff0c;回答时容易出现“看起来引用了资料&#xff0c;但其实资料不完整”的幻觉。</p>
<p>所以复杂文档的 RAG 优化&#xff0c;第一步不是调大 <code>chunk_size</code>&#xff0c;而是先问一个问题&#xff1a;</p>
<blockquote>
<p>当前分块方式有没有破坏原始文档的结构&#xff1f;</p>
</blockquote>
<hr />
<h3>二、哪些内容应该被保护&#xff1f;</h3>
<p>在 Markdown 文档里&#xff0c;有一些内容不能被普通分块器随意切开。我把这类内容称为&#xff1a;</p>
<blockquote>
<p><strong>Atomic Semantic Block&#xff0c;原子语义块。</strong></p>
</blockquote>
<p>所谓原子语义块&#xff0c;就是一旦被切断&#xff0c;语义就会明显受损的内容。</p>
<p>常见类型包括&#xff1a;</p>

<table><thead><tr><th>类型</th><th>为什么要保护</th><th>推荐处理方式</th></tr></thead><tbody><tr><td>Markdown 表格</td><td>表头、字段、行列关系一旦断开&#xff0c;语义就残缺</td><td>小表整体保留&#xff0c;大表按行分组但重复表头</td></tr><tr><td>代码块</td><td>函数、类、SQL、JSON 被截断后难以理解</td><td>fenced code block 整体保护</td></tr><tr><td>图片 OSS 地址</td><td>URL 本身没有语义&#xff0c;但图片可能包含关键证据</td><td>提取图片上下文&#xff0c;生成图片摘要</td></tr><tr><td>Mermaid / PlantUML</td><td>流程图关系不能被切断</td><td>整体保存&#xff0c;并生成流程摘要</td></tr><tr><td>JSON / YAML</td><td>层级结构被切断会导致字段含义丢失</td><td>按对象层级切分</td></tr><tr><td>SQL</td><td>join、where、group by 等逻辑断开会误导模型</td><td>按完整 SQL 或逻辑段切分</td></tr><tr><td>法条 / 合同条款</td><td>条款上下文连续性强</td><td>按条款编号和标题层级切分</td></tr><tr><td>公式 / 配置项</td><td>单独一部分通常没有意义</td><td>尽量整体保留</td></tr></tbody></table><p>这类内容不能简单按长度切。它们需要先被识别出来&#xff0c;然后作为独立 block 进入后续处理流程。</p>
<hr />
<h3>三、表格分块的关键&#xff1a;不是绝对不切&#xff0c;而是结构保真</h3>
<p>表格是 RAG 里最容易出问题的内容之一。</p>
<p>很多人会说&#xff1a;“那我把表格整体保留不就行了吗&#xff1f;”</p>
<p>这只适合小表格。真实企业文档里的表格可能非常长&#xff0c;比如项目清单、风险清单、预算明细、人员名单、设备清单、合同条款对照表。如果一个表格几千行&#xff0c;全部塞进一个 chunk 并不现实。</p>
<p>所以表格分块的核心不是“不切”&#xff0c;而是&#xff1a;</p>
<blockquote>
<p><strong>切完以后&#xff0c;仍然保留表格的语义结构。</strong></p>
</blockquote>
<p>可以分成几种情况处理。</p>
<h4>1. 小表格&#xff1a;整体保留</h4>
<p>如果表格本身不大&#xff0c;最好的方式就是完整作为一个 block 保存。</p>


```markdown
表格标题：用户字段说明表

| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | string | 用户唯一标识 |
| created_at | datetime | 创建时间 |
| status | int | 用户状态 |
```


<p>这种情况下&#xff0c;表头、字段、说明都在一起&#xff0c;检索和生成都比较稳定。</p>
<h4>2. 中等表格&#xff1a;整体保存原文&#xff0c;同时生成摘要</h4>
<p>中等表格可以完整保存 raw table&#xff0c;同时额外生成一份语义摘要。</p>
<p>原始表格用于回答&#xff0c;摘要用于检索。</p>
<p>例如原始表格是&#xff1a;</p>


```markdown
| 指标 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| 营收 | 1200 | 1500 | 2100 |
| 毛利率 | 28% | 31% | 35% |
```


<p>可以生成摘要&#xff1a;</p>


```text
该表展示了公司 2022 年至 2024 年的核心经营指标，包括营收和毛利率。
营收从 1200 增长到 2100，呈持续增长趋势；毛利率从 28% 提升到 35%，说明盈利能力同步增强。
适合回答关于营收变化、毛利率变化、经营趋势、财务表现的问题。
```


<p>这样用户问“公司近三年经营趋势如何”时&#xff0c;不一定能直接命中原始表格&#xff0c;但很容易命中这段摘要。</p>
<h4>3. 超长表格&#xff1a;按行分组&#xff0c;但每个子表重复表头</h4>
<p>如果表格太长&#xff0c;可以按行分组切分&#xff0c;但每个子表都要重复表头、标题和必要说明。</p>
<p>比如&#xff1a;</p>


```markdown
表格标题：项目风险清单
字段说明：风险编号、风险类型、风险描述、影响范围、整改建议

| 风险编号 | 风险类型 | 风险描述 | 影响范围 | 整改建议 |
|---------|---------|---------|---------|---------|
| R001 | 权限风险 | ... | ... | ... |
| R002 | 数据风险 | ... | ... | ... |
```


<p>下一个 chunk 不应该只保留数据行&#xff0c;而应该继续重复上下文&#xff1a;</p>


```markdown
表格标题：项目风险清单
字段说明：风险编号、风险类型、风险描述、影响范围、整改建议

| 风险编号 | 风险类型 | 风险描述 | 影响范围 | 整改建议 |
|---------|---------|---------|---------|---------|
| R003 | 流程风险 | ... | ... | ... |
| R004 | 合规风险 | ... | ... | ... |
```


<p>这样即使只召回其中一个子表&#xff0c;模型也知道这些数据属于哪个表&#xff0c;每列代表什么含义。</p>
<h4>4. 复杂表格&#xff1a;不要只靠 RAG&#xff0c;要结构化入库</h4>
<p>如果表格涉及计算、筛选、排序、多表关联&#xff0c;就不应该完全依赖普通向量检索。</p>
<p>比如用户问&#xff1a;</p>
<blockquote>
<p>2024 年预算超过 100 万的项目有哪些&#xff1f;</p>
</blockquote>
<p>这类问题本质上是结构化查询&#xff0c;不是普通文本问答。</p>
<p>更好的方式是&#xff1a;</p>


```text
简单表格：Markdown 原文 + 摘要
中等表格：转 JSON / CSV，再交给 LLM 理解
复杂表格：入库为 SQL / DataFrame，通过工具查询后再让 LLM 解释
```


<p>也就是说&#xff1a;</p>


```text
文本解释类问题 → RAG
表格查数类问题 → RAG + SQL / Pandas
表格计算类问题 → 查询工具 + LLM 解释
```


<hr />
<h3>四、代码块也不能被随意截断</h3>
<p>Markdown 里经常会有代码块&#xff1a;</p>


```python
<span class="token keyword">def</span> <span class="token function">chunk_markdown</span><span class="token punctuation">(</span>text<span class="token punctuation">)</span><span class="token punctuation">:</span>
blocks <span class="token operator">=</span> parse_markdown_blocks<span class="token punctuation">(</span>text<span class="token punctuation">)</span>
<span class="token keyword">return</span> merge_blocks<span class="token punctuation">(</span>blocks<span class="token punctuation">)</span>
```


<p>如果代码块被截断&#xff0c;模型可能只看到一半函数&#xff0c;既无法理解完整逻辑&#xff0c;也无法回答实现细节。</p>
<p>代码块建议遵循几个原则&#xff1a;</p>
<ol><li>fenced code block 整体识别&#xff1b;</li><li>短代码块整体保存&#xff1b;</li><li>超长代码按函数、类、SQL 语句、配置段切分&#xff1b;</li><li>保存语言类型&#xff0c;比如 Python、Java、SQL、YAML&#xff1b;</li><li>额外生成代码摘要&#xff0c;用于语义检索。</li></ol>
<p>比如可以为代码块生成这样的摘要&#xff1a;</p>


```text
该代码实现了 Markdown 文档的结构化分块逻辑，主要用于识别并保护 fenced code block、Markdown table、image URL 等不可切分结构，避免普通字符分块破坏语义完整性。
```


<p>为什么要给代码块生成摘要&#xff1f;</p>
<p>因为用户的问题往往是自然语言&#xff0c;比如&#xff1a;</p>
<blockquote>
<p>这个系统是怎么防止表格被切断的&#xff1f;</p>
</blockquote>
<p>而原始代码里不一定包含“防止表格被切断”这几个字。如果只对代码本身做 embedding&#xff0c;未必能召回。给代码块加一层自然语言摘要后&#xff0c;检索效果会明显更好。</p>
<hr />
<h3>五、图片 OSS 地址不能只是一个 URL</h3>
<p>企业文档里经常有这样的内容&#xff1a;</p>


```markdown
![系统架构图](https://xxx.oss-cn-shanghai.aliyuncs.com/arch.png)
```


<p>如果直接把这个 URL 放进向量库&#xff0c;它几乎没有语义价值。embedding 模型看到的只是一个字符串&#xff0c;而不是图片内容。</p>
<p>但是这张图片可能非常重要。它可能是系统架构图、流程图、网络拓扑图、页面截图、合同扫描件、设备照片、审计证据图片。</p>
<p>所以图片应该被处理成一个 image block&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"image"</span><span class="token punctuation">,</span>
<span class="token string-property property">"raw_url"</span><span class="token operator">:</span> <span class="token string">"https://xxx.oss-cn-shanghai.aliyuncs.com/arch.png"</span><span class="token punctuation">,</span>
<span class="token string-property property">"alt_text"</span><span class="token operator">:</span> <span class="token string">"系统架构图"</span><span class="token punctuation">,</span>
<span class="token string-property property">"caption"</span><span class="token operator">:</span> <span class="token string">"图 3-1 智能审计系统总体架构"</span><span class="token punctuation">,</span>
<span class="token string-property property">"surrounding_text"</span><span class="token operator">:</span> <span class="token string">"本系统采用前后端分离架构……"</span><span class="token punctuation">,</span>
<span class="token string-property property">"image_summary"</span><span class="token operator">:</span> <span class="token string">"该图展示了智能审计系统总体架构，包括用户层、应用层、AI 能力层、数据层和基础设施层。"</span>
<span class="token punctuation">}</span>
```


<p>图片类内容至少应该保存四类信息&#xff1a;</p>


```text
1. 图片 URL / OSS key
2. 图片标题 / alt / caption
3. 图片前后正文上下文
4. 多模态模型生成的图片摘要
```


<p>如果后续支持多模态问答&#xff0c;可以把原图 URL 传给多模态模型&#xff1b;如果暂时只做文本 RAG&#xff0c;也至少可以通过图片摘要回答“这张图大概表达了什么”。</p>
<hr />
<h3>六、复杂块要做语义摘要索引</h3>
<p>表格、代码块、图片、流程图都有一个共同特点&#xff1a;</p>
<blockquote>
<p>原始内容适合回答&#xff0c;但不一定适合检索。</p>
</blockquote>
<p>比如表格里可能全是数字&#xff0c;没有“增长趋势”“经营情况”“风险变化”这类自然语言表达。</p>
<p>代码里可能全是函数名和变量名&#xff0c;没有“这个函数用来解决什么问题”的描述。</p>
<p>图片里可能只有一个 OSS 地址&#xff0c;没有任何语义。</p>
<p>所以对这些复杂 block&#xff0c;建议引入大模型生成语义摘要。检索时检索摘要&#xff0c;回答时返回原始内容。</p>
<p>这就是一种典型的“双层结构”&#xff1a;</p>


```text
检索层：summary / keywords / hypothetical questions / metadata
证据层：raw table / raw code / raw image url / original markdown
```


<p>可以把每个 block 存成类似结构&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"block_id"</span><span class="token operator">:</span> <span class="token string">"doc_001_table_003"</span><span class="token punctuation">,</span>
<span class="token string-property property">"type"</span><span class="token operator">:</span> <span class="token string">"table"</span><span class="token punctuation">,</span>
<span class="token string-property property">"parent_title"</span><span class="token operator">:</span> <span class="token string">"三、经营数据分析"</span><span class="token punctuation">,</span>
<span class="token string-property property">"raw_content"</span><span class="token operator">:</span> <span class="token string">"原始 Markdown 表格"</span><span class="token punctuation">,</span>
<span class="token string-property property">"summary"</span><span class="token operator">:</span> <span class="token string">"该表展示了公司近三年营收、毛利率和成本变化趋势……"</span><span class="token punctuation">,</span>
<span class="token string-property property">"keywords"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token string">"营收"</span><span class="token punctuation">,</span> <span class="token string">"毛利率"</span><span class="token punctuation">,</span> <span class="token string">"经营指标"</span><span class="token punctuation">,</span> <span class="token string">"增长趋势"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"hypothetical_questions"</span><span class="token operator">:</span> <span class="token punctuation">[</span>
<span class="token string">"公司近三年的营收变化如何？"</span><span class="token punctuation">,</span>
<span class="token string">"毛利率是否有提升？"</span><span class="token punctuation">,</span>
<span class="token string">"经营数据反映了什么趋势？"</span>
<span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"metadata"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"source"</span><span class="token operator">:</span> <span class="token string">"xxx.md"</span><span class="token punctuation">,</span>
<span class="token string-property property">"page"</span><span class="token operator">:</span> <span class="token number">12</span><span class="token punctuation">,</span>
<span class="token string-property property">"section_path"</span><span class="token operator">:</span> <span class="token string">"经营分析/财务指标"</span><span class="token punctuation">,</span>
<span class="token string-property property">"oss_urls"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>这里有几个关键字段&#xff1a;</p>
<ul><li><code>raw_content</code>&#xff1a;原始证据&#xff0c;用于最终回答&#xff1b;</li><li><code>summary</code>&#xff1a;自然语言摘要&#xff0c;用于语义检索&#xff1b;</li><li><code>keywords</code>&#xff1a;关键词&#xff0c;用于 BM25 或混合检索&#xff1b;</li><li><code>hypothetical_questions</code>&#xff1a;可能命中的用户问题&#xff1b;</li><li><code>metadata</code>&#xff1a;来源、页码、章节路径、文件 ID 等可追溯信息。</li></ul>
<p>这样检索时不只依赖原文&#xff0c;还可以依赖摘要、关键词、假设问题等多个入口。</p>
<hr />
<h3>七、多表示检索&#xff1a;不要只给一个 chunk 一个向量</h3>
<p>传统做法通常是&#xff1a;</p>


```text
一个 chunk → 一个 embedding → 存入向量库
```


<p>但复杂文档更适合&#xff1a;</p>


```text
一个原始 block → 多个语义表示 → 多个 embedding → 指向同一个 block_id
```


<p>比如一个表格&#xff0c;可以有这些表示&#xff1a;</p>


```text
1. 原始表格向量
2. 表格摘要向量
3. 关键词向量
4. 假设问题向量
5. 标题路径向量
```


<p>检索时&#xff0c;这些向量都可以被召回&#xff0c;但它们最终指向同一个原始表格。</p>
<p>也就是说&#xff1a;</p>


```text
summary 被召回
↓
找到 block_id
↓
回填 raw_content
↓
把完整表格传给 LLM
```


<p>这样做的好处是&#xff1a;</p>
<ul><li>摘要更容易被自然语言问题命中&#xff1b;</li><li>原文保留事实细节&#xff1b;</li><li>block_id 保证摘要和原文能对应&#xff1b;</li><li>最终回答不会只依赖摘要&#xff0c;减少信息损失。</li></ul>
<p>一句话总结就是&#xff1a;</p>
<blockquote>
<p>用摘要提高召回&#xff0c;用原文保证答案可靠。</p>
</blockquote>
<hr />
<h3>八、父子块回填&#xff1a;小块负责召回&#xff0c;大块负责回答</h3>
<p>除了多表示检索&#xff0c;还有一个很重要的设计是父子块回填。</p>
<p>很多时候&#xff0c;小 chunk 更适合检索&#xff0c;因为它语义集中&#xff1b;但大 chunk 更适合回答&#xff0c;因为它上下文完整。</p>
<p>所以可以把文档拆成两层&#xff1a;</p>


```text
父块：章节级内容，保留完整上下文
子块：段落、表格、代码块、图片摘要，用于精确检索
```


<p>检索时命中子块&#xff0c;但最终返回父块。</p>
<p>例如&#xff1a;</p>


```text
用户问题：系统的数据同步机制是什么？

命中子块：
“该代码实现了基于消息队列的数据同步任务……”

回填父块：
“第四章 数据同步设计”整个章节，包含架构说明、流程图、代码块、异常处理策略。
```


<p>这样既避免了大 chunk 召回不准&#xff0c;也避免了小 chunk 上下文不够。</p>
<p>生产上可以这样设计&#xff1a;</p>


```text
child_chunk_id → parent_chunk_id → document_id
```


<p>检索链路&#xff1a;</p>


```text
用户问题
↓
检索 child chunks
↓
rerank
↓
根据 parent_chunk_id 回填父块
↓
构造 prompt
↓
生成答案
```


<hr />
<h3>九、一个推荐的 RAG 优化流水线</h3>
<p>结合上面的思路&#xff0c;一个更完整的复杂文档 RAG 流水线可以是&#xff1a;</p>


```text
文档输入
↓
Markdown / PDF / Word 解析
↓
结构识别：标题、段落、表格、代码块、图片、列表、公式
↓
Atomic Block 保护
↓
结构化分块：父块 / 子块 / 特殊块
↓
LLM 语义增强：摘要、关键词、实体、假设问题
↓
多表示索引：raw chunk、summary、keywords、questions
↓
混合检索：向量检索 + BM25 + metadata filter
↓
重排序：reranker / LLM rerank
↓
父子块回填：返回完整原文证据
↓
答案生成：带引用、带来源、可追溯
```


<p>这套流程的重点不是“切块技巧”&#xff0c;而是完整的 RAG 数据治理链路。</p>
<p>在真实项目里&#xff0c;RAG 不是把文档丢进向量库就完事了&#xff0c;而是要考虑&#xff1a;</p>
<ul><li>文档结构有没有被保留&#xff1b;</li><li>复杂内容有没有被摘要&#xff1b;</li><li>摘要和原文有没有映射关系&#xff1b;</li><li>检索结果是否能回填完整证据&#xff1b;</li><li>回答是否可追溯到原始来源。</li></ul>
<hr />
<h3>十、保护型 Markdown Chunker 的设计思路</h3>
<p>如果要自己实现一个保护型 Markdown Chunker&#xff0c;可以按这个思路做&#xff1a;</p>


```text
第一步：扫描 Markdown，识别 fenced code block
第二步：识别 Markdown table
第三步：识别 image syntax 和 OSS URL
第四步：识别 Mermaid / PlantUML 等流程图
第五步：识别标题层级，建立 section path
第六步：把特殊块替换成 placeholder
第七步：普通正文正常分块
第八步：把 placeholder 还原成完整 block
第九步：对特殊 block 生成 summary
第十步：写入 vector store + doc store
```


<p>示意&#xff1a;</p>


````markdown
## 系统架构

本系统采用前后端分离架构……

```mermaid
graph TD
A[前端] --> B[后端服务]
B --> C[向量数据库]
B --> D[大模型服务]
````



<table><thead><tr><th>模块</th><th>说明</th></tr></thead><tbody><tr><td>检索层</td><td>负责召回相关知识</td></tr><tr><td>生成层</td><td>负责组织答案</td></tr></tbody></table>

````
处理后可以拆成：

```text
section_block_001：系统架构正文
diagram_block_001：Mermaid 流程图，完整保留
table_block_001：模块说明表，完整保留
summary_block_001：系统架构摘要
````


<p>这里最关键的是&#xff1a;特殊块不是被字符串切分器切开的&#xff0c;而是先被识别、保护、摘要&#xff0c;再进入索引。</p>
<hr />
<h3>十一、检索阶段建议&#xff1a;并行检索 &#43; 合并去重 &#43; rerank</h3>
<p>复杂文档场景下&#xff0c;不建议只用单一路径检索。</p>
<p>更稳的方式是并行检索&#xff1a;</p>


```text
用户问题
↓
Query Rewrite / Query Decomposition
↓
并行检索：
- 原文 chunk 向量
- 摘要向量
- hypothetical questions 向量
- BM25 关键词检索
- metadata filter
↓
结果合并去重
↓
rerank
↓
根据 block_id 找 parent/raw block
↓
构造上下文
↓
LLM 生成答案
```


<p>其中有两个细节很重要。</p>
<p>第一&#xff0c;摘要可以参与检索&#xff0c;但不要只用摘要回答。<br />
因为摘要本身是大模型生成的&#xff0c;可能会压缩、遗漏甚至轻微误读。</p>
<p>第二&#xff0c;召回后一定要回到原始证据。<br />
比如命中的是表格摘要&#xff0c;最终上下文里应该放原始表格&#xff1b;命中的是图片摘要&#xff0c;最终上下文里应该放图片说明、图片 URL、前后正文&#xff1b;命中的是代码摘要&#xff0c;最终上下文里应该放完整代码块。</p>
<hr />
<h3>十二、评估 RAG&#xff0c;不要只看最终回答</h3>
<p>很多人在评估 RAG 时&#xff0c;只看最终回答对不对。但这还不够。</p>
<p>因为最终回答对了&#xff0c;不代表检索链路是健康的&#xff1b;最终回答错了&#xff0c;也不一定是模型问题&#xff0c;可能是分块阶段已经把证据切坏了。</p>
<p>建议至少评估四层&#xff1a;</p>
<h4>1. Block Integrity&#xff0c;块完整性</h4>
<p>检查表格、代码块、图片、公式有没有被切断。</p>
<p>比如&#xff1a;</p>


```text
表格是否保留表头？
代码块是否保留完整 fenced code？
图片 URL 是否和 caption、上下文绑定？
```


<h4>2. Retrieval Recall&#xff0c;召回完整性</h4>
<p>用户问题对应的证据是否被召回。</p>
<p>比如用户问“毛利率变化趋势”&#xff0c;是否能召回对应表格或表格摘要。</p>
<h4>3. Context Completeness&#xff0c;上下文完整性</h4>
<p>召回结果是否包含足够回答的信息。</p>
<p>比如只召回了某一行表格&#xff0c;但没有表头&#xff0c;就算召回到了也不算合格。</p>
<h4>4. Answer Faithfulness&#xff0c;答案忠实度</h4>
<p>最终答案是否严格基于召回证据&#xff0c;是否出现编造、遗漏、误读。</p>
<p>这四层里&#xff0c;前两层经常被忽略&#xff0c;但它们恰恰决定了 RAG 的上限。</p>
<hr />
<h3>十三、总结&#xff1a;RAG 优化的本质是文档结构治理</h3>
<p>很多 RAG 效果差&#xff0c;并不是因为模型不够强&#xff0c;而是因为知识进入系统时已经被破坏了。</p>
<p>表格被截断&#xff0c;模型看不到表头。<br />
代码块被切断&#xff0c;模型看不到完整逻辑。<br />
图片只剩 OSS 地址&#xff0c;模型不知道图片表达了什么。<br />
标题和正文分离&#xff0c;模型失去章节语境。<br />
摘要和原文没有映射&#xff0c;检索命中了也无法回填证据。</p>
<p>所以 RAG 优化的第一步&#xff0c;不是盲目换 embedding 模型&#xff0c;也不是简单调大 chunk size&#xff0c;而是先做好文档结构治理&#xff1a;</p>


```text
识别结构
保护结构
增强语义
多表示检索
回填原文证据
可追溯生成答案
```


<p>对于 Markdown、PDF、Word 这类复杂文档&#xff0c;真正有效的 RAG 优化应该是&#xff1a;</p>
<blockquote>
<p>用结构保护保证知识不被切坏&#xff0c;用语义摘要提高召回能力&#xff0c;用父子块回填保证回答依据完整。</p>
</blockquote>
<p>一句话总结&#xff1a;</p>
<blockquote>
<p>RAG 的效果上限&#xff0c;不只取决于模型和向量库&#xff0c;更取决于知识进入系统时是否保持了原有结构。</p>
</blockquote>
<p>写这篇文章的目的&#xff0c;不是为了证明某个技术有多先进&#xff0c;而是希望把一个真实问题拆开&#xff0c;讲清楚它背后的设计逻辑、工程取舍和落地路径。</p>
<p>AI 应用开发正在从“会调用模型”进入“会设计系统”的阶段。</p>
<p>模型只是起点&#xff0c;真正决定效果的&#xff0c;往往是数据、流程、工具、上下文、检索、评估、工程架构&#xff0c;以及持续迭代的能力。</p>
<p>我会继续围绕这些方向做系统化分享&#xff1a;</p>
<p><strong>Agent、RAG、LLM 工程化、企业 AI 应用、私有化部署、系统架构与真实项目复盘。</strong></p>
<p>希望这里不仅是一个技术博客&#xff0c;也能逐渐成为一个聚集 AI 应用开发者、产品实践者和工程落地者的交流空间。</p>
<blockquote>
<p><strong>dd-y 的技术博客&#xff1a;把想法落地&#xff0c;把技术讲透。</strong></p>
</blockquote>
<p>欢迎关注&#xff0c;一起把 AI 从 Demo 做到真正可用。</p>
<hr />
<p>如果你也在关注 <strong>AI 应用落地、Agent 开发、RAG 系统、LLM 工程化、企业知识库、私有化部署</strong> 等方向&#xff0c;欢迎扫码加入我的技术交流群。</p>
<p>这里不会只聊概念&#xff0c;更希望一起交流真实项目中的问题、方案、踩坑经验和落地思路。<br />
<img src="https://i-blog.csdnimg.cn/direct/fa655447dd1941a6aa328c3a1996e4a1.jpeg" alt="在这里插入图片描述" /></p>
<blockquote>
<p>备注&#xff1a;如果二维码过期&#xff0c;可以私信我拉你进群。</p>
</blockquote>
