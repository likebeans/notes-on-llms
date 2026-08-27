---
title: "从分布式系统到 Celery + Redis：一篇文章讲清异步任务工作流程"
description: "CSDN 原文全文镜像：异步任务处理：Celery + Redis 解决方案 本文介绍了在Web系统中处理耗时任务的异步解决方案。传统同步模式在处理PDF解析、OCR识别、批量邮件等长时间任务时存在超时、连接断开和服务阻塞等问题。通过Celery和Redis构……"
pageType: article
module: site
updated: '2026-08-04'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "practice"
  - "redis"
  - "数据库"
  - "缓存"
level: intermediate
prerequisites:
  - "/practice/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-04，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-04。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163476127](https://blog.csdn.net/m0_63309778/article/details/163476127)
- 站内分区：工程实践 / 异步任务架构
:::

<p><img src="https://i-blog.csdnimg.cn/direct/c1f769834c51471bbed7a58168d6c99a.png" alt="在这里插入图片描述" /></p>
<h3>前言</h3>
<p>在普通 Web 系统中&#xff0c;用户发送请求&#xff0c;服务端完成处理&#xff0c;然后返回结果。</p>


```text
用户请求
↓
Web 服务执行
↓
返回结果
```


<p>对于登录、查询、参数校验这类短操作&#xff0c;这种同步模式没有问题。</p>
<p>但当系统中出现以下任务时&#xff0c;同步执行就不合适了&#xff1a;</p>
<ul><li>解析几百页的 PDF&#xff1b;</li><li>执行 OCR 或图片识别&#xff1b;</li><li>调用大模型生成内容&#xff1b;</li><li>批量发送邮件&#xff1b;</li><li>抓取大量网页&#xff1b;</li><li>生成 Excel、Word 或 PDF&#xff1b;</li><li>建立 RAG 向量索引&#xff1b;</li><li>执行定时数据同步。</li></ul>
<p>这些任务可能需要几十秒&#xff0c;甚至几分钟。如果一直占用 HTTP 请求&#xff0c;容易出现超时、连接断开、服务卡顿等问题。</p>
<p>因此&#xff0c;我们通常会把耗时任务从 Web 服务中拆出去&#xff1a;</p>


```text
用户提交任务
↓
Web 服务记录任务
↓
将任务放入消息队列
↓
立即返回任务 ID

后台 Worker
↓
获取任务
↓
执行任务
↓
更新任务状态
```


<p>Celery &#43; Redis 就是 Python 项目中非常常见的一套异步任务解决方案。</p>
<hr />
<h2>一、为什么需要分布式任务系统</h2>
<h3>1. 同步执行的问题</h3>
<p>假设用户上传一个 300 页的 PDF&#xff0c;解析需要 5 分钟。</p>
<p>如果直接在接口中执行&#xff1a;</p>


```python
<span class="token keyword">def</span> <span class="token function">upload_and_parse</span><span class="token punctuation">(</span><span class="token builtin">file</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
save_file<span class="token punctuation">(</span><span class="token builtin">file</span><span class="token punctuation">)</span>
result <span class="token operator">=</span> parse_pdf<span class="token punctuation">(</span><span class="token builtin">file</span><span class="token punctuation">)</span>
<span class="token keyword">return</span> result
```


<p>整个请求会持续等待 5 分钟。</p>
<p>这期间可能出现&#xff1a;</p>
<ul><li>Nginx 请求超时&#xff1b;</li><li>浏览器连接断开&#xff1b;</li><li>Web 进程长期被占用&#xff1b;</li><li>其他用户请求变慢&#xff1b;</li><li>任务执行失败后难以重试&#xff1b;</li><li>用户不知道任务进行到哪一步。</li></ul>
<p>更严重的是&#xff0c;如果大量用户同时提交任务&#xff0c;Web 服务很快会被拖垮。</p>
<hr />
<h3>2. 异步执行的思路</h3>
<p>异步任务的核心是&#xff1a;</p>
<blockquote>
<p>Web 服务只负责接收任务&#xff0c;不负责完成所有耗时计算。</p>
</blockquote>
<p>处理流程变成&#xff1a;</p>


```text
用户上传文件
↓
Web 服务保存文件
↓
创建一条任务记录
↓
将任务发送到队列
↓
返回任务 ID
```


<p>后台 Worker 再慢慢处理&#xff1a;</p>


```text
Worker 获取任务
↓
下载文件
↓
解析文档
↓
保存结果
↓
更新任务状态
```


<p>这样做有几个明显好处&#xff1a;</p>
<h4>Web 接口响应更快</h4>
<p>用户提交任务后&#xff0c;可以很快收到响应&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"taskId"</span><span class="token operator">:</span> <span class="token number">10001</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"QUEUED"</span>
<span class="token punctuation">}</span>
```


<p>用户不需要一直保持连接。</p>
<h4>可以水平扩容</h4>
<p>任务变多时&#xff0c;可以增加 Worker&#xff1a;</p>


```text
一个 Worker
↓
三个 Worker
↓
十个 Worker
```


<p>多个 Worker 可以共同消费同一个任务队列。</p>
<h4>可以隔离不同类型任务</h4>
<p>例如&#xff1a;</p>


```text
文档解析 Worker
OCR Worker
爬虫 Worker
邮件 Worker
大模型 Worker
```


<p>爬虫服务崩溃&#xff0c;不一定影响文档解析和用户登录。</p>
<h4>可以重试和监控</h4>
<p>任务失败后可以自动重试&#xff0c;也可以记录&#xff1a;</p>
<ul><li>当前状态&#xff1b;</li><li>执行进度&#xff1b;</li><li>失败原因&#xff1b;</li><li>重试次数&#xff1b;</li><li>开始时间&#xff1b;</li><li>完成时间。</li></ul>
<hr />
<h2>二、Celery 和 Redis 分别是什么</h2>
<h3>1. Celery 是什么</h3>
<p>Celery 是 Python 中常用的分布式任务队列框架。</p>
<p>它主要负责&#xff1a;</p>
<ul><li>定义任务&#xff1b;</li><li>发送任务&#xff1b;</li><li>调度任务&#xff1b;</li><li>Worker 执行任务&#xff1b;</li><li>失败重试&#xff1b;</li><li>定时任务&#xff1b;</li><li>任务状态管理&#xff1b;</li><li>多任务编排。</li></ul>
<p>例如定义一个 Celery 任务&#xff1a;</p>


```python
<span class="token keyword">from</span> celery <span class="token keyword">import</span> Celery

app <span class="token operator">=</span> Celery<span class="token punctuation">(</span><span class="token string">"demo"</span><span class="token punctuation">)</span>

<span class="token decorator annotation punctuation">@app<span class="token punctuation">.</span>task</span>
<span class="token keyword">def</span> <span class="token function">add</span><span class="token punctuation">(</span>x<span class="token punctuation">,</span> y<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">return</span> x <span class="token operator">+</span> y
```


<p>提交任务&#xff1a;</p>


```python
result <span class="token operator">=</span> add<span class="token punctuation">.</span>delay<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">2</span><span class="token punctuation">)</span>
```


<p>这里的 <code>delay()</code> 并不会在当前 Web 进程中直接执行 <code>add()</code>。</p>
<p>它会把任务发送到消息队列&#xff0c;等待 Worker 执行。</p>
<hr />
<h3>2. Redis 是什么</h3>
<p>Redis 是一个高性能内存数据存储系统。</p>
<p>在 Celery 中&#xff0c;Redis 常见有两个作用。</p>
<h4>作为 Broker</h4>
<p>Broker 可以理解为任务中转站。</p>


```text
Web 服务
↓
Redis Broker
↓
Celery Worker
```


<p>Web 服务把任务消息写入 Redis&#xff0c;Worker 再从 Redis 读取任务。</p>
<h4>作为 Result Backend</h4>
<p>Result Backend 用来保存任务状态和返回结果。</p>


```text
Worker 执行完成
↓
Redis Result Backend
↓
系统查询任务状态
```


<p>因此需要区分&#xff1a;</p>


```text
Broker：任务要交给谁执行

Result Backend：任务执行得怎么样
```


<p>常见配置&#xff1a;</p>


```python
app <span class="token operator">=</span> Celery<span class="token punctuation">(</span>
<span class="token string">"demo"</span><span class="token punctuation">,</span>
broker<span class="token operator">=</span><span class="token string">"redis://localhost:6379/0"</span><span class="token punctuation">,</span>
backend<span class="token operator">=</span><span class="token string">"redis://localhost:6379/1"</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
```


<p>这里&#xff1a;</p>


```text
Redis DB 0：保存待执行任务
Redis DB 1：保存任务状态和结果
```


<hr />
<h2>三、Celery &#43; Redis 的核心组件</h2>
<p>一套完整的 Celery 系统&#xff0c;主要有以下几个角色。</p>
<div class="mermaid mermaid-newversion mermaid-flowchart"></div>
<h3>1. Producer</h3>
<p>Producer 是任务生产者。</p>
<p>它可以是&#xff1a;</p>
<ul><li>FastAPI&#xff1b;</li><li>Django&#xff1b;</li><li>Flask&#xff1b;</li><li>Python 脚本&#xff1b;</li><li>另一个 Celery 任务&#xff1b;</li><li>定时任务调度器。</li></ul>
<p>Producer 的职责是&#xff1a;</p>


```text
创建任务消息
↓
发送到 Broker
```


<hr />
<h3>2. Broker</h3>
<p>Broker 负责保存和转发任务消息。</p>
<p>它位于 Producer 和 Worker 之间&#xff0c;使二者解耦。</p>
<p>Producer 不需要知道&#xff1a;</p>
<ul><li>哪个 Worker 会执行任务&#xff1b;</li><li>Worker 当前是否空闲&#xff1b;</li><li>Worker 部署在哪台服务器&#xff1b;</li><li>任务什么时候开始。</li></ul>
<p>它只需要把任务成功交给 Broker。</p>
<hr />
<h3>3. Worker</h3>
<p>Worker 是真正执行任务的进程。</p>
<p>启动示例&#xff1a;</p>


```bash
celery <span class="token parameter variable">-A</span> app worker <span class="token parameter variable">--loglevel</span><span class="token operator">=</span>INFO
```


<p>Worker 启动后会&#xff1a;</p>
<ol><li>连接 Redis&#xff1b;</li><li>监听任务队列&#xff1b;</li><li>获取任务消息&#xff1b;</li><li>找到对应任务函数&#xff1b;</li><li>执行业务逻辑&#xff1b;</li><li>保存结果&#xff1b;</li><li>确认任务完成。</li></ol>
<hr />
<h3>4. Result Backend</h3>
<p>Result Backend 用于保存&#xff1a;</p>
<ul><li>PENDING&#xff1b;</li><li>STARTED&#xff1b;</li><li>SUCCESS&#xff1b;</li><li>FAILURE&#xff1b;</li><li>RETRY&#xff1b;</li><li>任务返回值&#xff1b;</li><li>异常信息。</li></ul>
<p>但在生产系统中&#xff0c;不能只依赖 Celery 状态。</p>
<p>因为 Celery 更关注执行状态&#xff0c;业务系统通常还需要自己的状态&#xff0c;例如&#xff1a;</p>


```text
UPLOADED
QUEUED
DOWNLOADING
PARSING
OCR_PROCESSING
INDEXING
SUCCESS
FAILED
```


<p>所以更推荐&#xff1a;</p>


```text
Celery 状态：任务执行层状态
业务数据库：真实业务状态
```


<hr />
<h3>5. Celery Beat</h3>
<p>Celery Beat 是定时任务调度器。</p>
<p>例如&#xff1a;</p>
<ul><li>每天凌晨清理临时文件&#xff1b;</li><li>每小时同步数据&#xff1b;</li><li>每五分钟检查一次任务&#xff1b;</li><li>每天生成业务日报。</li></ul>
<p>Beat 本身通常不执行任务&#xff0c;只负责定时把任务发送到 Broker。</p>
<hr />
<h2>四、Celery &#43; Redis 的完整工作流程</h2>
<p>下面通过一个“文档解析任务”说明整个过程。</p>
<p>用户上传文件后&#xff0c;系统提交任务&#xff1a;</p>


```python
parse_document<span class="token punctuation">.</span>delay<span class="token punctuation">(</span>file_id<span class="token operator">=</span><span class="token number">1001</span><span class="token punctuation">)</span>
```


<p>背后大致会经历以下步骤。</p>
<hr />
<h3>第一步&#xff1a;Web 服务创建业务任务</h3>
<p>系统先在数据库中创建一条任务记录&#xff1a;</p>


```text
task_id：80001
file_id：1001
status：CREATED
```


<p>为什么不能只使用 Celery 的 task ID&#xff1f;</p>
<p>因为业务系统通常还需要保存&#xff1a;</p>
<ul><li>文件 ID&#xff1b;</li><li>用户 ID&#xff1b;</li><li>租户 ID&#xff1b;</li><li>执行进度&#xff1b;</li><li>当前阶段&#xff1b;</li><li>错误代码&#xff1b;</li><li>结果地址。</li></ul>
<p>因此&#xff0c;一般会同时存在两个 ID&#xff1a;</p>


```text
业务任务 ID：用于业务查询
Celery task_id：用于 Celery 执行追踪
```


<hr />
<h3>第二步&#xff1a;发送任务</h3>
<p>Web 服务调用&#xff1a;</p>


```python
result <span class="token operator">=</span> parse_document<span class="token punctuation">.</span>delay<span class="token punctuation">(</span>
business_task_id<span class="token operator">=</span><span class="token number">80001</span><span class="token punctuation">,</span>
file_id<span class="token operator">=</span><span class="token number">1001</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
```


<p>Celery 会生成一个唯一的 task ID&#xff0c;并把任务转换成消息。</p>
<p>消息大致包含&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"task"</span><span class="token operator">:</span> <span class="token string">"tasks.parse_document"</span><span class="token punctuation">,</span>
<span class="token string-property property">"id"</span><span class="token operator">:</span> <span class="token string">"celery-task-uuid"</span><span class="token punctuation">,</span>
<span class="token string-property property">"args"</span><span class="token operator">:</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">,</span>
<span class="token string-property property">"kwargs"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"business_task_id"</span><span class="token operator">:</span> <span class="token number">80001</span><span class="token punctuation">,</span>
<span class="token string-property property">"file_id"</span><span class="token operator">:</span> <span class="token number">1001</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<hr />
<h3>第三步&#xff1a;任务写入 Redis</h3>
<p>Celery 将消息序列化后写入 Redis Broker。</p>
<p>这时任务只是进入了队列&#xff1a;</p>


```text
消息进入 Redis
≠
任务已经开始
≠
任务已经成功
```


<p>Web 服务可以立即返回&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"taskId"</span><span class="token operator">:</span> <span class="token number">80001</span><span class="token punctuation">,</span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"QUEUED"</span>
<span class="token punctuation">}</span>
```


<hr />
<h3>第四步&#xff1a;Worker 获取任务</h3>
<p>Celery Worker 一直监听 Redis。</p>
<p>当队列中出现新任务时&#xff0c;Worker 会取出消息&#xff0c;并根据任务名称找到对应的 Python 函数&#xff1a;</p>


```text
tasks.parse_document
↓
找到本地注册的 parse_document
```


<p>Worker 本地必须已经安装并加载对应代码。</p>
<p>任务消息中不会包含 Python 函数本身&#xff0c;只包含任务名称和参数。</p>
<hr />
<h3>第五步&#xff1a;Worker 执行业务逻辑</h3>
<p>任务开始执行&#xff1a;</p>


```python
<span class="token decorator annotation punctuation">@app<span class="token punctuation">.</span>task</span>
<span class="token keyword">def</span> <span class="token function">parse_document</span><span class="token punctuation">(</span>business_task_id<span class="token punctuation">,</span> file_id<span class="token punctuation">)</span><span class="token punctuation">:</span>
update_status<span class="token punctuation">(</span>business_task_id<span class="token punctuation">,</span> <span class="token string">"RUNNING"</span><span class="token punctuation">)</span>

file_path <span class="token operator">=</span> download_file<span class="token punctuation">(</span>file_id<span class="token punctuation">)</span>
result <span class="token operator">=</span> parse_pdf<span class="token punctuation">(</span>file_path<span class="token punctuation">)</span>
save_result<span class="token punctuation">(</span>business_task_id<span class="token punctuation">,</span> result<span class="token punctuation">)</span>

update_status<span class="token punctuation">(</span>business_task_id<span class="token punctuation">,</span> <span class="token string">"SUCCESS"</span><span class="token punctuation">)</span>

<span class="token keyword">return</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"business_task_id"</span><span class="token punctuation">:</span> business_task_id<span class="token punctuation">,</span>
<span class="token string">"page_count"</span><span class="token punctuation">:</span> result<span class="token punctuation">.</span>page_count<span class="token punctuation">,</span>
<span class="token punctuation">}</span>
```


<p>实际文档任务中可能包含&#xff1a;</p>


```text
下载文件
↓
解析 PDF
↓
执行 OCR
↓
提取表格和图片
↓
切分文本
↓
生成向量
↓
写入向量数据库
```


<hr />
<h3>第六步&#xff1a;保存任务结果</h3>
<p>任务成功后&#xff0c;Celery 可以将结果写入 Result Backend&#xff1a;</p>


```json
<span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"status"</span><span class="token operator">:</span> <span class="token string">"SUCCESS"</span><span class="token punctuation">,</span>
<span class="token string-property property">"result"</span><span class="token operator">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string-property property">"business_task_id"</span><span class="token operator">:</span> <span class="token number">80001</span><span class="token punctuation">,</span>
<span class="token string-property property">"page_count"</span><span class="token operator">:</span> <span class="token number">326</span>
<span class="token punctuation">}</span>
<span class="token punctuation">}</span>
```


<p>同时&#xff0c;业务数据库也应该更新&#xff1a;</p>


```text
status：SUCCESS
progress：100
finished_at：完成时间
result_path：结果地址
```


<p>重要结果不要直接全部放入 Redis。</p>
<p>例如&#xff0c;不要让 Celery 返回数百 MB 的文档内容&#xff1a;</p>


```python
<span class="token keyword">return</span> huge_document_result
```


<p>更合理的是把真实结果存到数据库或对象存储&#xff0c;只返回引用&#xff1a;</p>


```python
<span class="token keyword">return</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"result_path"</span><span class="token punctuation">:</span> <span class="token string">"parse-results/80001/result.json"</span>
<span class="token punctuation">}</span>
```


<hr />
<h2>五、消息确认、重复执行与幂等</h2>
<p>Celery 任务系统中&#xff0c;一个非常重要的问题是&#xff1a;</p>
<blockquote>
<p>Worker 执行到一半崩溃怎么办&#xff1f;</p>
</blockquote>
<p>这涉及消息确认&#xff0c;也就是 ACK。</p>
<h3>1. 提前确认</h3>
<p>提前确认的流程&#xff1a;</p>


```text
Worker 收到任务
↓
确认任务
↓
执行任务
```


<p>如果任务执行过程中 Worker 崩溃&#xff0c;Broker 可能认为任务已经完成&#xff0c;不再投递。</p>
<p>优点是任务不容易重复执行&#xff0c;缺点是任务可能丢失。</p>
<hr />
<h3>2. 执行后确认</h3>
<p>可以配置&#xff1a;</p>


```python
task_acks_late <span class="token operator">=</span> <span class="token boolean">True</span>
```


<p>流程变成&#xff1a;</p>


```text
Worker 收到任务
↓
执行任务
↓
执行成功
↓
确认任务
```


<p>如果 Worker 中途崩溃&#xff0c;任务可能重新进入队列&#xff0c;由其他 Worker 再执行一次。</p>
<p>这样可以减少任务丢失&#xff0c;但会带来另一个问题&#xff1a;</p>
<blockquote>
<p>同一个任务可能执行多次。</p>
</blockquote>
<p>因此&#xff0c;开启晚确认后&#xff0c;任务必须具备幂等性。</p>
<hr />
<h3>3. 什么是幂等</h3>
<p>幂等是指&#xff1a;</p>
<blockquote>
<p>同一个任务执行一次或多次&#xff0c;最终业务结果相同。</p>
</blockquote>
<p>例如下面的操作通常是幂等的&#xff1a;</p>


```text
将任务状态设置为 SUCCESS
```


<p>执行十次&#xff0c;最终还是 <code>SUCCESS</code>。</p>
<p>但下面的操作不是幂等的&#xff1a;</p>


```text
给用户发送一封邮件
账户扣款 100 元
插入一条新数据
```


<p>重复执行可能产生多封邮件、多次扣款或重复记录。</p>
<hr />
<h3>4. 常见幂等方式</h3>
<h4>唯一约束</h4>
<p>例如一个文件只能生成一份指定版本的解析结果&#xff1a;</p>


```sql
<span class="token keyword">UNIQUE</span><span class="token punctuation">(</span>file_id<span class="token punctuation">,</span> parse_version<span class="token punctuation">)</span>
```


<p>重复执行时&#xff0c;数据库阻止重复数据。</p>
<h4>条件更新</h4>


```sql
<span class="token keyword">UPDATE</span> task
<span class="token keyword">SET</span> <span class="token keyword">status</span> <span class="token operator">=</span> <span class="token string">'RUNNING'</span>
<span class="token keyword">WHERE</span> id <span class="token operator">=</span> <span class="token number">80001</span>
<span class="token operator">AND</span> <span class="token keyword">status</span> <span class="token operator">=</span> <span class="token string">'QUEUED'</span><span class="token punctuation">;</span>
```


<p>只有第一个 Worker 能成功把状态从 <code>QUEUED</code> 改成 <code>RUNNING</code>。</p>
<h4>幂等业务键</h4>


```text
document_parse:1001:v1
```


<p>每次任务执行前先判断该业务键是否已经成功处理。</p>
<h4>先计算&#xff0c;后原子提交</h4>
<p>推荐流程&#xff1a;</p>


```text
读取数据
↓
执行计算
↓
生成临时结果
↓
一次性提交最终结果
```


<p>减少任务执行一半时留下脏数据的风险。</p>
<hr />
<h2>六、失败重试与超时</h2>
<p>分布式系统中&#xff0c;失败是正常现象。</p>
<p>常见临时错误包括&#xff1a;</p>
<ul><li>网络抖动&#xff1b;</li><li>对象存储短暂不可用&#xff1b;</li><li>数据库连接失败&#xff1b;</li><li>第三方接口超时&#xff1b;</li><li>大模型接口限流&#xff1b;</li><li>Redis 短暂断开。</li></ul>
<p>这些错误通常可以重试。</p>


```python
<span class="token decorator annotation punctuation">@app<span class="token punctuation">.</span>task</span><span class="token punctuation">(</span>
bind<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span>
autoretry_for<span class="token operator">=</span><span class="token punctuation">(</span>ConnectionError<span class="token punctuation">,</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
retry_backoff<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span>
retry_jitter<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span>
max_retries<span class="token operator">=</span><span class="token number">5</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
<span class="token keyword">def</span> <span class="token function">process_task</span><span class="token punctuation">(</span>self<span class="token punctuation">,</span> task_id<span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">return</span> execute_task<span class="token punctuation">(</span>task_id<span class="token punctuation">)</span>
```


<p>这里&#xff1a;</p>


```text
autoretry_for：指定哪些异常自动重试
retry_backoff：使用指数退避
retry_jitter：加入随机抖动
max_retries：最大重试次数
```


<p>指数退避类似&#xff1a;</p>


```text
第一次失败：1 秒后重试
第二次失败：2 秒后重试
第三次失败：4 秒后重试
第四次失败：8 秒后重试
```


<p>这样可以避免下游系统已经过载时&#xff0c;所有任务立即重复请求。</p>
<p>并不是所有错误都应该重试。</p>
<p>例如&#xff1a;</p>
<ul><li>文件已经损坏&#xff1b;</li><li>参数错误&#xff1b;</li><li>用户无权限&#xff1b;</li><li>文件格式不支持&#xff1b;</li><li>业务数据不存在。</li></ul>
<p>这些属于永久错误&#xff0c;重试通常没有意义。</p>
<hr />
<h2>七、并发模型与队列划分</h2>
<p>Celery 可以同时执行多个任务。</p>
<p>常见 Worker 启动方式&#xff1a;</p>


```bash
celery <span class="token parameter variable">-A</span> app worker <span class="token parameter variable">--concurrency</span><span class="token operator">=</span><span class="token number">4</span>
```


<p>表示 Worker 同时拥有 4 个执行槽位。</p>
<p>但并发数不是越大越好&#xff0c;需要根据任务类型决定。</p>

<table><thead><tr><th>任务类型</th><th>特点</th><th>建议</th></tr></thead><tbody><tr><td>PDF 解析</td><td>CPU、内存密集</td><td>低到中等并发</td></tr><tr><td>OCR</td><td>CPU 或 GPU 密集</td><td>独立 Worker</td></tr><tr><td>接口调用</td><td>网络 I/O 密集</td><td>可以适当提高并发</td></tr><tr><td>邮件发送</td><td>网络 I/O 密集</td><td>中等并发</td></tr><tr><td>Playwright 爬虫</td><td>内存和浏览器资源密集</td><td>严格限制并发</td></tr><tr><td>大模型推理</td><td>GPU 显存密集</td><td>独立部署和限流</td></tr></tbody></table><p>不同任务最好拆到不同队列&#xff1a;</p>


```text
document_parse
ocr
crawler
notification
llm
```


<p>例如&#xff1a;</p>


```python
app<span class="token punctuation">.</span>conf<span class="token punctuation">.</span>task_routes <span class="token operator">=</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"tasks.parse_document"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"queue"</span><span class="token punctuation">:</span> <span class="token string">"document_parse"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string">"tasks.run_ocr"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"queue"</span><span class="token punctuation">:</span> <span class="token string">"ocr"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string">"tasks.send_email"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"queue"</span><span class="token punctuation">:</span> <span class="token string">"notification"</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span>
```


<p>再启动不同 Worker&#xff1a;</p>


```bash
celery <span class="token parameter variable">-A</span> app worker <span class="token parameter variable">-Q</span> document_parse <span class="token parameter variable">--concurrency</span><span class="token operator">=</span><span class="token number">4</span>
```




```bash
celery <span class="token parameter variable">-A</span> app worker <span class="token parameter variable">-Q</span> crawler <span class="token parameter variable">--concurrency</span><span class="token operator">=</span><span class="token number">2</span>
```


<p>这样可以避免一个耗时很长的爬虫任务&#xff0c;占用文档解析或邮件任务的执行资源。</p>
<hr />
<h2>八、生产环境中的推荐设计</h2>
<p>一套比较合理的架构如下&#xff1a;</p>
<div class="mermaid mermaid-newversion mermaid-flowchart"></div>
<p>核心设计原则有以下几条。</p>
<h3>1. Web 服务只负责提交任务</h3>
<p>不要在 API 中这样做&#xff1a;</p>


```python
result <span class="token operator">=</span> task<span class="token punctuation">.</span>delay<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">return</span> result<span class="token punctuation">.</span>get<span class="token punctuation">(</span>timeout<span class="token operator">=</span><span class="token number">600</span><span class="token punctuation">)</span>
```


<p>虽然使用了 Celery&#xff0c;但接口仍然在同步等待&#xff0c;失去了异步任务的意义。</p>
<p>正确方式是立即返回任务 ID。</p>
<hr />
<h3>2. 大文件不进入 Redis</h3>
<p>错误方式&#xff1a;</p>


```python
parse_document<span class="token punctuation">.</span>delay<span class="token punctuation">(</span>file_bytes<span class="token punctuation">)</span>
```


<p>正确方式&#xff1a;</p>


```python
parse_document<span class="token punctuation">.</span>delay<span class="token punctuation">(</span>file_id<span class="token punctuation">)</span>
```


<p>文件存入 MinIO 或 S3&#xff0c;消息中只传文件 ID 或对象存储 Key。</p>
<hr />
<h3>3. 业务状态存数据库</h3>
<p>Celery 状态只适合描述任务执行情况。</p>
<p>真正的业务状态应保存在数据库&#xff0c;例如&#xff1a;</p>


```text
QUEUED
DOWNLOADING
PARSING
OCR
INDEXING
SUCCESS
FAILED
```


<hr />
<h3>4. 不同任务使用不同队列</h3>
<p>不要把所有任务都放进默认队列。</p>
<p>应该根据&#xff1a;</p>
<ul><li>任务耗时&#xff1b;</li><li>资源类型&#xff1b;</li><li>优先级&#xff1b;</li><li>业务重要性&#xff1b;</li></ul>
<p>进行队列划分。</p>
<hr />
<h3>5. 任务必须设置超时</h3>
<p>网络请求、文件下载和第三方接口调用&#xff0c;都应该设置明确超时。</p>
<p>Celery 任务本身也可以设置&#xff1a;</p>


```python
soft_time_limit<span class="token operator">=</span><span class="token number">1800</span>
time_limit<span class="token operator">=</span><span class="token number">1860</span>
```


<p>防止任务永久卡死。</p>
<hr />
<h3>6. Worker 要定期回收</h3>
<p>某些文档解析、图片处理或浏览器任务可能存在内存无法完全释放的问题。</p>
<p>可以配置 Worker 子进程执行一定任务数后重启&#xff1a;</p>


```python
worker_max_tasks_per_child <span class="token operator">=</span> <span class="token number">100</span>
```


<p>避免单个进程长期运行后内存不断上涨。</p>
<hr />
<h3>7. 做好监控与日志</h3>
<p>至少需要关注&#xff1a;</p>
<ul><li>队列长度&#xff1b;</li><li>任务等待时间&#xff1b;</li><li>任务执行时间&#xff1b;</li><li>任务成功率&#xff1b;</li><li>任务失败率&#xff1b;</li><li>重试次数&#xff1b;</li><li>Worker 在线数量&#xff1b;</li><li>Redis 内存&#xff1b;</li><li>数据库连接数。</li></ul>
<p>日志中建议包含&#xff1a;</p>


```text
business_task_id
celery_task_id
user_id
file_id
worker_name
retry_count
current_stage
```


<p>这样才能从一次用户请求追踪到后台任务的完整链路。</p>
<hr />
<h2>九、一个简化的配置示例</h2>


```python
<span class="token keyword">from</span> celery <span class="token keyword">import</span> Celery

app <span class="token operator">=</span> Celery<span class="token punctuation">(</span>
<span class="token string">"document_service"</span><span class="token punctuation">,</span>
broker<span class="token operator">=</span><span class="token string">"redis://redis:6379/0"</span><span class="token punctuation">,</span>
backend<span class="token operator">=</span><span class="token string">"redis://redis:6379/1"</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>

app<span class="token punctuation">.</span>conf<span class="token punctuation">.</span>update<span class="token punctuation">(</span>
task_serializer<span class="token operator">=</span><span class="token string">"json"</span><span class="token punctuation">,</span>
result_serializer<span class="token operator">=</span><span class="token string">"json"</span><span class="token punctuation">,</span>
accept_content<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">"json"</span><span class="token punctuation">]</span><span class="token punctuation">,</span>

task_track_started<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span>
task_acks_late<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span>

worker_prefetch_multiplier<span class="token operator">=</span><span class="token number">1</span><span class="token punctuation">,</span>
worker_max_tasks_per_child<span class="token operator">=</span><span class="token number">100</span><span class="token punctuation">,</span>

result_expires<span class="token operator">=</span><span class="token number">86400</span><span class="token punctuation">,</span>

broker_connection_retry_on_startup<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span>

task_routes<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"tasks.parse_document"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"queue"</span><span class="token punctuation">:</span> <span class="token string">"document_parse"</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string">"tasks.run_ocr"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"queue"</span><span class="token punctuation">:</span> <span class="token string">"ocr"</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token string">"tasks.send_notification"</span><span class="token punctuation">:</span> <span class="token punctuation">{<!-- --></span>
<span class="token string">"queue"</span><span class="token punctuation">:</span> <span class="token string">"notification"</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
```


<p>这段配置表达了几个重要思路&#xff1a;</p>
<ul><li>使用 JSON 序列化&#xff1b;</li><li>记录任务开始状态&#xff1b;</li><li>执行完成后再确认&#xff1b;</li><li>长任务减少预取&#xff1b;</li><li>定期回收 Worker 子进程&#xff1b;</li><li>任务结果一天后过期&#xff1b;</li><li>不同任务进入不同队列。</li></ul>
<p>具体参数仍然需要根据业务任务耗时和资源情况调整。</p>
<hr />
<h2>十、总结</h2>
<p>Celery &#43; Redis 的完整工作流程可以概括为&#xff1a;</p>


```text
1. 用户提交任务
2. Web 服务创建业务任务记录
3. Celery 生成任务消息
4. 消息写入 Redis Broker
5. Web 服务立即返回任务 ID
6. Worker 从 Redis 获取任务
7. Worker 执行业务逻辑
8. 结果写入数据库或对象存储
9. Celery 保存任务执行状态
10. Worker 确认任务完成
11. 用户通过任务 ID 查询结果
```


<p>各组件的职责分别是&#xff1a;</p>


```text
Celery：
负责定义、发送、调度和执行任务

Redis Broker：
负责保存和传递任务消息

Celery Worker：
负责执行真正的业务代码

Result Backend：
负责保存 Celery 状态和轻量结果

业务数据库：
负责保存真实业务状态

对象存储：
负责保存文件和大型结果
```


<p>真正的生产级异步任务系统&#xff0c;并不是简单地安装 Celery 和 Redis。</p>
<p>还需要重点处理&#xff1a;</p>


```text
任务拆分
队列隔离
失败重试
超时控制
消息确认
幂等设计
业务状态
资源限制
监控告警
```


<p>可以把 Celery 理解为任务执行框架&#xff0c;把 Redis 理解为任务中转站。</p>
<p>而系统是否可靠&#xff0c;最终取决于业务层能否正确处理任务重复、任务失败、Worker 宕机和数据不一致等问题。</p>
