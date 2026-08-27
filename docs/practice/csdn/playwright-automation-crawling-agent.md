---
title: "Playwright 深度解析：从浏览器自动化到动态爬虫、自动化测试与 AI Agent"
description: "CSDN 原文全文镜像：Playwright 是一个由微软开发的开源浏览器自动化框架，支持Chromium、Firefox和WebKit三大浏览器引擎，提供多语言接口。它不仅能用于Web自动化测试，还能处理动态网页采集、自动登录、iframe操作等复杂场景。相……"
pageType: article
module: site
updated: '2026-07-30'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "practice"
  - "自动化"
  - "爬虫"
  - "运维"
level: intermediate
prerequisites:
  - "/practice/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-07-30，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-07-30。为适配本站结构，仅补充了站内元数据与来源说明，正文主体保持原文内容。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163335924](https://blog.csdn.net/m0_63309778/article/details/163335924)
- 站内分区：工程实践 / Playwright 自动化
:::

<p><img src="https://i-blog.csdnimg.cn/direct/b043066d1ad7435abab7519ba2cbfa31.png" alt="在这里插入图片描述" /></p>
<h3>前言</h3>
<p>在传统爬虫中&#xff0c;我们通常使用 <code>requests</code>、<code>httpx</code> 或 <code>curl</code> 向服务器发送 HTTP 请求&#xff0c;然后解析返回的 HTML。</p>
<p>这种方式简单、快速、资源消耗低&#xff0c;但它有一个明显的问题&#xff1a;</p>
<blockquote>
<p>它只能获取服务器直接返回的内容&#xff0c;无法像真实用户一样操作浏览器。</p>
</blockquote>
<p>现代网站大量使用 React、Vue、Angular 等前端框架。很多页面打开时&#xff0c;服务器只返回一个简单的 HTML 外壳&#xff0c;真正的数据需要浏览器执行 JavaScript、调用后端接口后才能渲染出来。</p>
<p>除此之外&#xff0c;很多业务系统还包含&#xff1a;</p>
<ul><li>登录状态与 Cookie&#xff1b;</li><li>动态 Token&#xff1b;</li><li>iframe 嵌套页面&#xff1b;</li><li>弹窗和新标签页&#xff1b;</li><li>文件上传与下载&#xff1b;</li><li>懒加载列表&#xff1b;</li><li>WebSocket 通信&#xff1b;</li><li>验证码和风控检测&#xff1b;</li><li>必须点击、滚动或输入后才加载的数据。</li></ul>
<p>在这些场景下&#xff0c;仅使用 HTTP 请求库往往很难完成任务。</p>
<p>Playwright 正是为浏览器自动化而设计的工具。它能够通过统一的 API 控制 Chromium、Firefox 和 WebKit&#xff0c;并提供 TypeScript、JavaScript、Python、Java 和 .NET 等语言接口。目前 Playwright 不仅用于自动化测试&#xff0c;也广泛应用于浏览器脚本、动态网页采集和 AI Agent 浏览器操作。</p>
<hr />
<h3>一、Playwright 是什么</h3>
<p>Playwright 是由 Microsoft 主导开发的开源浏览器自动化框架。</p>
<p>简单来说&#xff0c;Playwright 可以让程序像人一样操作浏览器&#xff0c;例如&#xff1a;</p>
<ul><li>打开网站&#xff1b;</li><li>点击按钮&#xff1b;</li><li>输入账号密码&#xff1b;</li><li>登录系统&#xff1b;</li><li>切换页面&#xff1b;</li><li>操作 iframe&#xff1b;</li><li>等待数据加载&#xff1b;</li><li>监听接口请求&#xff1b;</li><li>上传和下载文件&#xff1b;</li><li>截图和录制视频&#xff1b;</li><li>提取页面中的数据。</li></ul>
<p>它支持以下三类核心浏览器引擎&#xff1a;</p>

<table><thead><tr><th>浏览器引擎</th><th>常见浏览器</th></tr></thead><tbody><tr><td>Chromium</td><td>Chrome、Edge、Chromium</td></tr><tr><td>Firefox</td><td>Mozilla Firefox</td></tr><tr><td>WebKit</td><td>Safari 使用的浏览器引擎</td></tr></tbody></table><p>Playwright 可以使用同一套 API 控制 Chromium、Firefox 和 WebKit&#xff0c;也可以运行 Chrome、Edge 等品牌浏览器&#xff0c;并支持模拟桌面端、平板和移动设备。</p>
<p>可以把 Playwright 理解为&#xff1a;</p>
<blockquote>
<p>一套通过代码控制真实浏览器的工具。</p>
</blockquote>
<p>它不是一个简单的 HTML 解析器&#xff0c;也不是普通的 HTTP 请求库&#xff0c;而是真正启动并控制浏览器进程。</p>
<hr />
<h3>二、Playwright 能做什么</h3>
<h3>2.1 Web 自动化测试</h3>
<p>Playwright 最典型的用途是端到端测试&#xff0c;也就是 E2E 测试。</p>
<p>例如测试一个登录流程&#xff1a;</p>
<ol><li>打开登录页面&#xff1b;</li><li>输入用户名&#xff1b;</li><li>输入密码&#xff1b;</li><li>点击登录&#xff1b;</li><li>验证是否进入首页&#xff1b;</li><li>验证用户名称是否正确显示。</li></ol>


```python
<span class="token keyword">from</span> playwright<span class="token punctuation">.</span>sync_api <span class="token keyword">import</span> sync_playwright<span class="token punctuation">,</span> expect

<span class="token keyword">def</span> <span class="token function">test_login</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">with</span> sync_playwright<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">as</span> p<span class="token punctuation">:</span>
browser <span class="token operator">=</span> p<span class="token punctuation">.</span>chromium<span class="token punctuation">.</span>launch<span class="token punctuation">(</span>headless<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>
page <span class="token operator">=</span> browser<span class="token punctuation">.</span>new_page<span class="token punctuation">(</span><span class="token punctuation">)</span>

page<span class="token punctuation">.</span>goto<span class="token punctuation">(</span><span class="token string">"https://example.com/login"</span><span class="token punctuation">)</span>

page<span class="token punctuation">.</span>get_by_label<span class="token punctuation">(</span><span class="token string">"用户名"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>fill<span class="token punctuation">(</span><span class="token string">"admin"</span><span class="token punctuation">)</span>
page<span class="token punctuation">.</span>get_by_label<span class="token punctuation">(</span><span class="token string">"密码"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>fill<span class="token punctuation">(</span><span class="token string">"123456"</span><span class="token punctuation">)</span>
page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span><span class="token string">"button"</span><span class="token punctuation">,</span> name<span class="token operator">=</span><span class="token string">"登录"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>

expect<span class="token punctuation">(</span>page<span class="token punctuation">)</span><span class="token punctuation">.</span>to_have_url<span class="token punctuation">(</span><span class="token string">"https://example.com/home"</span><span class="token punctuation">)</span>
expect<span class="token punctuation">(</span>page<span class="token punctuation">.</span>get_by_text<span class="token punctuation">(</span><span class="token string">"欢迎回来"</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">.</span>to_be_visible<span class="token punctuation">(</span><span class="token punctuation">)</span>

browser<span class="token punctuation">.</span>close<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>这段代码不是直接调用登录接口&#xff0c;而是完整模拟用户操作浏览器。</p>
<p>因此&#xff0c;它能够验证的不仅是接口&#xff0c;还包括&#xff1a;</p>
<ul><li>页面是否正常渲染&#xff1b;</li><li>前后端是否正确联调&#xff1b;</li><li>按钮是否可点击&#xff1b;</li><li>表单校验是否生效&#xff1b;</li><li>页面跳转是否正确&#xff1b;</li><li>登录状态是否保存&#xff1b;</li><li>权限控制是否正确&#xff1b;</li><li>不同浏览器中是否表现一致。</li></ul>
<p>Playwright Test 本身还提供测试执行器、断言、自动等待、失败重试、并行执行和 Trace 调试等能力。</p>
<hr />
<h3>2.2 动态网页数据采集</h3>
<p>Playwright 也经常被用于动态爬虫。</p>
<p>假设一个网页的初始 HTML 只有&#xff1a;</p>


```html
<span class="token tag"><span class="token tag"><span class="token punctuation"><</span>div</span> <span class="token attr-name">id</span><span class="token attr-value"><span class="token punctuation attr-equals">=</span><span class="token punctuation">"</span>app<span class="token punctuation">"</span></span><span class="token punctuation">></span></span><span class="token tag"><span class="token tag"><span class="token punctuation"></</span>div</span><span class="token punctuation">></span></span>
<span class="token tag"><span class="token tag"><span class="token punctuation"><</span>script</span> <span class="token attr-name">src</span><span class="token attr-value"><span class="token punctuation attr-equals">=</span><span class="token punctuation">"</span>/assets/index.js<span class="token punctuation">"</span></span><span class="token punctuation">></span></span><span class="token script"></span><span class="token tag"><span class="token tag"><span class="token punctuation"></</span>script</span><span class="token punctuation">></span></span>
```


<p>真正的数据由 JavaScript 请求接口后渲染。</p>
<p>使用 <code>requests</code> 获取到的可能只是空壳页面&#xff0c;而 Playwright 会启动浏览器、加载 JavaScript、执行接口请求&#xff0c;最终拿到完整页面。</p>


```python
<span class="token keyword">from</span> playwright<span class="token punctuation">.</span>async_api <span class="token keyword">import</span> async_playwright

<span class="token keyword">async</span> <span class="token keyword">def</span> <span class="token function">fetch_page</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
<span class="token keyword">async</span> <span class="token keyword">with</span> async_playwright<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">as</span> p<span class="token punctuation">:</span>
browser <span class="token operator">=</span> <span class="token keyword">await</span> p<span class="token punctuation">.</span>chromium<span class="token punctuation">.</span>launch<span class="token punctuation">(</span>headless<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>

context <span class="token operator">=</span> <span class="token keyword">await</span> browser<span class="token punctuation">.</span>new_context<span class="token punctuation">(</span>
locale<span class="token operator">=</span><span class="token string">"zh-CN"</span><span class="token punctuation">,</span>
viewport<span class="token operator">=</span><span class="token punctuation">{<!-- --></span><span class="token string">"width"</span><span class="token punctuation">:</span> <span class="token number">1440</span><span class="token punctuation">,</span> <span class="token string">"height"</span><span class="token punctuation">:</span> <span class="token number">900</span><span class="token punctuation">}</span>
<span class="token punctuation">)</span>

page <span class="token operator">=</span> <span class="token keyword">await</span> context<span class="token punctuation">.</span>new_page<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">await</span> page<span class="token punctuation">.</span>goto<span class="token punctuation">(</span>
<span class="token string">"https://example.com"</span><span class="token punctuation">,</span>
wait_until<span class="token operator">=</span><span class="token string">"domcontentloaded"</span>
<span class="token punctuation">)</span>

<span class="token keyword">await</span> page<span class="token punctuation">.</span>locator<span class="token punctuation">(</span><span class="token string">".result-list"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>wait_for<span class="token punctuation">(</span><span class="token punctuation">)</span>

items <span class="token operator">=</span> <span class="token keyword">await</span> page<span class="token punctuation">.</span>locator<span class="token punctuation">(</span><span class="token string">".result-item"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>all_inner_texts<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">print</span><span class="token punctuation">(</span>items<span class="token punctuation">)</span>

<span class="token keyword">await</span> context<span class="token punctuation">.</span>close<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">await</span> browser<span class="token punctuation">.</span>close<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>这类方式适合处理&#xff1a;</p>
<ul><li>JavaScript 动态渲染页面&#xff1b;</li><li>无限滚动列表&#xff1b;</li><li>点击“查看更多”后加载的数据&#xff1b;</li><li>登录后才能访问的页面&#xff1b;</li><li>需要切换筛选条件的页面&#xff1b;</li><li>数据位于 iframe 中的页面&#xff1b;</li><li>通过接口异步加载内容的页面。</li></ul>
<p>不过&#xff0c;Playwright 的资源消耗通常比 <code>requests</code> 更高。因此&#xff0c;更合理的采集架构往往是&#xff1a;</p>


```text
普通静态页面
↓
requests / httpx 直接请求

动态页面或登录页面
↓
Playwright 获取登录状态、接口地址或页面内容

发现稳定数据接口
↓
尽可能切换回 HTTP 客户端批量请求
```


<p>也就是说&#xff1a;</p>
<blockquote>
<p>Playwright 不一定要承担全部采集工作&#xff0c;它也可以负责打开页面、完成登录和发现接口&#xff0c;再把后续批量请求交给更轻量的 HTTP 客户端。</p>
</blockquote>
<hr />
<h3>2.3 自动登录和登录状态保存</h3>
<p>很多系统需要登录后才能访问。</p>
<p>Playwright 可以操作登录页面&#xff0c;也可以保存 Cookie、LocalStorage 等浏览器状态。</p>


```python
context <span class="token operator">=</span> <span class="token keyword">await</span> browser<span class="token punctuation">.</span>new_context<span class="token punctuation">(</span><span class="token punctuation">)</span>

page <span class="token operator">=</span> <span class="token keyword">await</span> context<span class="token punctuation">.</span>new_page<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">await</span> page<span class="token punctuation">.</span>goto<span class="token punctuation">(</span><span class="token string">"https://example.com/login"</span><span class="token punctuation">)</span>

<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_label<span class="token punctuation">(</span><span class="token string">"账号"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>fill<span class="token punctuation">(</span><span class="token string">"admin"</span><span class="token punctuation">)</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_label<span class="token punctuation">(</span><span class="token string">"密码"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>fill<span class="token punctuation">(</span><span class="token string">"123456"</span><span class="token punctuation">)</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span><span class="token string">"button"</span><span class="token punctuation">,</span> name<span class="token operator">=</span><span class="token string">"登录"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">await</span> page<span class="token punctuation">.</span>wait_for_url<span class="token punctuation">(</span><span class="token string">"**/home"</span><span class="token punctuation">)</span>

<span class="token keyword">await</span> context<span class="token punctuation">.</span>storage_state<span class="token punctuation">(</span>path<span class="token operator">=</span><span class="token string">"auth.json"</span><span class="token punctuation">)</span>
```


<p>下次执行时&#xff0c;可以直接加载登录状态&#xff1a;</p>


```python
context <span class="token operator">=</span> <span class="token keyword">await</span> browser<span class="token punctuation">.</span>new_context<span class="token punctuation">(</span>
storage_state<span class="token operator">=</span><span class="token string">"auth.json"</span>
<span class="token punctuation">)</span>
```


<p>这样就不需要每次重新登录。</p>
<p>Playwright 的官方认证方案允许保存并复用已认证状态&#xff0c;从而减少重复登录并提高执行速度。需要注意的是&#xff0c;认证状态文件可能包含敏感 Cookie 和 Header&#xff0c;不应该提交到公开代码仓库。</p>
<hr />
<h3>2.4 操作 iframe</h3>
<p>很多政务网站、企业系统和老旧平台都会使用 iframe。</p>
<p>页面结构可能是&#xff1a;</p>


```html
<span class="token tag"><span class="token tag"><span class="token punctuation"><</span>html</span><span class="token punctuation">></span></span>
<span class="token tag"><span class="token tag"><span class="token punctuation"><</span>body</span><span class="token punctuation">></span></span>
<span class="token tag"><span class="token tag"><span class="token punctuation"><</span>iframe</span> <span class="token attr-name">id</span><span class="token attr-value"><span class="token punctuation attr-equals">=</span><span class="token punctuation">"</span>content-frame<span class="token punctuation">"</span></span> <span class="token attr-name">src</span><span class="token attr-value"><span class="token punctuation attr-equals">=</span><span class="token punctuation">"</span>/detail/content<span class="token punctuation">"</span></span><span class="token punctuation">></span></span><span class="token tag"><span class="token tag"><span class="token punctuation"></</span>iframe</span><span class="token punctuation">></span></span>
<span class="token tag"><span class="token tag"><span class="token punctuation"></</span>body</span><span class="token punctuation">></span></span>
<span class="token tag"><span class="token tag"><span class="token punctuation"></</span>html</span><span class="token punctuation">></span></span>
```


<p>正文并不在主页面中&#xff0c;而在 iframe 里。</p>
<p>Playwright 可以通过 <code>frame_locator</code> 进入 iframe&#xff1a;</p>


```python
frame <span class="token operator">=</span> page<span class="token punctuation">.</span>frame_locator<span class="token punctuation">(</span><span class="token string">"#content-frame"</span><span class="token punctuation">)</span>

title <span class="token operator">=</span> <span class="token keyword">await</span> frame<span class="token punctuation">.</span>locator<span class="token punctuation">(</span><span class="token string">"h1"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>inner_text<span class="token punctuation">(</span><span class="token punctuation">)</span>
content <span class="token operator">=</span> <span class="token keyword">await</span> frame<span class="token punctuation">.</span>locator<span class="token punctuation">(</span><span class="token string">".article-content"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>inner_text<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>Playwright 的 <code>FrameLocator</code> 专门用于定位 iframe&#xff0c;并继续查找 iframe 内部的元素。</p>
<p>这对动态采集非常重要。</p>
<p>例如有些网站的详情页实际上分为两层&#xff1a;</p>


```text
外层壳页面
└── iframe
└── 真正的正文页面
```


<p>如果直接在外层页面执行&#xff1a;</p>


```python
<span class="token keyword">await</span> page<span class="token punctuation">.</span>locator<span class="token punctuation">(</span><span class="token string">"body"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>inner_text<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>拿到的可能是导航栏、菜单、版权信息和 iframe 周边文本&#xff0c;而不是正文。</p>
<p>更合理的方式是&#xff1a;</p>
<ol><li>判断页面中是否存在 iframe&#xff1b;</li><li>找到当前激活的正文 iframe&#xff1b;</li><li>只在该 iframe 内提取正文&#xff1b;</li><li>iframe 不存在时&#xff0c;再回退到主页面提取。</li></ol>
<hr />
<h3>2.5 监听和拦截网络请求</h3>
<p>Playwright 不只能操作 DOM&#xff0c;还可以监听浏览器发送的网络请求。</p>


```python
page<span class="token punctuation">.</span>on<span class="token punctuation">(</span>
<span class="token string">"request"</span><span class="token punctuation">,</span>
<span class="token keyword">lambda</span> request<span class="token punctuation">:</span> <span class="token keyword">print</span><span class="token punctuation">(</span>
<span class="token string">"请求："</span><span class="token punctuation">,</span>
request<span class="token punctuation">.</span>method<span class="token punctuation">,</span>
request<span class="token punctuation">.</span>url
<span class="token punctuation">)</span>
<span class="token punctuation">)</span>

page<span class="token punctuation">.</span>on<span class="token punctuation">(</span>
<span class="token string">"response"</span><span class="token punctuation">,</span>
<span class="token keyword">lambda</span> response<span class="token punctuation">:</span> <span class="token keyword">print</span><span class="token punctuation">(</span>
<span class="token string">"响应："</span><span class="token punctuation">,</span>
response<span class="token punctuation">.</span>status<span class="token punctuation">,</span>
response<span class="token punctuation">.</span>url
<span class="token punctuation">)</span>
<span class="token punctuation">)</span>
```


<p>这项能力在动态页面分析中非常实用。</p>
<p>例如页面上显示了一张列表&#xff0c;但 DOM 结构非常复杂。此时可以监听页面加载过程&#xff0c;找到真正返回 JSON 数据的接口&#xff1a;</p>


```python
<span class="token keyword">async</span> <span class="token keyword">with</span> page<span class="token punctuation">.</span>expect_response<span class="token punctuation">(</span>
<span class="token keyword">lambda</span> response<span class="token punctuation">:</span> <span class="token string">"/api/project/list"</span> <span class="token keyword">in</span> response<span class="token punctuation">.</span>url
<span class="token punctuation">)</span> <span class="token keyword">as</span> response_info<span class="token punctuation">:</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span><span class="token string">"button"</span><span class="token punctuation">,</span> name<span class="token operator">=</span><span class="token string">"查询"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>

response <span class="token operator">=</span> <span class="token keyword">await</span> response_info<span class="token punctuation">.</span>value
data <span class="token operator">=</span> <span class="token keyword">await</span> response<span class="token punctuation">.</span>json<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>相比直接从页面文本中提取&#xff0c;读取接口 JSON 通常更加稳定。</p>
<p>Playwright 还支持&#xff1a;</p>
<ul><li>修改请求 Header&#xff1b;</li><li>阻止图片、字体和视频&#xff1b;</li><li>Mock API 返回值&#xff1b;</li><li>修改接口响应&#xff1b;</li><li>监听 WebSocket&#xff1b;</li><li>Mock WebSocket 通信&#xff1b;</li><li>模拟接口异常和超时。</li></ul>
<p>Playwright 官方网络能力覆盖 HTTP、HTTPS 和 WebSocket 的监听、修改与模拟。</p>
<p>例如&#xff0c;采集时可以阻止图片加载&#xff1a;</p>


```python
<span class="token keyword">async</span> <span class="token keyword">def</span> <span class="token function">handle_route</span><span class="token punctuation">(</span>route<span class="token punctuation">)</span><span class="token punctuation">:</span>
resource_type <span class="token operator">=</span> route<span class="token punctuation">.</span>request<span class="token punctuation">.</span>resource_type

<span class="token keyword">if</span> resource_type <span class="token keyword">in</span> <span class="token punctuation">{<!-- --></span><span class="token string">"image"</span><span class="token punctuation">,</span> <span class="token string">"font"</span><span class="token punctuation">,</span> <span class="token string">"media"</span><span class="token punctuation">}</span><span class="token punctuation">:</span>
<span class="token keyword">await</span> route<span class="token punctuation">.</span>abort<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">else</span><span class="token punctuation">:</span>
<span class="token keyword">await</span> route<span class="token punctuation">.</span>continue_<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">await</span> page<span class="token punctuation">.</span>route<span class="token punctuation">(</span><span class="token string">"**/*"</span><span class="token punctuation">,</span> handle_route<span class="token punctuation">)</span>
```


<p>这样可以减少网络流量&#xff0c;但需要谨慎使用。有些页面会通过图片、字体或其他资源的加载结果判断页面状态&#xff0c;过度拦截可能导致页面逻辑异常。</p>
<hr />
<h3>2.6 文件上传与下载</h3>
<p>Playwright 可以自动操作文件上传组件&#xff1a;</p>


```python
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_label<span class="token punctuation">(</span><span class="token string">"上传文件"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>set_input_files<span class="token punctuation">(</span>
<span class="token string">"documents/report.pdf"</span>
<span class="token punctuation">)</span>
```


<p>也可以监听文件下载&#xff1a;</p>


```python
<span class="token keyword">async</span> <span class="token keyword">with</span> page<span class="token punctuation">.</span>expect_download<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">as</span> download_info<span class="token punctuation">:</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span><span class="token string">"button"</span><span class="token punctuation">,</span> name<span class="token operator">=</span><span class="token string">"导出"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>

download <span class="token operator">=</span> <span class="token keyword">await</span> download_info<span class="token punctuation">.</span>value

<span class="token keyword">await</span> download<span class="token punctuation">.</span>save_as<span class="token punctuation">(</span>
<span class="token string-interpolation"><span class="token string">f"downloads/</span><span class="token interpolation"><span class="token punctuation">{<!-- --></span>download<span class="token punctuation">.</span>suggested_filename<span class="token punctuation">}</span></span><span class="token string">"</span></span>
<span class="token punctuation">)</span>
```


<p>Playwright 会为下载事件创建对应的 <code>Download</code> 对象&#xff0c;可以获取文件名称、下载地址并保存文件。下载文件默认与产生它的 BrowserContext 生命周期相关。</p>
<p>因此&#xff0c;它非常适合&#xff1a;</p>
<ul><li>自动下载报表&#xff1b;</li><li>自动导出 Excel&#xff1b;</li><li>上传标书或附件&#xff1b;</li><li>批量上传文档&#xff1b;</li><li>验证上传下载功能&#xff1b;</li><li>自动化处理后台管理系统。</li></ul>
<hr />
<h3>2.7 截图、视频和 Trace</h3>
<p>Playwright 可以对页面进行截图&#xff1a;</p>


```python
<span class="token keyword">await</span> page<span class="token punctuation">.</span>screenshot<span class="token punctuation">(</span>
path<span class="token operator">=</span><span class="token string">"page.png"</span><span class="token punctuation">,</span>
full_page<span class="token operator">=</span><span class="token boolean">True</span>
<span class="token punctuation">)</span>
```


<p>也可以只截取某个元素&#xff1a;</p>


```python
<span class="token keyword">await</span> page<span class="token punctuation">.</span>locator<span class="token punctuation">(</span><span class="token string">".article-content"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>screenshot<span class="token punctuation">(</span>
path<span class="token operator">=</span><span class="token string">"article.png"</span>
<span class="token punctuation">)</span>
```


<p>官方截图 API 支持全页面截图、指定区域截图和将图片读取为内存 Buffer。</p>
<p>除了截图&#xff0c;Playwright 还支持录制浏览器视频和 Trace。</p>
<p>Trace 可以记录&#xff1a;</p>
<ul><li>每一步执行的操作&#xff1b;</li><li>操作前后的页面状态&#xff1b;</li><li>DOM 快照&#xff1b;</li><li>网络请求&#xff1b;</li><li>Console 日志&#xff1b;</li><li>页面截图&#xff1b;</li><li>操作耗时&#xff1b;</li><li>使用的 Locator&#xff1b;</li><li>错误发生的位置。</li></ul>


```bash
npx playwright <span class="token builtin class-name">test</span> <span class="token parameter variable">--trace</span> on
```


<p>运行完成后可以打开 Trace&#xff1a;</p>


```bash
npx playwright show-trace trace.zip
```


<p>Trace Viewer 是 Playwright 提供的可视化分析工具&#xff0c;特别适合排查 CI 环境中偶发失败的问题。</p>
<hr />
<h3>2.8 模拟不同设备和网络环境</h3>
<p>Playwright 可以模拟&#xff1a;</p>
<ul><li>手机和平板&#xff1b;</li><li>不同屏幕尺寸&#xff1b;</li><li>不同 User-Agent&#xff1b;</li><li>不同语言&#xff1b;</li><li>不同时区&#xff1b;</li><li>地理位置&#xff1b;</li><li>摄像头和麦克风权限&#xff1b;</li><li>深色模式&#xff1b;</li><li>离线状态&#xff1b;</li><li>自定义 HTTP Header。</li></ul>


```python
context <span class="token operator">=</span> <span class="token keyword">await</span> browser<span class="token punctuation">.</span>new_context<span class="token punctuation">(</span>
viewport<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"width"</span><span class="token punctuation">:</span> <span class="token number">390</span><span class="token punctuation">,</span>
<span class="token string">"height"</span><span class="token punctuation">:</span> <span class="token number">844</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
user_agent<span class="token operator">=</span><span class="token string">"Mozilla/5.0 ..."</span><span class="token punctuation">,</span>
locale<span class="token operator">=</span><span class="token string">"zh-CN"</span><span class="token punctuation">,</span>
timezone_id<span class="token operator">=</span><span class="token string">"Asia/Shanghai"</span><span class="token punctuation">,</span>
color_scheme<span class="token operator">=</span><span class="token string">"dark"</span>
<span class="token punctuation">)</span>
```


<p>这使 Playwright 不仅能测试桌面网页&#xff0c;还能验证移动端适配、国际化、权限和网络异常场景。浏览器和 BrowserContext 级别均可以配置设备模拟、网络和录制能力。</p>
<hr />
<h3>2.9 作为 AI Agent 的浏览器执行工具</h3>
<p>随着 AI Agent 的发展&#xff0c;Playwright 也逐渐成为浏览器 Agent 的基础执行工具。</p>
<p>一个浏览器 Agent 的工作流程可能是&#xff1a;</p>


```text
用户提出任务
↓
大模型理解目标
↓
读取当前页面结构
↓
判断下一步操作
↓
调用 Playwright 点击、输入或跳转
↓
获取新的页面状态
↓
继续推理和执行
```


<p>例如用户说&#xff1a;</p>
<blockquote>
<p>登录后台&#xff0c;找到昨天失败的任务&#xff0c;把错误日志整理出来。</p>
</blockquote>
<p>Agent 可以通过 Playwright&#xff1a;</p>
<ol><li>打开系统&#xff1b;</li><li>填写账号密码&#xff1b;</li><li>进入任务中心&#xff1b;</li><li>选择时间范围&#xff1b;</li><li>筛选失败任务&#xff1b;</li><li>打开任务详情&#xff1b;</li><li>读取错误信息&#xff1b;</li><li>返回总结。</li></ol>
<p>Microsoft 目前提供了 Playwright MCP Server&#xff0c;使大模型能够通过 Model Context Protocol 调用浏览器自动化能力。官方实现主要使用结构化的可访问性快照帮助模型理解页面&#xff0c;而不是完全依赖截图和视觉识别。</p>
<hr />
<h2>三、Playwright 的核心对象模型</h2>
<p>学习 Playwright 时&#xff0c;最重要的是理解以下几个核心对象&#xff1a;</p>


```text
Playwright
├── Chromium
├── Firefox
└── WebKit
↓
Browser
↓
BrowserContext
↓
Page
↓
Frame / Locator
```


<h3>3.1 Playwright</h3>
<p><code>Playwright</code> 是整个框架的入口。</p>


```python
<span class="token keyword">async</span> <span class="token keyword">with</span> async_playwright<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">as</span> p<span class="token punctuation">:</span>
browser <span class="token operator">=</span> <span class="token keyword">await</span> p<span class="token punctuation">.</span>chromium<span class="token punctuation">.</span>launch<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>这里的 <code>p</code> 就是 Playwright 实例。</p>
<p>它提供三个主要的浏览器类型&#xff1a;</p>


```python
p<span class="token punctuation">.</span>chromium
p<span class="token punctuation">.</span>firefox
p<span class="token punctuation">.</span>webkit
```


<hr />
<h3>3.2 Browser</h3>
<p><code>Browser</code> 表示一个浏览器进程。</p>


```python
browser <span class="token operator">=</span> <span class="token keyword">await</span> p<span class="token punctuation">.</span>chromium<span class="token punctuation">.</span>launch<span class="token punctuation">(</span>
headless<span class="token operator">=</span><span class="token boolean">True</span>
<span class="token punctuation">)</span>
```


<p>常见配置包括&#xff1a;</p>


```python
browser <span class="token operator">=</span> <span class="token keyword">await</span> p<span class="token punctuation">.</span>chromium<span class="token punctuation">.</span>launch<span class="token punctuation">(</span>
headless<span class="token operator">=</span><span class="token boolean">False</span><span class="token punctuation">,</span>
slow_mo<span class="token operator">=</span><span class="token number">500</span>
<span class="token punctuation">)</span>
```


<p>其中&#xff1a;</p>
<ul><li><code>headless&#61;True</code>&#xff1a;无界面运行&#xff1b;</li><li><code>headless&#61;False</code>&#xff1a;显示浏览器界面&#xff1b;</li><li><code>slow_mo&#61;500</code>&#xff1a;每一步操作放慢 500 毫秒。</li></ul>
<p>在开发和排查问题时&#xff0c;可以使用有界面模式&#xff1b;在服务器和生产环境中&#xff0c;通常使用无头模式。</p>
<hr />
<h3>3.3 BrowserContext</h3>
<p><code>BrowserContext</code> 是 Playwright 非常重要的设计。</p>
<p>可以把它理解为一个独立的无痕浏览器环境。</p>
<p>每个 BrowserContext 都有自己独立的&#xff1a;</p>
<ul><li>Cookie&#xff1b;</li><li>LocalStorage&#xff1b;</li><li>SessionStorage&#xff1b;</li><li>页面&#xff1b;</li><li>权限&#xff1b;</li><li>User-Agent&#xff1b;</li><li>网络规则&#xff1b;</li><li>登录状态。</li></ul>


```python
context_1 <span class="token operator">=</span> <span class="token keyword">await</span> browser<span class="token punctuation">.</span>new_context<span class="token punctuation">(</span><span class="token punctuation">)</span>
context_2 <span class="token operator">=</span> <span class="token keyword">await</span> browser<span class="token punctuation">.</span>new_context<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>即使它们共用同一个 Browser 进程&#xff0c;也不会共享登录状态。</p>
<p>Playwright 使用 BrowserContext 实现测试隔离&#xff0c;每个测试可以拥有独立的浏览器上下文&#xff0c;从而降低状态污染和级联失败。</p>
<p>这意味着执行批量任务时&#xff0c;不一定要为每个任务都启动一个新的浏览器进程。</p>
<p>更常见的方式是&#xff1a;</p>


```text
一个 Browser
├── Context A：账号 A
├── Context B：账号 B
└── Context C：游客状态
```


<p>这样既实现状态隔离&#xff0c;也能减少频繁启动浏览器的成本。</p>
<hr />
<h3>3.4 Page</h3>
<p><code>Page</code> 表示浏览器中的一个标签页或弹窗页面。</p>


```python
page <span class="token operator">=</span> <span class="token keyword">await</span> context<span class="token punctuation">.</span>new_page<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>一个 BrowserContext 可以包含多个 Page。</p>


```python
page_1 <span class="token operator">=</span> <span class="token keyword">await</span> context<span class="token punctuation">.</span>new_page<span class="token punctuation">(</span><span class="token punctuation">)</span>
page_2 <span class="token operator">=</span> <span class="token keyword">await</span> context<span class="token punctuation">.</span>new_page<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>Playwright 中的 Page 可以&#xff1a;</p>
<ul><li>打开 URL&#xff1b;</li><li>获取 HTML&#xff1b;</li><li>执行 JavaScript&#xff1b;</li><li>操作页面元素&#xff1b;</li><li>监听请求&#xff1b;</li><li>监听弹窗&#xff1b;</li><li>监听下载&#xff1b;</li><li>截图&#xff1b;</li><li>获取 Console 日志。</li></ul>
<p>官方将 Page 定义为 BrowserContext 中的一个标签页或弹出窗口。</p>
<hr />
<h3>3.5 Locator</h3>
<p><code>Locator</code> 是 Playwright 元素定位机制的核心。</p>


```python
login_button <span class="token operator">=</span> page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span>
<span class="token string">"button"</span><span class="token punctuation">,</span>
name<span class="token operator">=</span><span class="token string">"登录"</span>
<span class="token punctuation">)</span>
```


<p>这里的 <code>login_button</code> 不是某一个固定 DOM 节点&#xff0c;而是一种“如何找到这个元素”的描述。</p>
<p>执行操作时&#xff0c;Playwright 会重新查找当前最新的 DOM 元素&#xff1a;</p>


```python
<span class="token keyword">await</span> login_button<span class="token punctuation">.</span>hover<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">await</span> login_button<span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>即使两次操作之间页面发生了重新渲染&#xff0c;Playwright 也会在每次操作前重新解析 Locator&#xff0c;而不是一直引用旧的 DOM 节点。</p>
<p>这对 React、Vue 等动态更新 DOM 的页面非常重要。</p>
<hr />
<h2>四、Playwright 的底层工作原理</h2>
<h3>4.1 整体调用链路</h3>
<p>Playwright 的工作过程可以抽象为&#xff1a;</p>
<div class="mermaid mermaid-newversion mermaid-flowchart"></div>
<p>开发者编写&#xff1a;</p>


```python
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span>
<span class="token string">"button"</span><span class="token punctuation">,</span>
name<span class="token operator">=</span><span class="token string">"登录"</span>
<span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>背后会经历以下过程&#xff1a;</p>
<ol><li>Python 客户端接收调用&#xff1b;</li><li>将操作转换为 Playwright 协议消息&#xff1b;</li><li>Playwright Driver 接收消息&#xff1b;</li><li>Driver 将操作发送给对应浏览器&#xff1b;</li><li>浏览器查找元素并执行点击&#xff1b;</li><li>浏览器产生页面、网络和事件变化&#xff1b;</li><li>Driver 将结果返回给 Python 客户端&#xff1b;</li><li>Python 代码继续执行。</li></ol>
<p>以 Playwright Python 为例&#xff0c;官方代码仓库说明&#xff0c;Python 客户端通过管道向内置的 Node.js Driver 发送 JSON 消息&#xff0c;通信协议由 Playwright 上游协议定义。</p>
<p>因此&#xff0c;Python 版本并不是完全用 Python 重新实现了一套浏览器控制核心。</p>
<p>更准确的理解是&#xff1a;</p>


```text
Python API
↓
Playwright Driver
↓
浏览器
```


<p>Java 和 .NET 等语言版本也采用类似的语言绑定与 Driver 通信方式。</p>
<hr />
<h3>4.2 Playwright 并不只是 CDP 的简单封装</h3>
<p>CDP 是 Chrome DevTools Protocol&#xff0c;也就是 Chromium 系浏览器使用的调试协议。</p>
<p>Playwright 确实支持通过 <code>connect_over_cdp</code> 连接现有 Chromium 浏览器&#xff1a;</p>


```python
browser <span class="token operator">=</span> <span class="token keyword">await</span> p<span class="token punctuation">.</span>chromium<span class="token punctuation">.</span>connect_over_cdp<span class="token punctuation">(</span>
<span class="token string">"http://localhost:9222"</span>
<span class="token punctuation">)</span>
```


<p>但 CDP 连接仅适用于 Chromium 浏览器。Playwright 官方还明确指出&#xff0c;通过 CDP 建立的连接&#xff0c;相比 Playwright 自己的协议连接功能完整度更低&#xff1b;复杂功能更适合使用 Playwright 原生连接方式。</p>
<p>因此&#xff0c;不能简单地把 Playwright 理解为&#xff1a;</p>
<blockquote>
<p>在 CDP 外面封装了一层 API。</p>
</blockquote>
<p>更准确的说法是&#xff1a;</p>
<blockquote>
<p>Playwright 提供统一的上层 API 和自己的通信协议&#xff0c;再针对 Chromium、Firefox 和 WebKit 实现浏览器控制能力。</p>
</blockquote>
<hr />
<h3>4.3 事件驱动模型</h3>
<p>浏览器中的很多行为不是同步发生的。</p>
<p>例如点击按钮后&#xff0c;可能产生&#xff1a;</p>
<ul><li>页面跳转&#xff1b;</li><li>弹出新窗口&#xff1b;</li><li>文件下载&#xff1b;</li><li>接口请求&#xff1b;</li><li>iframe 加载&#xff1b;</li><li>WebSocket 消息&#xff1b;</li><li>DOM 更新。</li></ul>
<p>因此 Playwright 大量使用事件驱动机制。</p>
<p>例如等待新标签页&#xff1a;</p>


```python
<span class="token keyword">async</span> <span class="token keyword">with</span> context<span class="token punctuation">.</span>expect_page<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">as</span> page_info<span class="token punctuation">:</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_text<span class="token punctuation">(</span><span class="token string">"打开详情"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>

new_page <span class="token operator">=</span> <span class="token keyword">await</span> page_info<span class="token punctuation">.</span>value
```


<p>等待下载&#xff1a;</p>


```python
<span class="token keyword">async</span> <span class="token keyword">with</span> page<span class="token punctuation">.</span>expect_download<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">as</span> download_info<span class="token punctuation">:</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_text<span class="token punctuation">(</span><span class="token string">"下载"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>

download <span class="token operator">=</span> <span class="token keyword">await</span> download_info<span class="token punctuation">.</span>value
```


<p>等待接口响应&#xff1a;</p>


```python
<span class="token keyword">async</span> <span class="token keyword">with</span> page<span class="token punctuation">.</span>expect_response<span class="token punctuation">(</span>
<span class="token keyword">lambda</span> response<span class="token punctuation">:</span> <span class="token string">"/api/detail"</span> <span class="token keyword">in</span> response<span class="token punctuation">.</span>url
<span class="token punctuation">)</span> <span class="token keyword">as</span> response_info<span class="token punctuation">:</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_text<span class="token punctuation">(</span><span class="token string">"查看详情"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>

response <span class="token operator">=</span> <span class="token keyword">await</span> response_info<span class="token punctuation">.</span>value
```


<p>这里有一个重要原则&#xff1a;</p>
<blockquote>
<p>先注册等待事件&#xff0c;再触发页面操作。</p>
</blockquote>
<p>错误写法&#xff1a;</p>


```python
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_text<span class="token punctuation">(</span><span class="token string">"下载"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>
download <span class="token operator">=</span> <span class="token keyword">await</span> page<span class="token punctuation">.</span>wait_for_event<span class="token punctuation">(</span><span class="token string">"download"</span><span class="token punctuation">)</span>
```


<p>点击完成时&#xff0c;下载事件可能已经发生&#xff0c;后续等待就可能超时。</p>
<p>正确写法&#xff1a;</p>


```python
<span class="token keyword">async</span> <span class="token keyword">with</span> page<span class="token punctuation">.</span>expect_download<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">as</span> download_info<span class="token punctuation">:</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_text<span class="token punctuation">(</span><span class="token string">"下载"</span><span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<hr />
<h2>五、Playwright 为什么比传统自动化脚本更稳定</h2>
<h3>5.1 自动等待</h3>
<p>传统浏览器脚本中经常出现大量固定等待&#xff1a;</p>


```python
<span class="token keyword">await</span> page<span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token string">"#submit"</span><span class="token punctuation">)</span>
<span class="token keyword">await</span> asyncio<span class="token punctuation">.</span>sleep<span class="token punctuation">(</span><span class="token number">3</span><span class="token punctuation">)</span>
```


<p>问题在于&#xff1a;</p>
<ul><li>网络快时&#xff0c;浪费三秒&#xff1b;</li><li>网络慢时&#xff0c;三秒仍然不够&#xff1b;</li><li>CI 环境负载变化时容易偶发失败。</li></ul>
<p>Playwright 在执行点击等操作之前&#xff0c;会自动检查元素状态。</p>
<p>以点击为例&#xff0c;Playwright 通常会检查&#xff1a;</p>
<ul><li>元素是否已经找到&#xff1b;</li><li>元素是否可见&#xff1b;</li><li>元素位置是否稳定&#xff1b;</li><li>元素是否启用&#xff1b;</li><li>元素是否能够接收点击事件&#xff1b;</li><li>元素是否被其他弹窗或遮罩层挡住。</li></ul>
<p>只有满足操作条件后&#xff0c;Playwright 才会执行点击。</p>


```python
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span>
<span class="token string">"button"</span><span class="token punctuation">,</span>
name<span class="token operator">=</span><span class="token string">"提交"</span>
<span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>这行代码背后实际上包含了查找、等待、状态检查、滚动和点击等过程。</p>
<hr />
<h3>5.2 自动重试断言</h3>
<p>Playwright 的 Web First Assertions 也会自动重试。</p>


```python
expect<span class="token punctuation">(</span>page<span class="token punctuation">.</span>locator<span class="token punctuation">(</span><span class="token string">".status"</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">.</span>to_have_text<span class="token punctuation">(</span>
<span class="token string">"处理完成"</span>
<span class="token punctuation">)</span>
```


<p>如果页面暂时显示的是&#xff1a;</p>


```text
处理中
```


<p>Playwright 不会立即判定失败&#xff0c;而是会在超时时间内持续检查&#xff0c;直到&#xff1a;</p>


```text
处理完成
```


<p>或者等待超时。</p>
<p>Playwright 官方断言支持对元素可见性、文本、属性、状态、URL 和响应等条件进行自动重试。</p>
<hr />
<h3>5.3 Locator 会重新解析元素</h3>
<p>传统代码可能先找到 DOM 元素&#xff0c;然后长期持有这个引用。</p>
<p>但 React 或 Vue 重新渲染后&#xff0c;原来的元素可能已经被删除并替换。</p>
<p>Playwright 推荐使用 Locator&#xff1a;</p>


```python
button <span class="token operator">=</span> page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span>
<span class="token string">"button"</span><span class="token punctuation">,</span>
name<span class="token operator">=</span><span class="token string">"提交"</span>
<span class="token punctuation">)</span>

<span class="token keyword">await</span> button<span class="token punctuation">.</span>hover<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">await</span> button<span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>
```


<p>每次真正执行操作时&#xff0c;Locator 都会根据当前页面重新寻找元素&#xff0c;从而减少旧元素引用导致的问题。</p>
<hr />
<h2>六、Playwright 常见定位方式</h2>
<p>Playwright 支持多种元素定位方式。</p>
<h3>6.1 根据角色定位</h3>


```python
page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span>
<span class="token string">"button"</span><span class="token punctuation">,</span>
name<span class="token operator">=</span><span class="token string">"登录"</span>
<span class="token punctuation">)</span>
```


<p>适合按钮、链接、输入框、复选框等标准元素。</p>
<hr />
<h3>6.2 根据 Label 定位</h3>


```python
page<span class="token punctuation">.</span>get_by_label<span class="token punctuation">(</span><span class="token string">"用户名"</span><span class="token punctuation">)</span>
page<span class="token punctuation">.</span>get_by_label<span class="token punctuation">(</span><span class="token string">"密码"</span><span class="token punctuation">)</span>
```


<p>适合表单元素。</p>
<hr />
<h3>6.3 根据文本定位</h3>


```python
page<span class="token punctuation">.</span>get_by_text<span class="token punctuation">(</span><span class="token string">"查看详情"</span><span class="token punctuation">)</span>
```


<hr />
<h3>6.4 根据 Placeholder 定位</h3>


```python
page<span class="token punctuation">.</span>get_by_placeholder<span class="token punctuation">(</span><span class="token string">"请输入关键词"</span><span class="token punctuation">)</span>
```


<hr />
<h3>6.5 根据 Test ID 定位</h3>


```python
page<span class="token punctuation">.</span>get_by_test_id<span class="token punctuation">(</span><span class="token string">"submit-button"</span><span class="token punctuation">)</span>
```


<p>前端页面&#xff1a;</p>


```html
<span class="token tag"><span class="token tag"><span class="token punctuation"><</span>button</span> <span class="token attr-name">data-testid</span><span class="token attr-value"><span class="token punctuation attr-equals">=</span><span class="token punctuation">"</span>submit-button<span class="token punctuation">"</span></span><span class="token punctuation">></span></span>
提交
<span class="token tag"><span class="token tag"><span class="token punctuation"></</span>button</span><span class="token punctuation">></span></span>
```


<hr />
<h3>6.6 CSS 选择器</h3>


```python
page<span class="token punctuation">.</span>locator<span class="token punctuation">(</span><span class="token string">".article-list .article-item"</span><span class="token punctuation">)</span>
```


<hr />
<h3>6.7 XPath</h3>


```python
page<span class="token punctuation">.</span>locator<span class="token punctuation">(</span>
<span class="token string">"xpath=//button[contains(text(),'提交')]"</span>
<span class="token punctuation">)</span>
```


<p>虽然 XPath 能力很强&#xff0c;但通常不应该优先使用过长的绝对 XPath&#xff1a;</p>


```python
<span class="token operator">/</span>html<span class="token operator">/</span>body<span class="token operator">/</span>div<span class="token punctuation">[</span><span class="token number">2</span><span class="token punctuation">]</span><span class="token operator">/</span>div<span class="token punctuation">[</span><span class="token number">3</span><span class="token punctuation">]</span><span class="token operator">/</span>div<span class="token operator">/</span>button
```


<p>页面结构稍微变化&#xff0c;定位就可能失效。</p>
<p>Playwright 官方最佳实践建议优先使用用户可感知的属性和明确契约&#xff0c;例如 Role、Label、Text 和 Test ID&#xff0c;而不是依赖脆弱的 DOM 层级。</p>
<hr />
<h2>七、一个相对完整的 Python 示例</h2>
<p>下面使用 Playwright 完成&#xff1a;</p>
<ol><li>启动浏览器&#xff1b;</li><li>创建独立上下文&#xff1b;</li><li>打开页面&#xff1b;</li><li>监听接口&#xff1b;</li><li>点击查询&#xff1b;</li><li>提取列表&#xff1b;</li><li>保存截图&#xff1b;</li><li>捕获异常&#xff1b;</li><li>释放资源。</li></ol>


```python
<span class="token keyword">import</span> asyncio
<span class="token keyword">from</span> pathlib <span class="token keyword">import</span> Path

<span class="token keyword">from</span> playwright<span class="token punctuation">.</span>async_api <span class="token keyword">import</span> <span class="token punctuation">(</span>
async_playwright<span class="token punctuation">,</span>
TimeoutError <span class="token keyword">as</span> PlaywrightTimeoutError<span class="token punctuation">,</span>
<span class="token punctuation">)</span>

<span class="token keyword">async</span> <span class="token keyword">def</span> <span class="token function">collect_data</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token operator">-</span><span class="token operator">></span> <span class="token builtin">list</span><span class="token punctuation">[</span><span class="token builtin">dict</span><span class="token punctuation">[</span><span class="token builtin">str</span><span class="token punctuation">,</span> <span class="token builtin">str</span><span class="token punctuation">]</span><span class="token punctuation">]</span><span class="token punctuation">:</span>
output_dir <span class="token operator">=</span> Path<span class="token punctuation">(</span><span class="token string">"artifacts"</span><span class="token punctuation">)</span>
output_dir<span class="token punctuation">.</span>mkdir<span class="token punctuation">(</span>parents<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span> exist_ok<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">)</span>

<span class="token keyword">async</span> <span class="token keyword">with</span> async_playwright<span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token keyword">as</span> playwright<span class="token punctuation">:</span>
browser <span class="token operator">=</span> <span class="token keyword">await</span> playwright<span class="token punctuation">.</span>chromium<span class="token punctuation">.</span>launch<span class="token punctuation">(</span>
headless<span class="token operator">=</span><span class="token boolean">True</span>
<span class="token punctuation">)</span>

context <span class="token operator">=</span> <span class="token keyword">await</span> browser<span class="token punctuation">.</span>new_context<span class="token punctuation">(</span>
locale<span class="token operator">=</span><span class="token string">"zh-CN"</span><span class="token punctuation">,</span>
viewport<span class="token operator">=</span><span class="token punctuation">{<!-- --></span>
<span class="token string">"width"</span><span class="token punctuation">:</span> <span class="token number">1440</span><span class="token punctuation">,</span>
<span class="token string">"height"</span><span class="token punctuation">:</span> <span class="token number">900</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>

page <span class="token operator">=</span> <span class="token keyword">await</span> context<span class="token punctuation">.</span>new_page<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">try</span><span class="token punctuation">:</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>goto<span class="token punctuation">(</span>
<span class="token string">"https://example.com/projects"</span><span class="token punctuation">,</span>
wait_until<span class="token operator">=</span><span class="token string">"domcontentloaded"</span><span class="token punctuation">,</span>
timeout<span class="token operator">=</span><span class="token number">30_000</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>

keyword_input <span class="token operator">=</span> page<span class="token punctuation">.</span>get_by_placeholder<span class="token punctuation">(</span>
<span class="token string">"请输入项目名称"</span>
<span class="token punctuation">)</span>

<span class="token keyword">await</span> keyword_input<span class="token punctuation">.</span>fill<span class="token punctuation">(</span><span class="token string">"人工智能"</span><span class="token punctuation">)</span>

<span class="token keyword">async</span> <span class="token keyword">with</span> page<span class="token punctuation">.</span>expect_response<span class="token punctuation">(</span>
<span class="token keyword">lambda</span> response<span class="token punctuation">:</span> <span class="token punctuation">(</span>
<span class="token string">"/api/project/list"</span> <span class="token keyword">in</span> response<span class="token punctuation">.</span>url
<span class="token keyword">and</span> response<span class="token punctuation">.</span>status <span class="token operator">==</span> <span class="token number">200</span>
<span class="token punctuation">)</span><span class="token punctuation">,</span>
timeout<span class="token operator">=</span><span class="token number">20_000</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span> <span class="token keyword">as</span> response_info<span class="token punctuation">:</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>get_by_role<span class="token punctuation">(</span>
<span class="token string">"button"</span><span class="token punctuation">,</span>
name<span class="token operator">=</span><span class="token string">"查询"</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span><span class="token punctuation">.</span>click<span class="token punctuation">(</span><span class="token punctuation">)</span>

response <span class="token operator">=</span> <span class="token keyword">await</span> response_info<span class="token punctuation">.</span>value
response_data <span class="token operator">=</span> <span class="token keyword">await</span> response<span class="token punctuation">.</span>json<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">await</span> page<span class="token punctuation">.</span>locator<span class="token punctuation">(</span>
<span class="token string">".project-list"</span>
<span class="token punctuation">)</span><span class="token punctuation">.</span>wait_for<span class="token punctuation">(</span>
state<span class="token operator">=</span><span class="token string">"visible"</span><span class="token punctuation">,</span>
timeout<span class="token operator">=</span><span class="token number">20_000</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>

rows <span class="token operator">=</span> page<span class="token punctuation">.</span>locator<span class="token punctuation">(</span><span class="token string">".project-item"</span><span class="token punctuation">)</span>
count <span class="token operator">=</span> <span class="token keyword">await</span> rows<span class="token punctuation">.</span>count<span class="token punctuation">(</span><span class="token punctuation">)</span>

results<span class="token punctuation">:</span> <span class="token builtin">list</span><span class="token punctuation">[</span><span class="token builtin">dict</span><span class="token punctuation">[</span><span class="token builtin">str</span><span class="token punctuation">,</span> <span class="token builtin">str</span><span class="token punctuation">]</span><span class="token punctuation">]</span> <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>

<span class="token keyword">for</span> index <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>count<span class="token punctuation">)</span><span class="token punctuation">:</span>
row <span class="token operator">=</span> rows<span class="token punctuation">.</span>nth<span class="token punctuation">(</span>index<span class="token punctuation">)</span>

title <span class="token operator">=</span> <span class="token keyword">await</span> row<span class="token punctuation">.</span>locator<span class="token punctuation">(</span>
<span class="token string">".project-title"</span>
<span class="token punctuation">)</span><span class="token punctuation">.</span>inner_text<span class="token punctuation">(</span><span class="token punctuation">)</span>

date <span class="token operator">=</span> <span class="token keyword">await</span> row<span class="token punctuation">.</span>locator<span class="token punctuation">(</span>
<span class="token string">".project-date"</span>
<span class="token punctuation">)</span><span class="token punctuation">.</span>inner_text<span class="token punctuation">(</span><span class="token punctuation">)</span>

results<span class="token punctuation">.</span>append<span class="token punctuation">(</span>
<span class="token punctuation">{<!-- --></span>
<span class="token string">"title"</span><span class="token punctuation">:</span> title<span class="token punctuation">.</span>strip<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
<span class="token string">"date"</span><span class="token punctuation">:</span> date<span class="token punctuation">.</span>strip<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
<span class="token punctuation">}</span>
<span class="token punctuation">)</span>

<span class="token keyword">await</span> page<span class="token punctuation">.</span>screenshot<span class="token punctuation">(</span>
path<span class="token operator">=</span>output_dir <span class="token operator">/</span> <span class="token string">"result.png"</span><span class="token punctuation">,</span>
full_page<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>

<span class="token keyword">print</span><span class="token punctuation">(</span>
<span class="token string">"接口返回数据量："</span><span class="token punctuation">,</span>
<span class="token builtin">len</span><span class="token punctuation">(</span>response_data<span class="token punctuation">.</span>get<span class="token punctuation">(</span><span class="token string">"data"</span><span class="token punctuation">,</span> <span class="token punctuation">[</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>

<span class="token keyword">return</span> results

<span class="token keyword">except</span> PlaywrightTimeoutError<span class="token punctuation">:</span>
<span class="token keyword">await</span> page<span class="token punctuation">.</span>screenshot<span class="token punctuation">(</span>
path<span class="token operator">=</span>output_dir <span class="token operator">/</span> <span class="token string">"timeout.png"</span><span class="token punctuation">,</span>
full_page<span class="token operator">=</span><span class="token boolean">True</span><span class="token punctuation">,</span>
<span class="token punctuation">)</span>
<span class="token keyword">raise</span> RuntimeError<span class="token punctuation">(</span><span class="token string">"页面操作超时"</span><span class="token punctuation">)</span>

<span class="token keyword">finally</span><span class="token punctuation">:</span>
<span class="token keyword">await</span> context<span class="token punctuation">.</span>close<span class="token punctuation">(</span><span class="token punctuation">)</span>
<span class="token keyword">await</span> browser<span class="token punctuation">.</span>close<span class="token punctuation">(</span><span class="token punctuation">)</span>

<span class="token keyword">if</span> __name__ <span class="token operator">==</span> <span class="token string">"__main__"</span><span class="token punctuation">:</span>
data <span class="token operator">=</span> asyncio<span class="token punctuation">.</span>run<span class="token punctuation">(</span>collect_data<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>

<span class="token keyword">for</span> item <span class="token keyword">in</span> data<span class="token punctuation">:</span>
<span class="token keyword">print</span><span class="token punctuation">(</span>item<span class="token punctuation">)</span>
```


<hr />
<h2>八、Playwright 在生产环境中的架构设计</h2>
<p>直接写一个 Playwright 脚本并不难&#xff0c;难的是将它建设成稳定的生产系统。</p>
<p>推荐将系统拆分为以下几层&#xff1a;</p>
<div class="mermaid mermaid-newversion mermaid-flowchart"></div>
<h3>8.1 Browser 不要频繁启动</h3>
<p>错误方式&#xff1a;</p>


```text
每来一个任务
↓
启动浏览器
↓
执行任务
↓
关闭浏览器
```


<p>浏览器启动属于相对昂贵的操作。</p>
<p>更合理的方式是&#xff1a;</p>


```text
Worker 启动
↓
创建 Browser
↓
任务到达
↓
创建 BrowserContext
↓
执行任务
↓
关闭 BrowserContext
↓
复用 Browser
```


<p>但是 Browser 也不能永久运行&#xff0c;需要配置&#xff1a;</p>
<ul><li>最大任务数&#xff1b;</li><li>最大存活时间&#xff1b;</li><li>内存阈值&#xff1b;</li><li>页面崩溃检测&#xff1b;</li><li>Browser 重启机制。</li></ul>
<hr />
<h3>8.2 任务之间使用 BrowserContext 隔离</h3>
<p>不同用户和任务不应该共用同一个 Context&#xff0c;否则可能产生&#xff1a;</p>
<ul><li>Cookie 串用&#xff1b;</li><li>登录状态污染&#xff1b;</li><li>LocalStorage 污染&#xff1b;</li><li>页面残留&#xff1b;</li><li>权限相互影响。</li></ul>
<p>推荐&#xff1a;</p>


```text
Browser：进程级复用
BrowserContext：任务级或账号级隔离
Page：具体页面操作
```


<hr />
<h3>8.3 设置分层超时</h3>
<p>不要只有一个总超时。</p>
<p>建议分别设置&#xff1a;</p>


```text
任务总超时
├── 页面打开超时
├── 元素等待超时
├── 接口等待超时
├── 下载超时
└── 单步操作超时
```


<p>例如&#xff1a;</p>


```python
context<span class="token punctuation">.</span>set_default_timeout<span class="token punctuation">(</span><span class="token number">10_000</span><span class="token punctuation">)</span>
context<span class="token punctuation">.</span>set_default_navigation_timeout<span class="token punctuation">(</span><span class="token number">30_000</span><span class="token punctuation">)</span>
```


<p>任务调度层还应设置更高层的总超时&#xff0c;避免整个 Worker 被某个任务长期占用。</p>
<hr />
<h3>8.4 做好可观测性</h3>
<p>生产环境中的 Playwright 任务失败时&#xff0c;只保存一条异常日志通常不够。</p>
<p>至少应该保存&#xff1a;</p>
<ul><li>当前 URL&#xff1b;</li><li>任务 ID&#xff1b;</li><li>页面标题&#xff1b;</li><li>异常堆栈&#xff1b;</li><li>页面截图&#xff1b;</li><li>页面 HTML&#xff1b;</li><li>Console 日志&#xff1b;</li><li>失败请求&#xff1b;</li><li>Trace&#xff1b;</li><li>浏览器版本&#xff1b;</li><li>脚本版本&#xff1b;</li><li>账号或会话标识。</li></ul>
<p>这样才能回答&#xff1a;</p>
<blockquote>
<p>是页面改版了、接口失败了、登录失效了、元素被遮挡了&#xff0c;还是浏览器崩溃了&#xff1f;</p>
</blockquote>
<hr />
<h3>8.5 设计任务状态机</h3>
<p>一个 Playwright 任务通常不是简单的成功或失败。</p>
<p>可以设计为&#xff1a;</p>


```text
PENDING
↓
RUNNING
↓
LOGIN_REQUIRED
↓
COLLECTING
↓
PARSING
↓
SUCCESS
```


<p>异常状态包括&#xff1a;</p>


```text
TIMEOUT
AUTH_EXPIRED
PAGE_CHANGED
NETWORK_ERROR
BROWSER_CRASHED
CAPTCHA_REQUIRED
FAILED
```


<p>通过明确的状态机&#xff0c;可以决定哪些错误能够自动重试&#xff0c;哪些错误需要人工介入。</p>
<hr />
<h2>九、Playwright 的局限性</h2>
<p>Playwright 很强&#xff0c;但并不是万能的。</p>
<h3>9.1 资源消耗较大</h3>
<p>每个浏览器进程和页面都需要消耗&#xff1a;</p>
<ul><li>CPU&#xff1b;</li><li>内存&#xff1b;</li><li>文件描述符&#xff1b;</li><li>网络连接&#xff1b;</li><li>临时磁盘空间。</li></ul>
<p>如果同时启动几百个浏览器实例&#xff0c;服务器很容易出现资源耗尽。</p>
<p>因此&#xff0c;需要限制&#xff1a;</p>
<ul><li>Worker 数量&#xff1b;</li><li>Browser 数量&#xff1b;</li><li>Context 数量&#xff1b;</li><li>Page 数量&#xff1b;</li><li>单任务并发&#xff1b;</li><li>单浏览器存活时间。</li></ul>
<hr />
<h3>9.2 无法天然解决验证码</h3>
<p>Playwright 可以操作验证码页面&#xff0c;但它并不能天然解决&#xff1a;</p>
<ul><li>图片验证码&#xff1b;</li><li>滑块验证码&#xff1b;</li><li>短信验证码&#xff1b;</li><li>扫码登录&#xff1b;</li><li>行为验证&#xff1b;</li><li>人机验证。</li></ul>
<p>遇到这些场景时&#xff0c;通常需要&#xff1a;</p>
<ul><li>人工介入&#xff1b;</li><li>官方接口&#xff1b;</li><li>合法授权的账号体系&#xff1b;</li><li>验证码识别服务&#xff1b;</li><li>会话状态复用。</li></ul>
<hr />
<h3>9.3 不能天然绕过所有反爬系统</h3>
<p>使用真实浏览器不等于完全无法被识别。</p>
<p>网站可能综合判断&#xff1a;</p>
<ul><li>IP&#xff1b;</li><li>Cookie&#xff1b;</li><li>浏览器指纹&#xff1b;</li><li>请求频率&#xff1b;</li><li>鼠标轨迹&#xff1b;</li><li>页面停留时间&#xff1b;</li><li>账号行为&#xff1b;</li><li>TLS 指纹&#xff1b;</li><li>Header&#xff1b;</li><li>访问路径&#xff1b;</li><li>浏览器自动化特征。</li></ul>
<p>因此&#xff0c;Playwright 的定位应该是&#xff1a;</p>
<blockquote>
<p>浏览器自动化工具&#xff0c;而不是万能的风控绕过工具。</p>
</blockquote>
<p>使用时也应遵守目标网站的服务条款、数据授权范围和相关法律法规。</p>
<hr />
<h3>9.4 页面改版仍然会导致脚本失效</h3>
<p>如果页面按钮名称、DOM 结构或业务流程发生变化&#xff0c;自动化脚本仍然可能失败。</p>
<p>稳定性需要依赖&#xff1a;</p>
<ul><li>更可靠的 Locator&#xff1b;</li><li>页面版本监控&#xff1b;</li><li>多种提取策略&#xff1b;</li><li>失败截图&#xff1b;</li><li>Trace&#xff1b;</li><li>结构变化检测&#xff1b;</li><li>回退逻辑&#xff1b;</li><li>自动化回归测试。</li></ul>
<hr />
<h2>十、Playwright、HTTP 请求和人工浏览器如何选择</h2>

<table><thead><tr><th>场景</th><th>推荐方案</th></tr></thead><tbody><tr><td>静态 HTML 页面</td><td>requests、httpx</td></tr><tr><td>已知并且稳定的 JSON 接口</td><td>直接请求接口</td></tr><tr><td>JavaScript 动态渲染页面</td><td>Playwright</td></tr><tr><td>必须登录才能访问</td><td>Playwright 或复用登录 Cookie</td></tr><tr><td>iframe 页面</td><td>Playwright</td></tr><tr><td>需要点击、输入、滚动</td><td>Playwright</td></tr><tr><td>高并发批量接口采集</td><td>HTTP 客户端</td></tr><tr><td>自动化回归测试</td><td>Playwright Test</td></tr><tr><td>自动下载后台报表</td><td>Playwright</td></tr><tr><td>AI Agent 操作网页</td><td>Playwright、Playwright MCP</td></tr><tr><td>复杂验证码或人工审批</td><td>人机协同</td></tr></tbody></table><p>一个成熟系统通常不会只使用一种方式。</p>
<p>更合理的组合是&#xff1a;</p>


```text
Playwright
负责登录、交互、页面分析和接口发现

HTTP 客户端
负责稳定接口的高并发数据请求

HTML 解析器
负责结构化提取

任务队列
负责并发、重试和调度

对象存储
负责保存截图、HTML、下载文件和 Trace
```


<hr />
<h2>十一、最佳实践总结</h2>
<p>在实际项目中使用 Playwright&#xff0c;可以重点遵循以下原则。</p>
<p>第一&#xff0c;优先使用 <code>Locator</code>&#xff0c;不要长期持有旧的 DOM ElementHandle。</p>
<p>第二&#xff0c;优先使用 Role、Label、Text 和 Test ID&#xff0c;尽量避免脆弱的绝对 XPath。</p>
<p>第三&#xff0c;减少固定 <code>sleep</code>&#xff0c;依赖 Locator 自动等待、接口事件和明确的页面状态。</p>
<p>第四&#xff0c;一个 Browser 可以复用&#xff0c;但不同任务尽量使用独立 BrowserContext。</p>
<p>第五&#xff0c;动态页面优先分析网络接口&#xff0c;不要所有数据都从页面文本中硬解析。</p>
<p>第六&#xff0c;出现 iframe 时&#xff0c;应明确进入目标 iframe&#xff0c;而不是直接扫描整个外层页面。</p>
<p>第七&#xff0c;等待下载、弹窗和接口时&#xff0c;要先注册事件&#xff0c;再触发操作。</p>
<p>第八&#xff0c;生产环境必须保存截图、HTML、Console、网络错误和 Trace。</p>
<p>第九&#xff0c;对浏览器、Context、Page 和任务设置资源上限与超时。</p>
<p>第十&#xff0c;不要把 Playwright 当作万能反爬工具&#xff0c;应在授权和合规范围内使用。</p>
<hr />
<h2>十二、结语</h2>
<p>Playwright 的价值并不只是“自动点击网页”。</p>
<p>它真正解决的是&#xff1a;</p>
<blockquote>
<p>如何让程序稳定地进入一个真实浏览器环境&#xff0c;并观察、操作和验证现代 Web 应用。</p>
</blockquote>
<p>从能力上看&#xff0c;Playwright 横跨了多个领域&#xff1a;</p>


```text
自动化测试
动态网页采集
后台流程自动化
文件上传下载
接口调试与 Mock
页面截图与录制
浏览器 Agent
AI 自动化执行
```


<p>它的核心优势来自几个关键设计&#xff1a;</p>
<ul><li>统一控制 Chromium、Firefox 和 WebKit&#xff1b;</li><li>使用 BrowserContext 实现会话隔离&#xff1b;</li><li>使用 Locator 应对动态 DOM&#xff1b;</li><li>通过自动等待减少脚本偶发失败&#xff1b;</li><li>通过网络监听获取页面背后的真实数据&#xff1b;</li><li>通过 Trace 提升自动化任务的可调试性&#xff1b;</li><li>通过 MCP 等方式成为 AI Agent 的浏览器执行层。</li></ul>
<p>如果只是采集一个简单的静态页面&#xff0c;使用 Playwright 可能显得过重。</p>
<p>但当你面对登录系统、动态渲染、iframe、复杂交互、文件下载、浏览器测试或 AI Agent 操作网页时&#xff0c;Playwright 往往是目前最值得掌握的浏览器自动化工具之一。</p>
