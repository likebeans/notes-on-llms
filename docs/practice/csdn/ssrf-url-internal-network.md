---
title: "一个 URL 参数，为什么可能打穿你的内网？"
description: "CSDN 原文全文镜像：SSRF，全称是 Server-Side Request Forgery，中文通常叫“服务端请求伪造”。它的核心问题是：攻击者无法直接访问某个目标资源，但可以诱导服务端替他访问。用户输入 URL↓服务端请求该 URL↓服务端把内容返回给……"
pageType: article
module: site
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "practice"
  - "架构"
  - "语言模型"
  - "人工智能"
  - "ssrf"
level: intermediate
prerequisites:
  - "/practice/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-06-10，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-06-10。为适配本站结构，补充了站内导读、元数据与来源说明，并清理代码高亮标记；原文观点与主体内容保留。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/161862209](https://blog.csdn.net/m0_63309778/article/details/161862209)
- 站内分区：工程实践 / SSRF 安全
:::

::: tip 站内导读：从 URL 输入追踪到真实网络请求
阅读重点是用户可控 URL 如何经过解析、DNS、重定向与连接，最终到达目标。对带抓取工具的 RAG/Agent，输入校验与执行时的出站边界需要一起考虑。

练习限于受控环境：覆盖重定向、不同地址表示和解析结果变化，检查允许列表和实际连接目标是否一致；不要只用字符串包含判断可信域名。

相关主线：[Agent 安全](/llms/agent/safety) · [MCP 高级功能](/llms/mcp/advanced)。本导读不代表对原文全部代码与结论的重新核验。
:::

<p><img src="https://i-blog.csdnimg.cn/direct/0eaf1dc8f4ac4e17a68666794c4d1ab4.png" alt="在这里插入图片描述" /></p>
<h2>一文讲清 SSRF&#xff1a;从漏洞原理到工程化防护方案</h2>
<p>在 Web 安全中&#xff0c;SSRF 是一个非常典型、也非常容易被低估的漏洞。</p>
<p>很多开发者第一次接触 SSRF 时&#xff0c;会觉得它只是“后端请求了一个用户传入的 URL”。但在真实系统中&#xff0c;SSRF 的影响远不止如此。它可能被用来访问内网服务、探测端口、读取云厂商元数据、绕过防火墙&#xff0c;甚至进一步扩大为权限泄露和横向移动。</p>
<p>本文会从工程实践角度&#xff0c;系统介绍 SSRF 的基本原理、常见攻击面、防护难点&#xff0c;以及一套相对完整的服务端防护方案。</p>
<hr />
<h3>1. 什么是 SSRF&#xff1f;</h3>
<p>SSRF&#xff0c;全称是 Server-Side Request Forgery&#xff0c;中文通常叫“服务端请求伪造”。</p>
<p>它的核心问题是&#xff1a;</p>
<blockquote>
<p>攻击者无法直接访问某个目标资源&#xff0c;但可以诱导服务端替他访问。</p>
</blockquote>
<p>举个简单例子&#xff0c;假设系统提供了一个“根据 URL 抓取网页内容”的功能&#xff1a;</p>


```text
用户输入 URL
↓
服务端请求该 URL
↓
服务端把内容返回给用户
```


<p>正常情况下&#xff0c;用户可能输入&#xff1a;</p>


```text
https://example.com/article/123
```


<p>但攻击者可能输入&#xff1a;</p>


```text
http://127.0.0.1:8080/admin
http://localhost:6379
http://169.254.169.254/latest/meta-data/
```


<p>如果后端没有任何限制&#xff0c;就会变成&#xff1a;</p>


```text
攻击者
↓
业务服务器
↓
内网服务 / 本机端口 / 云元数据服务
```


<p>这就是 SSRF 的本质&#xff1a;<strong>攻击者控制了服务端请求的目标地址。</strong></p>
<hr />
<h3>2. SSRF 为什么危险&#xff1f;</h3>
<p>SSRF 危险的原因在于&#xff0c;服务端通常比外部用户拥有更高的网络权限。</p>
<p>外部攻击者可能无法访问内网服务&#xff0c;但业务服务器可以。攻击者只要能让服务器发起请求&#xff0c;就可能借助服务器的身份进入本来无法触达的网络区域。</p>
<p>常见风险包括&#xff1a;</p>

<table><thead><tr><th>风险类型</th><th>说明</th></tr></thead><tbody><tr><td>访问本机服务</td><td>如 <code>127.0.0.1</code>、<code>localhost</code> 上的管理接口</td></tr><tr><td>访问内网服务</td><td>如数据库、Redis、Nacos、Elasticsearch、Prometheus 等</td></tr><tr><td>探测内网端口</td><td>通过响应差异判断内网服务是否存在</td></tr><tr><td>访问云元数据服务</td><td>获取实例信息、临时凭证、角色权限等敏感数据</td></tr><tr><td>绕过网络边界</td><td>借助服务器突破防火墙、VPN、ACL 等限制</td></tr><tr><td>敏感信息泄露</td><td>返回配置、环境变量、Token、密钥等</td></tr><tr><td>扩大攻击面</td><td>作为进一步攻击内网系统的跳板</td></tr></tbody></table><p>SSRF 的麻烦之处在于&#xff1a;它不是单纯的输入校验问题&#xff0c;而是涉及 URL 解析、DNS、HTTP 重定向、网络隔离、云环境权限等多个层面。</p>
<hr />
<h3>3. 哪些功能容易产生 SSRF&#xff1f;</h3>
<p>只要系统存在“服务端根据用户可控输入去请求外部资源”的逻辑&#xff0c;就需要考虑 SSRF。</p>
<p>常见场景包括&#xff1a;</p>
<h4>3.1 URL 内容抓取</h4>
<p>例如&#xff1a;</p>


```text
输入文章链接，系统自动抓取正文
输入图片链接，系统自动下载图片
输入网页地址，系统自动生成预览
输入远程文件地址，系统自动导入文件
```


<p>这类功能是 SSRF 的高发区域。</p>
<hr />
<h4>3.2 Webhook / Callback</h4>
<p>很多系统允许用户配置回调地址&#xff1a;</p>


```text
任务完成后，请求用户配置的 callbackUrl
```


<p>如果没有限制&#xff0c;攻击者可以把 callbackUrl 配成内网地址&#xff0c;让服务器主动访问内部系统。</p>
<hr />
<h4>3.3 图片代理和文件代理</h4>
<p>有些系统为了避免前端跨域&#xff0c;会提供图片代理功能&#xff1a;</p>


```text
/image-proxy?url=https://example.com/a.png
```


<p>后端读取 <code>url</code> 参数&#xff0c;再把图片内容返回给前端。</p>
<p>这个功能如果不做校验&#xff0c;很容易被改成&#xff1a;</p>


```text
/image-proxy?url=http://127.0.0.1:8080/admin
```


<hr />
<h4>3.4 HTML 转 PDF / Markdown 渲染 / 富文本解析</h4>
<p>这类场景经常被忽略。</p>
<p>例如系统支持 HTML 转 PDF&#xff0c;HTML 中可能包含&#xff1a;</p>


```html
<img src="http://127.0.0.1:8080/internal">
```


<p>渲染器在生成 PDF 时&#xff0c;会自动去加载图片、CSS、字体等远程资源。即使业务代码没有显式请求 URL&#xff0c;底层渲染工具也可能触发 SSRF。</p>
<hr />
<h4>3.5 文件上传后的自动解析</h4>
<p>比如上传一个文档&#xff0c;系统自动解析其中的远程图片、外部链接、XML 实体、模板引用等&#xff0c;也可能触发服务端请求。</p>
<hr />
<h3>4. SSRF 防护为什么不能只靠黑名单&#xff1f;</h3>
<p>很多人第一反应是&#xff1a;</p>


```text
禁止 127.0.0.1
禁止 localhost
禁止 169.254.169.254
```


<p>这当然有用&#xff0c;但远远不够。</p>
<p>因为 SSRF 可以通过很多方式绕过简单黑名单&#xff0c;例如&#xff1a;</p>


```text
使用域名解析到内网 IP
使用 IPv6 地址
使用特殊 IP 写法
使用短地址、整数地址、八进制地址
使用 DNS Rebinding
使用 302 重定向跳转到内网
使用 CNAME 指向内网地址
使用大小写、编码、URL 解析差异绕过
```


<p>例如你禁止了&#xff1a;</p>


```text
127.0.0.1
```


<p>但攻击者可能使用&#xff1a;</p>


```text
localhost
127.1
0.0.0.0
[::1]
某个解析到 127.0.0.1 的域名
```


<p>所以 SSRF 防护的核心不是“写几个黑名单字符串”&#xff0c;而是构建一整套请求安全边界。</p>
<hr />
<h3>5. SSRF 防护的总体思路</h3>
<p>更合理的 SSRF 防护应该是多层防御&#xff1a;</p>


```text
用户输入
↓
URL 解析与规范化
↓
协议校验
↓
域名 / IP 校验
↓
端口校验
↓
DNS 解析结果校验
↓
重定向校验
↓
统一安全 HTTP 客户端
↓
网络层出站控制
↓
日志审计与告警
```


<p>核心原则有三点&#xff1a;</p>


```text
1. 不要信任用户传入的 URL
2. 不要让业务服务拥有任意访问网络的能力
3. 不要只依赖代码校验，必须配合网络层隔离
```


<hr />
<h3>6. 第一层防护&#xff1a;统一收口外部请求</h3>
<p>最危险的写法是业务代码中到处直接请求用户传入的 URL。</p>
<p>错误示例&#xff1a;</p>


```java
String url = request.getParameter("url");
String result = restTemplate.getForObject(url, String.class);
```


<p>这样的问题是&#xff1a;</p>


```text
任何业务代码都可能绕过安全检查
安全规则分散在多个地方
后续难以审计
新增功能容易再次引入 SSRF
```


<p>更推荐的方式是封装统一组件&#xff1a;</p>


```text
SafeUrlValidator
SafeHttpClient
ExternalFetchService
WebhookDispatcher
ImageProxyService
```


<p>所有服务端请求外部 URL 的逻辑&#xff0c;都必须走统一的安全组件。</p>
<p>推荐结构&#xff1a;</p>


```text
业务代码
↓
SafeUrlValidator 校验 URL
↓
SafeHttpClient 发起请求
↓
Egress Proxy / 网络出口控制
↓
目标站点
```


<p>这样可以把 SSRF 防护逻辑集中到一个地方维护&#xff0c;而不是散落在各个业务模块中。</p>
<hr />
<h3>7. 第二层防护&#xff1a;优先使用白名单</h3>
<p>SSRF 防护中&#xff0c;白名单通常比黑名单更可靠。</p>
<h4>7.1 强白名单</h4>
<p>如果业务场景比较明确&#xff0c;最推荐的方式是只允许访问指定域名。</p>
<p>例如&#xff1a;</p>


```yaml
ssrf:
allowedDomains:
- example.com
- api.example.com
- static.example-cdn.com
```


<p>校验逻辑&#xff1a;</p>


```text
用户输入 URL
↓
解析 host
↓
判断 host 是否在 allowedDomains 中
↓
不在白名单则拒绝
```


<p>强白名单适合这些场景&#xff1a;</p>


```text
固定第三方接口调用
固定资源下载源
固定合作方域名
内部配置好的可信站点
```


<p>这种方式安全性最高&#xff0c;维护成本也相对可控。</p>
<hr />
<h4>7.2 弱白名单 &#43; 内网地址拦截</h4>
<p>如果业务必须支持用户输入任意公网 URL&#xff0c;就不能只依赖域名白名单。这时至少要做&#xff1a;</p>


```text
只允许 http / https
禁止访问内网 IP
禁止访问本机地址
禁止访问云元数据地址
禁止访问特殊网段
限制端口
限制重定向
限制响应大小
限制超时时间
```


<p>但需要注意&#xff1a;支持“任意公网 URL”本身就意味着更大的风险&#xff0c;因此更应该配合网络层隔离。</p>
<hr />
<h3>8. 第三层防护&#xff1a;限制 URL 协议</h3>
<p>服务端请求外部资源时&#xff0c;通常只应该允许&#xff1a;</p>


```text
http
https
```


<p>应该禁止&#xff1a;</p>


```text
file://
ftp://
gopher://
dict://
ldap://
jar://
netdoc://
data:
mailto:
```


<p>尤其是 <code>file://</code> 这类协议&#xff0c;可能导致读取本地文件。</p>
<p>示例校验逻辑&#xff1a;</p>


```java
URI uri = URI.create(rawUrl);
String scheme = uri.getScheme();

if (!"http".equalsIgnoreCase(scheme) && !"https".equalsIgnoreCase(scheme)) {<!-- -->
throw new IllegalArgumentException("Unsupported URL scheme");
}
```


<p>不要用简单的字符串判断&#xff0c;例如&#xff1a;</p>


```java
if (url.startsWith("http")) {<!-- -->
// allow
}
```


<p>因为 URL 解析可能存在各种边界情况&#xff0c;应该使用标准 URL / URI 解析库进行处理。</p>
<hr />
<h3>9. 第四层防护&#xff1a;限制端口</h3>
<p>如果没有特殊业务需求&#xff0c;建议只允许&#xff1a;</p>


```text
80
443
```


<p>危险端口包括&#xff1a;</p>


```text
22      SSH
2375    Docker API
3306    MySQL
5432    PostgreSQL
6379    Redis
9200    Elasticsearch
11211   Memcached
8848    Nacos
9000    对象存储服务
9001    管理控制台
10250   Kubelet
```


<p>推荐配置&#xff1a;</p>


```yaml
ssrf:
allowedPorts:
- 80
- 443
```


<p>处理默认端口时也要注意&#xff1a;</p>


```text
http 默认端口是 80
https 默认端口是 443
```


<p>例如&#xff1a;</p>


```java
int port = uri.getPort();

if (port == -1) {<!-- -->
port = "https".equalsIgnoreCase(uri.getScheme()) ? 443 : 80;
}

if (!allowedPorts.contains(port)) {<!-- -->
throw new IllegalArgumentException("Port not allowed");
}
```


<hr />
<h3>10. 第五层防护&#xff1a;禁止访问内网和特殊地址</h3>
<p>服务端应该禁止访问以下类型的地址&#xff1a;</p>


```text
127.0.0.0/8          Loopback 地址
10.0.0.0/8           私有地址
172.16.0.0/12        私有地址
192.168.0.0/16       私有地址
169.254.0.0/16       Link-local 地址
0.0.0.0/8            当前网络地址
100.64.0.0/10        Carrier-grade NAT
224.0.0.0/4          组播地址
240.0.0.0/4          保留地址
::1/128              IPv6 Loopback
fc00::/7             IPv6 Unique Local Address
fe80::/10            IPv6 Link-local
```


<p>尤其要注意&#xff1a;</p>


```text
127.0.0.1
localhost
0.0.0.0
169.254.169.254
10.x.x.x
172.16.x.x - 172.31.x.x
192.168.x.x
```


<p><code>169.254.169.254</code> 是云环境中特别敏感的地址&#xff0c;很多云厂商的实例元数据服务都和这个地址有关。一旦被 SSRF 访问&#xff0c;可能造成云资源凭证或实例信息泄露。</p>
<hr />
<h3>11. 第六层防护&#xff1a;DNS 解析结果校验</h3>
<p>很多系统只检查 URL 字符串&#xff0c;这是不够的。</p>
<p>攻击者可以使用一个看起来正常的域名&#xff0c;但该域名解析到内网地址。</p>
<p>例如&#xff1a;</p>


```text
http://safe-looking-domain.com
```


<p>解析结果可能是&#xff1a;</p>


```text
127.0.0.1
10.0.0.10
192.168.1.100
```


<p>因此&#xff0c;正确流程应该是&#xff1a;</p>


```text
解析域名
↓
获取所有 A / AAAA 记录
↓
逐个判断是否属于内网或特殊地址
↓
只要有一个非法地址，就拒绝
```


<p>Java 示例&#xff1a;</p>


```java
InetAddress[] addresses = InetAddress.getAllByName(host);

for (InetAddress address : addresses) {<!-- -->
if (isBlockedAddress(address)) {<!-- -->
throw new IllegalArgumentException("Blocked target address");
}
}
```


<p>注意&#xff1a;仅使用 <code>InetAddress.isSiteLocalAddress()</code> 还不够&#xff0c;因为它不能覆盖所有危险地址。真实生产环境中&#xff0c;建议使用 CIDR 匹配库对完整网段进行判断。</p>
<hr />
<h3>12. 第七层防护&#xff1a;防止 DNS Rebinding</h3>
<p>DNS Rebinding 是 SSRF 防护中的一个难点。</p>
<p>它的大致思路是&#xff1a;</p>


```text
第一次解析域名时，返回公网 IP
安全校验通过

实际请求或后续请求时，域名变成内网 IP
绕过校验
```


<p>这类问题本质上属于 TOCTOU 问题&#xff1a;</p>


```text
Time Of Check
检查时是安全的

Time Of Use
使用时已经变了
```


<p>防护建议&#xff1a;</p>


```text
1. 请求前解析域名，并校验所有解析结果
2. 不要只在任务创建时校验一次，真正请求前也要校验
3. 每次重定向后都重新解析和校验
4. 使用统一的出站代理，由代理完成 DNS 解析和网络限制
5. 在网络层禁止访问内网地址，作为最终兜底
```


<p>单纯应用层校验很难彻底解决 DNS Rebinding&#xff0c;因此网络层出站控制非常重要。</p>
<hr />
<h3>13. 第八层防护&#xff1a;禁用自动重定向</h3>
<p>重定向是 SSRF 中非常常见的绕过方式。</p>
<p>例如用户输入&#xff1a;</p>


```text
https://normal-domain.com/resource
```


<p>这个地址本身可能是安全的&#xff0c;但它返回&#xff1a;</p>


```http
HTTP/1.1 302 Found
Location: http://127.0.0.1:8080/admin
```


<p>如果 HTTP 客户端自动跟随重定向&#xff0c;就可能绕过第一次 URL 校验。</p>
<p>正确做法&#xff1a;</p>


```text
默认禁用自动重定向
如确实需要重定向，则每一次 Location 都重新做完整 SSRF 校验
限制最大重定向次数
```


<p>例如&#xff1a;</p>


```text
最多允许 3 次跳转
每次跳转都校验 scheme、host、port、DNS、IP
跳转到非法地址立即终止
```


<hr />
<h3>14. 第九层防护&#xff1a;限制请求和响应</h3>
<p>SSRF 防护不仅要校验目标地址&#xff0c;还要限制请求行为本身。</p>
<p>建议设置&#xff1a;</p>


```text
连接超时时间
读取超时时间
最大响应体大小
允许的 Content-Type
最大下载文件大小
最大重定向次数
禁止携带敏感 Header
禁止携带服务端 Cookie
禁止返回完整错误堆栈
```


<p>例如&#xff1a;</p>


```yaml
httpClient:
connectTimeoutMs: 3000
readTimeoutMs: 10000
maxResponseSizeMb: 10
maxRedirects: 0
```


<p>尤其要注意&#xff1a;不要把服务端请求到的原始响应直接返回给用户。</p>
<p>错误做法&#xff1a;</p>


```text
请求目标 URL
↓
把完整响应头、响应体、错误信息全部返回给前端
```


<p>更安全的方式&#xff1a;</p>


```text
只返回业务需要的字段
过滤响应头
限制响应体大小
错误信息脱敏
不暴露内部请求细节
```


<hr />
<h3>15. 第十层防护&#xff1a;云环境中的元数据服务保护</h3>
<p>云环境中的 SSRF 风险尤其高。</p>
<p>很多云服务器都提供元数据服务&#xff0c;用于让实例获取自身信息、临时凭证、角色权限等。攻击者如果通过 SSRF 访问到元数据服务&#xff0c;可能进一步获取云资源访问权限。</p>
<p>防护建议&#xff1a;</p>


```text
启用更安全的元数据访问机制
禁用旧版不安全访问方式
限制应用访问元数据地址
服务账号使用最小权限
不要给普通业务服务绑定高权限角色
容器网络层禁止访问 metadata endpoint
```


<p>以 AWS 为例&#xff0c;应优先使用 IMDSv2&#xff0c;并限制不必要的元数据访问能力。</p>
<p>同时要强调一点&#xff1a;不要只依赖云厂商元数据服务自身的保护机制。应用层、网络层、权限层都应该同时做约束。</p>
<hr />
<h3>16. 第十一层防护&#xff1a;网络层出站控制</h3>
<p>SSRF 最可靠的防护不是“每个开发都把 URL 校验写对”&#xff0c;而是&#xff1a;</p>
<blockquote>
<p>即使代码层有遗漏&#xff0c;服务本身也访问不到不该访问的地址。</p>
</blockquote>
<p>这就需要网络层出站控制。</p>
<p>推荐做法&#xff1a;</p>


```text
业务服务不能任意访问内网
外部请求必须走统一 Egress Proxy
防火墙 / 安全组限制出站流量
Kubernetes 使用 NetworkPolicy 控制 Pod 出站访问
禁止访问云元数据地址
禁止访问数据库、缓存、配置中心、容器管理接口
```


<p>理想结构&#xff1a;</p>


```text
业务服务
↓
统一出站代理
↓
公网目标地址
```


<p>统一出站代理负责&#xff1a;</p>


```text
域名白名单
IP 黑名单
CIDR 拦截
DNS 解析
重定向校验
访问日志
限流
告警
```


<p>这样可以把安全策略集中在基础设施层&#xff0c;而不是依赖每一个业务服务自己实现。</p>
<hr />
<h3>17. Java 中的 SafeUrlValidator 示例</h3>
<p>下面是一个简化版示例&#xff0c;用于说明基本思路。</p>
<p>注意&#xff1a;真实生产环境还需要补充完整 CIDR 判断、IPv6 处理、重定向校验、DNS Rebinding 防护和网络层隔离。</p>


```java
import java.net.*;
import java.util.Set;

public class SafeUrlValidator {<!-- -->

private static final Set<String> ALLOWED_SCHEMES = Set.of("http", "https");
private static final Set<Integer> ALLOWED_PORTS = Set.of(80, 443);

public void validate(String rawUrl) {<!-- -->
URI uri;

try {<!-- -->
uri = URI.create(rawUrl).normalize();
} catch (Exception e) {<!-- -->
throw new IllegalArgumentException("Invalid URL");
}

String scheme = uri.getScheme();
if (scheme == null || !ALLOWED_SCHEMES.contains(scheme.toLowerCase())) {<!-- -->
throw new IllegalArgumentException("Unsupported URL scheme");
}

String host = uri.getHost();
if (host == null || host.isBlank()) {<!-- -->
throw new IllegalArgumentException("Invalid host");
}

int port = uri.getPort();
if (port == -1) {<!-- -->
port = "https".equalsIgnoreCase(scheme) ? 443 : 80;
}

if (!ALLOWED_PORTS.contains(port)) {<!-- -->
throw new IllegalArgumentException("Port not allowed");
}

InetAddress[] addresses;
try {<!-- -->
addresses = InetAddress.getAllByName(host);
} catch (UnknownHostException e) {<!-- -->
throw new IllegalArgumentException("Cannot resolve host");
}

for (InetAddress address : addresses) {<!-- -->
if (isBlockedAddress(address)) {<!-- -->
throw new IllegalArgumentException("Blocked target address");
}
}
}

private boolean isBlockedAddress(InetAddress address) {<!-- -->
return address.isAnyLocalAddress()
|| address.isLoopbackAddress()
|| address.isLinkLocalAddress()
|| address.isMulticastAddress()
|| address.isSiteLocalAddress();
}
}
```


<p>这个示例只能作为基础版本&#xff0c;不能直接视为完整生产方案。</p>
<p>更完整的生产实现应该包括&#xff1a;</p>


```text
完整 CIDR 网段判断
IPv4 / IPv6 全覆盖
IDN / Punycode 规范化
禁止 userinfo
禁止异常端口
DNS 解析结果缓存策略
重定向逐跳校验
请求前和请求时一致性控制
统一出站代理
网络层兜底拦截
```


<hr />
<h3>18. HTTP 客户端也要统一封装</h3>
<p>除了 URL 校验器&#xff0c;还应该封装统一的 HTTP 请求客户端。</p>
<p>示例逻辑&#xff1a;</p>


```text
SafeHttpClient.fetch(url)
↓
SafeUrlValidator.validate(url)
↓
禁用自动重定向
↓
设置超时
↓
设置最大响应体
↓
过滤请求头
↓
发起请求
↓
校验响应
↓
返回安全结果
```


<p>不要让业务代码直接使用&#xff1a;</p>


```text
RestTemplate
WebClient
OkHttpClient
HttpClient
URLConnection
```


<p>而应该统一走&#xff1a;</p>


```text
SafeHttpClient
```


<p>这样才能保证新增功能不会绕过 SSRF 防护。</p>
<hr />
<h3>19. 常见错误做法总结</h3>
<h4>错误 1&#xff1a;只用字符串 contains 判断</h4>


```java
if (url.contains("127.0.0.1") || url.contains("localhost")) {<!-- -->
throw new RuntimeException("blocked");
}
```


<p>这种做法很容易被绕过。</p>
<hr />
<h4>错误 2&#xff1a;只校验用户输入的原始 URL</h4>
<p>原始 URL 安全&#xff0c;不代表最终请求地址安全。</p>
<p>还需要考虑&#xff1a;</p>


```text
DNS 解析
CNAME
重定向
URL 编码
IPv6
特殊 IP 表示法
```


<hr />
<h4>错误 3&#xff1a;自动跟随重定向</h4>
<p>自动重定向可能把请求带到内网地址。</p>
<p>建议默认禁用&#xff0c;如需启用&#xff0c;必须逐跳校验。</p>
<hr />
<h4>错误 4&#xff1a;允许任意端口</h4>
<p>如果允许访问任意端口&#xff0c;攻击者可能探测或访问内部管理端口。</p>
<p>建议默认只允许 80 和 443。</p>
<hr />
<h4>错误 5&#xff1a;把响应原样返回给用户</h4>
<p>如果目标地址是内部接口&#xff0c;原样返回响应可能泄露配置、密钥、环境变量、服务状态等敏感信息。</p>
<hr />
<h4>错误 6&#xff1a;只做应用层防护&#xff0c;没有网络层兜底</h4>
<p>应用层校验总有遗漏风险。</p>
<p>真正可靠的方案一定要配合&#xff1a;</p>


```text
防火墙
安全组
NetworkPolicy
Egress Proxy
云权限最小化
```


<hr />
<h3>20. 一套可落地的 SSRF 防护清单</h3>
<h4>输入校验</h4>


```text
限制 URL 长度
使用标准 URL parser
只允许 http / https
禁止 userinfo
禁止异常 schema
规范化 host
```


<h4>目标校验</h4>


```text
优先使用域名白名单
限制端口
解析 DNS
校验所有 A / AAAA 记录
禁止内网地址
禁止 link-local 地址
禁止 metadata 地址
禁止 IPv6 内网地址
```


<h4>请求控制</h4>


```text
统一 SafeHttpClient
禁用自动重定向
逐跳校验 Location
设置连接超时
设置读取超时
限制响应体大小
限制 Content-Type
禁止携带敏感 Header
```


<h4>响应处理</h4>


```text
不要原样返回响应
过滤响应头
错误信息脱敏
限制返回字段
记录请求摘要
```


<h4>网络隔离</h4>


```text
业务服务出站最小化
统一 Egress Proxy
禁止访问内网网段
禁止访问云元数据服务
禁止访问数据库、缓存、配置中心
容器和 Pod 使用最小网络权限
```


<h4>云环境</h4>


```text
启用安全版本的元数据服务
禁用不必要的元数据访问
服务账号最小权限
不要给普通应用绑定高权限角色
在网络层阻断 metadata endpoint
```


<h4>日志审计</h4>


```text
记录请求目标 host
记录解析后的 IP
记录端口
记录拦截原因
记录重定向链路
对访问内网地址的尝试做告警
```


<hr />
<h3>21. 推荐的配置示例</h3>


```yaml
ssrf:
allowedSchemes:
- http
- https

allowedPorts:
- 80
- 443

blockedCidrs:
- 127.0.0.0/8
- 10.0.0.0/8
- 172.16.0.0/12
- 192.168.0.0/16
- 169.254.0.0/16
- 0.0.0.0/8
- 100.64.0.0/10
- 224.0.0.0/4
- 240.0.0.0/4
- ::1/128
- fc00::/7
- fe80::/10

redirect:
enabled: false
maxHops: 0

response:
maxBodySizeMb: 10
connectTimeoutMs: 3000
readTimeoutMs: 10000
```


<p>如果业务必须支持重定向&#xff0c;可以改成&#xff1a;</p>


```yaml
redirect:
enabled: true
maxHops: 3
validateEachHop: true
```


<hr />
<h3>22. SSRF 防护的核心结论</h3>
<p>SSRF 看起来只是“服务端请求了一个 URL”&#xff0c;但本质上是服务端网络边界被用户输入控制的问题。</p>
<p>防护 SSRF 不能只靠一个简单的黑名单判断&#xff0c;而应该建立完整的安全链路&#xff1a;</p>


```text
统一请求入口
+
URL 解析与规范化
+
协议、端口、域名校验
+
DNS 解析结果校验
+
重定向逐跳校验
+
响应限制
+
网络层出站控制
+
云环境权限最小化
```


<p>一句话总结&#xff1a;</p>
<blockquote>
<p>SSRF 防护的关键&#xff0c;不是判断 URL 里有没有 localhost&#xff0c;而是不要让服务端拥有不受约束的网络访问能力。</p>
</blockquote>
<p>在工程实践中&#xff0c;最推荐的方案是&#xff1a;</p>


```text
白名单优先
统一 SafeHttpClient
禁用默认重定向
DNS + IP 双重校验
Egress Proxy 统一出站
网络层禁止访问内网和云元数据
```


<p>只有应用层和基础设施层一起做防护&#xff0c;才能真正降低 SSRF 风险。</p>
<blockquote>
<p>写这篇文章的目的&#xff0c;不是为了证明某个技术有多先进&#xff0c;而是希望把一个真实问题拆开&#xff0c;讲清楚它背后的设计逻辑、工程取舍和落地路径。</p>
</blockquote>
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
<img src="https://i-blog.csdnimg.cn/direct/fc8b100effab4c759d73d26a46885124.jpeg" alt="在这里插入图片描述" /></p>
<blockquote>
<p>备注&#xff1a;如果二维码过期&#xff0c;可以私信我拉你进群。</p>
</blockquote>
