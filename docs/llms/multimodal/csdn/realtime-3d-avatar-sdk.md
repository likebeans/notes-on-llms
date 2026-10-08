---
title: "实时 3D 数字人落地全记录：SDK 深拆、接大模型、7 个坑、开源工程一键部署"
description: "CSDN 原文全文镜像：本文介绍了如何利用魔珐星云具身驱动SDK实现\"会说话的3D数字人\"项目。核心采用\"参数流+AI端渲\"架构，通过下发音频、表情、动作等多路参数实现本地实时渲染，具有低延迟、轻量化的优势。文章详细解析了speak接口的流式调用、SSML动作……"
pageType: article
module: multimodal
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "multimodal"
  - "3d"
  - "开源"
  - "大模型"
  - "agent"
  - "数字人"
level: intermediate
prerequisites:
  - "/llms/multimodal/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-17，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-17。为适配本站结构，补充了站内元数据、来源说明与阅读导引，并修复代码高亮残留；原文主体与观点保留。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163828282](https://blog.csdn.net/m0_63309778/article/details/163828282)
- 站内分区：Multimodal / 实时 3D 数字人
:::

::: tip 站内导读：重点看流式状态与故障恢复
本文适合与 [多模态 Agent](/llms/multimodal/rag-agent)和 [部署评测](/llms/multimodal/deployment)连读。把 LLM 文本分块、TTS/动作流与浏览器播放视为三个独立状态机：记录会话 ID、分块序号、结束标记、取消与重连结果，测试中途打断是否继续播旧内容、重试是否重复播报。延迟需从用户输入到首音频端到端测量；文中的接口、版本与营销数字是原文记录。凭证示例只说明调用形态，上线需按服务商支持的鉴权流程避免公开长期 Secret。
:::


<p>我用魔珐星云的具身驱动 SDK 完整落地了一个「会说话的 3D 数字人」项目&#xff0c;从接入、深拆技术、接大模型、踩坑&#xff0c;到做成开源项目 &#43; CI/CD 一键部署&#xff0c;全程记录在这里。</p> 
<p>先看效果&#xff08;这是浏览器里<strong>实时渲染</strong>的数字人&#xff0c;不是录好的视频&#xff09;&#xff1a;</p> 
<p><img src="https://i-blog.csdnimg.cn/direct/08fa667c7d6a41cd9495357356c728b3.png" alt="在这里插入图片描述" /></p> 
<p>你可以直接打开在线版体验&#xff08;填个 App ID/Secret 就能玩&#xff09;&#xff1a;</p> 
<blockquote> 
<p>&#x1f517; https://likebeans.github.io/xingyun3D/</p> 
</blockquote> 
<p>注册魔珐星云时用邀请码 <strong>XDZARL7NEP</strong>&#xff0c;送 1000 积分。</p> 
<hr /> 
<h3>一、核心架构&#xff1a;参数流 &#43; AI 端渲</h3> 
<p>先理解一个关键点&#xff1a;这套 SDK <strong>不下发视频&#xff0c;下发「参数」</strong>。</p> 
<p>一次 <code>speak(text)</code> 之后&#xff0c;云端把文本处理成多路<strong>参数流</strong>下发&#xff1a;</p> 
<ul><li><strong>音频流</strong>&#xff1a;合成的语音&#xff1b;</li><li><strong>表情/口型参数</strong>&#xff1a;面部动作、口型对齐&#xff1b;</li><li><strong>动作参数</strong>&#xff1a;身体动作、KA 动作&#xff08;手势、跳舞、欢迎等&#xff09;&#xff1b;</li><li><strong>事件流</strong>&#xff1a;字幕、图片、视频等 Widget 事件。</li></ul> 
<p>浏览器端拿到参数后<strong>在本地实时渲染</strong>出画面。这就是官方说的「参数流 &#43; AI 端渲」。</p> 
<p>它的好处很直接&#xff1a;</p> 
<ol><li><strong>不用传输视频</strong> → 带宽小、能做大规模并发&#xff08;官方标称千万级&#xff09;&#xff1b;</li><li><strong>端侧渲染</strong> → 延迟低&#xff08;端到端 500ms 量级&#xff09;、画质随终端缩放&#xff1b;</li><li><strong>轻量</strong> → 不挑高端 GPU&#xff0c;官方说百元芯片也能跑。</li></ol> 
<p>这也解释了为什么它能「一套 SDK 适配屏幕、人形机器人、AR/VR 眼镜」——渲染逻辑和载体解耦&#xff0c;参数流可以驱动任何能渲染的终端。</p> 
<hr /> 
<h3>二、核心接口 <code>speak</code>&#xff1a;整句 vs 流式</h3> 
<p>SDK 最核心的接口就一个&#xff1a;</p> 


```js
sdk.speak(ssml, is_start, is_end)
```

 
<ul><li><strong>整句</strong>&#xff1a;<code>speak(&#39;欢迎使用魔珐星云&#39;, true, true)</code></li><li><strong>流式</strong>&#xff1a;第一段 <code>is_start&#61;true</code>&#xff0c;最后一段 <code>is_end&#61;true</code>&#xff0c;中间 <code>false</code></li></ul> 
<p><code>ssml</code> 参数既可传纯文本&#xff0c;也可传 SSML 标记语言&#xff08;下文&#xff09;。</p> 
<p>流式是接大模型的命门——下面第八章会详细讲。</p> 
<hr /> 
<h3>三、SSML &#43; KA 动作指令&#xff1a;让数字人「演」起来</h3> 
<p>光说话不够&#xff0c;数字人最大的差异化是<strong>情绪和动作</strong>。SDK 用 SSML 里的 <code>&lt;ue4event&gt;</code> 标签控制动作&#xff0c;我整理成三类&#xff1a;</p> 
<p><strong>1. 语义 KA&#xff08;根据语义触发动作&#xff09;</strong></p> 


```xml
<speak>
热烈
<ue4event><type>ka_intent</type><data><ka_intent>Welcome</ka_intent></data></ue4event>
欢迎各位贵宾莅临指导！
</speak>
```

 
<p><strong>2. 技能 KA&#xff08;指定动作&#xff0c;如跳舞&#xff09;</strong></p> 


```xml
<speak>
<ue4event><type>ka</type><data><action_semantic>dance</action_semantic></data></ue4event>
音乐响起来，一起跳舞吧！
</speak>
```

 
<p><strong>3. Speak KA&#xff08;动作 &#43; 台词&#xff09;</strong></p> 


```xml
<speak>
<ue4event><type>ka</type><data><action_semantic>Hello</action_semantic></data></ue4event>
欢迎来到星云具身 3D 数字人平台～
</speak>
```

 
<p>项目里的「开场秀」「跳舞」「打招呼」按钮&#xff0c;本质就是这三类 SSML。注意不同应用/角色支持的 KA 动作库不一样——我实测时发现某个角色对 <code>Welcome</code> 返回了 <code>ka intent not found</code>&#xff0c;说明 <strong>KA 动作要按你创建的应用实际支持情况来用</strong>。</p> 
<hr /> 
<h3>四、Widget 事件系统</h3> 
<p>SDK 内置了对几种事件的默认渲染&#xff08;<code>subtitle_on</code> 字幕、<code>subtitle_off</code>、<code>widget_pic</code> 图片&#xff09;。你可以用 <code>onWidgetEvent</code> 或 <code>proxyWidget</code> 自定义。</p> 
<p><strong>重点&#xff1a;优先级是 <code>onWidgetEvent</code> &gt; <code>proxyWidget</code> &gt; 默认事件</strong>——一旦定义了 <code>onWidgetEvent</code>&#xff0c;所有事件都走它&#xff0c;<code>proxyWidget</code> 不再触发。</p> 


```js
onWidgetEvent(data) {
if (data.type === 'subtitle_on')  { showSubtitle(data.text); return; }
if (data.type === 'subtitle_off') { hideSubtitle(); return; }
// 其它事件……
}
```

 
<hr /> 
<h3>五、回调体系&#xff1a;把状态「管起来」</h3> 
<p>一个健壮的接入&#xff0c;几乎要挂全这些回调&#xff1a;</p> 
<table><thead><tr><th>回调</th><th>作用</th></tr></thead><tbody><tr><td><code>onDownloadProgress</code></td><td>资源下载进度&#xff08;<code>init</code> 参数&#xff0c;<strong>必填</strong>&#xff09;</td></tr><tr><td><code>onVoiceStateChange</code></td><td>音频播放状态 <code>start</code> / <code>end</code>&#xff0c;用于管理说话状态</td></tr><tr><td><code>onStateChange</code></td><td>数字人状态变化&#xff08;idle / interactive_idle / speak…&#xff09;</td></tr><tr><td><code>onStateRenderChange</code></td><td>状态切换耗时&#xff08;发 action 到首帧渲染&#xff09;</td></tr><tr><td><code>onStatusChange</code></td><td>SDK 状态&#xff08;在线/离线/隐身/网络…&#xff09;</td></tr><tr><td><code>onMessage</code></td><td>错误/消息&#xff08;含错误码&#xff09;</td></tr><tr><td><code>onNetworkInfo</code></td><td>网络延迟 rtt、下行速率</td></tr><tr><td><code>onStartSessionWarning</code></td><td>数字人配置不正确的警告</td></tr></tbody></table>
<p>其中 <code>onMessage</code> 里会带<strong>错误码</strong>&#xff0c;是排查问题的第一现场&#xff08;比如 <code>10005</code> 房间并发超限&#xff09;。</p> 
<hr /> 
<h3>六、状态机</h3> 
<p>数字人有一组可主动切换的状态&#xff1a;</p> 
<table><thead><tr><th>方法</th><th>状态</th><th>说明</th></tr></thead><tbody><tr><td><code>idle()</code></td><td>待机</td><td>长时间无交互</td></tr><tr><td><code>interactiveidle()</code></td><td>待机互动</td><td>交互前的循环状态&#xff0c;<strong>也可用于打断当前说话</strong></td></tr><tr><td><code>speak()</code></td><td>说话</td><td>核心状态</td></tr><tr><td><code>offlineMode()</code> / <code>onlineMode()</code></td><td>离线/在线</td><td>离线不消耗积分</td></tr><tr><td><code>switchInvisibleMode()</code></td><td>隐身切换</td><td>主动切换隐身/在线</td></tr></tbody></table>
<hr /> 
<h3>七、消耗查询&#xff1a;一次完整的签名鉴权</h3> 
<p>SDK 之外&#xff0c;还有一个 HTTP 接口用来查积分消耗&#xff1a;</p> 


```
GET https://nebula-agent.xingyun3d.com/user/v1/external/consume_record
```

 
<p>它需要三个请求头&#xff0c;其中 <strong><code>X-TOKEN</code> 是签名</strong>&#xff0c;不是直接填 App Secret&#xff1a;</p> 


```
X-TOKEN = MD5( 小写路径 + 小写HTTP方法 + 排序JSON体 + Secret + 秒级时间戳 )
```

 
<p>这个算法官方 SDK 文档里没写&#xff0c;藏在另一篇 KA 接口文档里。调通之后&#xff0c;能在前端直接看到积分消耗记录&#xff1a;</p> 
<p><img src="https://img-home.csdnimg.cn/images/20230724024159.png?origin_url&#61;images%2Fblog-04-consume.png&amp;pos_id&#61;img-4MNVV8Bb-1786955588610" alt="外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传" /></p> 
<hr /> 
<h3>八、接大模型&#xff1a;从念稿到实时 AI 主播</h3> 
<p>这是整套 SDK 最值钱的地方。</p> 
<h4>8.1 为什么「流式」是关键</h4> 
<p>大模型生成回答是<strong>一个字一个字往外蹦的</strong>&#xff08;流式输出&#xff09;。如果等它全部生成完、再一次性丢给数字人去念&#xff0c;那用户要干等十几秒。正确姿势是&#xff1a;<strong>大模型每生成一小段&#xff0c;数字人就同步说一小段</strong>。而 <code>speak</code> 接口天生就是流式设计&#xff0c;专门为这个场景准备的。</p> 
<p><img src="https://img-home.csdnimg.cn/images/20230724024159.png?origin_url&#61;images%2Fblog-02-speaking.png&amp;pos_id&#61;img-l3wMUEwh-1786955588610" alt="外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传" /></p> 
<h4>8.2 先看「模拟流式」的实现</h4> 
<p>在开源项目的 <code>main.js</code> 里&#xff0c;我先用定时器模拟了大模型的流式输出&#xff08;<code>chunkText</code> 按标点切块&#xff0c;定时逐段喂给 <code>speak</code>&#xff09;&#xff1a;</p> 


```js
function streamSpeak(text, onDone) {
const chunks = chunkText(text);   // 按标点切成 8~12 字的小段
let i = 0;
streamTimer = setInterval(() => {
sdk.speak(chunks[i], i === 0, i === chunks.length - 1);
i++;
if (i >= chunks.length) done(); // 播完回调
}, 320);
}
```

 
<p>这段「模拟」就是给真实大模型留的接口——把 <code>setInterval</code> 换成大模型的流式回调即可。</p> 
<h4>8.3 接真实大模型&#xff1a;完整代码</h4> 
<p>下面是一个可落地的示例&#xff08;OpenAI 兼容接口&#xff0c;<code>/v1/chat/completions</code> &#43; <code>stream: true</code>&#xff09;&#xff1a;</p> 


```js
async function talkWithLLM(userText) {
sdk.interactiveidle(); // 先让数字人进入互动待机

const res = await fetch('https://your-llm-gateway/v1/chat/completions', {
method: 'POST',
headers: { 'Content-Type': 'application/json', Authorization: 'Bearer xxx' },
body: JSON.stringify({
model: 'your-model',
stream: true,
messages: [{ role: 'user', content: userText }],
}),
});

const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = '', pending = '', isStart = true;

while (true) {
const { done, value } = await reader.read();
if (done) break;
buffer += decoder.decode(value, { stream: true });

const lines = buffer.split('\n');
buffer = lines.pop(); // 最后一个可能不完整，留到下次

for (const line of lines) {
if (!line.startsWith('data:')) continue;
const payload = line.slice(5).trim();
if (payload === '[DONE]') continue;
let delta = '';
try { delta = JSON.parse(payload).choices?.[0]?.delta?.content || ''; } catch {}
if (!delta) continue;

pending += delta;
// 首段积攒一小段再开口，保证口型跟上后续输出速度
if (pending.length >= 12) {
sdk.speak(pending, isStart, false);
isStart = false;
pending = '';
}
}
}

if (pending) sdk.speak(pending, isStart, true); // 收尾：结束段
}
```

 
<h4>8.4 三个容易翻车的点</h4> 
<ol><li><strong><code>speak</code> 不允许连续多次调用</strong>&#xff1a;一次 <code>is_end &#61; true</code> 之后不能立刻接下一次&#xff0c;中间要用 <code>interactiveidle()</code> 做状态切换。</li><li><strong>用 <code>voice_end</code> 而不是靠猜</strong>&#xff1a;监听 <code>onVoiceStateChange</code> 的 <code>end</code> 事件判断「说完了」&#xff0c;不要用 <code>setTimeout</code> 估时长。</li><li><strong>首帧延迟是真实成本</strong>&#xff1a;从 <code>speak</code> 到数字人开口渲染首帧&#xff0c;我实测在 200ms~800ms 之间&#xff0c;所以官方强调「首段积攒缓冲」——把延迟藏在缓冲里。</li></ol> 
<h4>8.5 完整的多轮对话状态机</h4> 
<p>要做出真正的 AI 主播/客服&#xff0c;需要一个<strong>状态机</strong>管理「听 → 想 → 说 → 回待机」的循环&#xff1a;</p> 


```js
class AvatarChat {
state = 'idle';
pendingText = '';

async onUserSpeak(text) {
sdk.interactiveidle();          // 打断当前说话，回待机
await this.streamFromLLM(text);
}

async streamFromLLM(userText) {
const stream = await callLLMStream(userText);
let isStart = true;
for await (const delta of stream) {
this.pendingText += delta;
if (this.pendingText.length >= 12) {   // 首段积攒缓冲
sdk.speak(this.pendingText, isStart, false);
isStart = false;
this.pendingText = '';
}
}
if (this.pendingText) sdk.speak(this.pendingText, isStart, true);
}

onVoiceStateChange(status) {
if (status === 'end') {
sdk.interactiveidle();  // 说完回待机，等下一轮
this.state = 'idle';
}
}
}
```

 
<p>三个关键点&#xff1a;<strong><code>interactiveidle()</code> 做「打断」</strong>、<strong>状态由 <code>onVoiceStateChange</code> 驱动</strong>、<strong>首段缓冲阈值&#xff08;<code>12</code> 字&#xff09;可调</strong>。</p> 
<p>接语音识别&#xff08;ASR&#xff09;用 Web Speech API 快速跑通&#xff1a;</p> 


```js
const recognition = new webkitSpeechRecognition();
recognition.onresult = (e) => chat.onUserSpeak(e.results[0][0].transcript);
```

 
<p>「语音输入 → 大模型 → 数字人开口」的完整闭环就打通了。</p> 
<hr /> 
<h3>九、落地场景</h3> 
<p>技术最终要落到场景里。具身数字人比普通语音助手多了一条关键能力&#xff1a;<strong>表达</strong>——口型、表情、手势、动作都实时生成&#xff0c;解决的是「信任和氛围」。</p> 
<p><strong>六大场景</strong>&#xff1a;</p> 
<table><thead><tr><th>场景</th><th>说明</th></tr></thead><tbody><tr><td>直播带货 / 口播</td><td>7×24 在线&#xff0c;配大模型自动讲解</td></tr><tr><td>新闻播报 / 资讯</td><td>标准化高频内容&#xff0c;SSML 控语气动作</td></tr><tr><td>门店导购 / 品牌 IP</td><td>线下大屏&#xff0c;离线模式不消耗积分</td></tr><tr><td>展厅 / 发布会讲解</td><td>「开场秀」Demo 就是为此设计</td></tr><tr><td>智能客服 / 前台</td><td>大模型 &#43; ASR&#xff0c;面对面答疑</td></tr><tr><td>教育 / 陪伴</td><td>情感表达比语音更有温度</td></tr></tbody></table>
<p>一个「产品发布会」脚本示例&#xff08;改 <code>DEMO_SCRIPT</code> 即可复用&#xff09;&#xff1a;</p> 
<table><thead><tr><th>步骤</th><th>动作</th><th>台词&#xff08;节选&#xff09;</th></tr></thead><tbody><tr><td>1</td><td>打招呼</td><td>欢迎各位来宾莅临本次发布会&#xff01;</td></tr><tr><td>2</td><td>自我介绍</td><td>我是星云具身驱动演示官……</td></tr><tr><td>3</td><td>流式播报</td><td>&#xff08;产品卖点逐条流式讲解&#xff09;</td></tr><tr><td>4</td><td>强调动作</td><td>请看这里——我们的核心亮点是……</td></tr><tr><td>5</td><td>谢幕</td><td>感谢收看&#xff0c;欢迎到体验区亲身体验&#xff01;</td></tr></tbody></table>
<p><strong>不止屏幕</strong>&#xff1a;同一套 <code>speak()</code> 逻辑&#xff0c;可以搬到人形机器人、AR/VR 眼镜——一次开发、多端复用&#xff0c;这是官方「一套 SDK 全终端」的价值所在。</p> 
<hr /> 
<h3>十、我踩过的 7 个坑</h3> 
<p>这些坑官方文档要么没写、要么一笔带过&#xff0c;希望能帮你省几个小时。</p> 
<p><strong>坑 1&#xff1a;形象不显示&#xff0c;容器高度塌成 0</strong><br /> SDK 初始化会<strong>给容器写入自己的内联样式</strong>&#xff0c;覆盖你的 CSS 定位&#xff0c;导致高度塌陷。解决&#xff1a;给容器加 <code>width/height: 100% !important</code>。</p> 
<p><strong>坑 2&#xff1a;字幕被形象盖住</strong><br /> SDK 给 canvas 写了内联 <code>z-index: 100</code>&#xff0c;字幕条层级低于它就被压住。解决&#xff1a;字幕/角标提到 <code>z-index: 200</code>。</p> 
<p><strong>坑 3&#xff1a;房间并发超限&#xff0c;按钮全「失灵」</strong><br /> 日志里藏着 <code>[10005] 超出房间并发限制</code>。一个驱动应用<strong>同时只允许一个会话</strong>&#xff0c;抢不到房间的一方静默失效。解决&#xff1a;遇到 10005 给提示 &#43; 一键重连&#xff1b;平时只开一个标签页。</p> 
<p><strong>坑 4&#xff1a;只能 localhost 或 https</strong><br /> 用 IP &#43; 端口或裸 http 域名会报错。本地用 localhost&#xff0c;对外部署到 https。</p> 
<p><strong>坑 5&#xff1a;<code>X-TOKEN</code> 是签名不是明文</strong><br /> 调「消耗查询」时先报「签名超时」再报「签名有误」&#xff0c;真正的算法&#xff08;见第七章&#xff09;藏在另一篇 KA 接口文档里。</p> 
<p><strong>坑 6&#xff1a;Safari 舞台横向溢出</strong><br /> <code>aspect-ratio</code> &#43; <code>flex</code> 的组合在 Safari 有兼容性 bug。解决&#xff1a;改成显式宽度计算 <code>calc((100dvh - 136px) * 9 / 16)</code>。</p> 
<p><strong>坑 7&#xff1a;事件优先级</strong><br /> <code>onWidgetEvent</code> &gt; <code>proxyWidget</code> &gt; 默认事件&#xff0c;同时定义两者时后者不触发。</p> 
<hr /> 
<h3>十一、开源工程化&#xff1a;填个 env 就能玩</h3> 
<p>我把上面这些都做成了开源项目 <a href="https://github.com/likebeans/xingyun3D">likebeans/xingyun3D</a>&#xff0c;三种方式覆盖三类人&#xff1a;</p> 
<table><thead><tr><th>用户</th><th>方式</th><th>门槛</th></tr></thead><tbody><tr><td>想快速体验的访客</td><td>GitHub Pages 在线版</td><td>打开 URL&#xff0c;填自己的凭证</td></tr><tr><td>本机调试的开发者</td><td>本地 / Docker</td><td>一条命令</td></tr><tr><td>想二次开发的人</td><td>Codespaces</td><td>云端一键环境</td></tr></tbody></table>
<p><strong>三级配置自动降级</strong>&#xff1a;服务端环境变量 → 构建期注入 → 浏览器填写&#xff0c;同一个代码库既能当「打开即玩」演示站&#xff0c;也能当「自己填 key」的开放工具。</p> 
<p><strong>三条流水线</strong>&#xff08;GitHub Actions&#xff09;&#xff1a;</p> 
<ol><li><strong>CI</strong>&#xff1a;<code>node --check</code> 语法检查 &#43; 无凭证冒烟测试&#xff1b;</li><li><strong>Docker 镜像</strong>&#xff1a;push main 自动发布到 GHCR&#xff08;<code>ghcr.io/likebeans/xingyun3d:latest</code>&#xff09;&#xff1b;</li><li><strong>GitHub Pages</strong>&#xff1a;push main 自动部署在线版。</li></ol> 
<p><img src="https://img-home.csdnimg.cn/images/20230724024159.png?origin_url&#61;images%2Fblog-03-dark.png&amp;pos_id&#61;img-gbdtwM86-1786955588610" alt="外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传" /></p> 
<p>本地 / Docker 启动&#xff1a;</p> 


```bash
# 本地
cp .env.example .env   # 填 XMOV_APP_ID / XMOV_APP_SECRET
npm start              # http://localhost:3000

# Docker
docker run -d -p 3000:3000 \
-e XMOV_APP_ID=你的AppID \
-e XMOV_APP_SECRET=你的AppSecret \
ghcr.io/likebeans/xingyun3d:latest
```

 
<hr /> 
<h3>写在最后</h3> 
<p>从「一行代码让数字人开口」&#xff0c;到「接大模型做实时 AI 主播」&#xff0c;再到「CI/Docker/Pages 全自动交付」&#xff0c;全程可以零依赖、低成本跑通。<strong>参数流 &#43; AI 端渲</strong>这套架构&#xff0c;让数字人从「播放器」变成了能实时表达、交流的「具身智能体」。</p> 
<p>如果你也对这套东西感兴趣&#xff0c;直接上手玩最直观&#xff1a;</p> 
<ul><li>&#x1f381; 邀请码 <strong>XDZARL7NEP</strong>&#xff1a;注册魔珐星云送 1000 积分</li><li>&#x1f517; 在线体验&#xff1a;https://likebeans.github.io/xingyun3D/</li><li>&#x1f4e6; 开源仓库&#xff1a;https://github.com/likebeans/xingyun3D</li><li>&#x1f4d6; 官方文档&#xff1a;https://xingyun3d.com/developers/52-183</li><li>&#x1f310; 官网&#xff1a;https://xingyun3d.com/</li></ul>
