---
title: "5 分钟，让网页里站着一个会说话的 3D 数字人"
description: "CSDN 原文全文镜像：本文介绍了如何通过魔珐星云SDK在网页中实时渲染3D数字人，实现文本驱动语音、表情和动作同步。主要内容包括：效果展示（非预录视频，而是实时AI渲染）、接入步骤（引入SDK、创建实例、调用speak方法）、开源控制台项目封装（支持即兴对话……"
pageType: article
module: multimodal
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "multimodal"
  - "数字人"
  - "大模型"
  - "prompt"
level: intermediate
prerequisites:
  - "/llms/multimodal/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-08-17，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-08-17。为适配本站结构，补充了站内元数据、来源说明与阅读导引，并修复代码高亮残留；原文主体与观点保留。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/163828144](https://blog.csdn.net/m0_63309778/article/details/163828144)
- 站内分区：Multimodal / 网页 3D 数字人
:::

::: tip 站内导读：接通演示之后验证交互
本文演示“文本 → 语音/动作参数 → 浏览器渲染”的产品链路，可结合 [多模态部署](/llms/multimodal/deployment)阅读；它不等同于训练一个统一多模态模型。验收时分别测首次资源加载、首音频延迟、音画同步和低性能设备帧率，并覆盖断网/取消。`@latest`、SDK 方法、邀请码与赠送额度保留为发布时信息，复现需锁定版本并核对服务商文档。浏览器示例中的长期 Secret 不能作为生产公开页面的凭证方案，应按服务商支持的会话鉴权机制设计。
:::


<p>先放结论&#xff1a;这个效果不是我录的视频&#xff0c;是<strong>浏览器里实时渲染的 3D 数字人</strong>——你给一段文字&#xff0c;它实时合成语音、同步口型、配上表情和动作。</p>
<p><img src="https://i-blog.csdnimg.cn/direct/4e9152ad664b4a94b6be668517a5430f.png" alt="在这里插入图片描述" /></p>
<p>你可以直接打开在线版体验&#xff08;填个 App ID/Secret 就能玩&#xff09;&#xff1a;</p>
<blockquote>
<p>&#x1f517; https://likebeans.github.io/xingyun3D/</p>
</blockquote>
<p>如果你有魔珐星云的账号&#xff0c;用邀请码 <strong>XDZARL7NEP</strong> 注册还能送 1000 积分&#xff0c;够你跑很久的演示。</p>
<p>下面我带你从零把这个东西接进你自己的网页。</p>
<hr />
<h3>一、它到底做了什么</h3>
<p>这不是「一段提前录好的视频」&#xff0c;而是一个<strong>实时驱动的 3D 数字人</strong>&#xff1a;</p>
<ol><li>你调用一个 <code>speak(text)</code>&#xff1b;</li><li>云端把文本合成为语音、表情、口型、动作的<strong>参数流</strong>下发给浏览器&#xff1b;</li><li>浏览器端渲染&#xff08;AI 端渲&#xff09;出画面&#xff0c;和语音对齐。</li></ol>
<p>所以它能做到「<strong>所见即所听&#xff0c;毫秒级响应</strong>」。这也是后面接大模型做实时 AI 主播的基础&#xff08;后面有专门一篇讲&#xff09;。</p>
<hr />
<h3>二、环境要求</h3>
<p>先交代前置条件&#xff0c;免得你踩坑&#xff1a;</p>
<ul><li><strong>浏览器</strong>&#xff1a;建议最新版 Chrome / Edge / Safari&#xff08;SDK 依赖 WebGL2 硬件加速渲染&#xff09;&#xff1b;</li><li><strong>访问方式</strong>&#xff1a;SDK <strong>仅支持 <code>localhost</code> 或 <code>https</code></strong>&#xff0c;直接用「IP &#43; 端口」或裸 <code>http</code> 域名会报错&#xff08;这是我在踩坑篇里记录过的坑&#xff09;&#xff1b;</li><li><strong>账号凭证</strong>&#xff1a;登录 https://xingyun3d.com 在「应用中心」创建一个驱动应用&#xff0c;选择角色、音色、表演风格&#xff0c;即可拿到 App ID / App Secret&#xff1b;</li><li><strong>运行环境</strong>&#xff08;跑我的开源项目时&#xff09;&#xff1a;Node ≥ 18.17 即可&#xff0c;零依赖。</li></ul>
<h3>三、最小可运行代码</h3>
<p>整个接入只需要三步&#xff1a;引脚本、建实例、说话。</p>
<h4>1. 引入 SDK</h4>


```html
<div style="width: 540px; height: 960px">
<div id="sdk"></div>
</div>
<script src="https://media.xingyun3d.com/xingyun3d/general/litesdk/xmovAvatar@latest.js"></script>
```


<h4>2. 创建实例</h4>


```js
const sdk = new XmovAvatar({
containerId: '#sdk',           // 数字人渲染容器
appId: '你的 App ID',           // 在应用中心创建驱动应用后获取
appSecret: '你的 App Secret',
gatewayServer: 'https://nebula-agent.xingyun3d.com/user/v1/ttsa/session',
hardwareAcceleration: 'prefer-hardware', // 开启硬件加速
onMessage(message) { /* 处理错误/消息 */ },
onVoiceStateChange(status) { /* status: start / end */ },
});

// 初始化并监听资源下载进度
sdk.init({
initModel: 'normal',
onDownloadProgress: (progress) => console.log(progress + '%'), // 必填
});
```


<h4>3. 让它说话</h4>


```js
sdk.speak('欢迎使用魔珐星云', true, true);
```


<p>就这样&#xff0c;一个会说话的 3D 数字人就站到你的网页里了。</p>
<hr />
<h3>三、我把它封装成了一个「表演控制台」</h3>
<p>为了让演示更像样&#xff08;也更适合给产品做宣传&#xff09;&#xff0c;我写了个开源项目&#xff1a;</p>
<blockquote>
<p>&#x1f4e6; https://github.com/likebeans/xingyun3D</p>
</blockquote>
<p>它把 SDK 的能力包装成了一个<strong>零依赖、一个页面跑通的演示台</strong>&#xff1a;</p>
<ul><li><strong>即兴说话</strong>&#xff1a;输入框打什么它说什么&#xff0c;⌘/Ctrl &#43; Enter 发送</li><li><strong>开场秀 / 跳舞 / 打招呼</strong>&#xff1a;SSML &#43; KA 动作指令&#xff0c;情绪动作台词一起演</li><li><strong>流式播报</strong>&#xff1a;模拟大模型逐段输出&#xff0c;字幕、口型实时同步</li><li><strong>自动演示</strong>&#xff1a;一键跑完「打招呼 → 自我介绍 → 流式播报 → 跳舞 → 谢幕」完整节目</li><li><strong>状态遥控</strong>&#xff1a;待机 / 待机互动 / 在线 / 离线 / 隐身&#xff0c;外加音量调节</li><li><strong>实时事件流</strong>&#xff1a;语音、网络延迟、状态、积分消耗全都可视化</li><li><strong>亮 / 暗双主题</strong>&#xff1a;默认亮色&#xff0c;喜欢深色一键切</li></ul>
<p>说话时字幕、舞台光效实时联动&#xff0c;效果长这样&#xff1a;</p>
<p><img src="https://i-blog.csdnimg.cn/direct/e5dcfdf9e7fb40a0bc4ffadc7cb247d5.png" alt="在这里插入图片描述" /></p>
<p>本地跑起来只需要&#xff1a;</p>


```bash
# 1. 填环境变量
cp .env.example .env   # 编辑 .env 填入 XMOV_APP_ID / XMOV_APP_SECRET
# 2. 启动（零依赖，Node ≥ 18.17）
npm start
# 3. 打开 http://localhost:3000
```


<p>或者直接 Docker&#xff1a;</p>


```bash
docker run -d -p 3000:3000 \
-e XMOV_APP_ID=你的AppID \
-e XMOV_APP_SECRET=你的AppSecret \
ghcr.io/likebeans/xingyun3d:latest
```


<hr />
<h3>四、为什么我觉得这玩意儿值得关注</h3>
<p>&#xff08;插一句&#xff1a;如果你更喜欢深色界面&#xff0c;项目里一键就能切&#xff0c;效果如下——&#xff09;</p>
<p><img src="https://img-home.csdnimg.cn/images/20230724024159.png?origin_url&#61;images%2Fblog-03-dark.png&amp;pos_id&#61;img-q6uv4yHk-1786955226493" alt="外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传" /></p>
<p>魔珐&#xff08;xmov&#xff09;这家公司你可能没听过&#xff0c;但它背后的东西不简单&#xff1a;官方定位是「<strong>全球领先的 3D 具身交互智能体 AI 科技公司</strong>」&#xff0c;核心是自研的 <strong>LAM 文生 3D 多模态大模型</strong>&#xff0c;打造的「魔珐星云」是一个面向全终端的具身智能基础设施。</p>
<p>几个关键词值得记一下&#xff1a;</p>
<ul><li><strong>参数流 &#43; AI 端渲</strong>&#xff1a;不下发视频&#xff0c;下发「参数」&#xff08;语音、表情、动作、口型数据&#xff09;&#xff0c;端侧渲染&#xff0c;所以能做得轻、快&#xff1b;</li><li><strong>端到端 500ms 超低延迟</strong>&#xff1a;交互够实时&#xff0c;接大模型才不会「卡壳感」&#xff1b;</li><li><strong>千万级并发</strong>&#xff1a;这是给规模化场景准备的&#xff0c;不是玩具&#xff1b;</li><li><strong>百元芯片轻量化部署</strong>&#xff1a;不挑高端 GPU&#xff1b;</li><li><strong>一套 SDK 适配全终端</strong>&#xff1a;屏幕、人形机器人、AR/VR 眼镜通吃——同一个 <code>speak()</code>&#xff0c;从网页到机器人。</li></ul>
<p>这些我在后续的文章里会逐个拆解、验证&#xff0c;不是只念官网文案。</p>
<hr />
<h3>五、福利</h3>
<ul><li>&#x1f381; 邀请码 <strong>XDZARL7NEP</strong>&#xff1a;注册魔珐星云送 1000 积分</li><li>&#x1f517; 在线体验&#xff1a;https://likebeans.github.io/xingyun3D/</li><li>&#x1f4e6; 开源仓库&#xff1a;https://github.com/likebeans/xingyun3D</li><li>&#x1f4d6; 官方文档&#xff1a;https://xingyun3d.com/developers/52-183</li></ul>
<p>下一篇我会深入拆 SDK 的技术细节&#xff1a;语音、口型、表情、动作到底是怎么在 500ms 内联动的。</p>
