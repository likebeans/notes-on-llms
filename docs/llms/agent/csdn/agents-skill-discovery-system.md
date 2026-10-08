---
title: "构建下一代语境感知型 AI Agent：AGENTS.md 与 SKILL.md 发现系统的深度工程架构报告"
description: "CSDN 原文全文镜像：摘要： Agent技术正从对话式转向自主行动能力，其效能核心在于项目语境获取。行业通过标准化协议解决\"语境孤岛\"问题：AGENTS.md定义治理规则（宪法），SKILL.md封装可执行能力（技能包）。报告详细解析了构建此类Agent的完……"
pageType: article
module: agent
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "人工智能"
  - "架构"
level: advanced
prerequisites:
  - "/llms/agent/"
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-02-03，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-02-03。本站补充导读与相关主线链接，并修复代码展示；原文观点、来源与发布时间保留。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/157685083](https://blog.csdn.net/m0_63309778/article/details/157685083)
- 站内分区：Agent / AGENTS 与 SKILL 发现
:::

::: tip 站内导读与实践边界
本文可以作为发现与加载系统的架构草图阅读，文件命名、搜索层级、优先级和触发规则应按具体宿主核对，不能假定所有 Agent 都相同。加载到的技能文本不会自行获得工具权限；需防止不可信仓库内容扩大执行范围，并测试冲突、重复发现和缓存失效。

继续阅读：[安全与沙箱](/llms/agent/safety)、[工具调用](/llms/agent/tool-calling)。
:::

<div class="csdn-mirror-content">

<p><img src="https://i-blog.csdnimg.cn/direct/3012006ccf704f1fa35356976e3d1a30.png" alt="在这里插入图片描述" /></p>
<h3>1. 执行摘要</h3>
<p>随着 LLM 从对话界面演进为具备自主行动能力的 Agent&#xff0c;软件开发范式正从指令式编程转向基于意图和语境的编排。Agent 的效能瓶颈已不再是推理能力&#xff0c;而是其获取并遵循项目特定语境的能力。</p>
<p>为了解决“语境孤岛”&#xff0c;行业正向标准化协议收敛&#xff1a;</p>
<ul><li><strong>AGENTS.md</strong>&#xff1a;作为项目的“宪法”&#xff0c;定义架构规范与行为准则&#xff08;治理&#xff09;。</li><li><strong>SKILL.md</strong>&#xff1a;作为可执行的“技能包”&#xff0c;赋予动态调用的操作能力&#xff08;能力&#xff09;。</li></ul>
<p>本报告剖析了如何构建具备自主发现、解析并利用这些文件的 Agent&#xff0c;涵盖从文件系统遍历、AST 解析到基于 MCP 协议的安全沙箱执行环境的完整路线图。</p>
<hr />
<h3>2. 语境危机与 Agent 架构的演进</h3>
<h4>2.1 从“提示词工程”到“提示词运营&#xff08;PromptOps&#xff09;”</h4>
<p>早期依赖庞大的系统提示词&#xff08;System Prompt&#xff09;存在两大局限&#xff1a;</p>
<ol><li><strong>语境污染</strong>&#xff1a;全量语境注入&#xff08;Context Stuffing&#xff09;挤压上下文窗口&#xff0c;导致模型注意力下降。</li><li><strong>维护困难</strong>&#xff1a;硬编码的规则难以随代码库迭代。</li></ol>
<p>**“配置即代码&#xff08;Configuration as Code&#xff09;”**的理念应运而生&#xff0c;将行为准则定义为版本受控的文件。</p>
<h4>2.2 上下文感知的标准化协议</h4>
<ul><li><strong>AGENTS.md (项目的神经中枢)</strong>&#xff1a;解决 Agent “我是谁&#xff1f;我在哪里&#xff1f;遵循什么规则&#xff1f;”的问题。支持层级化继承&#xff0c;允许规则覆盖与合并。</li><li><strong>SKILL.md (模块化的能力单元)</strong>&#xff1a;采用渐进式披露模式。只有在识别到意图时才加载相关指令&#xff0c;优化 Token 效率。</li></ul>
<hr />
<h3>3. 行业实现深度剖析</h3>
<h4>3.1 OpenAI Codex 与 GitHub Copilot&#xff1a;语境分层</h4>
<ul><li><strong>发现机制</strong>&#xff1a;从用户主目录向上扫描至当前工作目录&#xff08;CWD&#xff09;。</li><li><strong>优先级逻辑</strong>&#xff1a;距离 CWD 越近的文件优先级越高&#xff0c;实现“局部优于全局”。</li></ul>
<h4>3.2 Anthropic (Claude Code)&#xff1a;虚拟机与技能挂载</h4>
<ul><li><strong>VM 架构</strong>&#xff1a;在受控虚拟机中运行。</li><li><strong>子智能体模式</strong>&#xff1a;主 Agent 编排&#xff0c;根据 <code>SKILL.md</code> 启动加载特定上下文的子 Agent 以隔离干扰。</li></ul>
<h4>3.3 Cursor&#xff1a;语义检索增强 (RAG)</h4>
<ul><li><strong>精细化控制</strong>&#xff1a;引入 <code>.mdc</code> 格式&#xff0c;利用 Glob 模式&#xff08;如 <code>*.ts</code>&#xff09;限定规则生效范围。</li><li><strong>语义路由</strong>&#xff1a;对技能描述进行向量索引&#xff0c;动态匹配用户查询。</li></ul>
<hr />
<h3>4. 自研 Agent 核心模块架构</h3>
<p>系统架构划分为四个核心层级&#xff1a;</p>

<table><thead><tr><th>模块名称</th><th>核心职责</th><th>关键技术栈</th></tr></thead><tbody><tr><td><strong>I. 发现引擎</strong></td><td>遍历文件系统&#xff0c;识别配置文件&#xff0c;处理忽略规则</td><td>Python <code>pathlib</code>, <code>pathspec</code></td></tr><tr><td><strong>II. 认知解析层</strong></td><td>解析 Markdown AST&#xff0c;验证 YAML 元数据&#xff0c;处理继承逻辑</td><td><code>markdown-it-py</code>, <code>PyYAML</code>, <code>Pydantic</code></td></tr><tr><td><strong>III. 语境管理器</strong></td><td>动态构建 Prompt&#xff0c;实现渐进式披露&#xff0c;管理 Token 预算</td><td>LangChain, Vector DB</td></tr><tr><td><strong>IV. 执行运行时</strong></td><td>安全执行脚本&#xff0c;工具调用&#xff0c;沙箱隔离</td><td>MCP Protocol, Docker/Firecracker</td></tr></tbody></table><hr />
<h3>5. 模块实现详解</h3>
<h4>5.1 发现引擎&#xff1a;智能遍历</h4>
<p>实现基于 <code>.gitignore</code> 感知的遍历器。</p>
<ul><li><strong>自底向上</strong>&#xff1a;寻找 <code>AGENTS.md</code> 以构建规则链。</li><li><strong>自顶向下</strong>&#xff1a;全局索引 <code>SKILL.md</code> 构建技能库。</li></ul>
<h4>5.2 认知解析&#xff1a;AST 与元数据</h4>
<ul><li><strong>语义合并</strong>&#xff1a;解析 <code>AGENTS.md</code> 为 AST&#xff0c;基于标题&#xff08;Header&#xff09;进行内容覆盖或追加。</li><li><strong>SKILL.md 验证</strong>&#xff1a;严格校验 YAML 中的 <code>name</code>、<code>description</code> 和 <code>allowed-tools</code>&#xff08;安全白名单&#xff09;。</li></ul>
<h4>5.3 语境管理&#xff1a;动态 Prompt 构建</h4>
<p>采用三层结构构建系统提示词&#xff1a;</p>
<ol><li><strong>身份层</strong>&#xff1a;基本人设。</li><li><strong>宪法层</strong>&#xff1a;合并后的 <code>AGENTS.md</code>&#xff08;始终在线&#xff09;。</li><li><strong>能力索引层</strong>&#xff1a;仅包含技能名称与描述&#xff0c;具体指令按需加载&#xff08;Lazy Loading&#xff09;。</li></ol>
<h4>5.4 执行运行时&#xff1a;基于 MCP 的沙箱</h4>
<p>为了防御 RCE 风险&#xff0c;必须采用隔离环境&#xff1a;</p>
<ul><li><strong>MCP 协议</strong>&#xff1a;将文件系统和工具调用标准化。</li><li><strong>沙箱化</strong>&#xff1a;使用 <strong>Docker</strong> 容器或 <strong>Firecracker MicroVMs</strong> 运行 <code>SKILL.md</code> 中定义的脚本。</li></ul>
<hr />
<h3>6. 战略价值</h3>
<ol><li><strong>知识资产化</strong>&#xff1a;将团队隐性知识转化为可执行的代码。</li><li><strong>跨平台互操作</strong>&#xff1a;遵循 <code>AGENTS.md</code> 等标准&#xff0c;避免供应商锁定。</li><li><strong>无限扩展性</strong>&#xff1a;通过添加技能文件夹即可赋予 Agent 新能力&#xff0c;无需微调模型。</li></ol>
<h3>7. 总结与展望</h3>
<p>构建此类 Agent 是 AI 辅助开发走向工业级的关键。未来&#xff0c;<strong>自我进化型 Agent</strong> 将成为主流&#xff1a;它们不仅读取规范&#xff0c;还能通过观察项目偏差&#xff0c;主动发起 PR 更新 <code>AGENTS.md</code>&#xff0c;实现 DevOps 闭环。</p>
<hr />
<h4>附录&#xff1a;核心技术规格对比表</h4>

<table><thead><tr><th>特性</th><th>AGENTS.md</th><th>SKILL.md</th><th>.cursorrules</th></tr></thead><tbody><tr><td><strong>主要职责</strong></td><td>全局治理、行为准则</td><td>任务操作、工具封装</td><td>编辑器上下文注入</td></tr><tr><td><strong>作用域</strong></td><td>递归继承/覆盖</td><td>模块化独立定义</td><td>基于 Glob 文件匹配</td></tr><tr><td><strong>执行能力</strong></td><td>被动&#xff08;规则&#xff09;</td><td>主动&#xff08;含 scripts/&#xff09;</td><td>被动&#xff08;Prompt&#xff09;</td></tr><tr><td><strong>加载策略</strong></td><td>全量合并加载</td><td>按需延迟加载</td><td>基于相关性 RAG</td></tr></tbody></table>

</div>
