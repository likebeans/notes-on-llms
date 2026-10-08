---
title: "AI Agent 工程化实践：Skill as Prompt 与渐进式加载机制"
description: "CSDN 原文全文镜像：本文探讨了AI Agent系统中Prompt管理失控的问题，并提出Skill体系解决方案。传统方式将所有能力集中在一个系统Prompt中，导致上下文浪费、能力污染等问题。Skill体系将能力拆分为独立模块，每个Skill包含元数据和提示……"
pageType: article
module: agent
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - "csdn-mirror"
  - "agent"
  - "人工智能"
  - "prompt"
  - "大模型"
  - "软件工程"
level: advanced
prerequisites:
  - "/llms/agent/"
  - "/llms/prompt/"
reviewed: '2026-08-27'
techVersion: "CSDN 原文镜像，原文发布于 2026-05-19，站内同步于 2026-08-27"
author: likebeans
---

::: info CSDN 原文镜像
本文为作者 CSDN 博客的全文镜像，原文发布于 2026-05-19。本站补充导读与相关主线链接，并修复代码展示；原文观点、来源与发布时间保留。

- 原文链接：[https://blog.csdn.net/m0_63309778/article/details/161198674](https://blog.csdn.net/m0_63309778/article/details/161198674)
- 站内分区：Agent / Skill as Prompt
:::

::: tip 站内导读与实践边界
本文重点在按需提供任务说明和资源，减少无关上下文。Skill 不限于一段提示，也可能引用脚本、模板和工具；加载技能不等于已执行这些资源。渐进加载是否省成本要统计检索与重复读取开销，并验证关键约束是否在需要时真正进入上下文。

继续阅读：[记忆系统](/llms/agent/memory)、[资源感知优化](/llms/agent/resource-optimization)。
:::

<p><img src="https://i-blog.csdnimg.cn/direct/598709959eda421690a290978aa9c8a1.png" alt="在这里插入图片描述" /></p> 
<h3>前言</h3> 
<p>在构建 AI Agent 系统时&#xff0c;我们经常会遇到一个非常现实的问题&#xff1a;</p> 
<blockquote> 
<p>Agent 的能力越做越多&#xff0c;Prompt 越写越长&#xff0c;系统也越来越难维护。</p> 
</blockquote> 
<p>一开始&#xff0c;我们可能只需要一个简单的系统提示词&#xff1a;</p> 


```text
你是一个智能助手，请根据用户的问题进行回答。
```

 
<p>但随着业务不断复杂&#xff0c;系统 Prompt 很快就会膨胀成这样&#xff1a;</p> 


```text
你是一个智能助手。
你可以分析代码。
你可以读取文件。
你可以生成日报。
你可以编写接口文档。
你可以总结会议纪要。
你可以根据公司模板输出方案。
你可以调用 OA 系统。
你可以处理请假、外勤、会议室预定。
你需要遵守公司的输出规范。
你需要注意安全边界。
你需要……
```

 
<p>最后&#xff0c;一个 Agent 的系统提示词会变成一个巨大的“提示词泥球”。</p> 
<p>这种方式在 Demo 阶段问题不大&#xff0c;但一旦进入真实项目&#xff0c;尤其是企业级 Agent 场景&#xff0c;就会暴露出很多问题&#xff1a;</p> 
<ol><li>Prompt 过长&#xff0c;浪费上下文窗口。</li><li>所有能力一次性加载&#xff0c;哪怕当前任务根本用不到。</li><li>不同能力的规则互相干扰&#xff0c;导致模型输出不稳定。</li><li>新增能力需要频繁修改主 Prompt&#xff0c;维护成本越来越高。</li><li>多人协作困难&#xff0c;很难进行版本管理、测试和灰度发布。</li></ol> 
<p>所以&#xff0c;我们需要一种更加工程化的方式来管理 Agent 的能力。</p> 
<p>这就是本文要介绍的核心方案&#xff1a;</p> 
<blockquote> 
<p>用一套 Skill 体系&#xff0c;把 Agent 的能力拆成一个个独立、可加载、可触发、可缓存的能力模块。</p> 
</blockquote> 
<p>这套体系可以概括为四个关键词&#xff1a;</p> 
<ol><li><strong>Skill as Prompt</strong></li><li><strong>Progressive Disclosure</strong></li><li><strong>智能触发</strong></li><li><strong>LRU 缓存</strong></li></ol> 
<p>Anthropic 在 Agent Skills 的设计中也采用了类似思路&#xff1a;Skill 可以被组织成包含 <code>SKILL.md</code> 的目录&#xff0c;<code>SKILL.md</code> 中包含 YAML frontmatter 和具体说明&#xff0c;系统可以根据任务动态加载相关能力。Claude Help Center 也将 Skills 描述为“可动态加载的指令、脚本和资源文件夹”&#xff0c;用于让模型更好地完成特定任务。(<a href="https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills?utm_source&#61;chatgpt.com" title="Equipping agents for the real world with Agent Skills" rel="nofollow">Anthropic</a>)</p> 
<hr /> 
<h2>一、传统 Agent Prompt 为什么会失控&#xff1f;</h2> 
<p>很多 Agent 项目最开始都是这样做的&#xff1a;</p> 


```text
系统提示词 = 角色设定 + 所有工具说明 + 所有业务规则 + 所有输出规范 + 所有异常处理逻辑
```

 
<p>这种写法简单直接&#xff0c;但扩展性很差。</p> 
<p>假设我们正在做一个企业级 AI 助手&#xff0c;它可能需要支持&#xff1a;</p> 
<ul><li>代码审查</li><li>接口文档生成</li><li>项目日报生成</li><li>周报和月报总结</li><li>RAG 知识库问答</li><li>数据库 SQL 分析</li><li>OA 请假流程填报</li><li>外勤申请</li><li>会议室预定</li><li>项目方案生成</li><li>招投标文件分析</li><li>API 测试用例生成</li></ul> 
<p>如果把所有能力都写进一个系统 Prompt&#xff0c;用户只是问一句&#xff1a;</p> 


```text
帮我看一下这段 FastAPI 代码有没有问题。
```

 
<p>模型却同时看到了&#xff1a;</p> 


```text
日报规范
OA 请假流程
会议纪要模板
招投标方案格式
知识库引用规范
接口文档输出规范
数据库分析规则
……
```

 
<p>这显然是不合理的。</p> 
<p>它会导致两个核心问题。</p> 
<h3>1. 上下文浪费</h3> 
<p>模型每次推理都要处理大量无关内容。</p> 
<p>这不仅会增加 token 成本&#xff0c;还会降低响应速度。</p> 
<p>更严重的是&#xff0c;长 Prompt 会占用原本应该留给用户输入、工具结果、历史上下文和模型推理空间的上下文窗口。</p> 
<h3>2. 能力污染</h3> 
<p>不同任务的提示词规则可能互相冲突。</p> 
<p>例如&#xff1a;</p> 
<ul><li>“日报生成 Skill”要求输出固定日报格式。</li><li>“代码审查 Skill”要求输出问题列表和修改建议。</li><li>“方案生成 Skill”要求按照项目背景、建设目标、技术路线、实施计划输出。</li><li>“知识库问答 Skill”要求必须引用检索来源。</li></ul> 
<p>如果这些规则同时进入上下文&#xff0c;模型就可能混用格式。</p> 
<p>最后结果可能是&#xff1a;</p> 


```text
用户让模型审查代码，模型却按照项目日报格式输出。
```

 
<p>所以更合理的方式是&#xff1a;</p> 
<blockquote> 
<p>当前任务需要什么能力&#xff0c;就只加载什么能力。</p> 
</blockquote> 
<p>这就是 Skill 体系要解决的问题。</p> 
<hr /> 
<h2>二、什么是 Skill as Prompt&#xff1f;</h2> 
<p>所谓 <strong>Skill as Prompt</strong>&#xff0c;就是把 Agent 的每一种能力封装成一个独立的提示词模块。</p> 
<p>每个 Skill 对应一个目录&#xff0c;每个目录下有一个 <code>SKILL.md</code> 文件。</p> 
<p>例如&#xff1a;</p> 


```text
skills/
code-review/
SKILL.md

api-doc-generator/
SKILL.md

work-report/
SKILL.md

rag-answer/
SKILL.md

oa-form-fill/
SKILL.md
```

 
<p>每个 <code>SKILL.md</code> 文件由两部分组成&#xff1a;</p> 
<ol><li><strong>frontmatter 元数据</strong></li><li><strong>body 提示词正文</strong></li></ol> 
<p>一个典型的 <code>SKILL.md</code> 可以这样写&#xff1a;</p> 


```markdown
---
name: code-review
version: 1.0.0
description: 用于审查代码质量、发现潜在 bug、提出优化建议
trigger:
type: prefix
patterns:
- "审查代码"
- "帮我看看这段代码"
- "code review"
priority: 80
cache: true
tags:
- code
- review
- backend
---

# Code Review Skill

你是一个资深代码审查专家。

当用户提供代码时，你需要从以下角度进行分析：

1. 是否存在明显 bug
2. 是否存在性能问题
3. 是否存在安全风险
4. 命名是否清晰
5. 代码结构是否合理
6. 是否符合工程化最佳实践

输出格式：

## 总体评价

## 主要问题

## 优化建议

## 修改后的示例代码
```

 
<p>这里的 frontmatter 是机器可读的元数据。</p> 


```yaml
name: code-review
version: 1.0.0
description: 用于审查代码质量、发现潜在 bug、提出优化建议
trigger:
type: prefix
patterns:
- "审查代码"
- "帮我看看这段代码"
priority: 80
cache: true
```

 
<p>它主要给 Agent Runtime 使用&#xff0c;用来判断&#xff1a;</p> 
<ul><li>这个 Skill 叫什么&#xff1f;</li><li>它适合解决什么问题&#xff1f;</li><li>它应该什么时候被触发&#xff1f;</li><li>它的优先级是多少&#xff1f;</li><li>是否允许缓存&#xff1f;</li></ul> 
<p>而 body 是真正给模型看的提示词正文。</p> 


```markdown
你是一个资深代码审查专家。
当用户提供代码时，你需要……
```

 
<p>这种设计的关键是&#xff1a;</p> 
<blockquote> 
<p>Skill 既是一个 Prompt 文件&#xff0c;也是一个可调度的能力模块。</p> 
</blockquote> 
<p>YAML frontmatter 本身是一种常见的 Markdown 元数据组织方式&#xff0c;GitHub Docs 也将它描述为位于 Markdown 文件顶部的 key-value 元数据块。(<a href="https://docs.github.com/en/contributing/writing-for-github-docs/using-yaml-frontmatter?utm_source&#61;chatgpt.com" title="Using YAML frontmatter" rel="nofollow">GitHub Docs</a>)</p> 
<hr /> 
<h2>三、Progressive Disclosure&#xff1a;渐进式披露</h2> 
<p>Skill 体系最核心的设计思想是 <strong>Progressive Disclosure</strong>&#xff0c;也就是“渐进式披露”。</p> 
<p>它的原则非常简单&#xff1a;</p> 
<blockquote> 
<p>初始化时只读取 Skill 的元数据&#xff0c;真正命中时才加载完整提示词正文。</p> 
</blockquote> 
<p>也就是说&#xff0c;系统启动时不会把所有 <code>SKILL.md</code> 的完整内容都塞进上下文&#xff0c;而是只解析 frontmatter。</p> 
<p>例如系统启动后只拿到类似这样的 Skill Registry&#xff1a;</p> 


```json
[
{
"name": "code-review",
"description": "用于审查代码质量、发现潜在 bug、提出优化建议",
"trigger": {
"type": "prefix",
"patterns": ["审查代码", "code review"]
},
"priority": 80
},
{
"name": "work-report",
"description": "用于根据工作记录生成日报、周报、月报",
"trigger": {
"type": "llm"
},
"priority": 60
}
]
```

 
<p>此时系统并没有读取完整的 Prompt 正文。</p> 
<p>只有当用户输入&#xff1a;</p> 


```text
审查代码：下面这段 FastAPI 代码有没有问题？
```

 
<p>系统匹配到 <code>code-review</code> 之后&#xff0c;才会真正读取&#xff1a;</p> 


```text
skills/code-review/SKILL.md
```

 
<p>并把 body 部分加入本轮模型上下文。</p> 
<p>这就是“渐进式披露”。</p> 
<p>Anthropic 对 Agent Skills 的介绍中&#xff0c;也强调了 Skill 可以通过按需加载的方式扩展 Agent 能力&#xff0c;而不是一次性把所有任务说明塞进上下文。(<a href="https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills?utm_source&#61;chatgpt.com" title="Equipping agents for the real world with Agent Skills" rel="nofollow">Anthropic</a>)</p> 
<hr /> 
<h2>四、为什么 Progressive Disclosure 很重要&#xff1f;</h2> 
<h3>1. 减少初始化成本</h3> 
<p>系统启动时只扫描元数据&#xff0c;不读取完整 Prompt。</p> 
<p>当 Skill 数量很多时&#xff0c;这个优化非常明显。</p> 
<p>假设系统里有 100 个 Skill&#xff0c;每个 Skill 的 Prompt 正文平均 2000 字&#xff0c;如果全部加载&#xff0c;就会变成一个非常庞大的上下文负担。</p> 
<p>但如果只加载 metadata&#xff0c;系统只需要处理几十 KB 的轻量索引。</p> 
<h3>2. 节省上下文窗口</h3> 
<p>用户没有触发的 Skill&#xff0c;不应该进入模型上下文。</p> 
<p>例如用户只是让 Agent 写接口文档&#xff0c;就不需要加载&#xff1a;</p> 
<ul><li>日报 Skill</li><li>会议纪要 Skill</li><li>SQL 分析 Skill</li><li>OA 请假 Skill</li><li>招投标方案 Skill</li></ul> 
<p>这样可以让模型把注意力集中在当前任务上。</p> 
<h3>3. 降低指令冲突</h3> 
<p>每次只加载相关 Skill&#xff0c;可以减少不同能力之间的提示词污染。</p> 
<p>这对于复杂 Agent 非常重要。</p> 
<p>尤其是企业场景中&#xff0c;不同业务的输出格式、审批逻辑、权限要求都不一样&#xff0c;如果全部放进系统 Prompt&#xff0c;很容易互相干扰。</p> 
<h3>4. 让 Prompt 工程变成软件工程</h3> 
<p>传统 Prompt 是一整坨文本。</p> 
<p>Skill 体系则让 Prompt 拥有了类似软件模块的能力&#xff1a;</p> 
<ul><li>可以拆分</li><li>可以复用</li><li>可以版本管理</li><li>可以测试</li><li>可以灰度发布</li><li>可以按需加载</li></ul> 
<p>这才是复杂 Agent 项目长期可维护的关键。</p> 
<hr /> 
<h2>五、Skill 的推荐文件结构</h2> 
<p>一个比较完整的 Skill 可以这样设计&#xff1a;</p> 


```text
skills/
api-doc-generator/
SKILL.md
examples.md
output_schema.json
tests.yaml
```

 
<p>其中&#xff1a;</p> 


```text
SKILL.md           核心元数据和提示词
examples.md        示例输入输出
output_schema.json 结构化输出约束
tests.yaml         回归测试用例
```

 
<p><code>SKILL.md</code> 示例&#xff1a;</p> 


```markdown
---
name: api-doc-generator
version: 1.0.0
description: 根据接口信息生成标准 API 文档
trigger:
type: prefix
patterns:
- "生成接口文档"
- "帮我写 API 文档"
- "生成 API 文档"
priority: 70
cache: true
tags:
- api
- document
- backend
---

# API 文档生成 Skill

你是一个专业的后端接口文档编写助手。

用户会提供接口路径、请求参数、响应结构、业务说明等内容。

你需要输出结构清晰、适合研发团队使用的 API 文档。

## 输出格式

### 1. 接口说明

### 2. 请求地址

### 3. 请求方式

### 4. 请求参数

### 5. 响应参数

### 6. 示例请求

### 7. 示例响应

### 8. 错误码说明

## 约束要求

- 参数说明要清晰。
- 字段类型要标明。
- 示例 JSON 要格式化。
- 如果信息缺失，需要明确指出缺失项。
- 不要自行编造不存在的字段。
```

 
<p>这样做的好处是&#xff1a;</p> 
<ul><li>frontmatter 负责调度。</li><li>body 负责执行。</li><li>examples 负责示范。</li><li>schema 负责输出约束。</li><li>tests 负责质量回归。</li></ul> 
<p>这时 Skill 就不再只是一个 Prompt&#xff0c;而是一个完整的能力包。</p> 
<hr /> 
<h2>六、三种智能触发模式</h2> 
<p>Skill 体系的核心不只是“怎么存储”&#xff0c;更重要的是“怎么触发”。</p> 
<p>一个实用的 Skill Runtime 至少应该支持三种触发模式&#xff1a;</p> 
<ol><li><strong>前缀匹配</strong></li><li><strong>always-on</strong></li><li><strong>LLM 判断</strong></li></ol> 
<hr /> 
<h3>1. 前缀匹配</h3> 
<p>前缀匹配是最简单、最快的触发方式。</p> 
<p>适合命令式任务。</p> 
<p>例如&#xff1a;</p> 


```yaml
trigger:
type: prefix
patterns:
- "审查代码"
- "code review"
- "帮我看看这段代码"
```

 
<p>当用户输入&#xff1a;</p> 


```text
审查代码：下面这段代码有没有问题？
```

 
<p>系统可以直接命中 <code>code-review</code> Skill。</p> 
<p>伪代码如下&#xff1a;</p> 


```python
def match_prefix_skill(user_input, skill_metadata_list):
matched = []

for skill in skill_metadata_list:
trigger = skill.get("trigger", {})

if trigger.get("type") != "prefix":
continue

patterns = trigger.get("patterns", [])

for pattern in patterns:
if user_input.startswith(pattern) or pattern in user_input:
matched.append(skill)
break

return sorted(
matched,
key=lambda x: x.get("priority", 0),
reverse=True
)
```

 
<p>前缀匹配的优点是&#xff1a;</p> 
<ul><li>快</li><li>稳定</li><li>成本低</li><li>可控性强</li></ul> 
<p>适合这些任务&#xff1a;</p> 


```text
生成日报：……
生成接口文档：……
审查代码：……
分析 SQL：……
总结会议纪要：……
```

 
<p>但它也有缺点&#xff1a;对自然语言表达不够灵活。</p> 
<p>例如用户说&#xff1a;</p> 


```text
这段代码我总感觉哪里不对，你帮我看看。
```

 
<p>这时候未必能命中前缀规则。</p> 
<p>于是我们需要 LLM 判断。</p> 
<hr /> 
<h3>2. always-on</h3> 
<p>有些 Skill 不是某个具体任务&#xff0c;而是全局规范。</p> 
<p>例如&#xff1a;</p> 
<ul><li>安全规范</li><li>公司统一输出风格</li><li>禁止泄露敏感信息</li><li>所有技术方案都要包含风险说明</li><li>所有回答都要先给结论</li><li>所有不确定信息都要明确说明</li></ul> 
<p>这种 Skill 可以设计成 always-on。</p> 
<p>示例&#xff1a;</p> 


```markdown
---
name: company-style
version: 1.0.0
description: 公司统一输出风格规范
trigger:
type: always-on
priority: 100
cache: true
---

# 公司统一输出规范

所有回答都应该遵守以下规则：

1. 先给结论，再解释原因。
2. 技术方案要包含优点、缺点和适用场景。
3. 涉及生产环境时，需要补充风险点。
4. 不确定的信息要明确说明。
5. 不要编造接口、字段、数据和政策。
```

 
<p>always-on Skill 会在每次对话中默认加载。</p> 
<p>但这里必须注意&#xff1a;</p> 
<blockquote> 
<p>always-on Skill 一定要克制。</p> 
</blockquote> 
<p>如果 always-on 太多&#xff0c;系统又会退化成“巨型 Prompt”。</p> 
<p>建议 always-on 只放&#xff1a;</p> 
<ul><li>安全边界</li><li>全局风格</li><li>项目级硬约束</li><li>用户长期偏好</li><li>输出底线规范</li></ul> 
<p>具体业务能力不要放 always-on。</p> 
<hr /> 
<h3>3. LLM 判断</h3> 
<p>LLM 判断适合处理语义模糊的场景。</p> 
<p>例如用户输入&#xff1a;</p> 


```text
我这里有一段接口返回，你帮我看看这个设计合不合理。
```

 
<p>这个请求可能命中&#xff1a;</p> 
<ul><li>API 设计 Skill</li><li>Code Review Skill</li><li>后端架构 Skill</li><li>接口文档 Skill</li></ul> 
<p>单靠关键词不一定准确。</p> 
<p>这时候可以设计一个轻量级 Skill Router&#xff0c;让模型根据 Skill metadata 判断应该加载哪些 Skill。</p> 
<p>注意&#xff1a;这个阶段只给模型看 metadata&#xff0c;不给完整 Prompt。</p> 
<p>示例路由 Prompt&#xff1a;</p> 


```text
你是一个 Skill 路由器。

请根据用户输入，从候选 Skill 中选择最适合的一个或多个。

你只能返回 JSON，不要输出多余内容。

用户输入：
{<!-- -->{user_input}}

候选 Skill：
{<!-- -->{skill_metadata_list}}

返回格式：
{
"matched_skills": [
{
"name": "skill_name",
"reason": "为什么选择它",
"confidence": 0.0
}
]
}
```

 
<p>候选 Skill metadata&#xff1a;</p> 


```json
[
{
"name": "code-review",
"description": "用于审查代码质量、发现潜在 bug、提出优化建议",
"tags": ["code", "review", "bug"]
},
{
"name": "api-design",
"description": "用于分析接口设计是否合理，包括参数、响应结构、错误码、幂等等",
"tags": ["api", "backend", "design"]
}
]
```

 
<p>模型返回&#xff1a;</p> 


```json
{
"matched_skills": [
{
"name": "api-design",
"reason": "用户关注接口返回结构和设计合理性，更符合 API 设计分析任务",
"confidence": 0.86
}
]
}
```

 
<p>然后系统再加载 <code>api-design</code> 的完整 Skill body。</p> 
<p>这就是一个比较优雅的两阶段加载流程&#xff1a;</p> 


```text
先用轻量 metadata 判断是否需要
再按需加载完整 Prompt
```

 
<hr /> 
<h2>七、Skill 加载流程设计</h2> 
<p>整体流程可以设计成这样&#xff1a;</p> 


```text
系统启动
↓
扫描 skills 目录
↓
只解析每个 SKILL.md 的 frontmatter
↓
构建 Skill Registry
↓
用户输入
↓
执行 Skill 触发判断
↓
匹配到相关 Skill
↓
读取 Skill body
↓
加入模型上下文
↓
执行任务
```

 
<p>可以抽象成下面这个架构&#xff1a;</p> 


```text
+-------------------+
|    User Input     |
+---------+---------+
|
v
+-------------------+
|   Skill Router    |
| prefix / always   |
| LLM judge         |
+---------+---------+
|
v
+-------------------+
|  Skill Registry   |
| metadata only     |
+---------+---------+
|
v
+-------------------+
|  Load Skill Body  |
| on demand         |
+---------+---------+
|
v
+-------------------+
|   LLM Execution   |
+-------------------+
```

 
<p>这里有一个关键点&#xff1a;</p> 
<blockquote> 
<p>Skill Registry 只保存轻量级索引&#xff0c;不保存所有 Prompt 正文。</p> 
</blockquote> 
<p>这也是它能够扩展到几十个、上百个 Skill 的基础。</p> 
<hr /> 
<h2>八、LRU 缓存&#xff1a;避免重复磁盘 I/O</h2> 
<p>如果每次触发 Skill 都从磁盘读取 <code>SKILL.md</code>&#xff0c;在高并发场景下会产生额外开销。</p> 
<p>因此可以引入 LRU 缓存。</p> 
<p>LRU 是 <strong>Least Recently Used</strong> 的缩写&#xff0c;意思是“最近最少使用”。</p> 
<p>它的策略很简单&#xff1a;</p> 
<blockquote> 
<p>最近使用过的 Skill 保留在内存里&#xff0c;很久没用过的 Skill 被淘汰。</p> 
</blockquote> 
<p>例如缓存容量设置为 32&#xff1a;</p> 


```python
CACHE_SIZE = 32
```

 
<p>当系统第一次加载 <code>code-review</code> Skill 后&#xff0c;把它放入缓存。</p> 
<p>下次再次触发 <code>code-review</code> 时&#xff0c;就不需要重新读取磁盘&#xff0c;直接从内存获取。</p> 
<p>Python 标准库中的 <code>functools.lru_cache</code> 就提供了类似能力&#xff0c;它可以缓存函数调用结果&#xff0c;并通过 <code>maxsize</code> 控制缓存容量。(<a href="https://docs.python.org/3/library/functools.html?utm_source&#61;chatgpt.com" title="functools — Higher-order functions and operations on callable ..." rel="nofollow">Python documentation</a>)</p> 
<p>示例&#xff1a;</p> 


```python
from functools import lru_cache
from pathlib import Path
import frontmatter

SKILL_DIR = Path("./skills")

@lru_cache(maxsize=32)
def load_skill_body(skill_name: str) -> str:
skill_file = SKILL_DIR / skill_name / "SKILL.md"

if not skill_file.exists():
raise FileNotFoundError(f"Skill not found: {skill_name}")

post = frontmatter.load(skill_file)
return post.content
```

 
<p>也可以自己实现一个简单 LRU&#xff1a;</p> 


```python
from collections import OrderedDict

class LRUCache:
def __init__(self, capacity: int = 32):
self.capacity = capacity
self.cache = OrderedDict()

def get(self, key: str):
if key not in self.cache:
return None

self.cache.move_to_end(key)
return self.cache[key]

def put(self, key: str, value: str):
if key in self.cache:
self.cache.move_to_end(key)

self.cache[key] = value

if len(self.cache) > self.capacity:
self.cache.popitem(last=False)
```

 
<p>使用方式&#xff1a;</p> 


```python
skill_cache = LRUCache(capacity=32)

def get_skill_body(skill_name: str) -> str:
cached = skill_cache.get(skill_name)
if cached:
return cached

body = read_skill_body_from_disk(skill_name)
skill_cache.put(skill_name, body)
return body
```

 
<p>这样可以减少重复磁盘读取&#xff0c;提高系统性能。</p> 
<hr /> 
<h2>九、完整代码示例&#xff1a;实现一个简单 Skill Runtime</h2> 
<p>下面给出一个简化版本。</p> 
<p>目录结构&#xff1a;</p> 


```text
project/
main.py
skills/
code-review/
SKILL.md
work-report/
SKILL.md
company-style/
SKILL.md
```

 
<p><code>main.py</code>&#xff1a;</p> 


```python
from pathlib import Path
from functools import lru_cache
import frontmatter

SKILL_DIR = Path("./skills")

def load_skill_metadata():
"""
系统启动时只加载每个 SKILL.md 的 frontmatter。
不读取完整 Prompt 正文。
"""
registry = {}

for skill_file in SKILL_DIR.glob("*/SKILL.md"):
post = frontmatter.load(skill_file)

metadata = dict(post.metadata)
skill_name = metadata.get("name") or skill_file.parent.name

registry[skill_name] = {
"name": skill_name,
"path": str(skill_file),
"description": metadata.get("description", ""),
"trigger": metadata.get("trigger", {}),
"priority": metadata.get("priority", 0),
"cache": metadata.get("cache", True),
"tags": metadata.get("tags", []),
"version": metadata.get("version", "0.0.0"),
}

return registry

@lru_cache(maxsize=32)
def load_skill_body(skill_path: str) -> str:
"""
只有 Skill 被命中时，才加载 body。
并且使用 LRU 缓存减少重复磁盘 I/O。
"""
post = frontmatter.load(skill_path)
return post.content

def match_always_on_skills(registry):
return [
skill for skill in registry.values()
if skill.get("trigger", {}).get("type") == "always-on"
]

def match_prefix_skills(user_input, registry):
matched = []

for skill in registry.values():
trigger = skill.get("trigger", {})

if trigger.get("type") != "prefix":
continue

patterns = trigger.get("patterns", [])

for pattern in patterns:
if user_input.startswith(pattern) or pattern in user_input:
matched.append(skill)
break

return matched

def select_skills(user_input, registry):
matched = []

# 1. always-on Skill 默认加载
matched.extend(match_always_on_skills(registry))

# 2. prefix Skill 根据用户输入匹配
matched.extend(match_prefix_skills(user_input, registry))

# 3. 去重
unique = {}
for skill in matched:
unique[skill["name"]] = skill

# 4. 按优先级排序
return sorted(
unique.values(),
key=lambda x: x.get("priority", 0),
reverse=True
)

def build_prompt(user_input, selected_skills):
skill_prompts = []

for skill in selected_skills:
body = load_skill_body(skill["path"])
skill_prompts.append(
f"## Skill: {skill['name']}\n\n{body}"
)

final_prompt = f"""
你是一个智能 Agent。

下面是本次任务需要使用的 Skill：

{chr(10).join(skill_prompts)}

用户输入：

{user_input}

请根据以上 Skill 完成任务。
"""

return final_prompt

if __name__ == "__main__":
registry = load_skill_metadata()

user_input = "审查代码：下面这段 FastAPI 代码有没有问题？"

selected_skills = select_skills(user_input, registry)

prompt = build_prompt(user_input, selected_skills)

print(prompt)
```

 
<p>这个简化版实现了几个关键能力&#xff1a;</p> 
<ol><li>启动时只加载 metadata。</li><li>用户输入后进行 Skill 匹配。</li><li>命中 Skill 后才加载 body。</li><li>使用 LRU 缓存 Skill body。</li><li>动态组装最终 Prompt。</li></ol> 
<p>虽然它只是一个基础版本&#xff0c;但已经具备 Skill Runtime 的核心雏形。</p> 
<hr /> 
<h2>十、Skill 不应该只是 Prompt&#xff0c;还可以绑定工具</h2> 
<p>在真实 Agent 系统中&#xff0c;Skill 不应该只是一段提示词。</p> 
<p>它还可以声明自己需要哪些工具。</p> 
<p>例如一个 OA 请假 Skill&#xff1a;</p> 


```markdown
---
name: oa-leave-request
version: 1.0.0
description: 用于根据自然语言帮助员工填写请假申请
trigger:
type: llm
priority: 90
tools:
- get_user_profile
- query_leave_balance
- submit_leave_form
cache: true
---

# OA 请假申请 Skill

你是一个企业 OA 助手。

当用户表达请假意图时，你需要：

1. 识别请假类型。
2. 识别开始时间和结束时间。
3. 识别请假原因。
4. 查询用户剩余假期。
5. 检查信息是否完整。
6. 在用户确认后提交请假申请。

如果缺少必要字段，需要向用户追问。

在提交表单前，必须让用户确认。
```

 
<p>这里的 frontmatter 中声明了工具&#xff1a;</p> 


```yaml
tools:
- get_user_profile
- query_leave_balance
- submit_leave_form
```

 
<p>这样 Agent Runtime 在加载 Skill 时&#xff0c;可以同步挂载对应工具。</p> 
<p>也就是说&#xff1a;</p> 


```text
Skill = Metadata + Prompt + Tools + Examples + Output Schema + Permissions
```

 
<p>进一步可以扩展成&#xff1a;</p> 


```text
Skill = 能力描述 + 触发规则 + 提示词 + 工具声明 + 输出协议 + 权限控制
```

 
<p>这时 Skill 就成为 Agent 系统中的最小能力单元。</p> 
<hr /> 
<h2>十一、Skill 的版本管理与测试</h2> 
<p>如果 Skill 要用于生产环境&#xff0c;就不能只靠“感觉可用”。</p> 
<p>它应该像代码一样被管理。</p> 
<p>建议每个 Skill 至少包含&#xff1a;</p> 


```text
version: 1.0.0
author: ai-team
updated_at: 2026-05-18
```

 
<p>并且放入 Git 仓库。</p> 
<p>推荐目录&#xff1a;</p> 


```text
skills/
code-review/
SKILL.md
examples.md
tests.yaml
CHANGELOG.md
```

 
<p><code>tests.yaml</code> 可以这样设计&#xff1a;</p> 


```yaml
cases:
- name: "FastAPI 代码审查"
input: "审查代码：下面这段 FastAPI 代码有没有问题？"
expected_contains:
- "总体评价"
- "主要问题"
- "优化建议"

- name: "SQL 性能分析"
input: "分析 SQL：select * from user where name like '%abc%'"
expected_contains:
- "索引"
- "性能"
- "优化建议"
```

 
<p>这样每次修改 Skill 后&#xff0c;可以跑一批回归测试。</p> 
<p>目的不是保证模型每次输出完全一致&#xff0c;而是保证&#xff1a;</p> 
<ul><li>没有偏离任务目标。</li><li>没有丢失关键结构。</li><li>没有违反输出规范。</li><li>没有出现明显幻觉。</li><li>没有触发错误工具。</li></ul> 
<hr /> 
<h2>十二、Skill 体系中的安全问题</h2> 
<p>Skill 体系虽然优雅&#xff0c;但也会带来新的安全风险。</p> 
<p>因为 <code>SKILL.md</code> 本质上是“可被模型读取并执行的自然语言指令”。</p> 
<p>如果 Skill 来自第三方&#xff0c;或者可以被用户上传&#xff0c;就可能出现类似“Prompt Supply Chain Attack”的问题。</p> 
<p>近期也有研究关注 <code>SKILL.md</code> 类机制中的语义供应链风险&#xff1a;攻击者可以通过 skill 描述、触发词、说明文本来影响 Agent 的发现、选择和加载过程。该研究指出&#xff0c;<code>SKILL.md</code> 并不只是被动文档&#xff0c;它会影响 Agent 如何发现、信任和使用第三方能力。(<a href="https://arxiv.org/abs/2605.11418?utm_source&#61;chatgpt.com" title="Under the Hood of SKILL.md: Semantic Supply-chain Attacks on AI Agent Skill Registry" rel="nofollow">arXiv</a>)</p> 
<p>所以生产环境中要注意&#xff1a;</p> 
<h3>1. Skill 来源必须可信</h3> 
<p>不要随便加载未知来源的 Skill。</p> 
<p>第三方 Skill 至少需要经过审核。</p> 
<h3>2. Skill 权限要隔离</h3> 
<p>不同 Skill 能调用的工具应该不同。</p> 
<p>例如&#xff1a;</p> 


```yaml
permissions:
tools:
- query_leave_balance
forbidden_tools:
- submit_payment
- delete_database
```

 
<p>高风险工具必须要求用户确认。</p> 
<h3>3. Skill 不应该拥有无限权限</h3> 
<p>不要因为某个 Skill 被命中&#xff0c;就把所有工具都暴露给模型。</p> 
<p>应该按 Skill 挂载最小工具集。</p> 
<p>这符合最小权限原则。</p> 
<h3>4. Skill 修改要有审计</h3> 
<p>Skill 文件修改应该记录&#xff1a;</p> 
<ul><li>修改人</li><li>修改时间</li><li>修改原因</li><li>版本号</li><li>diff 内容</li></ul> 
<p>因为 Skill 的变化可能直接影响 Agent 行为。</p> 
<hr /> 
<h2>十三、Skill 设计的最佳实践</h2> 
<h3>1. 一个 Skill 只解决一类问题</h3> 
<p>不要让一个 Skill 过于庞大。</p> 
<p>例如下面这种就不好&#xff1a;</p> 


```text
backend-all-in-one-skill
```

 
<p>它同时处理&#xff1a;</p> 
<ul><li>代码审查</li><li>接口设计</li><li>SQL 优化</li><li>架构设计</li><li>日志规范</li><li>部署方案</li></ul> 
<p>这会重新变成巨型 Prompt。</p> 
<p>更合理的是拆成&#xff1a;</p> 


```text
code-review
api-design
sql-optimization
backend-architecture
logging-best-practice
deployment-advice
```

 
<h3>2. Skill 也不能拆得太碎</h3> 
<p>过粗不好&#xff0c;过细也不好。</p> 
<p>例如&#xff1a;</p> 


```text
daily-report-skill
weekly-report-skill
monthly-report-skill
```

 
<p>这三个其实可以合并成&#xff1a;</p> 


```text
work-report-skill
```

 
<p>然后在内部根据用户意图区分日报、周报、月报。</p> 
<p>一个经验判断是&#xff1a;</p> 
<blockquote> 
<p>如果多个任务共享同一套角色、规则、输出结构&#xff0c;只是参数不同&#xff0c;就可以放在同一个 Skill。</p> 
</blockquote> 
<h3>3. description 要写清楚</h3> 
<p>LLM Router 很依赖 Skill metadata。</p> 
<p>尤其是 description。</p> 
<p>不好的写法&#xff1a;</p> 


```yaml
description: 用于处理代码
```

 
<p>好的写法&#xff1a;</p> 


```yaml
description: 用于审查用户提供的代码，发现 bug、性能问题、安全风险，并给出可执行的修改建议
```

 
<p>description 越清晰&#xff0c;路由越准确。</p> 
<h3>4. trigger patterns 要覆盖常见表达</h3> 
<p>例如代码审查 Skill&#xff1a;</p> 


```yaml
patterns:
- "审查代码"
- "帮我看看这段代码"
- "这段代码有没有问题"
- "code review"
- "review this code"
```

 
<p>既要覆盖命令式表达&#xff0c;也要覆盖自然语言表达。</p> 
<h3>5. Prompt 正文要包含边界</h3> 
<p>每个 Skill 都应该明确告诉模型&#xff1a;</p> 
<ul><li>你擅长什么。</li><li>你不应该做什么。</li><li>信息不足时怎么办。</li><li>输出格式是什么。</li><li>是否允许猜测。</li><li>是否需要用户确认。</li></ul> 
<p>例如&#xff1a;</p> 


```markdown
如果用户没有提供代码，不要直接开始审查。
你需要先提醒用户补充代码片段或文件内容。
```

 
<p>这类边界非常重要。</p> 
<hr /> 
<h2>十四、在企业 Agent 中的典型应用</h2> 
<p>这套 Skill 体系非常适合企业级 Agent。</p> 
<p>例如一个企业 OA Agent 可以有&#xff1a;</p> 


```text
skills/
company-style/
employee-profile-query/
leave-request/
business-trip-request/
meeting-room-booking/
todo-query/
schedule-query/
work-report/
policy-qa/
```

 
<p>当用户输入&#xff1a;</p> 


```text
我明天下午想请半天年假。
```

 
<p>系统流程是&#xff1a;</p> 


```text
1. 读取用户输入
2. Skill Router 判断命中 leave-request
3. 加载 leave-request/SKILL.md
4. 挂载 get_user_profile、query_leave_balance 等工具
5. 抽取请假类型、时间、原因
6. 缺字段则追问
7. 信息完整后让用户确认
8. 确认后调用 submit_leave_form
```

 
<p>当用户输入&#xff1a;</p> 


```text
帮我根据今天的工作内容生成日报。
```

 
<p>系统则只加载&#xff1a;</p> 


```text
work-report Skill
company-style Skill
```

 
<p>而不会加载 OA 请假、会议室预定、SQL 分析等无关能力。</p> 
<p>这就是 Skill 体系的价值&#xff1a;</p> 
<blockquote> 
<p>让 Agent 根据任务动态“长出”需要的能力&#xff0c;而不是一开始就背上所有能力。</p> 
</blockquote> 
<hr /> 
<h2>十五、和传统工具调用有什么区别&#xff1f;</h2> 
<p>很多人可能会问&#xff1a;</p> 
<blockquote> 
<p>Skill 和 Tool 有什么区别&#xff1f;</p> 
</blockquote> 
<p>可以这样理解&#xff1a;</p> 


```text
Tool 解决“能做什么”
Skill 解决“怎么做得好”
```

 
<p>Tool 是具体能力接口&#xff0c;例如&#xff1a;</p> 


```text
query_database()
submit_form()
read_file()
search_knowledge_base()
```

 
<p>Skill 是完成某类任务的方法论&#xff0c;例如&#xff1a;</p> 


```text
如何分析 SQL
如何生成日报
如何填写请假单
如何根据知识库回答问题
```

 
<p>Tool 更像函数。</p> 
<p>Skill 更像操作手册。</p> 
<p>在 Agent 系统里&#xff0c;两者应该配合使用&#xff1a;</p> 


```text
Skill 决定任务策略
Tool 执行具体动作
```

 
<p>例如&#xff1a;</p> 


```text
请假 Skill：
- 识别请假意图
- 抽取请假字段
- 判断字段是否完整
- 让用户确认
- 调用 submit_leave_form 工具
```

 
<p>所以&#xff0c;Skill 不是 Tool 的替代品&#xff0c;而是 Tool 的上层调度说明。</p> 
<hr /> 
<h2>十六、最终推荐架构</h2> 
<p>一个比较完整的 Skill Runtime 可以分成五层&#xff1a;</p> 


```text
+-----------------------------+
|        User Input           |
+-----------------------------+
|
v
+-----------------------------+
|        Skill Router         |
| prefix / always / llm judge |
+-----------------------------+
|
v
+-----------------------------+
|        Skill Registry       |
| metadata / version / tags   |
+-----------------------------+
|
v
+-----------------------------+
|        Skill Loader         |
| body / examples / schema    |
+-----------------------------+
|
v
+-----------------------------+
|        Agent Executor       |
| prompt + tools + memory     |
+-----------------------------+
```

 
<p>其中&#xff1a;</p> 
<ul><li>Skill Router 负责判断加载哪些 Skill。</li><li>Skill Registry 负责管理元数据。</li><li>Skill Loader 负责按需读取完整 Skill。</li><li>LRU Cache 负责优化重复加载。</li><li>Agent Executor 负责最终模型调用和工具执行。</li></ul> 
<hr /> 
<h2>十七、总结</h2> 
<p>传统 Agent 系统最大的问题之一&#xff0c;是把所有能力都堆进一个巨大的 Prompt 里。</p> 
<p>这种方式短期看简单&#xff0c;长期看一定会失控。</p> 
<p>它会带来&#xff1a;</p> 
<ul><li>上下文浪费</li><li>响应变慢</li><li>能力污染</li><li>维护困难</li><li>扩展困难</li><li>测试困难</li><li>权限边界不清晰</li></ul> 
<p>而 Skill 体系提供了一种更加工程化的解决方案。</p> 
<p>它的核心思想是&#xff1a;</p> 
<ol><li>使用 <strong>Skill as Prompt</strong>&#xff0c;把每个能力封装成独立的 <code>SKILL.md</code> 文件。</li><li>使用 <strong>Progressive Disclosure</strong>&#xff0c;启动时只读取元数据&#xff0c;触发时才加载完整提示词。</li><li>使用 <strong>智能触发机制</strong>&#xff0c;支持前缀匹配、always-on 和 LLM 判断。</li><li>使用 <strong>LRU 缓存</strong>&#xff0c;缓存最近使用的 Skill&#xff0c;减少重复磁盘 I/O。</li><li>使用 <strong>版本管理、测试和权限控制</strong>&#xff0c;让 Skill 真正具备生产可用性。</li></ol> 
<p>一句话总结&#xff1a;</p> 
<blockquote> 
<p>Skill 体系的本质&#xff0c;是把 Prompt 从“临时文本”升级为“工程化能力模块”。</p> 
</blockquote> 
<p>对于复杂 Agent 项目来说&#xff0c;这一步非常关键。</p> 
<p>因为未来的 Agent 不应该是一个塞满规则的超长 Prompt&#xff0c;而应该是一个能够按需加载能力、动态组合工具、具备清晰边界和可维护结构的智能运行时。</p> 
<p>也就是说&#xff1a;</p> 


```text
好的 Agent，不是把所有能力都写进 Prompt。
好的 Agent，是知道什么时候该加载什么能力。
```

 
<p>这就是 Skill 体系真正优雅的地方。</p>
