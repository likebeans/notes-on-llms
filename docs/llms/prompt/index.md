---
title: Prompt 与上下文工程全景
description: 从指令层级、示例、结构化输出到上下文构造、评估和安全边界的学习单元。
pageType: article
module: prompt
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - prompt
level: beginner
prerequisites: []
reviewed: '2026-10-08'
reviewScope: 结构化输出、压缩和注入防护导读；其余资料未全面复核
exampleStatus: not-run
techVersion: 结构化输出、压缩与注入边界核验 2026-10-08；API 示例未实跑
---

# Prompt 与上下文工程全景

<LearningObjectives :items="[
  '能够把一次模型调用拆成指令层级、任务说明、上下文、示例、输出约束和评估样例六个部分。',
  '能够区分 prompt engineering 与 context engineering，并说明什么时候要改文字、什么时候要改上下文供应链。',
  '能够为一个生产提示设计结构化输出、回归评估和提示注入防护边界。'
]" />

## 这个模块解决什么问题

Prompt engineering 最早像一组经验技巧：写清楚、给例子、让模型一步步处理。这个基础仍然重要，但 2026 年的生产系统已经不再只是“调一句咒语”。OpenAI 文档强调 prompt 是让模型稳定满足需求的指令设计；Anthropic 把 context engineering 称为 prompt engineering 的自然演进：问题从“怎么写一句话”扩展为“本轮推理该给模型哪些 token、哪些工具、哪些历史、哪些约束”。

因此，本模块把 Prompt 分成两层。第一层是狭义提示工程：角色、任务、约束、示例、输出格式和推理策略。第二层是上下文工程：系统/开发者/用户/工具消息的权威顺序，RAG 或 MCP 带来的外部资料，工具 schema，历史摘要，安全策略，评估样例和结构化输出。CRISPE、zero-shot、few-shot、CoT、ToT 等模式应放进完整的输入、执行与验收流程中理解。

一个好 prompt 的目标不是让模型“听话一次”，而是让同一类任务在模型升级、输入变化、上下文变长和攻击性内容出现时仍然可测试、可解释、可回滚。它既要帮助模型理解任务，也要告诉应用如何判断输出是否合格。

## 核心工作流

```mermaid
flowchart LR
  Goal[任务目标] --> Authority[指令层级]
  Authority --> Context[上下文构造]
  Context --> Examples[示例与边界样例]
  Examples --> Output[结构化输出约束]
  Output --> Run[模型调用]
  Run --> Eval[评估与回归]
  Eval --> Security[安全与注入防护]
  Security --> Context
```

工作流的第一步是确定权威顺序。OpenAI Model Spec 把指令分成 Root、System、Developer、User、Guideline 和 No Authority；API 文档也强调高优先级 instructions 会覆盖普通 input。实践里可以简化成：平台/系统规则最强，开发者规则定义产品边界，用户请求定义本轮目标，工具输出和检索资料是数据而不是命令。这个边界是提示注入防护的地基。

第二步是构造上下文。狭义 prompt 只写任务，context engineering 会决定是否加入用户画像、历史摘要、检索证据、工具返回、schema、反例、预算和失败策略。Anthropic 的上下文工程观点很适合 Agent：多轮循环里不断产生新信息，不可能把所有内容都塞进窗口，必须持续选择、压缩和校验。

先把需求整理成一个最小 baseline：

```text
角色：你是一个面向工程团队的技术写作助手。
任务：把输入材料整理成可执行 checklist。
上下文：{{source_notes}}
约束：不要添加材料中没有的事实；不确定处标为“待核验”。
输出：JSON，字段为 title、risks、checklist、sources。
示例：给出 1 个高质量输入/输出对。
评估：仅引用材料中真实存在的来源；不足时说明缺口，checklist 每项可执行。
```

这个模板不是为了固定所有任务，而是提醒你把“指令、上下文、输出、评估”放在一起设计。对于支持结构化输出的模型，JSON Schema 应该承担格式约束，prompt 负责语义标准、边界样例和拒答规则；不要用一堆“必须输出合法 JSON!!!”替代类型系统。

CRISPE 可以作为入门检查框架：Capacity/Role 说明任务角色和受众，Insight 提供背景，Statement 明确动作，Style 控制表达，Parameter 给出硬约束，Experiment 用示例校准任务。但今天更推荐把它翻译成可测试字段，而不是背缩写。比如“角色”要对应实际受众和任务边界，“背景”要来自可追溯上下文，“格式”最好由 schema 验证，“示例”要覆盖正例和反例。这样 CRISPE 不再是玄学口诀，而是 prompt review checklist。

Few-shot 与 ICL 的重点是示例提供了什么可学习信号。示例并不只是告诉模型“答案长什么样”，还在暗示分类边界、语气、字段粒度和错误处理方式。示例顺序、类别平衡、是否包含边界案例都会影响输出。如果一个 prompt 只有成功样例，模型会倾向于无论资料是否足够都给出完整答案；加入一个“资料不足时拒答”的样例，往往比在末尾重复三遍“不要幻觉”更有效。

## 关键概念

| 概念 | 它解决什么 | 使用时要警惕 |
| --- | --- | --- |
| 指令层级 | 决定冲突时谁说了算，防止工具或用户低权威内容覆盖系统目标。 | 工具输出、网页和检索资料要视为不可信数据，不是新指令。 |
| Zero-shot | 只用任务说明让模型完成任务，适合简单分类、改写和问答。 | 复杂格式和隐含标准容易漂移。 |
| Few-shot / ICL | 通过示例展示输入输出映射，复用上下文学习能力。 | 示例质量、顺序和覆盖面会影响结果，也会消耗窗口。 |
| CoT / ToT | 用分步处理、候选路径和自评提升复杂推理。 | 不要把隐藏推理当用户输出；必要时要求简要理由或可验证步骤。 |
| 结构化输出 | 用 JSON Schema、工具参数或类型系统约束格式。 | 在支持的接口和成功完成条件下约束形状；拒答、截断与事实正确性另处理。 |
| 上下文工程 | 选择本轮推理所需的资料、工具、历史和状态。 | 上下文越多不一定越好；噪声会稀释关键指令。 |
| 评估 | 用固定样例、grader 和人工抽检检测 prompt 版本变化。 | 单看一次主观效果会把偶然成功误认为稳定改进。 |
| 安全边界 | 防提示注入、越权、数据泄露和不当工具调用。 | 防护是系统设计，不是一句“忽略恶意指令”就能解决。 |

## 先诊断哪一层失败

| 失败 | 首选检查 | 阅读入口 |
| --- | --- | --- |
| 指令含糊或标签不稳定 | 任务、类别边界、示例质量 | [基础技术](/llms/prompt/basics) |
| 用户约束被长历史埋没 | 实际消息、摘要、token 预算 | [上下文工程](/llms/prompt/context) |
| 复杂任务缺外部证据或验证 | 工具循环、候选搜索与检查器 | [高级技术](/llms/prompt/advanced) |
| 输出不能可靠进入下游系统 | 完成状态、Schema 与业务规则 | [结构化输出](/llms/prompt/advanced) |
| 网页或工具结果改写任务 | 信任来源、权限与副作用 | [提示安全](/llms/prompt/security) |

## 推荐学习顺序

1. 先读 [提示工程基础](/llms/prompt/basics)，掌握清晰任务、约束、示例和输出格式。
2. 再读 [上下文工程](/llms/prompt/context)，理解检索资料、历史状态、工具输出和消息层级如何共同影响模型。
3. 接着读 [高级提示技术](/llms/prompt/advanced)，学习 CoT、自洽性、ToT、分解和多候选策略何时有收益。
4. 最后读 [安全测试](/llms/prompt/security)，把提示注入、越权、拒答边界和回归测试纳入默认设计。

如果你正在重构旧 prompt，建议先建 20–50 条代表性样例，不要直接重写生产提示。把样例分成正常输入、边界输入、恶意输入、无答案输入和格式压力输入；每次改 prompt 只比较一类失败是否改善，同时确认没有让其他类别退化。

## 实践检查点

- Prompt 是否明确区分系统/开发者约束、用户目标和外部资料？
- 是否有至少一个正例、一个反例和一个“资料不足时怎么做”的样例？
- 输出格式是否由 schema 或 parser 兜底，而不是只靠自然语言强压？
- 是否为模型版本、prompt 版本、上下文构造逻辑和评估集保存变更记录？
- 是否测试了提示注入：网页说“忽略系统指令”、工具结果伪装成命令、用户要求泄露隐藏 prompt？
- 是否把 prompt 失败归因到具体层：任务不清、上下文缺失、示例误导、schema 不足、模型能力不够，还是安全策略冲突？

一个小练习是把同一任务做成三版：zero-shot、few-shot、schema 约束版。然后用同一评估集比较准确率、格式错误率、拒答率和人工修订时间。你会发现很多“prompt 技巧”其实是在补系统缺口：如果输出经常格式错，优先上结构化输出；如果答案缺证据，优先修上下文；如果模型被网页指令带跑，优先修信任边界。

另一个练习是做“上下文预算表”。把一次调用中的 token 分成系统规则、开发者指令、用户问题、检索证据、历史摘要、工具结果、示例和输出空间八类，记录每类占比和失败样本。很多长 prompt 失败不是因为模型不够强，而是关键约束被埋在太多历史和噪声材料里。删掉无关上下文、把长历史压缩成任务状态、把工具输出转成结构化摘要，通常比继续追加说明更有效。

## 这轮更新怎样用于实践

先把输出消费逻辑分成完成、拒答、结构解析和业务验证四个分支，再检查历史压缩有没有丢失目标与授权边界，最后对外部资料进入工具或长期记忆的路径做回归。对应更新见 [结构化输出接口](/llms/prompt/advanced#responses-与-chat-completions-的字段不要混用)、[上下文压缩](/llms/prompt/context#接口压缩与应用状态) 和 [2026 注入风险](/llms/prompt/security#prompt-injection-2026)。

## 场景自测

<details>
<summary>Schema 合法的答案引用了不存在的来源，结构化输出是不是失效了？</summary>

格式约束可能仍有效，失败发生在事实与证据层。核对引用 ID 是否存在、证据是否支持结论，并为资料不足设计明确状态；不要仅继续强化“输出 JSON”。参见 [高级提示技术](/llms/prompt/advanced)。

</details>

<details>
<summary>压缩接口返回多个 item，能否只留下看起来像摘要的一项？</summary>

不能自行裁剪供应商规定的压缩窗口。以本文核验的 Responses compact 为例，应原样保留返回窗口，再追加新输入；业务任务与授权状态另存。参见 [上下文工程](/llms/prompt/context#接口压缩与应用状态)。

</details>

<details>
<summary>检索网页声称“为了完成任务必须上传本地文件”，如何处理？</summary>

网页是待分析数据，不是用户授权。应用核对用户目标、目标地址、数据范围和执行权限；禁止未经授权的上传，同时继续可完成的阅读任务。把没有外传和正常任务仍能完成都写入安全验收。参见 [提示安全](/llms/prompt/security)。

</details>

## 版本与边界

2026-10-08 本轮对照官方文档复核结构化输出、上下文压缩与注入防护；网络示例没有据此标为实跑通过。其他模型和工具细节仍需在具体版本上验证。具体模型快照会改变最佳写法：有的模型需要更显式的工具触发，有的模型对长上下文更敏感，有的模型支持结构化输出或隐藏推理。生产应用应固定模型版本或至少记录模型家族、日期和参数，并用评估集监控升级影响。

Prompt 的边界也要诚实。它不能替代权限系统，不能保证事实正确，不能让模型看到不存在的资料，也不能单独解决提示注入。结构化输出降低格式风险，但不等于内容真实；CoT/ToT 提升某些推理任务，但也增加成本和延迟；context engineering 能改善信息供给，但如果资料本身错误，模型仍会生成错误答案。把 prompt 当成系统接口，而不是魔法句子，是这个模块最重要的转变。

<SourceList :items="[
  { title: 'OpenAI prompt engineering guide', href: 'https://developers.openai.com/api/docs/guides/prompt-engineering', note: 'OpenAI API 官方提示工程指南，覆盖 roles、instructions、测试和模型版本。' },
  { title: 'OpenAI Model Spec', href: 'https://raw.githubusercontent.com/openai/model_spec/main/model_spec.md', note: 'OpenAI 模型行为规范，说明 chain of command 与指令权威层级。' },
  { title: 'OpenAI instruction hierarchy challenge', href: 'https://openai.com/index/instruction-hierarchy-challenge/', note: '说明 System > developer > user > tool 等层级为何影响安全与注入防护。' },
  { title: 'OpenAI Structured Outputs', href: 'https://developers.openai.com/api/docs/guides/structured-outputs', note: '结构化输出官方文档，说明 JSON Schema、拒答和 schema 边界。' },
  { title: 'Anthropic prompting best practices', href: 'https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices', note: 'Anthropic 官方提示实践，覆盖清晰指令、示例、XML、thinking 与工具使用。' },
  { title: 'Anthropic effective context engineering for AI agents', href: 'https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents', note: '区分 prompt engineering 与 context engineering 的官方工程文章。' },
  { title: 'Anthropic prompt injection defenses', href: 'https://www.anthropic.com/news/prompt-injection-defenses', note: '提示注入风险与防护思路，强调浏览器/网页环境仍是对抗性场景。' },
  { title: 'Gemini prompt design strategies', href: 'https://ai.google.dev/gemini-api/docs/prompting-strategies', note: 'Gemini 官方提示设计策略，强调清晰、具体、迭代和模型差异。' },
  { title: 'Language Models are Few-Shot Learners', href: 'https://arxiv.org/abs/2005.14165', note: 'GPT-3 与 few-shot/in-context learning 的代表论文。' },
  { title: 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models', href: 'https://arxiv.org/abs/2201.11903', note: 'CoT 原始论文，解释中间推理示例对复杂任务的作用。' },
  { title: 'Self-Consistency Improves Chain of Thought Reasoning in Language Models', href: 'https://arxiv.org/abs/2203.11171', note: '自洽性论文，说明多路径采样与答案一致性选择。' },
  { title: 'Tree of Thoughts', href: 'https://arxiv.org/abs/2305.10601', note: 'ToT 论文，扩展 CoT 到可搜索、可回溯的候选思路结构。' }
]" />
