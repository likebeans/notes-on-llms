---
title: RAG 技术全景
description: 检索增强生成从数据、检索、重排到评估与生产实践的学习单元。
pageType: article
module: rag
updated: '2026-08-26'
contentStatus: verified
tags:
  - rag
level: intermediate
prerequisites:
  - /llms/prompt/
reviewed: '2026-08-26'
techVersion: 2026-08（RAG、GraphRAG、评估）
---

# RAG 技术全景

<LearningObjectives :items="[
  '能够把一个 RAG 系统拆成数据准备、索引、检索、重排、生成、评估与运维七个可观察环节。',
  '能够根据问题类型选择稀疏检索、向量检索、混合检索、late interaction 或 GraphRAG，而不是把“向量库”当成唯一答案。',
  '能够设计一套离线评估与线上监控指标，定位回答错误到底来自语料、召回、排序、上下文组装还是生成阶段。'
]" />

## 这个模块解决什么问题

RAG（Retrieval-Augmented Generation）要解决的不是“让模型背更多知识”，而是把模型生成能力和外部可更新知识源连接起来。原始 RAG 论文把语言模型的参数化记忆和外部非参数化记忆结合：模型负责理解、推理和表达，检索系统负责把当前任务所需的证据带到上下文里。这个分工在今天仍然成立，只是组件更细、评估更工程化，数据源也从纯文本扩展到了表格、代码、结构化知识图谱和业务系统。

最常见的误区是把 RAG 简化成“切文档 + embedding + top-k + 拼 prompt”。这套最小实现适合 demo，却很难支撑生产：文档边界会切断语义，embedding 召回会漏掉关键词强相关材料，top-k 会把相似但无用的段落挤进上下文，模型会在证据不足时编造答案，线上更新又可能让索引和权限状态不同步。一个可靠的 RAG 系统必须把“检索是否命中”“证据是否足够”“回答是否忠于证据”“引用是否可追溯”拆开看。

旧稿里有一个值得保留的经验判断：embedding 的本质是把查询和文档投到同一高维空间里，再用距离度量做近邻搜索；但“向量距离近”并不等于“这段材料能回答问题”。所以生产里不要迷信单一路径：专有名词、报错码、合同条款常常需要 BM25 这类关键词召回；语义改写后的自然语言问题适合密集检索；召回集变小之后再交给 rerank 或 cross-encoder 判断“能不能回答”。旧稿提到稀疏/密集加权常被设成某个经验比例，这个提醒仍然有用：权重不是标准答案，应该由评估集和线上反馈决定。

学习本模块时，可以把 RAG 当成一条证据供应链：用户问题进入系统后，先被改写、路由或分解；随后在不同索引中召回候选证据；再通过重排、过滤和上下文压缩控制质量与成本；最后由模型生成答案，并把失败样本回流到评估集、索引策略和数据治理流程里。

## 核心工作流

```mermaid
flowchart LR
  Q[用户问题] --> Rewrite[查询改写与路由]
  Rewrite --> Retrieve[检索：BM25/向量/图]
  Retrieve --> Rerank[重排与过滤]
  Rerank --> Context[上下文组装]
  Context --> LLM[生成回答]
  LLM --> Cite[引用与置信度]
  Cite --> Eval[评估与反馈]
  Eval --> Rewrite
```

这条链路的关键不是每一步都堆最复杂的算法，而是让每一步都有明确职责和可验证输出。查询改写要回答“用户真正想问什么”；检索要回答“候选证据是否覆盖答案”；重排要回答“最有用的证据是否排在前面”；上下文组装要回答“模型看到的信息是否足够且不互相污染”；生成要回答“答案是否只基于证据”；评估要回答“系统哪里退化了”。

在复杂知识库里，RAG 往往不止一条路径。事实型问题适合短 chunk + 混合检索；流程型问题需要保留章节结构；跨文档综合问题可能需要 GraphRAG 这类先抽取实体关系、再做全局摘要或社区检索的方案；代码、日志、合同等材料还需要语法、权限和时效过滤。真正的架构决策通常来自评估集，而不是来自某个工具的默认模板。

旧稿中关于 ETL、OCR、分块和 rerank 的长解释被拆到后续文章里，但总览页仍建议保留一个最小可复现 trace。它不要求你一开始就买复杂平台，只要求每次问答都能把链路摊开：

```text
query: 用户原始问题
rewrite: 系统改写后的检索问题
retrieval: [关键词命中文档, 向量命中文档, 图检索命中文档]
rerank: 进入上下文的证据片段与分数
context: 实际拼入模型的上下文窗口
answer: 模型回答、引用、拒答理由
label: 人工标注的命中/相关/忠实/可用结果
```

这个模板继承了旧稿“数据质量决定上限”的判断。文档解析错了，后面再强的模型也只能漂亮地复述错误；chunk 太小会丢上下文，chunk 太大会塞进噪声；上下文压缩如果只看词频，可能删掉关键限定条件；GraphRAG 如果实体抽取错了，会把错误传播到社区摘要。trace 的作用就是让你知道该修哪一层，而不是把所有问题都归咎于模型。

同时保留旧稿“Prompt 构建”这个可落地动作。一个最小 RAG 回答模板可以这样写，先让模型承认上下文边界，再要求引用证据；这比只把 top-k 文档粗暴拼到问题后面可靠得多：

```text
你是一个基于证据回答问题的助手。

已检索到的资料：
{{context_chunks_with_source_ids}}

请回答用户问题：{{user_query}}

规则：
1. 只使用“已检索到的资料”中的信息回答。
2. 每个关键结论后标注来源编号，例如 [doc-3]。
3. 如果资料不足以回答，直接说“当前资料不足”，并说明缺少哪类证据。
4. 不要编造未出现在资料中的日期、数字、人物或政策。
```

这个模板不是最终答案，而是 baseline。后续优化可以把 `context_chunks_with_source_ids` 换成 rerank 后片段、图社区摘要或压缩上下文；也可以把第 3 条拒答规则做成单独 grader。保留它的价值在于：当答案出错时，你能区分是证据没召回、证据被错误压缩，还是 Prompt 没有把“只基于证据回答”约束住。

## 关键概念

| 概念 | 它解决什么 | 使用时要警惕 |
| --- | --- | --- |
| 参数化记忆 vs 非参数化记忆 | 模型权重负责通用能力，外部索引负责可更新事实和私有知识。 | RAG 不能自动修复模型推理错误，也不能替代数据治理。 |
| 稀疏检索 | BM25 等关键词方法对专有名词、编号、报错信息很强。 | 语义改写能力弱，问法变化大时容易漏召回。 |
| 密集检索 | DPR 类双塔或 embedding 检索能捕捉语义相似问题。 | 相似不等于相关；负样本、领域术语和切分策略会显著影响召回。 |
| Late interaction / 重排 | ColBERT 等方法保留 token 级交互，cross-encoder 重排能提高 top-k 质量。 | 成本和延迟更高，通常放在候选集变小之后。 |
| GraphRAG | 把实体、关系和社区摘要纳入检索，适合跨文档综合和“全局性”问题。 | 构图、抽取和摘要会引入新误差；当前实现生态变化快，应先验证收益。 |
| RAG 评估 | 分别度量检索命中、上下文相关性、答案忠实度、引用正确性和端到端任务成功率。 | 单看最终答案分数会掩盖召回失败、排序失败和生成幻觉的不同原因。 |

## 推荐学习顺序

1. 先读 [范式演进](/llms/rag/paradigms)，建立从原始 RAG、检索-生成融合到图增强检索的时间线。
2. 再读 [文档切分](/llms/rag/chunking) 与 [Embedding](/llms/rag/embedding)，理解索引质量为什么通常决定上限。
3. 接着读 [向量数据库](/llms/rag/vector-db)，把索引、过滤、权限、增量更新和成本放进同一张图。
4. 然后读 [检索策略](/llms/rag/retrieval) 与 [重排序](/llms/rag/rerank)，学习如何从“召回更多”走向“把正确证据放到模型面前”。
5. 最后读 [评估](/llms/rag/evaluation) 和 [生产实践](/llms/rag/production)，把离线数据集、线上监控、回归测试和灰度发布串起来。
6. 如果你想看旧版长文整理，可以把 [CSDN 文章合集](/llms/rag/csdn_articles) 当作归档材料阅读；它不作为当前主线结论的唯一依据。

一条实用路线是：先用一个小而真实的问答集跑通 baseline，不急着上 GraphRAG 或复杂 Agent；当失败样本显示“召回不到”时改索引和检索，当显示“召回到了但没用好”时改重排和上下文组装，当显示“证据足够但答案仍错”时再改提示、生成约束或模型。

## 实践检查点

- 能否为同一批问题保存 query、召回文档、重排分数、最终上下文、模型答案和人工标签？如果不能，后续优化会变成玄学。
- 是否准备了覆盖事实查询、对比查询、跨文档综合、无答案问题和权限受限问题的评估集？
- 是否同时跑了关键词、向量和混合检索 baseline，并记录各自的召回率、延迟和成本？
- 是否把“无证据时拒答”作为正向能力评估，而不是只奖励模型给出流畅答案？
- 是否有索引更新、文档删除、权限变更、embedding 模型升级后的回归检查？

一个很实用的练习是做三组 ablation：第一组只用 BM25，第二组只用 embedding，第三组使用混合检索加 rerank。不要只比较“答案看起来哪个好”，而要记录每个问题的 top-k 是否包含标准证据、正确证据排在第几位、上下文 token 消耗多少、模型是否引用了错误材料。你会很快看到：有些失败是 query 太口语化，需要重写；有些失败是文档切分把标题和正文拆散；有些失败是 top-k 太小；还有些失败是模型拿到了证据却忽略了否定条件。这个练习比盲目替换向量库更能提高系统质量。

## 版本与边界

截至 2026-08，RAG 的基本分工仍然稳定：外部检索提供可追溯证据，生成模型负责把证据转化成答案。但工具生态变化很快，尤其是 GraphRAG、评估平台和托管向量数据库。微软 GraphRAG 的开源仓库已明确偏研究项目和维护模式，因此本模块把 GraphRAG 视为一种适合特定问题形态的架构模式，而不是默认生产依赖。

RAG 也不是万能补丁。它不能保证知识库内容正确，不能绕过权限治理，不能消除模型在复杂推理中的错误，也不能在证据互相冲突时自动做出业务判断。生产系统最好把 RAG 输出定位为“带证据的候选答案”，并通过引用、置信度、人工复核、拒答策略和持续评估来管理风险。

<SourceList :items="[
  { title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', href: 'https://arxiv.org/abs/2005.11401', note: 'RAG 原始论文，提出参数化记忆与外部非参数化记忆结合。' },
  { title: 'Dense Passage Retrieval for Open-Domain Question Answering', href: 'https://arxiv.org/abs/2004.04906', note: 'DPR 论文，理解双塔密集检索在开放域问答中的作用。' },
  { title: 'ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT', href: 'https://arxiv.org/abs/2004.12832', note: 'late interaction 检索代表作，解释 token 级交互为何能提升排序质量。' },
  { title: 'ColBERTv2: Efficient and Effective Retrieval via Lightweight Late Interaction', href: 'https://arxiv.org/abs/2112.01488', note: 'ColBERT 后续版本，关注压缩、效率与质量平衡。' },
  { title: 'Microsoft GraphRAG documentation', href: 'https://microsoft.github.io/graphrag/', note: 'GraphRAG 官方文档，适合了解索引、查询和全局检索工作流。' },
  { title: 'Microsoft GraphRAG repository', href: 'https://github.com/microsoft/graphrag', note: '官方仓库；截至本次审查，项目定位偏研究与维护模式。' },
  { title: 'OpenAI Evals', href: 'https://github.com/openai/evals', note: '开源评估框架，可参考数据集、grader 和回归评估组织方式。' },
  { title: 'LangSmith RAG evaluation tutorial', href: 'https://docs.langchain.com/langsmith/evaluate-rag-tutorial', note: '官方教程，展示如何分解 RAG 的检索、上下文和答案评估。' }
]" />
