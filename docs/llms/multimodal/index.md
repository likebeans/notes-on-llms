---
title: 多模态大模型全景
description: 从视觉编码、模态连接、指令微调到理解、生成、部署评测的学习单元。
pageType: article
module: multimodal
updated: '2026-08-26'
contentStatus: verified
tags:
  - multimodal
level: advanced
prerequisites:
  - /guide/prerequisites
reviewed: '2026-08-26'
techVersion: 2026-08（视觉语言模型、多模态生成与评测）
---

# 多模态大模型全景

<LearningObjectives :items="[
  '能够把多模态系统拆成编码器、连接器、语言/统一模型、生成器、评估与部署边界。',
  '能够区分视觉理解、跨模态检索、图像生成、视频/音频理解和多模态 Agent 的不同训练目标。',
  '能够设计带证据区域、任务标签和安全标注的多模态样本，避免只凭模型描述是否流畅来判断质量。'
]" />

## 这个模块解决什么问题

多模态大模型要解决的是“让模型在同一个任务里理解、对齐和生成多种信息形态”。旧稿从图像编码器、文本编码器、连接层和 LLM 讲起，这条主线仍然正确：模型需要把图像、视频、音频、文本等不同模态转成可比较或可共同推理的表示，再通过连接器、交叉注意力、Q-Former、投影层或统一 token 空间送入语言模型或生成模型。真正困难的不是把图片塞进模型，而是让模型知道哪些像素、文本、时间片或区域证据支撑它的结论。

在实践里，多模态至少包含两类不同目标。第一类是理解：图像问答、OCR、图表分析、视频问答、文档理解、遥感/医学图像辅助分析等，输出通常是文本、结构化答案或行动计划。第二类是生成：文本生成图像、图像编辑、视频生成、音频生成、3D 或跨模态创作，输出本身是另一个模态。CLIP 代表了“用自然语言监督学习可迁移视觉表示”的重要路线；BLIP-2、LLaVA 等工作展示了如何用冻结视觉编码器、连接模块和视觉指令数据把视觉能力接入 LLM；Latent Diffusion 和 DALL·E 3 则代表了图像生成路线中从像素空间效率到高质量 caption 对齐的演进。

旧稿里有一些具体榜单数字和模型排名，这次不继续保留为主线结论。多模态 benchmark 更新非常快，数据污染、提示格式、图像分辨率、工具使用和安全过滤都会影响结果。比“谁在某个日期的榜单最高”更稳定的是架构判断：视觉编码器负责把输入变成高层表示；连接器负责压缩和对齐；语言模型负责指令跟随、推理和输出；生成器负责把语义条件变成图像或视频；评估负责确认答案是否基于正确视觉证据，而不是只看语言是否自信。

学习本模块时，可以把多模态系统当成一条证据与表示的管线：输入先被分块、采样或编码；模型把模态表示对齐到共享语义空间；指令调优让模型学会按照人类问题使用这些表示；理解任务需要引用或定位证据；生成任务需要检查 prompt following、真实性、安全和版权风险；部署阶段还要处理分辨率、帧数、上下文长度、延迟、缓存、审计和人工复核。

## 核心工作流

```mermaid
flowchart LR
  Media[图像 / 视频 / 音频 / 文档] --> Encoder[模态编码器：ViT / CLIP / Audio]
  Encoder --> Connector[连接器：Projection / Q-Former / Cross-Attention]
  Text[文本指令与上下文] --> Core[语言模型或统一多模态模型]
  Connector --> Core
  Core --> Understand[理解：问答 / OCR / 图表 / 视频推理]
  Core --> Generate[生成：图像 / 视频 / 编辑]
  Understand --> Eval[评估：证据 / 安全 / 业务]
  Generate --> Eval
  Eval --> Deploy[部署：延迟 / 成本 / 审计]
```

这条链路里的关键工程问题是“信息压缩”。一张高分辨率图像、一个长视频或一份扫描文档都可能远大于模型上下文窗口。ViT 把图像切成 patch 再做 Transformer 编码；CLIP 用图文对比学习把图像和文本拉到共享空间；BLIP-2 的 Q-Former 用少量 query token 从冻结视觉编码器中抽取对语言模型有用的信息；LLaVA 用视觉指令数据训练模型把图像表示转化为对话能力；Gemini 1.5 把长上下文扩展到跨文档、音频和视频场景。这些路线看似不同，本质都在回答同一个问题：哪些视觉/听觉/文档证据应该进入模型推理，哪些可以被压缩或丢弃？

一个可复用的多模态训练或评估样本，最好同时保留输入、问题、答案、证据和风险标签：

```yaml
id: chart_qa_042
media:
  type: image
  uri: s3://eval/charts/revenue_q4.png
question: "图中 Q4 收入相比 Q3 增长了多少？"
answer: "Q4 比 Q3 增长 18%。"
evidence:
  regions:
    - label: "Q3 bar"
      bbox: [120, 310, 180, 460]
    - label: "Q4 bar"
      bbox: [210, 250, 270, 460]
risk_tags: ["chart_reasoning", "numeric_accuracy"]
```

这个模板比“图片 + 问题 + 答案”更有价值，因为它让错误可诊断：模型是没看见图、读错坐标、算错比例、忽略单位，还是语言上编造了一个合理答案？对于视频，可以把 `regions` 换成时间片和帧号；对于文档，可以换成页码、表格单元格或 OCR token；对于生成任务，可以把 evidence 改成 prompt constraints、negative constraints、人工偏好和安全标签。

## 关键概念

| 概念 | 它解决什么 | 使用时要警惕 |
| --- | --- | --- |
| 视觉编码器 | 把像素、patch 或视频帧转成高层表示。 | 分辨率、裁剪、帧采样和 OCR 会显著影响结果。 |
| CLIP 式对齐 | 通过图文对比学习获得可迁移的跨模态表示。 | 相似度不是因果证据；零样本能力也会受数据偏差影响。 |
| 连接器 | 用投影、Q-Former 或交叉注意力把视觉表示接入 LLM。 | 连接器压缩过强会丢小字、空间关系和细粒度计数。 |
| 视觉指令微调 | 用图像-指令-回答数据让模型学会视觉对话。 | 合成指令可能让模型语言流畅但视觉 grounding 不足。 |
| 统一多模态模型 | 在同一模型或上下文中处理文本、图像、音频、视频。 | 能力边界依赖输入长度、采样策略和安全策略，不能只看 demo。 |
| 扩散生成 | 通过噪声去除或潜空间生成输出图像/视频。 | prompt following、文字渲染、版权、人物相似性和安全过滤都要评估。 |
| 多模态评估 | 同时检查答案、证据区域、推理步骤、安全与业务可用性。 | 语言判断器容易被流畅回答骗过，需要人类校准和结构化标签。 |

## 推荐学习顺序

1. 先读 [视觉编码器](/llms/multimodal/vision-encoder)，理解 ViT、CLIP、patch、分辨率和图文表示空间。
2. 再读 [模态连接器](/llms/multimodal/connector)，比较线性投影、Q-Former、cross-attention 和 token 压缩。
3. 接着读 [架构范式](/llms/multimodal/architecture) 与 [数据构造](/llms/multimodal/data)，把预训练、图文对齐、视觉指令数据和安全标注连起来。
4. 然后读 [Diffusion 与生成](/llms/multimodal/diffusion)，理解 latent diffusion、caption 重写、prompt following 和生成安全。
5. 再读 [多模态 RAG 与 Agent](/llms/multimodal/rag-agent)，学习如何让模型使用图像、文档、工具和外部知识。
6. 最后读 [统一模型](/llms/multimodal/unified) 与 [部署实践](/llms/multimodal/deployment)，把长上下文、多模态 serving、评估、审计和成本放进生产系统。

如果你刚开始做项目，建议从“理解型任务”而不是“全能多模态 Agent”入手：选一个真实场景，例如票据抽取、图表问答、界面截图诊断或产品图片审核；构造 100 条带证据标注的样本；分别测试 OCR + 文本模型、视觉语言模型、RAG + 图像描述三种 baseline。等你知道错误主要来自 OCR、视觉 grounding、数字推理还是业务规则后，再决定是否需要微调、工具调用或多模态 RAG。

## 实践检查点

- 输入预处理是否记录了原始分辨率、裁剪、缩放、帧采样、OCR 版本和文档页码？
- 评估样本是否包含证据区域、时间片、表格单元格或引用页，而不只是最终答案？
- 是否覆盖了小字、遮挡、低清、图表、表格、跨页文档、长视频和“无法判断”的负例？
- 生成任务是否分别评估 prompt following、主体一致性、文字渲染、风格约束、安全与版权风险？
- 多模态 Agent 是否限制了工具权限，并记录每次截图、文件读取、搜索、点击或外部写入？
- serving 配置是否按真实图片尺寸、视频长度、并发和上下文长度测试，而不是只用压缩 demo 图？

一个很实用的对照实验是：同一批图表问答样本，分别让模型直接看图回答、先 OCR/解析成表格再回答、以及图像 + OCR 文本共同输入。直接看图可能擅长布局理解但容易读错小字；OCR 路线可能数字更准但丢失视觉关系；混合路线成本更高但可解释性更好。不要提前假设哪条路线最好，用 evidence-level 标签统计错误来源，结果会比单一总分更能指导架构。

## 版本与边界

截至 2026-08，多模态的稳定共识是：视觉编码、跨模态对齐、指令微调、生成建模和任务级评估都必须分开看。ViT、CLIP、BLIP-2、LLaVA、Latent Diffusion、DALL·E 3、Gemini 1.5、MMMU 等资料代表了几条重要路线，但它们并不构成固定终局。新模型的输入模态、上下文长度、视频理解、图像生成和工具使用能力仍在快速变化，因此本文不写无日期的榜单结论，也不把某个公开 benchmark 当成生产质量的替代品。

多模态系统还有几个硬边界。第一，模型会用语言填补视觉不确定性，答案自信不代表看对了证据。第二，生成模型的安全与版权风险不能靠 prompt 一次性解决，需要内容过滤、来源审查、水印或人工复核。第三，医学、法律、金融、安防等高风险场景必须把模型输出当作辅助线索，并保留可追溯证据与人类责任链。第四，多模态 Agent 的外部动作比纯文本问答风险更高；它看到屏幕、文件或摄像头内容时，权限、审计和最小暴露原则要先于模型能力炫技。

<SourceList :items="[
  { title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', href: 'https://arxiv.org/abs/2010.11929', note: 'ViT 论文，理解图像 patch 与 Transformer 视觉编码。' },
  { title: 'Learning Transferable Visual Models From Natural Language Supervision', href: 'https://arxiv.org/abs/2103.00020', note: 'CLIP 论文，说明图文对比学习和可迁移视觉表示。' },
  { title: 'BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models', href: 'https://arxiv.org/abs/2301.12597', note: 'BLIP-2 论文，介绍 Q-Former 连接冻结视觉编码器与 LLM。' },
  { title: 'Visual Instruction Tuning', href: 'https://arxiv.org/abs/2304.08485', note: 'LLaVA 论文，展示视觉指令数据与视觉对话模型训练。' },
  { title: 'High-Resolution Image Synthesis with Latent Diffusion Models', href: 'https://arxiv.org/abs/2112.10752', note: 'Latent Diffusion 论文，理解潜空间扩散生成路线。' },
  { title: 'DALL·E 3 technical report', href: 'https://cdn.openai.com/papers/dall-e-3.pdf', note: 'OpenAI 技术报告，讨论通过更高质量 caption 改善图像生成 prompt following。' },
  { title: 'GPT-4V(ision) System Card', href: 'https://cdn.openai.com/papers/GPTV_System_Card.pdf', note: 'OpenAI 系统卡，适合理解视觉模型能力边界与安全评估。' },
  { title: 'Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context', href: 'https://arxiv.org/abs/2403.05530', note: 'Gemini 1.5 技术报告，覆盖长上下文、多文档、音频和视频理解。' },
  { title: 'MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI', href: 'https://arxiv.org/abs/2311.16502', note: 'MMMU 论文，代表多学科多模态理解与推理评测。' }
]" />
