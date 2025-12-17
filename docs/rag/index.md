---
title: RAG 知识体系
description: 检索增强生成（RAG）技术全景图谱
---

# RAG 知识体系

> 从入门到精通，掌握 RAG 系统的设计与实现

## 🗺️ RAG 知识图谱

<div class="knowledge-map">
  <div class="map-center">
    <span class="map-title">RAG</span>
  </div>
  <div class="map-branches">
    <div class="branch branch-1">
      <div class="branch-title">📚 基础概念</div>
      <ul>
        <li>什么是 RAG</li>
        <li>为什么需要 RAG</li>
        <li>RAG vs 微调</li>
        <li>RAG 演进历程</li>
      </ul>
    </div>
    <div class="branch branch-2">
      <div class="branch-title">⚙️ 核心流程</div>
      <ul>
        <li><strong>离线索引</strong>：解析 → 切分 → 向量化 → 入库</li>
        <li><strong>在线检索</strong>：Query → 检索 → Rerank → 生成</li>
      </ul>
    </div>
    <div class="branch branch-3">
      <div class="branch-title">🔧 关键技术</div>
      <ul>
        <li><strong>Embedding</strong>：原理 · 选型 · 中文优化</li>
        <li><strong>Chunking</strong>：固定 · 语义 · 层次切分</li>
        <li><strong>检索</strong>：稠密 · 稀疏 · 混合</li>
        <li><strong>Rerank</strong>：Cross-Encoder · LLM</li>
      </ul>
    </div>
    <div class="branch branch-4">
      <div class="branch-title">🚀 高级架构</div>
      <ul>
        <li>Query 改写 / HyDE</li>
        <li>多路召回</li>
        <li>GraphRAG</li>
        <li>Agentic RAG</li>
      </ul>
    </div>
    <div class="branch branch-5">
      <div class="branch-title">🏭 工程实践</div>
      <ul>
        <li>评估指标</li>
        <li>生产部署</li>
        <li>性能优化</li>
      </ul>
    </div>
  </div>
</div>

<style>
.knowledge-map {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px;
  padding: 2rem;
  margin: 2rem 0;
  color: white;
}
.map-center {
  text-align: center;
  margin-bottom: 1.5rem;
}
.map-title {
  display: inline-block;
  background: white;
  color: #667eea;
  font-size: 2rem;
  font-weight: bold;
  padding: 1rem 2rem;
  border-radius: 50px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.map-branches {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}
.branch {
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 1rem;
  transition: transform 0.2s;
}
.branch:hover {
  transform: translateY(-4px);
}
.branch-title {
  font-weight: bold;
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid rgba(255,255,255,0.3);
}
.branch ul {
  margin: 0;
  padding-left: 1.2rem;
  font-size: 0.9rem;
}
.branch li {
  margin: 0.3rem 0;
}
</style>

---

## 📊 核心流程

::: info 离线索引
📄 **文档** → 🔪 **切分** → 🧮 **Embedding** → 💾 **向量库**
:::

::: tip 在线查询
❓ **Query** → 🧮 **Embedding** → 🔍 **检索** → 📊 **Rerank** → 🤖 **LLM** → 💬 **答案**
:::

---

## � 技术大纲

### 一、基础概念

| 主题 | 核心内容 |
|------|----------|
| **RAG 定义** | 检索增强生成 = 外部知识 + LLM 推理 |
| **解决的问题** | 知识过时、幻觉、私有数据、上下文限制 |
| **RAG vs 微调** | RAG 更新快/成本低，微调效果深/需数据 |
| **演进历程** | Naive → Advanced → Modular → Agentic |

### 二、关键技术模块

#### 2.1 文档处理

| 阶段 | 技术要点 | 难点 |
|------|----------|------|
| **文档解析** | PDF/Word/HTML、OCR、表格识别 | 复杂布局保持 |
| **文档切分** | 固定长度、语义切分、递归切分、父子分段 | 粒度平衡 |
| **元数据提取** | 标题、时间、来源、层级结构 | 结构化保持 |

#### 2.2 Embedding 技术

::: details 模型演进路线
**Word2Vec** → **GloVe** → **BERT** → **专用 Embedding**（BGE、M3E、text-embedding-3）
:::

| 选型要点 | 说明 |
|----------|------|
| 维度大小 | 768/1024/1536，越大越精准但越慢 |
| 中文支持 | BGE、M3E、text2vec 等 |
| 领域适配 | 通用 vs 垂直领域微调 |

#### 2.3 检索策略

| 策略 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| **稠密检索** | 向量相似度 | 语义理解强 | 依赖 Embedding 质量 |
| **稀疏检索** | BM25 词频统计 | 精确匹配好 | 缺乏语义理解 |
| **混合检索** | 稠密 + 稀疏融合 | 兼顾两者 | 需要权重调优 |

#### 2.4 重排序（Rerank）

::: warning 粗排 → 精排
**召回 Top-100** → Reranker（Cross-Encoder / LLM / BGE-Reranker）→ **精排 Top-10** → 送入 LLM
:::

### 三、高级架构

#### 3.1 Query 优化

| 技术 | 原理 | 效果 |
|------|------|------|
| **Query 改写** | LLM 重写用户问题 | 提升检索准确率 |
| **HyDE** | 先生成假设答案再检索 | 跨越 Query-Doc 语义鸿沟 |
| **多 Query** | 分解为多个子问题 | 覆盖更多相关文档 |
| **Step-back** | 抽象为更通用问题 | 获取背景知识 |

#### 3.2 GraphRAG vs 传统 RAG

<div class="compare-box">
  <div class="compare-item">
    <div class="compare-title">传统 RAG</div>
    <div class="compare-flow">文档 → 向量化 → 相似度检索 → <strong>单跳检索</strong></div>
  </div>
  <div class="compare-vs">VS</div>
  <div class="compare-item highlight">
    <div class="compare-title">GraphRAG</div>
    <div class="compare-flow">文档 → 实体抽取 → 关系构建 → 知识图谱 → <strong>多跳推理</strong></div>
  </div>
</div>

<style>
.compare-box {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin: 1.5rem 0;
  flex-wrap: wrap;
}
.compare-item {
  flex: 1;
  min-width: 200px;
  padding: 1rem;
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
}
.compare-item.highlight {
  background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
  border-color: #667eea;
}
.compare-title {
  font-weight: bold;
  margin-bottom: 0.5rem;
  color: var(--vp-c-brand);
}
.compare-flow {
  font-size: 0.9rem;
}
.compare-vs {
  font-weight: bold;
  color: var(--vp-c-text-2);
}
</style>

#### 3.3 Agentic RAG

::: tip Agent 循环
**用户问题** → Agent 任务分解 → 检索决策 → 多轮检索 → 结果综合 → 自我反思 → ✅ 满意则输出 / ❌ 不满意则继续检索
:::

### 四、工程实践

#### 4.1 评估体系

| 维度 | 指标 | 说明 |
|------|------|------|
| **检索质量** | Recall@K, MRR, NDCG | 检索器独立评估 |
| **生成质量** | Faithfulness, Relevance | 生成器独立评估 |
| **端到端** | Answer Accuracy, F1 | 整体效果评估 |

#### 4.2 常见问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 检索不准 | Embedding 不适配 | 换模型/微调 |
| 答案幻觉 | 检索内容不相关 | 加 Rerank/过滤 |
| 上下文丢失 | 切分粒度不当 | 父子分段/重叠切分 |
| 响应慢 | 检索/生成延迟 | 缓存/并行/量化 |

---

## 📚 学习路线

<div class="learning-path">
  <div class="path-step step-1">
    <div class="step-num">1</div>
    <div class="step-title">入门</div>
    <ul>
      <li>理解 RAG 概念</li>
      <li>跑通最小 Demo</li>
    </ul>
  </div>
  <div class="path-arrow">→</div>
  <div class="path-step step-2">
    <div class="step-num">2</div>
    <div class="step-title">进阶</div>
    <ul>
      <li>Embedding 原理</li>
      <li>切分策略</li>
      <li>检索优化</li>
    </ul>
  </div>
  <div class="path-arrow">→</div>
  <div class="path-step step-3">
    <div class="step-num">3</div>
    <div class="step-title">精通</div>
    <ul>
      <li>GraphRAG</li>
      <li>Agentic RAG</li>
      <li>评估体系</li>
      <li>生产部署</li>
    </ul>
  </div>
</div>

<style>
.learning-path {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  margin: 2rem 0;
  flex-wrap: wrap;
  justify-content: center;
}
.path-step {
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  padding: 1rem;
  min-width: 150px;
  border: 2px solid var(--vp-c-border);
}
.step-1 { border-color: #10b981; }
.step-2 { border-color: #3b82f6; }
.step-3 { border-color: #8b5cf6; }
.step-num {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--vp-c-brand);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  margin-bottom: 0.5rem;
}
.step-1 .step-num { background: #10b981; }
.step-2 .step-num { background: #3b82f6; }
.step-3 .step-num { background: #8b5cf6; }
.step-title {
  font-weight: bold;
  margin-bottom: 0.5rem;
}
.path-step ul {
  margin: 0;
  padding-left: 1rem;
  font-size: 0.85rem;
}
.path-arrow {
  font-size: 1.5rem;
  color: var(--vp-c-text-2);
  align-self: center;
}
</style>

---

## 📖 我的 RAG 系列文章

### 🎯 RAG 综述与原理

| 文章 | 简介 |
|------|------|
| [检索增强生成（RAG）综述：技术范式、核心组件与未来展望](https://dd-ff.blog.csdn.net/article/details/149274498) | Naive → Advanced → Modular RAG 三种范式详解 |
| [从"失忆"到"过目不忘"：RAG技术如何给LLM装上"外挂大脑"？](https://dd-ff.blog.csdn.net/article/details/149348018) | RAG 入门必读，理解 RAG 解决的核心问题 |
| [高级RAG技术全景：从原理到实战，解锁大模型应用的进阶密码](https://dd-ff.blog.csdn.net/article/details/149396526) | 查询转换、多源检索、索引优化等高级技巧 |

### 📄 文档处理与切分

| 文章 | 简介 |
|------|------|
| [解锁RAG效能：15种分块策略秘籍（附实战案例）](https://dd-ff.blog.csdn.net/article/details/149529161) | 15种实用分块策略，含应用场景和示例 |
| [超越纯文本：解锁高级RAG中复杂文档预处理的艺术](https://dd-ff.blog.csdn.net/article/details/152045489) | 文档布局分析、OCR、表格识别等预处理技术 |
| [从"拆文档"到"通语义"：RAG+知识图谱如何破解大模型"失忆+幻觉"难题？](https://dd-ff.blog.csdn.net/article/details/149354855) | 父子分段、层次结构、知识图谱增强 |

### 🧮 Embedding 技术

| 文章 | 简介 |
|------|------|
| [从意义到机制：深入剖析Embedding模型原理及其在RAG中的作用](https://dd-ff.blog.csdn.net/article/details/152809855) | Word2Vec → GloVe → BERT 演进路线 |
| [从潜在空间到实际应用：Embedding模型架构与训练范式的综合解析](https://dd-ff.blog.csdn.net/article/details/152815637) | Transformer 架构、孪生网络、池化技术 |
| [从文本到上下文：深入解析Tokenizer、Embedding及高级RAG架构的底层原理](https://dd-ff.blog.csdn.net/article/details/152819135) | Tokenizer + Embedding + RAG 完整技术栈 |

### 🔍 检索与向量数据库

| 文章 | 简介 |
|------|------|
| [RAGFlow的检索神器-HNSW：高维向量空间中的高效近似最近邻搜索算法](https://dd-ff.blog.csdn.net/article/details/149275016) | HNSW 算法原理，毫秒级向量检索 |

### 🚀 高级架构

| 文章 | 简介 |
|------|------|
| [GraphRAG 技术教程：从核心概念到高级架构](https://dd-ff.blog.csdn.net/article/details/154530805) | 知识图谱增强 RAG，多跳推理 |
| [OpenAI Agent 工具全面开发者指南——从 RAG 到 Computer Use](https://dd-ff.blog.csdn.net/article/details/154445828) | OpenAI file_search 托管式 RAG |

### 📊 评估与反思

| 文章 | 简介 |
|------|------|
| [检索增强生成（RAG）系统综合评估：从核心指标到前沿框架](https://dd-ff.blog.csdn.net/article/details/152823514) | RAGAs、ARES、TruLens 评估框架 |
| [别再卷了！你引以为傲的 RAG，正在杀死你的 AI 创业公司](https://dd-ff.blog.csdn.net/article/details/150944979) | RAG 架构反思，长上下文时代的思考 |
| [LLM 上下文退化：当越长的输入让AI变得越"笨"](https://dd-ff.blog.csdn.net/article/details/149531324) | 上下文长度与 RAG 的权衡 |

---

## 🔗 章节导航

| 章节 | 内容 | 状态 |
|------|------|------|
| [RAG范式演进](/rag/paradigms) | Naive→Advanced→Modular→Graph→Agentic | ✅ |
| [文档切分](/rag/chunking) | 15种切分策略详解 | ✅ |
| [Embedding](/rag/embedding) | 向量化原理与模型选型 | ✅ |
| [向量数据库](/rag/vector-db) | HNSW算法/Chroma/Milvus/Qdrant | ✅ |
| [检索策略](/rag/retrieval) | 稠密/稀疏/混合检索 | ✅ |
| [重排序](/rag/rerank) | Cross-Encoder/LLM重排序 | ✅ |
| [RAG评估](/rag/evaluation) | RAGAs/ARES评估框架 | ✅ |
| [生产实践](/rag/production) | 架构设计/部署/监控/安全 | ✅ |

---

## 🌐 外部学习资源

### 权威综述与论文

| 资源 | 说明 |
|------|------|
| [RAG技术的5种范式](https://hub.baai.ac.cn/view/43613) | 智源社区：NaiveRAG→AgenticRAG完整梳理 |
| [Searching for Best Practices in RAG](https://arxiv.org/abs/2407.01219) | EMNLP 2024：RAG最佳实践研究 |
| [GraphRAG综述论文](https://arxiv.org/abs/2408.08921) | 知识图谱增强RAG系统性综述 |

### 开源框架与工具

| 工具 | 用途 | 链接 |
|------|------|------|
| **LlamaIndex** | RAG开发框架 | [llamaindex.ai](https://www.llamaindex.ai/) |
| **LangChain** | LLM应用框架 | [langchain.com](https://www.langchain.com/) |
| **RAGAs** | RAG评估框架 | [ragas.io](https://docs.ragas.io/) |
| **Chroma** | 向量数据库 | [trychroma.com](https://www.trychroma.com/) |
| **Milvus** | 分布式向量数据库 | [milvus.io](https://milvus.io/) |

### RAG 12个常见痛点

::: details 点击展开查看
1. **内容缺失** - 知识库缺少上下文时返回错误答案
2. **错过排名靠前文档** - 重要文档未出现在Top结果
3. **上下文整合限制** - 整合长度超过LLM窗口
4. **文档信息未提取** - 关键信息未被抽取
5. **格式错误** - 输出格式与预期不符
6. **答案不正确** - 缺乏具体细节导致错误
7. **回答不完整** - 答案不全面
8. **数据提取可扩展性** - 数据摄入性能问题
9. **结构化数据QA** - 表格等结构化数据处理
10. **复杂PDF提取** - 复杂布局PDF处理困难
11. **后备模型策略** - 需要fallback机制
12. **LLM安全性** - 安全防护问题
:::

> 来源：[RAG的12个痛点](https://hub.baai.ac.cn/view/43613)
