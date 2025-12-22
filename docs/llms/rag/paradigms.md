---
title: RAG 范式演进
description: 从 Naive RAG 到 Agentic RAG 的技术演进历程
---

# RAG 范式演进

> 理解RAG技术的发展脉络，掌握从基础到前沿的完整技术栈

## 🎯 概述

RAG（检索增强生成）技术自2020年提出以来，经历了多轮范式迭代。2024年被称为"RAG发展元年"，全年产生了超过1000篇相关论文。

### 五大范式演进路线

```
Naive RAG → Advanced RAG → Modular RAG → GraphRAG → Agentic RAG
```

<div class="paradigm-timeline">
  <div class="paradigm-item">
    <div class="paradigm-year">2020</div>
    <div class="paradigm-name">Naive RAG</div>
    <div class="paradigm-desc">基础检索-生成流程</div>
  </div>
  <div class="paradigm-arrow">→</div>
  <div class="paradigm-item">
    <div class="paradigm-year">2022</div>
    <div class="paradigm-name">Advanced RAG</div>
    <div class="paradigm-desc">预检索/后检索优化</div>
  </div>
  <div class="paradigm-arrow">→</div>
  <div class="paradigm-item">
    <div class="paradigm-year">2023</div>
    <div class="paradigm-name">Modular RAG</div>
    <div class="paradigm-desc">模块化可组合架构</div>
  </div>
  <div class="paradigm-arrow">→</div>
  <div class="paradigm-item highlight">
    <div class="paradigm-year">2024</div>
    <div class="paradigm-name">GraphRAG</div>
    <div class="paradigm-desc">知识图谱增强</div>
  </div>
  <div class="paradigm-arrow">→</div>
  <div class="paradigm-item highlight">
    <div class="paradigm-year">2024+</div>
    <div class="paradigm-name">Agentic RAG</div>
    <div class="paradigm-desc">智能体驱动</div>
  </div>
</div>


---

## 📚 范式一：Naive RAG

### 核心特点

Naive RAG是最基础的RAG实现，遵循简单的"索引-检索-生成"流程。这种架构仅包含三个核心步骤：

1. **索引**：将文档切分成小块，计算每个块的Embedding向量并存储到向量库
2. **检索**：将用户问题转为向量，从向量库中找到最相似的文本块
3. **生成**：将检索到的文本与问题拼接，让大模型生成回答

```python
# Naive RAG 基本流程
def naive_rag(query, knowledge_base):
    # 1. 索引阶段（离线）
    chunks = split_documents(knowledge_base)
    vectors = embed_chunks(chunks)
    index = build_vector_index(vectors)
    
    # 2. 检索阶段
    query_vector = embed_query(query)
    relevant_chunks = index.search(query_vector, top_k=5)
    
    # 3. 生成阶段
    context = "\n".join(relevant_chunks)
    prompt = f"基于以下信息回答问题：\n{context}\n\n问题：{query}"
    answer = llm.generate(prompt)
    
    return answer
```

### 流程图

::: info Naive RAG 流程
📄 **文档** → 🔪 **切分** → 🧮 **Embedding** → 💾 **向量库** → 🔍 **检索** → 🤖 **LLM** → 💬 **答案**
:::

这个过程就像学生做阅读理解：先圈出关键段落，再结合问题总结答案。虽然简单，但已经能解决80%的基础场景——比如让LLM回答"公司年假政策"，只要把员工手册拆成向量，就能快速定位到相关条款。
### 局限性

| 问题 | 表现 | 原因 |
|------|------|------|
| **检索质量差** | 召回不相关内容 | 语义鸿沟、Embedding局限 |
| **冗余信息** | 重复内容干扰生成 | 切分策略简单 |
| **上下文丢失** | 答案不完整 | 固定切分破坏语义 |
| **幻觉问题** | 生成虚假内容 | 检索内容不足以支撑回答 |
| **信息整合难** | 面对复杂问题单轮检索不足 | 缺乏迭代检索机制 |

### 为什么需要RAG？

大模型存在三大"天然缺陷"，RAG提供了完美的弥补逻辑：

| 缺陷 | 表现 | RAG解决方案 |
|------|------|-------------|
| **领域知识缺失** | 预训练数据覆盖广但不深，面对医疗、法律等专业领域如同"门外汉" | 接入私有知识库，让大模型瞬间"精通"特定领域 |
| **实时信息滞后** | 训练数据有时间截止线，无法回答最新政策等问题 | 通过实时检索外部数据源，打破时间壁垒 |
| **幻觉生成风险** | 自信地编造不存在的信息（虚假引用、错误数据） | 让生成内容锚定检索到的真实信息，从源头减少幻觉 |

> **相关文章**：[从"失忆"到"过目不忘"：RAG技术如何给LLM装上"外挂大脑"？](https://dd-ff.blog.csdn.net/article/details/149348018)

---

## ⚡ 范式二：Advanced RAG

### 核心改进

Advanced RAG在Naive RAG基础上引入**预检索优化**和**后检索优化**。

引入**预检索**和**后检索**优化模块：
即检索前和检索后进行优化，提升检索质量。
问题扩展，问题重写都可以将用户原本不专业，不容易被大模型理解的问题，转换成专业，容易被大模型理解的问题。
HyDE（Hypothetical Document Embeddings）是一种改进检索的方法，它**生成可用于回答用户输入问题的假设文档**。这些文档来自LLM自身学习到的知识，向量化后用于从索引中检索文档。HyDE方法认为原始问题一般都比较短，而生成的假设文档可能会更好地与索引文档对齐。[HyDE](https://arxiv.org/abs/2206.05266)

高级RAG中的检索一般都是使用混合检索，稠密向量检索和稀疏向量检索相结合，稠密向量检索是基于向量空间的，稀疏向量检索是基于关键词的。二者各有优势，稠密向量检索可以更好地表征文档的语义，稀疏向量检索可以更好地表征文档的关键词。二者加权后，得到最后的评分（默认是7：3的权重，即稠密向量检索占70%，稀疏向量检索占30%，但是好像这一直都是个玄学）。

后检索，即Rerank，是根据文档的语义和用户的问题进行重新排序，选择最相关的文档。这个就跟embedding不太类似了，embedding是基于向量空间的，而Rerank是基于语义的。embedding是为了快，能够在向量空间中快速找到最相关的文档，而Rerank是基于语义的，能够找到最相关的文档。先找出top——k个相关文档，这个相关是基于向量距离度量的，不是基于语义的，即这些文档分块并不一定就可以解决用户的问题，只有经历rerank后，使用cross-encoder，即交叉编码器，就可以相较于Bi-Encoder，即双编码器，能够更好地表征文档的语义，找到最相关的文档。

```python
class AdvancedRAG:
    def __init__(self):
        self.query_rewriter = QueryRewriter()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
    
    def answer(self, query):
        # 1. 预检索优化：查询改写
        enhanced_query = self.query_rewriter.rewrite(query)
        
        # 2. 检索：混合检索（稠密+稀疏）
        candidates = self.retriever.search(enhanced_query, top_k=50)
        
        # 3. 后检索优化：重排序
        reranked = self.reranker.rerank(query, candidates, top_k=5)
        
        # 4. 生成
        context = self.build_context(reranked)
        answer = self.generate(query, context)
        
        return answer
```

### 关键技术

#### 预检索优化

| 技术 | 原理 | 效果 |
|------|------|------|
| **Query改写** | LLM重写用户问题 | 提升检索准确率 |
| **HyDE** | 先生成假设答案再检索 | 跨越Query-Doc语义鸿沟 |
| **Query扩展** | 添加同义词、相关概念 | 提高召回率 |
| **Query分解** | 复杂问题拆分为子问题 | 覆盖更多相关文档 |

#### 后检索优化

| 技术 | 原理 | 效果 |
|------|------|------|
| **重排序** | Cross-Encoder精排 | 提升Top-K精度 |
| **上下文压缩** | 提取关键信息 | 减少噪音干扰 |
| **多样性优化** | 去重+多样性采样 | 覆盖不同角度 |

### 索引优化策略

::: tip 分层索引
**摘要索引**：文档级摘要，快速定位  
**内容索引**：段落级细节，精确检索  
**父子分段**：小块检索，大块返回
:::

### 分块策略详解

分块是RAG的核心矛盾点：**块太大，噪声多；块太小，丢上下文**。高级RAG采用更精细的处理策略：

#### 分块大小选择

实践证明，**1024 token** 的块大小在多数场景中表现更优——既能抓住核心逻辑，又不至于被噪声淹没。但需根据文档类型动态调整：

| 文档类型 | 推荐策略 | 原因 |
|----------|----------|------|
| **长文档（论文）** | 较小块（512-768 token） | 避免丢失细节问题 |
| **短文档（邮件）** | 较大块（1024-2048 token） | 保留完整语境 |
| **结构化文档** | 按章节/标题分块 | 保持语义完整性 |

#### 重叠分块

让相邻块重叠 **20%-30%**（如1024 token块重叠200 token），避免"一句话被劈成两半"的尴尬。

#### 格式适配分块

| 格式 | 分块方式 |
|------|----------|
| **Markdown** | 按标题层级拆分 |
| **代码文件** | 按函数/类拆分 |
| **PDF** | 按表格/文本分离处理 |
| **普通文本** | 固定大小 + 语义边界 |

### 从小到大检索（父子分段）

如果说分块是"拆蛋糕"，那"从小到大检索"就是"先尝小块，再吃大块"：

```python
# 父子分段检索策略
class ParentChildRetriever:
    def __init__(self):
        self.child_chunks = []  # 256 token 细粒度块（用于检索）
        self.parent_chunks = []  # 1024 token 大块（用于生成）
    
    def retrieve(self, query, top_k=5):
        # 1. 用小块精准检索
        relevant_children = self.search_children(query, top_k=top_k)
        
        # 2. 找到相关小块的父块
        parent_ids = set(child.parent_id for child in relevant_children)
        
        # 3. 返回完整的父块作为上下文
        return [self.parent_chunks[pid] for pid in parent_ids]
```

**核心思想**：
- **检索用小块**：用256 token的细粒度块做向量检索，精准定位相关片段
- **生成用大块**：找到相关小块后，自动关联其所属的1024 token大块，为LLM提供完整上下文

### 文档层次结构索引

如果把文档比作一本书，传统分块就是"把书撕成散页"，而层次结构则是"给书做目录+章节摘要"：

```
┌─────────────────────────────────────────┐
│      根节点：全文摘要                      │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
┌───▼───────────┐     ┌────────▼──────────┐
│ 中间节点：章节摘要 │     │ 中间节点：章节摘要   │
└───────┬───────┘     └────────┬──────────┘
        │                      │
   ┌────┴────┐            ┌────┴────┐
   │         │            │         │
┌──▼──┐   ┌──▼──┐      ┌──▼──┐   ┌──▼──┐
│叶节点│   │叶节点│      │叶节点│   │叶节点│
│段落  │   │段落  │      │段落  │   │段落  │
└─────┘   └─────┘      └─────┘   └─────┘
```

**优势**：先通过章节摘要快速排除无关部分，再深入细粒度内容，检索效率提升50%以上。

### 查询转换技术详解

用户的原始输入往往模糊、冗余或依赖上下文，需通过技术手段将其转为"检索友好型"查询。

#### 查询扩展

将复杂问题拆分为子问题，分步骤检索后整合结果：

```python
def query_expansion(query: str):
    """查询扩展示例"""
    # 原始问题："2023年女足世界杯冠军及该国GDP"
    sub_queries = [
        "2023女足世界杯冠军是哪个国家",
        "[冠军国家]2023年GDP是多少"
    ]
    return sub_queries
```

**RAG Fusion进阶技巧**：生成多个相似问题，再用倒数排名融合（RRF）算法合并结果，提升召回率。

#### 假设性文档嵌入（HyDE）详解

当问题与文档表述差异大时，先让大模型生成"假答案"，再用假答案的Embedding检索真实文档：

```python
def hyde_retrieval(query: str):
    """HyDE检索流程"""
    # 1. 让LLM生成假设性答案
    hypothetical_answer = llm.generate(
        f"请回答以下问题（即使不确定也尝试回答）：{query}"
    )
    
    # 2. 用假设性答案的向量检索
    hyde_embedding = embed(hypothetical_answer)
    
    # 3. 检索真实文档
    real_docs = vector_db.search(hyde_embedding, top_k=10)
    
    return real_docs
```

**原理**：HyDE方法认为原始问题一般都比较短，而生成的假设文档可能会更好地与索引文档对齐。

#### 查询重写

处理冗余或上下文依赖的问题：

| 原始查询 | 重写后 | 场景 |
|----------|--------|------|
| "那里有什么好吃的" | "合肥特色美食" | 上文提到"合肥" |
| "我叫张三，想问Maisie Peters是谁" | "Maisie Peters的身份" | 提取核心查询 |
| "怎么提高效率" | "企业流程优化方法" | 模糊问题转精准 |

### 查询路由

当数据分散在多种存储中（向量库、SQL数据库、图数据库），需通过路由技术匹配最合适的检索方式：

| 路由类型 | 技术 | 示例 |
|----------|------|------|
| **文本→结构化数据** | Text-to-SQL | "2023年每季度销售额" → `SELECT 季度, SUM(销售额)...` |
| **文本→图数据** | Text-to-Cypher | "马云和阿里巴巴的关系" → 图查询三元组 |
| **多源混合** | 路由器动态选择 | "特斯拉销量及股价" → SQL + 实时API |

### 后处理优化详解

#### 重排序（Reranking）

用专门模型优化检索结果顺序，如Cohere Rerank或SentenceTransformer交叉编码器对初筛结果重新打分。

#### 上下文压缩

通过句子级相似度计算，从检索到的段落中筛选与问题相关的句子，**减少50%以上的token消耗**。

#### 来源追踪

为每个生成片段标注来源（如"XX报告P3"），增强可信度，便于验证。

> **相关文章**：
> - [高级RAG技术全景：从原理到实战](https://dd-ff.blog.csdn.net/article/details/149396526)
> - [从"拆文档"到"通语义"：RAG+知识图谱如何破解大模型"失忆+幻觉"难题？](https://dd-ff.blog.csdn.net/article/details/149354855)

---

## 🧩 范式三：Modular RAG

### 核心理念

Modular RAG将RAG系统拆分为**可插拔的独立模块**，支持灵活组合。

```python
class ModularRAG:
    """模块化RAG架构"""
    
    def __init__(self, config):
        # 可配置的模块
        self.modules = {
            'indexing': self._init_indexing(config),
            'pre_retrieval': self._init_pre_retrieval(config),
            'retrieval': self._init_retrieval(config),
            'post_retrieval': self._init_post_retrieval(config),
            'generation': self._init_generation(config)
        }
        
        # 可配置的流程
        self.pipeline = config.get('pipeline', [
            'pre_retrieval', 'retrieval', 'post_retrieval', 'generation'
        ])
    
    def execute(self, query):
        """执行配置的流程"""
        context = {'query': query, 'results': []}
        
        for module_name in self.pipeline:
            module = self.modules[module_name]
            context = module.process(context)
        
        return context['answer']
```

### 六大核心模块

| 模块 | 职责 | 可选组件 |
|------|------|----------|
| **索引模块** | 文档处理与存储 | 向量索引、倒排索引、图索引 |
| **预检索模块** | 查询优化 | 改写、扩展、分解、路由 |
| **检索模块** | 文档召回 | 向量检索、BM25、混合检索 |
| **后检索模块** | 结果优化 | 重排序、过滤、压缩 |
| **生成模块** | 答案生成 | 提示工程、输出格式化 |
| **编排模块** | 流程控制 | 条件分支、迭代、并行 |

### 模块组合示例

```python
# 简单问答场景
simple_qa_config = {
    'pipeline': ['retrieval', 'generation'],
    'retrieval': {'type': 'dense', 'top_k': 5}
}

# 复杂分析场景
complex_analysis_config = {
    'pipeline': [
        'query_decomposition',  # 问题分解
        'parallel_retrieval',   # 并行检索
        'result_fusion',        # 结果融合
        'reranking',            # 重排序
        'iterative_generation'  # 迭代生成
    ]
}

# 多模态场景
multimodal_config = {
    'pipeline': ['pre_retrieval', 'multimodal_retrieval', 'generation'],
    'retrieval': {
        'text_retriever': 'dense',
        'image_retriever': 'clip',
        'fusion': 'late_fusion'
    }
}
```

> **相关文章**：[检索增强生成（RAG）综述：技术范式、核心组件与未来展望](https://dd-ff.blog.csdn.net/article/details/149274498)

---

## 🕸️ 范式四：GraphRAG

### 核心思想

GraphRAG通过**知识图谱**增强RAG，解决传统RAG在**关系推理**和**全局理解**方面的不足。

GraphRAG（Graph-based Retrieval-Augmented Generation）是在传统 RAG 基础上引入**知识图谱/关系结构**的一种方法，核心优势体现在“**理解关系**”和“**复杂推理**”上，主要优点如下：

1. **更强的关系理解能力**
   通过图结构显式建模实体与实体、概念与概念之间的关系，避免只靠相似度检索带来的信息割裂。

2. **提升复杂问题的回答质量**
   对多跳问题、因果链问题、跨文档关联问题更友好，能沿着图进行推理，而不是拼接零散文本。

3. **减少幻觉（Hallucination）**
   回答基于结构化关系和可追溯节点，模型更容易“有据可依”，降低胡编乱造的概率。

4. **上下文更高效**
   图检索能精准定位相关子图，减少无关文本进入上下文，提升 token 利用率。

5. **可解释性更强**
   检索和推理路径可以用“节点—边—路径”展示，便于调试、审计和业务解释。

6. **适合知识密集型场景**
   特别适用于企业知识库、技术文档、科研文献、法律/医疗知识等关系复杂、概念稳定的领域。

**一句话总结：**

> 传统 RAG 擅长“找相似文本”，GraphRAG 更擅长“理解和利用知识之间的关系”。


### 传统RAG vs GraphRAG

<div class="compare-box">
  <div class="compare-item">
    <div class="compare-title">传统 RAG</div>
    <ul>
      <li>基于向量相似度检索</li>
      <li>平面数据表示</li>
      <li>单跳检索</li>
      <li>局部信息理解</li>
    </ul>
  </div>
  <div class="compare-vs">VS</div>
  <div class="compare-item highlight">
    <div class="compare-title">GraphRAG</div>
    <ul>
      <li>基于图结构检索</li>
      <li>实体关系网络</li>
      <li>多跳推理</li>
      <li>全局信息聚合</li>
    </ul>
  </div>
</div>


### GraphRAG工作流程

```python
class GraphRAG:
    """GraphRAG核心实现"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.graph_db = Neo4jClient()
        self.community_detector = LouvainCommunity()
    
    def build_knowledge_graph(self, documents):
        """1. 图索引构建（G-Indexing）"""
        for doc in documents:
            # 实体抽取
            entities = self.entity_extractor.extract(doc)
            
            # 关系抽取
            relations = self.relation_extractor.extract(doc, entities)
            
            # 构建图谱
            for entity in entities:
                self.graph_db.create_node(entity)
            for relation in relations:
                self.graph_db.create_edge(relation)
        
        # 社区检测（用于全局查询）
        communities = self.community_detector.detect(self.graph_db)
        self.generate_community_summaries(communities)
    
    def retrieve(self, query):
        """2. 图引导检索（G-Retrieval）"""
        # 提取查询实体
        query_entities = self.entity_extractor.extract(query)
        
        # 子图检索
        subgraph = self.graph_db.get_subgraph(
            entities=query_entities,
            hops=2  # 多跳检索
        )
        
        # 社区摘要检索（全局查询）
        relevant_communities = self.search_communities(query)
        
        return subgraph, relevant_communities
    
    def generate(self, query, subgraph, communities):
        """3. 图增强生成（G-Generation）"""
        # 构建图结构化上下文
        context = self.format_graph_context(subgraph, communities)
        
        # 生成答案
        answer = self.llm.generate(query, context)
        return answer
```

### 传统RAG的四大局限（GraphRAG解决的问题）

深入分析后发现，传统RAG存在根本性局限：

| 局限 | 表现 | GraphRAG解决方案 |
|------|------|------------------|
| **关系盲目性** | 擅长发现语义相似的独立片段，但无法理解信息间的隐含和显式关系 | 图结构显式建模实体关系 |
| **无法"连接点"** | 需要综合多个独立文档/分离片段时往往失败 | 通过图遍历进行多跳推理 |
| **上下文冗余** | 检索大量重叠信息，导致"迷失在中间"问题 | 精准定位相关子图，减少无关文本 |
| **全局问题失败** | 无法回答"数据集有哪些主要议题？"这类全局性问题 | 社区检测 + 层级摘要支持全局理解 |

### 两大技术流派

::: warning 重要洞察
"GraphRAG" 并非单一标准化技术，而是涵盖了两种截然不同哲学思想的范式。实践者必须根据具体用例来选择合适的架构。
:::

| 流派 | 代表 | 核心焦点 | 技术特点 | 适用场景 |
|------|------|----------|----------|----------|
| **流派A：图作为事实数据库** | 蚂蚁集团(DB-GPT+TuGraph) | 事实、精度、路径 | 三元组抽取、Cypher/GQL查询、子图遍历 | 精确推理、事实问答、多跳推理 |
| **流派B：图作为洞察结构** | 微软GraphRAG | 主题、摘要、社区 | Louvain社区检测、层级摘要、Map-Reduce聚合 | 全局理解、主题分析、数据集概览 |

#### 流派A详解：三元组提取与子图遍历

此流派将知识图谱视为一个结构化的、可验证的"真理"存储库。核心目标是通过检索明确的事实及其间的路径来回答问题。

##### 什么是三元组？

知识图谱的基本原子单元是 **(主语, 谓语, 宾语)** 的命题：
- 例如：`(The Beatles, performed, 'Hello, Goodbye')`
- 例如：`(马云, 创立, 阿里巴巴)`

```python
# 流派A核心：从文本中提取精确的原子化事实
class FactDatabaseGraphRAG:
    """蚂蚁集团技术栈示例：DB-GPT + OpenSPG + TuGraph"""
    
    def g_indexing(self, text):
        """G-Indexing：LLM提取三元组"""
        prompt = """从以下文本中提取实体和关系三元组：
        文本：{text}
        输出格式：[(主语, 谓语, 宾语), ...]"""
        triplets = self.llm.extract(prompt.format(text=text))
        self.graph_db.insert_triplets(triplets)
    
    def g_retrieval(self, query):
        """G-Retrieval：关键词提取 + 图遍历"""
        # 方法1：BFS/DFS遍历N跳子图
        entities = self.llm.extract_entities(query)
        subgraph = self.graph_db.traverse(entities, hops=2)
        
        # 方法2：Text-to-Cypher查询
        cypher = self.llm.generate_cypher(query)
        results = self.graph_db.execute(cypher)
        return results
```

##### 蚂蚁集团技术栈

| 组件 | 角色 | 功能 |
|------|------|------|
| **DB-GPT** | AI开发框架 | 协调LLM的交互 |
| **OpenSPG** | 知识图谱引擎 | 定义图的"模式"(Schema)，语义增强可编程图框架 |
| **TuGraph** | 图数据库 | 高性能存储和管理三元组（节点和边） |

##### G-Retrieval的两种方式

**方法1：关键词提取 + 图遍历**
1. LLM从用户查询中提取关键词（通常是实体）
2. 在图数据库中定位这些实体节点
3. 使用BFS/DFS算法从这些节点"向外探索"N跳，获取局部子图

**方法2：Text-to-Cypher/GQL**

将自然语言查询直接翻译为形式化的图查询语言：

```cypher
-- 用户提问："哪些演员出演了汤姆·汉克斯导演的电影？"
MATCH (p:Person {name: "Tom Hanks"})-->(m:Movie)<--(a:Person) 
RETURN a.name
```

| 方式 | 特点 | 权衡 |
|------|------|------|
| **Cypher模板** | 像单一块玩具 | 僵硬但安全 |
| **动态Cypher生成** | 像玩乐高积木 | 更灵活，仍相对安全 |
| **Text-to-Cypher** | 像绘画 | 完全自由，但也极易失败 |

#### 流派B详解：社区检测与摘要聚合

此流派对个体事实的兴趣较小，更关注知识的宏观结构和浮现的主题。核心解决的问题是**查询聚焦型摘要（Query-Focused Summarization, QFS）**——回答"这份报告的核心主题是什么？"或"A项目和B项目是如何相互关联的？"。

##### 微软GraphRAG索引工作流

这是一个复杂的、由多个工作流组成的数据管道：

```
┌────────────────────────────────────────────────────────────────┐
│                    微软 GraphRAG 索引流程                        │
├────────────────────────────────────────────────────────────────┤
│  1. 输入与分块                                                   │
│     读入输入数据（.txt, .csv），分割成 TextUnits（文本单元）         │
├────────────────────────────────────────────────────────────────┤
│  2. 图提取（LLM密集型）                                          │
│     LLM扫描每个TextUnit，提取：                                  │
│     - 实体（人、地点、组织）                                      │
│     - 关系（实体间的联系）                                        │
│     - 主张（文本中的关键断言或事实）                               │
├────────────────────────────────────────────────────────────────┤
│  3. 图构建                                                      │
│     将提取的实体、关系、主张组装成内存中的图结构                    │
├────────────────────────────────────────────────────────────────┤
│  4. 社区检测（算法核心）                                          │
│     运行 Leiden/Louvain 算法，将图分割成"社区"或"主题集群"         │
├────────────────────────────────────────────────────────────────┤
│  5. 层级摘要（LLM密集型）                                         │
│     为每个社区预生成摘要，形成层级结构                             │
└────────────────────────────────────────────────────────────────┘
```

##### Louvain算法详解

这是一种高效的层级聚类算法，目标是最大化图的"模块性"(Modularity)：

- **模块性**：衡量社区内部连接密度与社区之间连接密度的对比
- **结果**：图被自动分割成多个"社区"或"主题集群"，同一社区内的节点彼此间联系更紧密

```python
class InsightStructureGraphRAG:
    """微软GraphRAG：解决全局性问题"""
    
    def g_indexing(self, documents):
        """构建社区层级结构"""
        # 1. 实体关系抽取
        kg = self.extract_entities_relations(documents)
        
        # 2. Louvain社区检测
        communities = self.louvain_community_detection(kg)
        
        # 3. 为每个社区预生成摘要
        for community in communities:
            summary = self.llm.summarize(community.entities)
            community.summary = summary
    
    def g_retrieval(self, query):
        """Map-Reduce风格的全局查询"""
        # Map：每个社区摘要生成部分响应
        partial_responses = []
        for community in self.communities:
            response = self.llm.answer(query, community.summary)
            partial_responses.append(response)
        
        # Reduce：汇总所有部分响应
        final_answer = self.llm.aggregate(partial_responses)
        return final_answer
```

##### Map-Reduce风格的全局查询

1. **Map阶段**：每个社区摘要独立生成针对查询的部分响应
2. **Reduce阶段**：汇总所有部分响应，生成最终答案

这种架构特别适合回答"这个数据集中有哪些主要议题？"这类全局性问题。

### GRAG：知识图谱+RAG的"强强联合"

单独的RAG擅长文本语义匹配，单独的KG擅长关系推理，而**GRAG（图增强RAG）** 则是"1+1>2"的组合：

```
┌─────────────────────────────────────────────────────────────┐
│                 GRAG 工作流程                                │
├─────────────────────────────────────────────────────────────┤
│  用户查询: "波士顿红袜队的主场有什么特色？"                    │
│                      │                                      │
│          ┌──────────┴──────────┐                           │
│          ▼                     ▼                            │
│   ┌────────────┐        ┌────────────┐                     │
│   │ RAG模块    │        │  KG模块    │                     │
│   │ 检索相关   │        │ 补充实体   │                     │
│   │ 文本片段   │        │ 关系       │                     │
│   └─────┬──────┘        └─────┬──────┘                     │
│         │                     │                             │
│         ▼                     ▼                             │
│   芬威公园建于1912年,    红袜队→主场→芬威公园                │
│   有绿色怪物墙...        芬威公园→特色→绿色怪物墙            │
│         │                     │                             │
│         └──────────┬──────────┘                            │
│                    ▼                                        │
│             ┌────────────┐                                  │
│             │ LLM 融合   │                                  │
│             └─────┬──────┘                                  │
│                   ▼                                         │
│   答案：红袜队的主场是芬威公园（1912年建成），                │
│   其标志性特色是左外野的"绿色怪物墙"...                      │
└─────────────────────────────────────────────────────────────┘
```

**优势**：既保留了RAG对文本细节的捕捉能力，又借助KG的关系推理，让回答更有深度。

### 轻量级变体

| 变体 | 特点 | 优势 |
|------|------|------|
| **LightRAG** | 去掉社区检测，双级检索 | 低层具体信息 + 高层广泛话题，更轻量 |
| **LazyGraphRAG** | 按需构建图谱 | 降低初始化成本，适合动态数据 |
| **KAG** | LLM友好的知识表示 | 语义增强，结合符号推理 |
| **nano-graphrag** | 极简实现 | 适合学习和小规模应用 |

### 图数据源：构建 vs. 购买

GraphRAG系统依赖的图数据可以来自两个主要来源：

| 来源 | 类型 | 示例 |
|------|------|------|
| **开放知识图谱** | 百科全书式 | Wikidata, Freebase, DBpedia, YAGO |
|  | 常识知识 | ConceptNet, ATOMIC |
|  | 领域知识 | CMeKG (生物医学), Wiki-Movies (电影) |
| **自建知识图谱** | 企业私有 | 从公司内部文档、法律档案、研究论文动态构建 |

> **相关文章**：[GraphRAG 技术教程：从核心概念到高级架构](https://dd-ff.blog.csdn.net/article/details/154530805)

---

## 🤖 范式五：Agentic RAG

### 核心理念

Agentic RAG将**AI Agent**与RAG结合，实现**自主决策**、**动态检索**和**迭代优化**。

**Agentic RAG** 是在传统 RAG 基础上，引入 **Agent（智能体）决策与执行能力** 的增强范式，核心思想是：

> **不只是“检索后生成”，而是“会思考、会规划、会多步行动的 RAG”。**

### Agentic RAG 的主要特点与优势

1. **具备规划与决策能力**
   Agent 会先分析问题，决定是否需要检索、检索几次、用什么策略，而不是固定的一次性 RAG 流程。

2. **支持多步推理与迭代检索**
   可以“检索 → 反思 → 再检索 → 总结”，适合复杂、开放式或信息不完整的问题。

3. **工具与环境感知能力强**
   不仅能用向量检索，还可调用：

   * 搜索引擎
   * 数据库 / API
   * 代码执行、计算工具
   * 知识图谱（可与 GraphRAG 结合）

4. **动态上下文构建**
   根据当前推理状态动态选择最有价值的信息进入上下文，而非一次性塞满。

5. **任务导向更强**
   不只是回答问题，更适合完成任务，如：

   * 调研总结
   * 方案设计
   * 故障排查
   * 企业内部流程执行

6. **可扩展性与自治性高**
   Agent 可拆解子任务、并行执行、失败重试，适合复杂系统和长流程。

### 与传统 RAG / GraphRAG 的对比

| 维度   | 传统 RAG | GraphRAG | Agentic RAG |
| ---- | ------ | -------- | ----------- |
| 核心能力 | 相似度检索  | 关系建模     | 决策 + 行动     |
| 推理方式 | 单步     | 结构化多跳    | 规划式多步       |
| 检索次数 | 1 次    | 多跳       | 动态、多次       |
| 工具使用 | 基本无    | 图查询      | 多工具协同       |
| 适用场景 | 简单 QA  | 知识密集     | 复杂任务        |

### 典型应用场景

* 企业智能助理（跨系统查资料 + 执行动作）
* 技术/运维排障
* 法律、投研、咨询类深度分析
* 自动化工作流（Agent + RAG）

**一句话总结：**

> **Agentic RAG = RAG + 会思考、会规划、会行动的 Agent。**


::: tip Agent循环
**用户问题** → Agent任务分解 → 检索决策 → 多轮检索 → 结果综合 → 自我反思 → ✅满意输出 / ❌继续迭代
:::

### 核心实现

```python
class AgenticRAG:
    """Agentic RAG核心实现"""
    
    def __init__(self):
        self.tools = {
            'retriever': VectorRetriever(),
            'web_search': WebSearchTool(),
            'calculator': CalculatorTool(),
            'code_executor': CodeExecutor()
        }
        self.planner = TaskPlanner()
        self.reflector = SelfReflector()
    
    def answer(self, query, max_iterations=5):
        # 1. 任务规划
        plan = self.planner.create_plan(query)
        
        context = {'query': query, 'results': [], 'history': []}
        
        for iteration in range(max_iterations):
            # 2. 执行当前步骤
            current_step = plan.get_next_step()
            
            if current_step is None:
                break
            
            # 3. 选择工具并执行
            tool = self.select_tool(current_step)
            result = tool.execute(current_step.params)
            context['results'].append(result)
            
            # 4. 自我反思
            reflection = self.reflector.evaluate(
                query=query,
                current_result=result,
                history=context['history']
            )
            
            if reflection['is_sufficient']:
                break
            
            # 5. 调整计划
            if reflection['needs_replanning']:
                plan = self.planner.replan(query, context)
            
            context['history'].append({
                'step': current_step,
                'result': result,
                'reflection': reflection
            })
        
        # 6. 生成最终答案
        answer = self.synthesize_answer(query, context)
        return answer
    
    def select_tool(self, step):
        """动态工具选择"""
        tool_name = self.planner.recommend_tool(step)
        return self.tools[tool_name]
```

### 架构分类

| 架构 | 特点 | 适用场景 |
|------|------|----------|
| **单Agent** | 一个Agent处理全流程 | 简单任务 |
| **多Agent协作** | 多个专业Agent分工 | 复杂任务 |
| **层级Agent** | Manager + Worker模式 | 大规模系统 |
| **图结构Agent** | Agent间动态交互 | 灵活任务流 |

### SELF-RAG：自反思检索

**SELF-RAG（Self-Reflective Retrieval-Augmented Generation）** 是一种强调**“自我反思与自我纠错”**能力的 RAG 架构，核心思想是：

> **模型在生成过程中，主动判断“要不要检索、检索是否有用、回答是否可信”，并据此自我调整。**

---

## SELF-RAG 的核心机制

1. **是否需要检索（Retrieve or Not）**
   模型先评估当前问题是否需要外部知识，避免无意义检索。

2. **检索质量自评（Critique Retrieval）**
   对检索结果进行自我打分或判断相关性，不合格则重新检索或放弃使用。

3. **生成过程中的反思（Reflection）**
   在生成答案时不断检查：

   * 是否有证据支持？
   * 是否出现臆断？
   * 是否偏离问题？

4. **回答可信度控制（Groundedness）**
   通过显式标注或内部判断，减少幻觉，提升事实一致性。

---

## SELF-RAG 的主要优点

1. **显著降低幻觉问题**
   模型会拒绝或修正“无依据的生成”。

2. **检索更“省”和更“准”**
   不必要时不检索，必要时多次检索，提高效率与质量。

3. **回答更稳健**
   对不确定问题更倾向于给出保守、可解释的答案。

4. **无需复杂外部 Agent 框架**
   相比 Agentic RAG，结构更轻量，易于落地。

---

## 与其他 RAG 形态的对比

| 维度     | 传统 RAG | SELF-RAG | Agentic RAG |
| ------ | ------ | -------- | ----------- |
| 是否自我评估 | ❌      | ✅        | ✅           |
| 检索决策   | 固定     | 自适应      | 自主规划        |
| 推理深度   | 低      | 中        | 高           |
| 系统复杂度  | 低      | 中        | 高           |
| 幻觉控制   | 一般     | 强        | 强           |

---

## 适用场景

* 高准确性要求的问答（法律、金融、医疗）
* 企业内部知识库 QA
* 对成本、延迟敏感的 RAG 系统

---

**一句话总结：**

> **SELF-RAG 让模型在 RAG 中学会“先想清楚，再去查，再敢说”。**


```python
class SelfRAG:
    """SELF-RAG: 自反思RAG"""
    
    def __init__(self):
        self.retriever = Retriever()
        self.critic = CriticModel()  # 反思标记生成
    
    def answer(self, query):
        # 1. 判断是否需要检索
        need_retrieval = self.critic.should_retrieve(query)
        
        if need_retrieval:
            # 2. 检索
            documents = self.retriever.search(query)
            
            # 3. 评估每个文档的相关性
            relevant_docs = []
            for doc in documents:
                relevance = self.critic.evaluate_relevance(query, doc)
                if relevance > 0.7:
                    relevant_docs.append(doc)
            
            context = relevant_docs
        else:
            context = []
        
        # 4. 生成答案
        answer = self.generate(query, context)
        
        # 5. 评估答案质量
        quality = self.critic.evaluate_answer(query, answer, context)
        
        if quality['needs_improvement']:
            # 迭代优化
            answer = self.refine_answer(query, answer, quality['feedback'])
        
        return answer
```

> **相关文章**：
> - [OpenAI Agent 工具全面开发者指南——从 RAG 到 Computer Use](https://dd-ff.blog.csdn.net/article/details/154445828)
> - [LangChainv1 模型模块全面教程](https://dd-ff.blog.csdn.net/article/details/155068085)

---

## 🧩 技术组合：像搭乐高一样拼出最优解

高级RAG的终极玩法是**技术组合**——没有"最好的技术"，只有"最适合的组合"。

### 组合策略

| 组合方式 | 技术要素 | 适用场景 |
|----------|----------|----------|
| **多索引并行** | 向量索引（语义匹配）+ 知识图谱（关系匹配）+ 摘要索引（快速概览） | 复杂知识库 |
| **多检索融合** | BM25关键词过滤 + 向量检索语义相关 + KG补充关系 | 高精度需求 |
| **多引擎协同** | 子问题引擎拆题 + 查询增强引擎补全 + GRAG引擎融合 | 复杂问答 |

### 组合的权衡

::: warning 复杂度警示
"乐高块"越多，系统越复杂：
- **响应时间变长**：多检索叠加增加延迟
- **成本上升**：多模型调用消耗更多资源
- **调试困难**：故障定位更加复杂
:::

**组合的关键是"按需叠加"**：
- **简单问题**：用"基础分块 + 单向量检索"
- **复杂问题**：上"层次结构 + KG + 子问题规划"

### RAG仍面临的挑战

尽管技术突飞猛进，RAG仍有三座大山要翻：

| 挑战 | 表现 | 现状 |
|------|------|------|
| **没有"万能处理方案"** | 分块大小、KG实体粒度、检索组合都需按文档类型定制 | 无法一刀切 |
| **动态文档处理难** | 文档更新时如何高效更新向量索引和KG关系 | 往往需重建整个库 |
| **成本与效果平衡** | 高级技术虽好，但高算力+长耗时 | 中小企业望而却步 |

---

## 📊 范式对比总结

| 范式 | 检索方式 | 优势 | 劣势 | 适用场景 |
|------|----------|------|------|----------|
| **Naive RAG** | 向量相似度 | 简单快速 | 精度有限 | 原型验证 |
| **Advanced RAG** | 混合检索+重排序 | 精度提升 | 复杂度增加 | 生产环境 |
| **Modular RAG** | 可配置流程 | 灵活可扩展 | 设计复杂 | 多场景适配 |
| **GraphRAG** | 图结构检索 | 关系推理强 | 构建成本高 | 知识密集型 |
| **Agentic RAG** | 动态决策 | 自主适应 | 延迟较高 | 复杂任务 |
| **SELF-RAG** | 自适应检索 | 幻觉控制强 | 需要微调 | 高准确性场景 |

### 如何选择？

```python
def choose_rag_paradigm(requirements):
    """根据需求选择RAG范式"""
    
    if requirements['simplicity'] and not requirements['high_accuracy']:
        return "Naive RAG"
    
    if requirements['production_ready'] and requirements['moderate_complexity']:
        return "Advanced RAG"
    
    if requirements['multi_scenario'] and requirements['flexibility']:
        return "Modular RAG"
    
    if requirements['relation_reasoning'] or requirements['global_understanding']:
        return "GraphRAG"
    
    if requirements['complex_tasks'] or requirements['multi_step_reasoning']:
        return "Agentic RAG"
    
    return "Advanced RAG"  # 默认推荐
```

---

## 🔗 相关阅读

- [文档切分策略](/llms/rag/chunking) - 索引阶段的核心技术
- [Embedding技术](/llms/rag/embedding) - 向量化的原理与实践
- [检索策略](/llms/rag/retrieval) - 多种检索方法详解
- [RAG评估](/llms/rag/evaluation) - 系统性能评估方法

> **核心参考**：
> - [RAG技术的5种范式](https://hub.baai.ac.cn/view/43613) - 智源社区
> - [检索增强生成（RAG）综述](https://dd-ff.blog.csdn.net/article/details/149274498)
> - [GraphRAG 技术教程：从核心概念到高级架构](https://dd-ff.blog.csdn.net/article/details/154530805)
> - [高级RAG技术全景：从原理到实战](https://dd-ff.blog.csdn.net/article/details/149396526)
> - [从"拆文档"到"通语义"：RAG+知识图谱如何破解大模型"失忆+幻觉"难题？](https://dd-ff.blog.csdn.net/article/details/149354855)
