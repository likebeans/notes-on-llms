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

Naive RAG是最基础的RAG实现，遵循简单的"索引-检索-生成"流程。

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

### 局限性

| 问题 | 表现 | 原因 |
|------|------|------|
| **检索质量差** | 召回不相关内容 | 语义鸿沟、Embedding局限 |
| **冗余信息** | 重复内容干扰生成 | 切分策略简单 |
| **上下文丢失** | 答案不完整 | 固定切分破坏语义 |
| **幻觉问题** | 生成虚假内容 | 检索内容不足以支撑回答 |

> **相关文章**：[从"失忆"到"过目不忘"：RAG技术如何给LLM装上"外挂大脑"？](https://dd-ff.blog.csdn.net/article/details/149348018)

---

## ⚡ 范式二：Advanced RAG

### 核心改进

Advanced RAG在Naive RAG基础上引入**预检索优化**和**后检索优化**。

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

> **相关文章**：[高级RAG技术全景：从原理到实战](https://dd-ff.blog.csdn.net/article/details/149396526)

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

### 两大技术流派

::: warning 重要洞察
"GraphRAG" 并非单一标准化技术，而是涵盖了两种截然不同哲学思想的范式。实践者必须根据具体用例来选择合适的架构。
:::

| 流派 | 代表 | 核心焦点 | 技术特点 | 适用场景 |
|------|------|----------|----------|----------|
| **流派A：图作为事实数据库** | 蚂蚁集团(DB-GPT+TuGraph) | 事实、精度、路径 | 三元组抽取、Cypher/GQL查询、子图遍历 | 精确推理、事实问答、多跳推理 |
| **流派B：图作为洞察结构** | 微软GraphRAG | 主题、摘要、社区 | Louvain社区检测、层级摘要、Map-Reduce聚合 | 全局理解、主题分析、数据集概览 |

#### 流派A详解：三元组提取与子图遍历

```python
# 流派A核心：从文本中提取精确的原子化事实
# 三元组格式：(主语, 谓语, 宾语)
# 例如：(The Beatles, performed, 'Hello, Goodbye')

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

#### 流派B详解：社区检测与摘要聚合

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

### 传统RAG的四大局限（GraphRAG解决的问题）

1. **关系盲目性**：无法理解信息片段之间的隐含和显式关系
2. **无法"连接点"**：需要综合多个独立文档时往往失败
3. **上下文冗余**：检索大量重叠信息导致"迷失在中间"问题
4. **全局问题失败**："数据集中有哪些主要议题？"这类问题无法回答

### 轻量级变体

| 变体 | 特点 | 优势 |
|------|------|------|
| **LightRAG** | 去掉社区检测，双级检索 | 低层具体信息 + 高层广泛话题，更轻量 |
| **LazyGraphRAG** | 按需构建图谱 | 降低初始化成本，适合动态数据 |
| **KAG** | LLM友好的知识表示 | 语义增强，结合符号推理 |

> **相关文章**：[GraphRAG 技术教程：从核心概念到高级架构](https://dd-ff.blog.csdn.net/article/details/154530805)

---

## 🤖 范式五：Agentic RAG

### 核心理念

Agentic RAG将**AI Agent**与RAG结合，实现**自主决策**、**动态检索**和**迭代优化**。

::: tip Agent循环
**用户问题** → Agent任务分解 → 检索决策 → 多轮检索 → 结果综合 → 自我反思 → ✅满意输出 / ❌继续迭代
:::

### 与传统RAG的区别

| 特性 | 传统RAG | Agentic RAG |
|------|---------|-------------|
| **检索策略** | 固定流程 | 动态决策 |
| **迭代能力** | 单次检索 | 多轮迭代 |
| **工具使用** | 仅检索 | 多工具组合 |
| **自我反思** | 无 | 结果验证与修正 |
| **任务分解** | 无 | 复杂问题拆解 |

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

## 📊 范式对比总结

| 范式 | 检索方式 | 优势 | 劣势 | 适用场景 |
|------|----------|------|------|----------|
| **Naive RAG** | 向量相似度 | 简单快速 | 精度有限 | 原型验证 |
| **Advanced RAG** | 混合检索+重排序 | 精度提升 | 复杂度增加 | 生产环境 |
| **Modular RAG** | 可配置流程 | 灵活可扩展 | 设计复杂 | 多场景适配 |
| **GraphRAG** | 图结构检索 | 关系推理强 | 构建成本高 | 知识密集型 |
| **Agentic RAG** | 动态决策 | 自主适应 | 延迟较高 | 复杂任务 |

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
> - [GraphRAG 技术教程](https://dd-ff.blog.csdn.net/article/details/154530805)
