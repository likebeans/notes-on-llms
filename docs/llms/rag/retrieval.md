---
title: 检索策略优化
description: RAG系统中的检索技术与策略详解
pageType: article
module: rag
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - rag
level: intermediate
prerequisites:
  - /llms/prompt/
reviewed: '2026-08-26'
techVersion: 2026-08（检索策略，部分工具细节待复核）
---

# 检索策略优化

::: info 代码阅读约定
本页纯函数展示度量与 RRF；涉及 `vector_db`、`llm`、查询改写和多路路由的类是接口示意。实际适配器须统一文档 ID、返回类型、分数方向、超时与权限过滤，不能直接拼接不同库的返回对象。
:::

> 掌握多种检索技术，构建高效准确的RAG检索系统

## 2026 阅读提示

检索策略的核心不再是“向量库选哪一个”，而是如何让正确证据稳定进入模型上下文。读这一页时，可以把检索链路拆成四个可测问题：

1. **查询是否被理解**：原始 query 是否需要改写、拆分、路由或补充同义词？
2. **候选是否召回**：标准答案所需证据是否进入 top-k？关键词、向量、混合检索分别表现如何？
3. **证据是否排前**：召回到了但排在后面时，应优先看 rerank、过滤规则和字段权重。
4. **上下文是否可用**：进入 prompt 的片段是否保留标题、时间、权限和相互关系？

生产系统里，BM25、向量检索、混合检索、rerank 和 Graph/metadata 过滤通常不是互斥选项，而是一组可组合的召回通道。真正的选择依据应该来自评估集和失败样本，而不是工具默认值。

## 🎯 核心概念

### 什么是检索策略？

**检索策略**是RAG系统中负责从向量数据库中找到与用户查询最相关文档的技术方案。它直接影响RAG系统的准确性和响应速度。

**检索的核心挑战**：
- **召回率 vs 精确率**：如何在检索更多相关内容的同时减少噪音
- **语义理解 vs 精确匹配**：平衡语义相似性和关键词匹配
- **效率 vs 质量**：在检索速度和结果质量间找到平衡

### 当前检索方法的局限性

> 来源：[RAG技术的5种范式](https://hub.baai.ac.cn/view/43613)

::: warning 关键洞察
当前RAG系统的大多数检索方法依赖于**关键词和相似性搜索**，这限制了RAG系统的整体准确性。如果检索(R)部分提供的上下文不相关，无论生成(G)部分如何优化，答案也将不准确。
:::

| 检索方法 | 技术原理 | 局限性 |
|----------|----------|--------|
| **BM25** | 基于词频(TF)、逆文档频率(IDF)和文档长度 | 无法捕捉语义关系 |
| **密集向量** | k近邻(KNN)算法，余弦相似度 | 依赖Embedding模型质量 |
| **稀疏编码器** | 扩展术语映射，保持高维解释性 | 处理复杂查询能力有限 |

### 检索评估指标速查

| 指标 | 公式要点 | 作用 |
|------|----------|------|
| **Recall@K** | 检索到的相关文档 / 总相关文档 | 衡量召回能力 |
| **Precision@K** | 检索到的相关文档 / 检索的总文档 | 衡量精确度 |
| **F1@K** | Recall和Precision的调和平均 | 综合评估 |
| **MAP** | 平均精度均值 | 排序质量 |

---

## 📊 检索策略分类

### 按检索方式分类

| 检索类型 | 原理 | 优势 | 劣势 | 适用场景 |
|----------|------|------|------|----------|
| **稠密检索** | 向量相似度计算 | 语义理解强 | 依赖模型质量 | 概念性查询 |
| **稀疏检索** | 关键词匹配（BM25） | 精确匹配好 | 缺乏语义理解 | 特定术语查询 |
| **混合检索** | 稠密+稀疏融合 | 兼顾两者优势 | 复杂度高 | 通用场景 |

### 按查询处理分类

| 策略 | 技术要点 | 效果 |
|------|----------|------|
| **原始查询** | 直接使用用户输入 | 简单直接 |
| **查询改写** | LLM重写查询语句 | 提升匹配度 |
| **查询扩展** | 添加同义词、相关词 | 提高召回率 |
| **多查询** | 分解为多个子查询 | 覆盖更全面 |

---

## 🔍 稠密检索（Dense Retrieval）

### 核心原理

稠密检索通过计算查询向量与文档向量的相似度来匹配相关内容：

```python
# 稠密检索的数学原理
similarity = cosine_similarity(query_vector, document_vector)
# 或使用点积
similarity = dot_product(query_vector, document_vector)
```

### 相似度计算方法

#### 1. 余弦相似度（推荐）

标题保留旧链接；度量必须遵循模型训练与索引配置，余弦不是所有模型的通用最优选择。
```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("余弦相似度不接受零向量")
    return dot_product / (norm1 * norm2)

# 示例
query_vec = [0.1, 0.2, 0.3]
doc_vec = [0.15, 0.18, 0.32]
sim = cosine_similarity(query_vec, doc_vec)
print(f"相似度: {sim:.3f}")  # 具体数值由上面的向量计算
```

#### 2. 欧几里得距离
```python
def euclidean_distance(vec1, vec2):
    """欧几里得距离（越小越相似）"""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

# 转换为相似度分数
def euclidean_similarity(vec1, vec2):
    distance = euclidean_distance(vec1, vec2)
    return 1 / (1 + distance)  # 距离越小，相似度越高
```

### 实战代码

下例是适配器接口示意，`vector_db.search` 需实现为所选数据库调用，并明确 `score` 越大越好；数据库返回距离时先转换方向。查询与入库须使用匹配的编码配置。

```python
class DenseRetriever:
    def __init__(self, embedding_model, vector_db):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float | None = None):
        """稠密检索实现"""
        # 1. 查询向量化
        query_vector = self.embedding_model.encode(query)
        
        # 2. 向量检索
        results = self.vector_db.search(
            vector=query_vector,
            top_k=top_k * 2,  # 多检索一些候选
            metric="cosine"
        )
        
        # 3. 相似度过滤
        filtered_results = []
        for result in results:
            if threshold is None or result.score >= threshold:
                filtered_results.append(result)
        
        return filtered_results[:top_k]

# 使用示例
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
retriever = DenseRetriever(model, vector_db)

results = retriever.retrieve("什么是RAG技术？", top_k=5)
for result in results:
    print(f"相似度: {result.score:.3f} | 内容: {result.text[:100]}...")
```

---

## 🔤 稀疏检索（Sparse Retrieval）

### BM25算法详解

BM25（Best Matching 25）是最经典的稀疏检索算法，基于词频-逆文档频率（TF-IDF）改进：

**BM25公式**：
```
BM25(q,d) = Σ IDF(qi) * (f(qi,d) * (k1 + 1)) / (f(qi,d) + k1 * (1 - b + b * |d|/avgdl))
```

其中：
- `f(qi,d)`：词qi在文档d中的频率
- `|d|`：文档d的长度
- `avgdl`：平均文档长度
- `k1`, `b`：调节参数

### 实战实现

```python
from rank_bm25 import BM25Okapi
import jieba

class SparseRetriever:
    def __init__(self, documents):
        # 中文分词
        self.tokenized_docs = [list(jieba.cut(doc)) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.documents = documents
    
    def retrieve(self, query: str, top_k: int = 5):
        """BM25检索"""
        # 查询分词
        tokenized_query = list(jieba.cut(query))
        
        # 计算BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 排序获取top-k
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'score': scores[idx],
                'doc_id': str(idx),
                'index': idx
            })
        
        return results

# 使用示例
documents = [
    "检索增强生成（RAG）技术结合了信息检索和文本生成",
    "向量数据库是存储高维向量并支持相似性搜索的数据库",
    "自然语言处理中的预训练模型如BERT改变了NLP领域"
]

sparse_retriever = SparseRetriever(documents)
results = sparse_retriever.retrieve("RAG技术原理", top_k=2)

for result in results:
    print(f"BM25分数: {result['score']:.3f}")
    print(f"内容: {result['text']}")
    print("---")
```

---

## 🔀 混合检索（Hybrid Retrieval）

### 核心思想

混合检索结合稠密检索和稀疏检索的优势，通过加权融合获得更好的检索效果。

### 融合策略

#### 1. 分数加权融合

加权融合的前提是两路分数经过明确的尺度变换，并使用相同的稳定文档 ID。余弦分数范围为 `[-1, 1]`，BM25 的范围依赖语料、分词器和实现；把 BM25 直接过 Sigmoid 并不能让两路分数具有相同语义。没有标注集时，先用下面的 RRF 建立基线。

若要做线性融合，应在训练/验证集上拟合归一化或排序模型，再在独立测试集比较 Recall@K、NDCG@K、无答案误接受率及延迟。未被某一路召回表示“本路未观测”，不能不加说明地当成原始分数 0；可以对候选并集补算两路分数，或给学习排序器增加缺失标记。权重 0.7/0.3 只是一组待实验配置，不是行业默认值。

两路检索适配器统一返回：

```python
# 接口约定示例；不是某个向量库的原生返回类型。
result = {"doc_id": "handbook:v3:chunk-12", "text": "年假规定……", "score": 0.62}
```

#### 2. 倒数排名融合（RRF）

以下函数可独立运行；调用示例中的两路检索器需按上节契约返回同一文档空间的 `doc_id`。单路内部先去重，避免同一文档重复加分。

```python
def reciprocal_rank_fusion(results_list, k=60):
    """输入为已按分数降序排列且包含稳定 doc_id 的字典列表。"""
    if k < 0:
        raise ValueError("k 必须非负")
    doc_scores = {}
    
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            doc_id = result["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            
            # RRF公式：1/(k + rank)
            rrf_score = 1 / (k + rank + 1)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += rrf_score
            else:
                doc_scores[doc_id] = {
                    'score': rrf_score,
                    'text': result.get('text', result.get('content', ''))
                }
    
    # 按RRF分数排序
    sorted_results = sorted(
        doc_scores.items(), 
        key=lambda x: x[1]['score'], 
        reverse=True
    )
    
    return [
        {
            'doc_id': doc_id,
            'text': data['text'],
            'rrf_score': data['score']
        }
        for doc_id, data in sorted_results
    ]

# 使用示例
dense_results = dense_retriever.retrieve("RAG技术", top_k=10)
sparse_results = sparse_retriever.retrieve("RAG技术", top_k=10)

rrf_results = reciprocal_rank_fusion([dense_results, sparse_results])
print("RRF融合结果:")
for result in rrf_results[:5]:
    print(f"RRF分数: {result['rrf_score']:.3f}")
    print(f"内容: {result['text'][:100]}...")
    print("---")
```

---

## 🚀 高级检索策略

### 1. 查询改写与扩展

#### Query Rewriting
```python
from openai import OpenAI

class QueryRewriter:
    def __init__(self):
        self.client = OpenAI()
    
    def rewrite_query(self, original_query: str):
        """使用LLM改写查询"""
        prompt = f"""
        请将以下用户查询改写为更适合检索的形式，要求：
        1. 保持原意不变
        2. 使用更精确的技术术语
        3. 扩展关键概念
        4. 如果查询模糊，请提供多个可能的解释
        
        原查询：{original_query}
        
        改写后的查询：
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()

# 使用示例
rewriter = QueryRewriter()

original = "RAG是什么"
rewritten = rewriter.rewrite_query(original)
print(f"原查询: {original}")
print(f"改写后: {rewritten}")

# 使用改写后的查询进行检索
results = retriever.retrieve(rewritten, top_k=5)
```

#### HyDE（Hypothetical Document Embeddings）
```python
class HyDERetriever:
    def __init__(self, llm_client, embedding_model, vector_db):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.vector_db = vector_db
    
    def generate_hypothetical_answer(self, query: str):
        """生成假设性回答"""
        prompt = f"""
        请基于以下问题生成一个详细、准确的回答。即使你不确定答案，也要生成一个合理的假设性回答。
        
        问题：{query}
        
        回答：
        """
        
        response = self.llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def retrieve(self, query: str, top_k: int = 5):
        """HyDE检索策略"""
        # 1. 生成假设性文档
        hypothetical_doc = self.generate_hypothetical_answer(query)
        
        # 2. 对假设性文档进行向量化
        hypo_vector = self.embedding_model.encode(hypothetical_doc)
        
        # 3. 使用假设性文档向量进行检索
        results = self.vector_db.search(
            vector=hypo_vector,
            top_k=top_k,
            metric="cosine"
        )
        
        return results

# 使用示例
hyde_retriever = HyDERetriever(openai_client, embedding_model, vector_db)
results = hyde_retriever.retrieve("RAG系统的优缺点", top_k=5)
```

### 2. 多路召回

```python
class MultiPathRetriever:
    def __init__(self, retrievers_config):
        self.retrievers = retrievers_config
    
    def multi_retrieve(self, query: str, top_k_per_path: int = 10, final_top_k: int = 5):
        """多路召回策略"""
        all_results = []
        
        # 1. 多种策略并行检索
        for name, retriever in self.retrievers.items():
            try:
                results = retriever.retrieve(query, top_k_per_path)
                # 添加来源标识
                for result in results:
                    result['source_retriever'] = name
                all_results.extend(results)
                print(f"{name} 检索到 {len(results)} 条结果")
            except Exception as e:
                print(f"{name} 检索失败: {e}")
        
        # 2. 去重合并
        unique_results = self._deduplicate_results(all_results)
        
        # 3. 重新排序
        final_results = self._rerank_results(unique_results, query)
        
        return final_results[:final_top_k]
    
    def _deduplicate_results(self, results):
        """结果去重"""
        seen_texts = set()
        unique_results = []
        
        for result in results:
            text_hash = hash(result['text'][:100])  # 使用前100字符去重
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _rerank_results(self, results, query):
        """结果重排序"""
        # 这里可以使用更复杂的排序逻辑
        # 简单示例：按分数排序
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)

# 配置多个检索器
retrievers_config = {
    'dense': dense_retriever,
    'sparse': sparse_retriever,
    'hyde': hyde_retriever
}

multi_retriever = MultiPathRetriever(retrievers_config)
results = multi_retriever.multi_retrieve("RAG系统设计原则", final_top_k=5)

print("多路召回结果:")
for result in results:
    print(f"来源: {result['source_retriever']}")
    print(f"分数: {result.get('score', 'N/A')}")
    print(f"内容: {result['text'][:150]}...")
    print("---")
```

---

## 📊 检索效果优化

### 1. 参数调优指南

| 参数 | 建议值 | 影响 | 调优策略 |
|------|--------|------|----------|
| **top_k** | 5-20 | 召回数量 | 根据下游处理能力调整 |
| **相似度阈值** | 无通用数值 | 结果质量 | 通过验证集确定最优值 |
| **混合权重α** | 在验证集搜索 | 检索策略平衡 | A/B测试确定 |
| **chunk_size** | 500-1000 | 文档粒度 | 平衡上下文与精确性 |

### 2. 检索质量评估

统一使用[评估章节](/llms/rag/evaluation)中的去重 ID 指标函数，避免不同文章各实现一套 Recall/MRR/NDCG。查询的相关文档集合不能为空却仍被按普通 Recall 除零；无答案问题应单独评估。检索器返回对象或字典时先通过适配器转换，不要将 `.score` 与 `result['score']` 混写。

先冻结语料与切分，对同题比较检索路径；调参使用开发集，最终报告使用独立测试集。除了平均分，保留专名、跨语言、时间条件、多跳、权限过滤、无答案各桶的样本数与表现。若有答案查询的召回提高但无答案误接受率恶化，应报告这一取舍。

---

## 🔧 混合检索分数归一化

### 融合困境：支配性特征问题

归一化解决数值尺度问题，校准解决分数与真实事件频率是否一致的问题，二者不能互换。同为 0.8 的余弦、排序分数和校准概率，含义不同。排序有效也不代表分数可以跨查询比较。

### 归一化方法对比

#### 1. Min-Max归一化

对当前候选集做 `(s - min) / (max - min)` 可用于相对融合，但候选集变化会改变数值；全是无关文档时，最高分仍被映射为 1。空列表应返回空列表，所有分数相等时应明确固定输出或退回排名融合。

#### 2. Sigmoid函数变换（推荐）

这里保留旧标题以兼容链接，但不再把 Sigmoid 作为通用推荐。`sigmoid(s)` 只是单调变换，不会修复模型偏差，也不会自动得到 `P(相关 | s)`。按本次候选均值和标准差计算的 Sigmoid 仍是相对分数；均值、标准差本身也会受到离群值影响。

需要概率时，应先定义标签（例如“片段足以支持该问题的关键答案”），在代表线上流量的独立校准集拟合 Platt scaling 或 isotonic regression，再检查可靠性分桶、Brier 分数和领域迁移后的偏差。阈值依据误接受/误拒绝成本选择，模型、语料或候选池变化后重新评估。

### Cross-Encoder Logits的Sigmoid变换

部分模型返回 logits，部分返回经过激活的分数或多个类别的输出。先确认模型卡和 SDK，避免重复 Sigmoid。对单个标量 logit 做 Sigmoid 不改变排序，因此只做 rerank 时不必转换。[Sentence Transformers 官方示例](https://sbert.net/docs/cross_encoder/usage/usage.html) 明确说明了这一点。

```python
from math import exp

def sigmoid(logit):
    # 数值稳定的标量实现：输出只是压缩分数，不宣称已校准。
    if logit >= 0:
        return 1 / (1 + exp(-logit))
    value = exp(logit)
    return value / (1 + value)

logits = [8.5, 2.1, -0.5, -2.3]
scores = [sigmoid(value) for value in logits]
```

拒答应综合证据覆盖、冲突、权限和校准后的阈值。仅有高排序分数，不能证明资料足以回答，更不能保证生成过程不会出错。

## 🔴 异构向量空间失配问题

### 核心问题：Embedding模型不一致

检索要求查询与文档编码器属于**兼容、共同训练或明确对齐的检索空间**。它们不必共享权重：[DPR](https://arxiv.org/abs/2004.04906) 就使用查询与段落编码器。不能把两个任意模型的输出仅因维数相等就混用。

### 负相似度的数学本质

余弦为负只表示夹角大于 90°，不表示语言上的否定、反义，也不能据此诊断模型失配。同一合法模型的无关句对也可产生负分。随机单位向量的高维近正交现象需要特定分布假设；它不能推出真实模型混用后必有一半负分，Johnson–Lindenstrauss 引理也不是这条诊断的依据。

### 失配的三大根源

#### 1. 分词器（Tokenizer）失配

将 A 模型的 tokenizer 直接接到 B 权重上属于输入契约错误；两个独立模型各用自己的 tokenizer，则不是 token ID 相撞的问题，而是其输出坐标和训练目标没有保证兼容。查询前缀、文档前缀和池化方法也属于模型契约。

#### 2. 各向异性与锥形效应

嵌入分布可能有偏置，但不能仅凭示意图推断不同模型的中心轴随机、相互排斥或必然负分。应对同一批正负样本绘制分数分布，检查方差、范数、重复向量和召回变化。

#### 3. 训练目标函数差异

句子相似度、问答检索、分类和代码检索的相关性定义并不相同。模型即使维数、架构一致，也可能因训练任务或版本变化而不兼容。

### 解决方案

将模型 revision、tokenizer、query/document 前缀、池化、归一化、维数、距离函数与索引版本共同保存。先验证原始文本到向量的链路，再用小规模精确搜索与 ANN 对照：精确搜索也差时查编码/语料；只有 ANN 差时查索引参数和过滤。

升级时构建独立新索引，用对应查询编码器做影子流量和回归；回填完成后切换别名，并保留回滚版本。线性空间对齐需要同一批文本的成对旧/新向量和独立测试，不能在只有一组旧向量时凭空求解。

## 📉 短查询高分异常与Rerank修正

> 来源：[混合检索中短查询高分异常的深度剖析与神经重排序的修正机制](https://dd-ff.blog.csdn.net/article/details/156067548)

### 问题现象

::: danger 反直觉的病态现象
输入"Hello"、"系统"、"测试"等**短查询或高频通用词**，混合检索系统往往以**较高排序分数**返回大量**完全不相关**的文档。
:::

在RAG系统中，这种召回噪声是致命的——它直接污染LLM的输入上下文，导致幻觉。

### 稀疏检索（BM25）的病理

#### 1. IDF权重崩溃

高频词通常区分度低，但 IDF 的符号取决于实现，并非所有 BM25 都返回负值。还应检查中文分词、停用词、专名和字段权重；“Hello”也可能是代码检索中的有效关键词。先确认查询意图，不能按长度一刀切。

#### 2. 文档长度归一化的副作用

长度归一化会改变排序，短片段有时得分更高，但并不意味着内容贫乏。用长短文档分桶检查错误，再调 `b`、字段或切分策略。

### 稠密检索的几何陷阱

#### 1. 语义熵与向量模糊性

“系统”缺少任务范围，可能有多个合理解释；这种歧义应通过对话状态、澄清或路由处理。不能由查询长短推导向量位置或“语义熵”。

#### 2. 各向异性与枢纽点问题

若相同通用文档在大量无关查询中反复进入 top-k，应统计文档命中频率，并排查模板文本、重复切块和向量范数。枢纽效应只是可能解释，需要分布证据支持。

### RRF融合的放大效应

RRF 根据名次融合，不读取原始分数，也不判断问题是否有答案。多路检索同时把无关片段排高时，融合仍会保留它；这是方法的边界，不能把融合分数称为“较高排序分数”。公式及候选窗口参数见 [Elasticsearch RRF 文档](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)。

### 解决方案：神经重排序（Rerank）

#### Cross-Encoder vs Bi-Encoder

| 架构 | 计算方式 | 优势 | 劣势 |
|------|----------|------|------|
| **Bi-Encoder** | 独立编码，向量点积 | 快速，支持ANN | 受几何陷阱影响 |
| **Cross-Encoder** | `[CLS] Q [SEP] D`联合编码 | 联合建模词元交互，仍可能误排 | 计算成本高 |

#### Rerank如何修正短查询异常

联合编码可补充匹配信号，但不是短查询问题的万能修复。先区分有效短查询（产品型号、错误码）与缺少任务意图的输入；后者可能应澄清或走普通对话。可运行的排序过滤函数与分数契约见[重排章节](/llms/rag/rerank)，此处不重复一套固定 0.3 阈值实现。

#### Rerank修正机制

1. **细粒度交互**：评估候选是否真正覆盖问题，而非只共享主题词
2. **截断检查**：确认输入上限未删掉关键条件，联合编码也可能受长度偏差影响
3. **分数校准**：如需拒答阈值，另用带标签数据校准；输出在 0–1 范围不等于已经校准

::: tip 工程建议
- 召回阶段多检索一些候选（如Top-100），容忍噪声
- Rerank阶段使用高质量Cross-Encoder进行精排
- 在验证集上确定阈值，并评估无答案误接受率和有答案误拒绝率；无证据时明确拒答
:::

---

## ⚠️ 常见问题与解决

### 问题1：检索结果不相关

**现象**：返回的文档与查询语义不匹配  
**原因分析**：
- Embedding模型不适配
- 查询表达不准确
- 文档切分粒度不当

**解决方案**：
```python
# 1. 查询预处理
def preprocess_query(query):
    """查询预处理"""
    # 去除停用词
    query = remove_stopwords(query)
    # 添加上下文信息
    if len(query.split()) < 3:
        query = f"请详细介绍{query}"
    return query

# 2. 结果后处理
def postprocess_results(results, query, threshold=0.6):
    """结果后处理"""
    filtered = []
    for result in results:
        # 语义相关性二次验证
        if semantic_similarity(query, result['text']) > threshold:
            filtered.append(result)
    return filtered
```

### 问题2：检索速度慢

**现象**：检索响应时间过长  
**优化策略**：

```python
# 1. 向量缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text):
    return embedding_model.encode(text)

# 2. 批量检索优化
class BatchRetriever:
    def __init__(self, retriever, batch_size=32):
        self.retriever = retriever
        self.batch_size = batch_size
    
    def batch_retrieve(self, queries):
        results = {}
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            # 批量向量化
            vectors = embedding_model.encode(batch)
            # 批量检索
            for query, vector in zip(batch, vectors):
                results[query] = self.retriever.search_by_vector(vector)
        return results
```

---

## 检索实验与故障定位

固定语料快照、权限过滤和 chunk ID，按专名/编号、口语改写、跨文档、无答案分桶，比较 BM25、向量、RRF 三条基线。分别记录候选 Recall@K、重排后 NDCG@K、最终证据覆盖、p95 延迟；不要用答案流畅度代替检索标签。

- 原始文档有答案但标准证据未入库：修解析、切分和版本。
- 精确向量搜索命中、ANN 不命中：调索引或过滤，别先换模型。
- 候选池已命中但 top-k 丢失：修融合或重排。
- top-k 有依据但答案错误：转查上下文截断、引用和生成。

## 相关阅读

- [RAG范式演进](/llms/rag/paradigms) - 了解RAG技术发展脉络
- [文档切分策略](/llms/rag/chunking) - 影响检索粒度的切分技术
- [Embedding技术](/llms/rag/embedding) - 稠密检索的基础
- [向量数据库](/llms/rag/vector-db) - 检索的底层存储
- [重排序优化](/llms/rag/rerank) - 检索后的精排技术

> **相关文章**：
> - [混合搜索中的分数归一化方法深度解析](https://dd-ff.blog.csdn.net/article/details/156072979)
> - [异构向量空间失配机制与负余弦相似度的深层拓扑学解析](https://dd-ff.blog.csdn.net/article/details/156068492)
> - [混合检索中短查询高分异常的深度剖析与神经重排序的修正机制](https://dd-ff.blog.csdn.net/article/details/156067548)
> - [高级RAG技术全景：从原理到实战](https://dd-ff.blog.csdn.net/article/details/149396526)
> - [从“拆文档”到“通语义”：RAG+知识图谱如何破解大模型“失忆+幻觉”难题？](https://dd-ff.blog.csdn.net/article/details/149354855)
> - [从“失忆”到“过目不忘”：RAG技术如何给LLM装上“外挂大脑”？](https://dd-ff.blog.csdn.net/article/details/149348018)

> **外部资源**：
> - [LlamaIndex检索指南](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/) - 检索器详细文档
> - [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/) - 多种检索器实现
