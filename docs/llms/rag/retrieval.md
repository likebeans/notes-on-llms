---
title: 检索策略优化
description: RAG系统中的检索技术与策略详解
---

# 检索策略优化

> 掌握多种检索技术，构建高效准确的RAG检索系统

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
```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 示例
query_vec = [0.1, 0.2, 0.3]
doc_vec = [0.15, 0.18, 0.32]
sim = cosine_similarity(query_vec, doc_vec)
print(f"相似度: {sim:.3f}")  # 输出：0.999
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

```python
class DenseRetriever:
    def __init__(self, embedding_model, vector_db):
        self.embedding_model = embedding_model
        self.vector_db = vector_db
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.7):
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
            if result.score >= threshold:
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

```python
class HybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, alpha=0.7):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha  # 稠密检索权重
    
    def retrieve(self, query: str, top_k: int = 5):
        """混合检索实现"""
        # 1. 分别获取稠密和稀疏检索结果
        dense_results = self.dense_retriever.retrieve(query, top_k * 2)
        sparse_results = self.sparse_retriever.retrieve(query, top_k * 2)
        
        # 2. 构建文档ID到分数的映射
        doc_scores = {}
        
        # 稠密检索分数
        for result in dense_results:
            doc_id = result.get('doc_id', result.get('index'))
            doc_scores[doc_id] = doc_scores.get(doc_id, {})
            doc_scores[doc_id]['dense'] = result.score
            doc_scores[doc_id]['text'] = result.text
        
        # 稀疏检索分数
        for result in sparse_results:
            doc_id = result.get('doc_id', result.get('index'))
            doc_scores[doc_id] = doc_scores.get(doc_id, {})
            doc_scores[doc_id]['sparse'] = result['score']
            doc_scores[doc_id]['text'] = result['text']
        
        # 3. 分数归一化和融合
        final_results = []
        for doc_id, scores in doc_scores.items():
            dense_score = scores.get('dense', 0)
            sparse_score = scores.get('sparse', 0)
            
            # 归一化处理
            dense_norm = self._normalize_score(dense_score, 'cosine')
            sparse_norm = self._normalize_score(sparse_score, 'bm25')
            
            # 加权融合
            final_score = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm
            
            final_results.append({
                'doc_id': doc_id,
                'text': scores['text'],
                'final_score': final_score,
                'dense_score': dense_score,
                'sparse_score': sparse_score
            })
        
        # 4. 按最终分数排序
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        return final_results[:top_k]
    
    def _normalize_score(self, score, score_type):
        """分数归一化"""
        if score_type == 'cosine':
            # 余弦相似度已在[0,1]范围内
            return score
        elif score_type == 'bm25':
            # BM25分数归一化到[0,1]
            return 1 / (1 + np.exp(-score))  # sigmoid归一化
        return score

# 使用示例
hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    alpha=0.7  # 70%稠密检索，30%稀疏检索
)

results = hybrid_retriever.retrieve("RAG系统架构设计", top_k=5)
for result in results:
    print(f"综合分数: {result['final_score']:.3f}")
    print(f"稠密分数: {result['dense_score']:.3f}")
    print(f"稀疏分数: {result['sparse_score']:.3f}")
    print(f"内容: {result['text'][:100]}...")
    print("---")
```

#### 2. 倒数排名融合（RRF）

```python
def reciprocal_rank_fusion(results_list, k=60):
    """倒数排名融合算法"""
    doc_scores = {}
    
    for results in results_list:
        for rank, result in enumerate(results):
            doc_id = result.get('doc_id', result.get('index'))
            
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
| **相似度阈值** | 0.7-0.85 | 结果质量 | 通过验证集确定最优值 |
| **混合权重α** | 0.6-0.8 | 检索策略平衡 | A/B测试确定 |
| **chunk_size** | 500-1000 | 文档粒度 | 平衡上下文与精确性 |

### 2. 检索质量评估

```python
class RetrievalEvaluator:
    def __init__(self, test_queries, ground_truth):
        self.test_queries = test_queries
        self.ground_truth = ground_truth
    
    def evaluate_retrieval(self, retriever, top_k=5):
        """评估检索性能"""
        metrics = {
            'recall': [],
            'precision': [],
            'mrr': [],  # Mean Reciprocal Rank
            'ndcg': []  # Normalized Discounted Cumulative Gain
        }
        
        for query_id, query in self.test_queries.items():
            results = retriever.retrieve(query, top_k)
            relevant_docs = self.ground_truth[query_id]
            
            # 计算各项指标
            retrieved_docs = [r.get('doc_id') for r in results]
            
            # Recall@K
            recall = len(set(retrieved_docs) & set(relevant_docs)) / len(relevant_docs)
            metrics['recall'].append(recall)
            
            # Precision@K
            precision = len(set(retrieved_docs) & set(relevant_docs)) / len(retrieved_docs)
            metrics['precision'].append(precision)
            
            # MRR
            mrr = self._calculate_mrr(retrieved_docs, relevant_docs)
            metrics['mrr'].append(mrr)
        
        # 计算平均值
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        return avg_metrics
    
    def _calculate_mrr(self, retrieved, relevant):
        """计算平均倒数排名"""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0

# 使用示例
test_queries = {
    'q1': 'RAG技术原理',
    'q2': '向量数据库选型',
    # ... 更多测试查询
}

ground_truth = {
    'q1': ['doc_1', 'doc_5', 'doc_12'],  # 相关文档ID
    'q2': ['doc_3', 'doc_8'],
    # ... 对应的相关文档
}

evaluator = RetrievalEvaluator(test_queries, ground_truth)
metrics = evaluator.evaluate_retrieval(dense_retriever)

print("检索性能评估:")
for metric, value in metrics.items():
    print(f"{metric.upper()}: {value:.3f}")
```

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

## 相关阅读

- [RAG范式演进](/llms/rag/paradigms) - 了解RAG技术发展脉络
- [文档切分策略](/llms/rag/chunking) - 影响检索粒度的切分技术
- [Embedding技术](/llms/rag/embedding) - 稠密检索的基础
- [向量数据库](/llms/rag/vector-db) - 检索的底层存储
- [重排序优化](/llms/rag/rerank) - 检索后的精排技术

> **相关文章**：
> - [高级RAG技术全景：从原理到实战](https://dd-ff.blog.csdn.net/article/details/149396526)
> - [从“拆文档”到“通语义”：RAG+知识图谱如何破解大模型“失忆+幻觉”难题？](https://dd-ff.blog.csdn.net/article/details/149354855)
> - [从“失忆”到“过目不忘”：RAG技术如何给LLM装上“外挂大脑”？](https://dd-ff.blog.csdn.net/article/details/149348018)

> **外部资源**：
> - [LlamaIndex检索指南](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/) - 检索器详细文档
> - [LangChain Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/) - 多种检索器实现
