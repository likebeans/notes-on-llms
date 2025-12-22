---
title: 重排序技术详解
description: RAG系统中的检索结果重排序与精排技术
---

# 重排序技术详解

> 提升检索精度的关键技术，从粗排到精排的核心环节

## 🎯 核心概念

### 什么是重排序（Rerank）？

**重排序**是RAG检索流程中的精排环节，对初步检索得到的候选文档进行二次排序，筛选出最相关的内容。

**典型流程**：
```
用户查询 → 粗排检索(Top-100) → 重排序(Top-10) → LLM生成
```

### 为什么需要重排序？

::: tip 核心价值
**精度提升**：通过更复杂的模型提高匹配精度  
**计算平衡**：在速度和质量间找到最佳平衡点  
**多模态融合**：结合多种相关性信号进行综合判断
:::

**重排序的优势**：
- 使用更精确但计算量大的模型
- 考虑查询和文档的交互特征
- 整合多种相关性信号
- 减少传递给LLM的噪音

---

## 📊 重排序方法分类

### 按模型架构分类

| 类型 | 原理 | 优势 | 劣势 | 适用场景 |
|------|------|------|------|----------|
| **Cross-Encoder** | 查询-文档联合编码 | 精度最高 | 计算量大 | 高质量要求 |
| **Bi-Encoder** | 查询和文档分别编码 | 速度快 | 交互不足 | 大规模检索 |
| **Late Interaction** | 延迟交互计算 | 平衡精度速度 | 实现复杂 | 平衡场景 |

### 按技术路径分类

| 方法 | 技术要点 | 特点 |
|------|----------|------|
| **基于传统ML** | 特征工程+分类器 | 可解释性强 |
| **基于深度学习** | 神经网络相关性建模 | 效果更好 |
| **基于LLM** | 大模型判断相关性 | 理解能力强 |
| **多信号融合** | 结合多种相关性指标 | 综合性能好 |

---

## 🧠 Cross-Encoder 重排序

### 核心原理

Cross-Encoder将查询和文档拼接输入，通过Transformer进行联合编码，输出相关性分数：

```python
# Cross-Encoder架构
input = "[CLS] query [SEP] document [SEP]"
score = CrossEncoder(input)  # 输出0-1相关性分数
```

### 实战实现

```python
from sentence_transformers import CrossEncoder
import numpy as np

class CrossEncoderReranker:
    def __init__(self, model_name='BAAI/bge-reranker-large'):
        """初始化Cross-Encoder重排序器"""
        self.model = CrossEncoder(model_name)
        
    def rerank(self, query: str, documents: list, top_k: int = 5):
        """重排序实现"""
        if not documents:
            return []
        
        # 1. 构建查询-文档对
        pairs = [(query, doc['text']) for doc in documents]
        
        # 2. 批量计算相关性分数
        scores = self.model.predict(pairs)
        
        # 3. 重新排序
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = float(score)
            scored_docs.append(doc_copy)
        
        # 4. 按分数降序排列
        ranked_docs = sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)
        
        return ranked_docs[:top_k]

# 使用示例
reranker = CrossEncoderReranker()

# 假设从第一轮检索获得的候选文档
candidates = [
    {'text': 'RAG技术结合了检索和生成，提升了大模型的知识获取能力', 'id': 'doc1'},
    {'text': '向量数据库是存储高维向量的专用数据库系统', 'id': 'doc2'},
    {'text': '检索增强生成通过外部知识库增强语言模型的生成质量', 'id': 'doc3'},
]

query = "什么是RAG技术？"
reranked = reranker.rerank(query, candidates, top_k=3)

print("重排序结果:")
for i, doc in enumerate(reranked):
    print(f"{i+1}. 分数: {doc['rerank_score']:.3f}")
    print(f"   内容: {doc['text']}")
    print("---")
```

### Cross-Encoder输出处理：Logits到概率

> 来源：[混合搜索中的分数归一化方法深度解析](https://dd-ff.blog.csdn.net/article/details/156072979)

::: warning 关键注意
Cross-Encoder（如bge-reranker）输出的是**原始Logits**（对数几率），定义域为(-∞, +∞)。直接将Logits与其他分数（如余弦相似度）混合是**数学谬误**。
:::

```python
import numpy as np

class CrossEncoderRerankerWithCalibration:
    """带概率校准的Cross-Encoder重排序器"""
    
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
    
    def _sigmoid(self, x):
        """将Logits转换为概率"""
        return 1 / (1 + np.exp(-np.array(x)))
    
    def rerank(self, query: str, documents: list, top_k: int = 5, 
               return_probabilities: bool = True):
        """
        重排序并返回校准后的概率分数
        
        Cross-Encoder训练目标是BCEWithLogitsLoss:
        - Logit > 0 意味着 P(相关) > 0.5
        - Logit = 8.5  -> P = 0.9998 (高相关)
        - Logit = -2.3 -> P = 0.0911 (低相关)
        """
        if not documents:
            return []
        
        pairs = [(query, doc['text']) for doc in documents]
        
        # 获取原始Logits
        logits = self.model.predict(pairs)
        
        # 转换为概率（推荐）
        if return_probabilities:
            scores = self._sigmoid(logits)
        else:
            scores = logits
        
        scored_docs = []
        for doc, score, logit in zip(documents, scores, logits):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = float(score)
            doc_copy['raw_logit'] = float(logit)
            scored_docs.append(doc_copy)
        
        ranked_docs = sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)
        return ranked_docs[:top_k]

# 使用示例
reranker = CrossEncoderRerankerWithCalibration()
results = reranker.rerank("什么是RAG？", candidates)

for doc in results:
    print(f"概率: {doc['rerank_score']:.3f} (Logit: {doc['raw_logit']:.2f})")
    # 概率: 0.998 (Logit: 6.21)  <- 高相关
    # 概率: 0.124 (Logit: -1.95) <- 低相关
```

**为什么必须转换为概率？**
- **分数可比性**：概率值[0,1]可与余弦相似度直接融合
- **阈值截断**：概率支持设置绝对质量阈值（如P<0.3拒绝回答）
- **幻觉抑制**：即使所有文档都不相关，也能识别出低概率

### 开源重排序模型对比

| 模型 | 语言 | 参数量 | MTEB排名 | 特点 |
|------|------|--------|----------|------|
| **bge-reranker-v2-m3** | 多语言 | 568M | Top 1 | 最新版本，推荐 |
| **bge-reranker-large** | 中英 | 560M | Top 3 | 性能优秀，中文友好 |
| **bge-reranker-base** | 中英 | 278M | Top 10 | 平衡性能与速度 |
| **jina-reranker-v2** | 多语言 | 278M | - | 多语言支持 |
| **ms-marco-cross-encoder** | 英文 | 340M | - | 经典英文模型 |

---

## 🛡️ Rerank修正检索异常

> 来源：[混合检索中短查询高分异常的深度剖析与神经重排序的修正机制](https://dd-ff.blog.csdn.net/article/details/156067548)

### 短查询高分异常问题

::: danger 病态现象
输入"Hello"、"系统"、"测试"等短查询时，混合检索往往以**极高置信度**返回**完全不相关**的文档。这在RAG中是致命的——噪声上下文直接导致LLM幻觉。
:::

**根本原因分析**：

| 检索阶段 | 失效机制 | 后果 |
|----------|----------|------|
| **BM25** | IDF权重崩溃 + 长度偏置 | 短碎片高分 |
| **向量检索** | 各向异性 + 枢纽点效应 | 通用文档高分 |
| **RRF融合** | 盲信排名，放大错误 | 噪声居榜首 |

### Cross-Encoder如何修正

**Bi-Encoder vs Cross-Encoder 对比**：

```
Bi-Encoder（向量检索）：
  Query  ────→ [Encoder] ────→ q_vec ─┐
                                       ├─→ cosine(q, d) → 受几何陷阱影响
  Doc    ────→ [Encoder] ────→ d_vec ─┘

Cross-Encoder（重排序）：
  [CLS] Query [SEP] Doc [SEP] ────→ [Transformer] ────→ 相关性分数
                                    ↑
                                    逐词交互，消除几何噪声
```

**修正机制**：

1. **消除几何噪声**：通过自注意力机制逐词分析，识别"Hello"与"用户协议"无语义蕴含关系
2. **解决长度偏置**：阅读完整上下文，识别文档中的"Hello"若只是孤立词汇则无法回答查询
3. **分数校准**：输出概率值，支持绝对阈值截断

### 阈值截断与幻觉抑制

```python
import numpy as np

class ThresholdedReranker:
    """带阈值截断的重排序器，用于抑制RAG幻觉"""
    
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', 
                 threshold=0.3, min_results=0):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.threshold = threshold
        self.min_results = min_results  # 最少返回数量（0表示可返回空）
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.array(x)))
    
    def rerank(self, query: str, documents: list, top_k: int = 5):
        """
        重排序并应用阈值截断
        
        关键：若所有文档相关性都低于阈值，返回空列表
        这优于返回噪声——让下游系统知道"无可靠答案"
        """
        if not documents:
            return [], "no_candidates"
        
        pairs = [(query, doc['text']) for doc in documents]
        logits = self.model.predict(pairs)
        probs = self._sigmoid(logits)
        
        scored_docs = []
        for doc, prob in zip(documents, probs):
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = float(prob)
            scored_docs.append(doc_copy)
        
        # 按分数排序
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # 阈值过滤
        filtered = [d for d in scored_docs if d['rerank_score'] >= self.threshold]
        
        # 判断结果状态
        if len(filtered) == 0:
            if self.min_results > 0:
                # 强制返回top结果，但标记为低置信
                return scored_docs[:self.min_results], "low_confidence"
            else:
                # 返回空，触发"无法回答"逻辑
                return [], "no_relevant_docs"
        
        return filtered[:top_k], "success"

# 使用示例
reranker = ThresholdedReranker(threshold=0.3)

# 正常查询
results, status = reranker.rerank("RAG技术的核心原理是什么？", candidates)
# status: "success", results: [相关文档...]

# 短查询/无关查询
results, status = reranker.rerank("Hello", candidates)
# status: "no_relevant_docs", results: []
# 下游系统应返回"抱歉，未找到相关信息"而非幻觉回答
```

::: tip 幻觉抑制的关键
- **Min-Max归一化失败**：即使全是烂文档，也会制造出1.0分，LLM强行回答
- **Sigmoid概率胜利**：提供绝对阈值，低于0.3时果断拒绝，避免污染LLM上下文
:::

### 完整两阶段检索流水线

```python
class TwoStageRAGRetriever:
    """生产级两阶段检索器"""
    
    def __init__(self, hybrid_retriever, reranker, 
                 recall_k=100, rerank_k=10, threshold=0.3):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.recall_k = recall_k
        self.rerank_k = rerank_k
        self.threshold = threshold
    
    def retrieve(self, query: str):
        """
        阶段1：召回（容忍噪声，追求高召回率）
        阶段2：精排（消除噪声，保证高精度）
        """
        # 阶段1：混合检索快速召回
        candidates = self.hybrid_retriever.retrieve(query, top_k=self.recall_k)
        
        if not candidates:
            return {
                'documents': [],
                'status': 'no_candidates',
                'message': '未检索到任何候选文档'
            }
        
        # 阶段2：Cross-Encoder精排
        pairs = [(query, doc['text']) for doc in candidates]
        logits = self.reranker.predict(pairs)
        probs = 1 / (1 + np.exp(-np.array(logits)))
        
        for doc, prob in zip(candidates, probs):
            doc['rerank_score'] = float(prob)
        
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # 阈值过滤
        filtered = [d for d in candidates if d['rerank_score'] >= self.threshold]
        
        if not filtered:
            return {
                'documents': [],
                'status': 'low_relevance',
                'message': '未找到与查询相关的高质量文档',
                'max_score': candidates[0]['rerank_score'] if candidates else 0
            }
        
        return {
            'documents': filtered[:self.rerank_k],
            'status': 'success',
            'message': f'找到 {len(filtered)} 个相关文档'
        }

# 集成到RAG系统
class RAGSystem:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def answer(self, query: str):
        result = self.retriever.retrieve(query)
        
        if result['status'] != 'success':
            # 关键：拒绝回答而非幻觉
            return f"抱歉，{result['message']}，无法回答您的问题。"
        
        context = "\n\n".join([d['text'] for d in result['documents']])
        return self.llm.generate(query, context)
```

---

## ⚡ 高效重排序策略

### 1. 分层重排序

```python
class HierarchicalReranker:
    def __init__(self, fast_reranker, precise_reranker):
        self.fast_reranker = fast_reranker      # 轻量级模型
        self.precise_reranker = precise_reranker # 精确模型
    
    def rerank(self, query: str, documents: list, 
               stage1_top_k: int = 20, final_top_k: int = 5):
        """分层重排序：先快速筛选，再精确排序"""
        
        # 第一层：快速筛选
        if len(documents) > stage1_top_k:
            stage1_results = self.fast_reranker.rerank(
                query, documents, top_k=stage1_top_k
            )
        else:
            stage1_results = documents
        
        # 第二层：精确重排
        final_results = self.precise_reranker.rerank(
            query, stage1_results, top_k=final_top_k
        )
        
        return final_results

# 使用示例
from sentence_transformers import SentenceTransformer

# 配置两层重排序器
fast_model = SentenceTransformer('BAAI/bge-base-zh-v1.5')  # 快速模型
precise_reranker = CrossEncoderReranker('BAAI/bge-reranker-large')  # 精确模型

class FastReranker:
    def __init__(self, model):
        self.model = model
    
    def rerank(self, query, documents, top_k):
        query_emb = self.model.encode(query)
        doc_embs = self.model.encode([doc['text'] for doc in documents])
        
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([query_emb], doc_embs)[0]
        
        scored_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy['fast_score'] = float(score)
            scored_docs.append(doc_copy)
        
        return sorted(scored_docs, key=lambda x: x['fast_score'], reverse=True)[:top_k]

fast_reranker = FastReranker(fast_model)
hierarchical = HierarchicalReranker(fast_reranker, precise_reranker)

# 处理大量候选文档
large_candidates = [{'text': f'文档{i}内容...', 'id': f'doc{i}'} for i in range(100)]
results = hierarchical.rerank("查询内容", large_candidates, stage1_top_k=20, final_top_k=5)
```

### 2. LLM-as-Judge 重排序

```python
from openai import OpenAI

class LLMReranker:
    def __init__(self, model="gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model = model
    
    def rerank(self, query: str, documents: list, top_k: int = 5):
        """使用LLM进行重排序"""
        if len(documents) <= top_k:
            return documents
        
        # 构建重排序提示词
        doc_list = ""
        for i, doc in enumerate(documents):
            doc_list += f"[{i+1}] {doc['text'][:200]}...\n\n"
        
        prompt = f"""
请根据查询内容对以下文档按相关性进行排序，只需要返回最相关的{top_k}个文档的编号。

查询：{query}

文档列表：
{doc_list}

请返回最相关的{top_k}个文档编号，按相关性从高到低排列，格式如：[1, 3, 5, 2, 4]
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # 解析LLM返回的排序结果
            result_text = response.choices[0].message.content.strip()
            
            # 提取数字序列
            import re
            numbers = re.findall(r'\d+', result_text)
            selected_indices = [int(n)-1 for n in numbers[:top_k] if 0 <= int(n)-1 < len(documents)]
            
            # 按LLM排序返回文档
            reranked_docs = []
            for idx in selected_indices:
                doc_copy = documents[idx].copy()
                doc_copy['llm_rank'] = len(reranked_docs) + 1
                reranked_docs.append(doc_copy)
            
            return reranked_docs
            
        except Exception as e:
            print(f"LLM重排序失败: {e}")
            # 降级到原始排序
            return documents[:top_k]

# 使用示例
llm_reranker = LLMReranker()
results = llm_reranker.rerank(query, candidates, top_k=3)

print("LLM重排序结果:")
for doc in results:
    print(f"排名: {doc['llm_rank']}")
    print(f"内容: {doc['text'][:100]}...")
    print("---")
```

### 3. 多信号融合重排序

```python
class MultiSignalReranker:
    def __init__(self, rerankers, weights=None):
        """
        多信号融合重排序器
        rerankers: 不同的重排序器列表
        weights: 各重排序器的权重
        """
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)
    
    def rerank(self, query: str, documents: list, top_k: int = 5):
        """融合多个重排序信号"""
        all_scores = {}
        
        # 1. 获取各个重排序器的分数
        for i, reranker in enumerate(self.rerankers):
            try:
                ranked_docs = reranker.rerank(query, documents, top_k=len(documents))
                
                for j, doc in enumerate(ranked_docs):
                    doc_id = doc.get('id', j)
                    if doc_id not in all_scores:
                        all_scores[doc_id] = {'doc': doc, 'scores': []}
                    
                    # 归一化分数（排名转分数）
                    normalized_score = (len(ranked_docs) - j) / len(ranked_docs)
                    all_scores[doc_id]['scores'].append(normalized_score * self.weights[i])
                    
            except Exception as e:
                print(f"重排序器{i}失败: {e}")
                continue
        
        # 2. 计算融合分数
        final_docs = []
        for doc_id, data in all_scores.items():
            doc = data['doc'].copy()
            # 加权平均
            final_score = sum(data['scores']) / len(data['scores'])
            doc['fusion_score'] = final_score
            final_docs.append(doc)
        
        # 3. 按融合分数排序
        final_docs.sort(key=lambda x: x['fusion_score'], reverse=True)
        return final_docs[:top_k]

# 使用示例：融合三种重排序方法
rerankers = [
    CrossEncoderReranker('BAAI/bge-reranker-base'),
    fast_reranker,  # 基于向量相似度
    llm_reranker    # 基于LLM判断
]

weights = [0.5, 0.3, 0.2]  # Cross-Encoder权重最高
fusion_reranker = MultiSignalReranker(rerankers, weights)

results = fusion_reranker.rerank(query, candidates, top_k=5)
print("融合重排序结果:")
for doc in results:
    print(f"融合分数: {doc['fusion_score']:.3f}")
    print(f"内容: {doc['text'][:100]}...")
    print("---")
```

---

## 📊 重排序性能优化

### 1. 批量处理优化

```python
class BatchReranker:
    def __init__(self, base_reranker, batch_size=32):
        self.base_reranker = base_reranker
        self.batch_size = batch_size
    
    def batch_rerank(self, query_doc_pairs: list):
        """批量重排序处理"""
        results = []
        
        for i in range(0, len(query_doc_pairs), self.batch_size):
            batch = query_doc_pairs[i:i + self.batch_size]
            
            # 批量处理
            batch_queries = [pair['query'] for pair in batch]
            batch_docs = [pair['documents'] for pair in batch]
            
            # 这里需要根据具体reranker实现批量接口
            batch_results = []
            for query, docs in zip(batch_queries, batch_docs):
                result = self.base_reranker.rerank(query, docs)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
```

### 2. 缓存机制

```python
import hashlib
import json
from functools import lru_cache

class CachedReranker:
    def __init__(self, base_reranker, cache_size=10000):
        self.base_reranker = base_reranker
        self.cache = {}
        self.cache_size = cache_size
    
    def _get_cache_key(self, query, documents):
        """生成缓存键"""
        doc_texts = [doc['text'] for doc in documents]
        content = f"{query}:{':'.join(doc_texts)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def rerank(self, query: str, documents: list, top_k: int = 5):
        """带缓存的重排序"""
        cache_key = self._get_cache_key(query, documents)
        
        # 尝试从缓存获取
        if cache_key in self.cache:
            return self.cache[cache_key][:top_k]
        
        # 计算重排序结果
        results = self.base_reranker.rerank(query, documents, top_k)
        
        # 缓存结果
        if len(self.cache) >= self.cache_size:
            # 简单的FIFO清理策略
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = results
        return results

# 使用示例
cached_reranker = CachedReranker(
    CrossEncoderReranker('BAAI/bge-reranker-base'),
    cache_size=5000
)
```

---

## 🔧 重排序评估与调优

### 1. 重排序效果评估

```python
import numpy as np
from sklearn.metrics import ndcg_score

class RerankEvaluator:
    def __init__(self, test_data):
        """
        test_data: [
            {
                'query': 'query text',
                'documents': [{'text': '...', 'relevance': 0/1}],
            }
        ]
        """
        self.test_data = test_data
    
    def evaluate_reranker(self, reranker, metrics=['ndcg', 'map', 'mrr']):
        """评估重排序效果"""
        results = {metric: [] for metric in metrics}
        
        for item in self.test_data:
            query = item['query']
            documents = item['documents']
            
            # 获取重排序结果
            reranked = reranker.rerank(query, documents, top_k=len(documents))
            
            # 提取相关性标签和预测分数
            y_true = [doc.get('relevance', 0) for doc in documents]
            y_pred = []
            
            for doc in reranked:
                # 找到原文档的相关性
                original_idx = documents.index(doc)
                y_pred.append(doc.get('rerank_score', 1.0))
            
            # 计算各项指标
            if 'ndcg' in metrics:
                ndcg = ndcg_score([y_true], [y_pred])
                results['ndcg'].append(ndcg)
            
            if 'map' in metrics:
                map_score = self._calculate_map(y_true, y_pred)
                results['map'].append(map_score)
            
            if 'mrr' in metrics:
                mrr_score = self._calculate_mrr(y_true, y_pred)
                results['mrr'].append(mrr_score)
        
        # 计算平均值
        avg_results = {k: np.mean(v) for k, v in results.items()}
        return avg_results
    
    def _calculate_map(self, y_true, y_pred):
        """计算平均精度均值"""
        # 实现MAP计算逻辑
        pass
    
    def _calculate_mrr(self, y_true, y_pred):
        """计算平均倒数排名"""
        # 实现MRR计算逻辑
        pass

# 使用示例
evaluator = RerankEvaluator(test_data)
metrics = evaluator.evaluate_reranker(reranker)
print("重排序评估结果:")
for metric, value in metrics.items():
    print(f"{metric.upper()}: {value:.3f}")
```

### 2. 参数调优指南

| 参数 | 建议值 | 影响 | 调优策略 |
|------|--------|------|----------|
| **top_k** | 5-10 | 精排候选数量 | 根据下游LLM处理能力调整 |
| **阈值** | 0.5-0.8 | 相关性过滤 | 通过验证集确定 |
| **融合权重** | [0.6, 0.3, 0.1] | 多信号重要性 | A/B测试优化 |
| **批量大小** | 16-64 | 处理效率 | 根据GPU显存调整 |

---

## ⚠️ 常见问题与解决

### 问题1：重排序速度慢

**现象**：重排序成为系统瓶颈  
**解决方案**：

```python
# 1. 异步重排序
import asyncio
import concurrent.futures

class AsyncReranker:
    def __init__(self, base_reranker, max_workers=4):
        self.base_reranker = base_reranker
        self.max_workers = max_workers
    
    async def async_rerank(self, query_batches):
        """异步批量重排序"""
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []
            for query, docs in query_batches:
                task = loop.run_in_executor(
                    executor, 
                    self.base_reranker.rerank, 
                    query, docs
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results

# 2. 预计算优化
class PrecomputedReranker:
    def __init__(self):
        self.precomputed_scores = {}  # 预计算的查询-文档对分数
    
    def precompute_common_pairs(self, common_queries, document_pool):
        """预计算常见查询的重排序分数"""
        for query in common_queries:
            for doc in document_pool:
                key = (query, doc['id'])
                score = self._compute_score(query, doc['text'])
                self.precomputed_scores[key] = score
    
    def rerank(self, query, documents, top_k=5):
        """使用预计算分数的快速重排序"""
        scored_docs = []
        for doc in documents:
            key = (query, doc['id'])
            if key in self.precomputed_scores:
                score = self.precomputed_scores[key]
            else:
                score = self._compute_score(query, doc['text'])
            
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = score
            scored_docs.append(doc_copy)
        
        return sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
```

### 问题2：重排序效果不佳

**现象**：重排序后相关性仍然不高  
**解决策略**：

```python
# 1. 模型微调
class FineTunedReranker:
    def __init__(self, base_model_path, training_data):
        self.model_path = base_model_path
        self.training_data = training_data
    
    def fine_tune(self, epochs=3, learning_rate=2e-5):
        """在特定数据上微调重排序模型"""
        from sentence_transformers import CrossEncoder, InputExample
        
        # 准备训练数据
        train_examples = []
        for item in self.training_data:
            query = item['query']
            for doc in item['documents']:
                example = InputExample(
                    texts=[query, doc['text']], 
                    label=doc['relevance']
                )
                train_examples.append(example)
        
        # 加载并微调模型
        model = CrossEncoder(self.model_path)
        model.fit(
            train_examples,
            epochs=epochs,
            warmup_steps=100,
            output_path=f"{self.model_path}_finetuned"
        )
        
        return f"{self.model_path}_finetuned"

# 2. 领域适配
class DomainAdaptedReranker:
    def __init__(self, general_reranker, domain_keywords):
        self.general_reranker = general_reranker
        self.domain_keywords = domain_keywords
    
    def rerank(self, query, documents, top_k=5):
        """领域适配的重排序"""
        # 先进行通用重排序
        general_results = self.general_reranker.rerank(query, documents, top_k * 2)
        
        # 领域关键词加权
        for doc in general_results:
            domain_boost = 0
            for keyword in self.domain_keywords:
                if keyword.lower() in doc['text'].lower():
                    domain_boost += 0.1
            
            doc['rerank_score'] += domain_boost
        
        # 重新排序
        final_results = sorted(general_results, key=lambda x: x['rerank_score'], reverse=True)
        return final_results[:top_k]
```

---

## 🔗 相关阅读

- [RAG范式演进](/llms/rag/paradigms) - 了解RAG技术发展脉络
- [检索策略优化](/llms/rag/retrieval) - 重排序的上游环节
- [RAG评估方法](/llms/rag/evaluation) - 重排序效果评估
- [向量数据库](/llms/rag/vector-db) - 检索的底层存储
- [生产实践指南](/llms/rag/production) - 重排序的部署优化

> **相关文章**：
> - [混合搜索中的分数归一化方法深度解析](https://dd-ff.blog.csdn.net/article/details/156072979)
> - [混合检索中短查询高分异常的深度剖析与神经重排序的修正机制](https://dd-ff.blog.csdn.net/article/details/156067548)
> - [高级RAG技术全景：从原理到实战](https://dd-ff.blog.csdn.net/article/details/149396526)
> - [检索增强生成（RAG）系统综合评估](https://dd-ff.blog.csdn.net/article/details/152823514)
> - [检索增强生成（RAG）综述：技术范式、核心组件与未来展望](https://dd-ff.blog.csdn.net/article/details/149274498)

> **外部资源**：
> - [BGE-Reranker模型](https://huggingface.co/BAAI/bge-reranker-v2-m3) - 智源开源重排序模型
> - [Cohere Rerank API](https://docs.cohere.com/docs/rerank) - 商业重排序服务
> - [Cross-Encoder原理](https://www.sbert.net/examples/applications/cross-encoder/README.html) - Sentence-Transformers文档
