---
title: 重排序技术详解
description: RAG系统中的检索结果重排序与精排技术
pageType: article
module: rag
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - rag
level: intermediate
prerequisites:
  - /llms/prompt/
reviewed: '2026-08-25'
techVersion: 待复核（2026-08）
---

# 重排序技术详解

::: info 代码阅读约定
模型调用示例需安装匹配依赖并下载权重；自定义打分、批量/缓存/微调组件是接口骨架。文中不提供跨语言、硬件和领域通用的得分阈值或性能保证。
:::

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
| **Cross-Encoder** | 查询-文档联合编码 | 可建模更细的交互 | 计算量大 | 高质量要求 |
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
score = reranker(input)  # 架构伪代码；输出范围由具体模型决定
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

先确认模型输出契约：单标量 logits、已激活分数、多分类 logits 不能混用。Sigmoid 对单标量严格单调，因此只为排序时可直接用 logits；变为 0–1 不会提高排序质量，也不代表获得已校准的相关概率。[Sentence Transformers 官方文档](https://sbert.net/docs/cross_encoder/usage/usage.html) 给出了原始 logits 与显式激活的用法。

若要比较分数与阈值，必须固定模型、候选分布和输入模板，再用标注集拟合与检查校准。不要把未经校准的 rerank 分数与余弦直接相加，也不要对 SDK 已做过的激活再做一次 Sigmoid。

```python
# pip install sentence-transformers torch
import torch
from sentence_transformers import CrossEncoder

# 英文教学模型：显式使用 Identity，确保下面得到原始标量 logits。
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    activation_fn=torch.nn.Identity(),
)
pairs = [
    ("What is RAG?", "RAG combines retrieval with text generation."),
    ("What is RAG?", "Bananas are yellow."),
]
logits = model.predict(pairs)
ranked = sorted(zip(pairs, logits), key=lambda item: float(item[1]), reverse=True)
```

中文场景应换为适合中文的模型并重新评估，不应沿用这个教学模型的分数阈值。

### 开源重排序模型对比

下列是候选家族，不是实时榜单。相同系列的参数规模、窗口、许可证和推理接口可能不同，选择时固定具体模型卡与 revision。

| 候选家族 | 先核对什么 | 需要实测什么 |
| --- | --- | --- |
| BGE reranker | 中文/多语言支持、输入长度、分数定义 | 专有名词、否定条件与长片段截断 |
| Jina reranker | 具体版本与运行时依赖 | 跨语言排序和吞吐 |
| MS MARCO Cross-Encoder | 训练域与语言覆盖 | 从英文搜索数据迁移到业务域的退化 |

以固定候选集上的 NDCG/MRR 增益、最终证据覆盖率及 p95 延迟选型。参数量大、榜单名次高或标注“多语言”都不能替代业务测试。

---

## 🛡️ Rerank修正检索异常

> 来源：[混合检索中短查询高分异常的深度剖析与神经重排序的修正机制](https://dd-ff.blog.csdn.net/article/details/156067548)

### 短查询高分异常问题

::: danger 病态现象
输入"Hello"、"系统"、"测试"等短查询时，混合检索往往以**较高排序分数**返回**完全不相关**的文档。这在RAG中是致命的——噪声上下文会增加无依据回答的风险。
:::

**根本原因分析**：

| 检索阶段 | 失效机制 | 后果 |
|----------|----------|------|
| **BM25** | 分词、词频和长度归一化不适配 | 短碎片高分 |
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
                                    逐词交互，补充相关性信号
```

**修正机制**：

1. **补充交互信号**：联合阅读问题与片段，可改善一些仅凭向量相似度难区分的候选
2. **检查实际输入**：超过模型长度的片段会截断；关键信息必须在可见范围内
3. **校准拒答决策**：用带标签验证集选择阈值，检查无答案查询的误接受率

### 阈值截断与幻觉抑制

阈值是一项经过验证的应用决策，不是模型的通用属性。先标注“可支持回答”与“仅主题相关”的区别，在验证集选择可接受的误接受/误拒绝折中；最后用未参与选择的测试集报告结果。模型、切分、语言或候选数量改变时重新检查。

```python
# 独立可用的后处理函数；threshold 由独立验证集确定。
from math import isfinite

def select_evidence(documents, scores, *, threshold, top_k=5):
    if top_k <= 0:
        raise ValueError("top_k 必须为正")
    if len(documents) != len(scores):
        raise ValueError("候选与分数数量不一致")
    if not isfinite(threshold) or not all(isfinite(float(s)) for s in scores):
        raise ValueError("拒绝非有限分数")
    ranked = sorted(
        ({**doc, "rerank_score": float(score)} for doc, score in zip(documents, scores)),
        key=lambda doc: doc["rerank_score"], reverse=True,
    )
    selected = [doc for doc in ranked if doc["rerank_score"] >= threshold][:top_k]
    return {"documents": selected, "status": "ok" if selected else "insufficient_evidence"}
```

不为凑满 top-k 强制补入低分文档。`ok` 只表示通过排序过滤，不代表每个答案要点都有依据；仍需检查证据覆盖、时间冲突和引用。重排器无法找回候选池之外的材料。

### 完整两阶段检索流水线

生产链路应保存每一步的输入与输出，尤其是候选 ID、原始分数、模型 revision、截断后的文本和最终证据 ID。

```text
认证并确定可访问语料
→ BM25 / 向量召回（统一稳定文档 ID、去重）
→ 检查候选 Recall@K 的离线基线
→ 对查询与候选成对打分（分数定义由模型卡决定）
→ 按验证过的阈值与上下文预算选择证据
→ 无充分依据时澄清或拒答
→ 生成答案并校验引用是否支持对应声明
```

召回 K、送入模型的证据数量和 token 预算是三个不同参数。重复切块可能挤占多个位置；多跳问题可能要求保留若干互补片段，单靠逐片相关性分数无法保证完整覆盖。

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

下面是历史 Chat Completions 接口示例，模型 ID 与 SDK 支持需在运行时确认；正文前 200 个字符的裁剪会丢失后文证据，仅用于演示。生产应按 token 和证据边界裁剪，使用结构约束返回唯一候选 ID，并监控降级率。

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
            import json
            indices = json.loads(result_text)
            if not isinstance(indices, list) or not all(type(n) is int for n in indices):
                raise ValueError("排序输出必须是整数数组")
            if len(set(indices)) != len(indices) or any(n < 1 or n > len(documents) for n in indices):
                raise ValueError("排序包含重复或越界编号")
            selected_indices = [n - 1 for n in indices[:top_k]]
            
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
    print(f"排名: {doc.get('llm_rank', '原始排序降级')}")
    print(f"内容: {doc['text'][:100]}...")
    print("---")
```

### 3. 多信号融合重排序

示例采用名次变换而非概率；需校验权重数量和非负性、路内去重，并记录失败的通道。任一路失败会改变融合标度，结果不可与全通道时的阈值直接比较。

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
                    doc_id = doc['id']  # 稳定 ID；不能使用各路排序名次代替
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
            final_score = sum(data['scores'])  # 权重和已在配置中约定；缺失路不奖励
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

评估必须通过稳定 ID 对齐标签与候选。重排后的字典可能新增分数字段，不能再用 `documents.index(doc)` 找原对象，也不能把原始顺序的标签与重排顺序的分数直接交给指标函数。

做法一：按原始 ID 顺序构造 `y_true` 与预测分数，再交给同序的 NDCG 实现；做法二：直接按重排名次取得相关标签，计算 DCG 并用标注全集构造 IDCG。二值示例可复用[评估章节](/llms/rag/evaluation)的 `retrieval_metrics`。

未标注相关性不等于不相关；评估集须说明标注覆盖。候选池固定时比较重排能力，候选池变化时另外报告 Recall@K；多跳问题再检查全部必要证据的联合覆盖。报表不得输出未实现的 MAP/MRR 或空列表平均值后将其当有效分数。

### 2. 参数调优指南

| 参数 | 建议值 | 影响 | 调优策略 |
|------|--------|------|----------|
| **top_k** | 由证据预算确定 | 最终保留数量 | 根据下游LLM处理能力调整 |
| **阈值** | 无通用数值 | 相关性过滤 | 通过验证集确定 |
| **融合权重** | 验证集选择 | 多信号重要性 | A/B测试优化 |
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

## 重排验收：先固定候选，再比较模型

第一轮只更换 reranker，冻结候选 ID 与内容，比较 NDCG@K、MRR、证据覆盖率和 p95 延迟；第二轮才联调召回 K 与 token 预算。否则候选池变化会掩盖真正收益。重排准确率提升但端到端质量不变时，检查重复证据、否定条件截断、引用错位和生成器利用率。

LLM-as-Judge 重排还要测试输入顺序、长度偏好和候选文本中的注入内容。通过交换候选顺序、人工复核分歧样本来校准判断；缓存键包含查询、文档版本、模型与模板版本，不能只含查询字符串。

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
