---
title: 训练数据处理
description: 高质量微调数据的准备与处理
pageType: article
module: training
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - training
level: advanced
prerequisites:
  - /guide/prerequisites
reviewed: '2026-10-08'
techVersion: 原理与示例复核于 2026-10；依赖接口需锁定版本
---

# 训练数据处理

> "垃圾进，垃圾出"——数据质量决定模型上限

## 🎯 核心原则

> 来源：[垃圾进，垃圾出：打造高质量LLM微调数据集的终极指南](https://dd-ff.blog.csdn.net/article/details/152254276)

![数据质量金字塔](https://pic2.zhimg.com/v2-44f397a445692fe8631990b251d10bdf_r.jpg)
*数据是 LLM 训练的基石*

::: tip 先确定数据要改变什么
格式遵循需要一致示范，领域任务需要覆盖真实问题，偏好优化需要可解释的相对选择。数据量与质量没有通用换算关系；[LIMA](https://arxiv.org/abs/2305.11206)的小样本结果不意味着任何任务都能用 1,000 条替代 100,000 条。先建立按任务与来源分组的留出集，再衡量增量数据的价值。
:::

### 数据质量维度

| 维度 | 说明 | 检查方法 |
|------|------|----------|
| **准确性** | 内容正确无误 | 专家审核、事实核查 |
| **相关性** | 响应切题 | 指令-响应匹配度 |
| **多样性** | 覆盖广泛 | 聚类分析、嵌入可视化 |
| **一致性** | 风格统一 | 模板检查、格式验证 |
| **安全性** | 无有害内容 | 敏感词过滤、人工审核 |

---

## 📊 数据格式

### SFT 数据集格式至关重要

SFT 数据集需包含明确的结构：
- **指令（Instruction）**：任务描述
- **输入（Input）**：可选的上下文信息
- **期望输出（Output）**：标准答案
- 若需多语言理解或复杂上下文处理，还需标注**对话历史**和**角色身份**

### 主流格式对比

| 格式 | 结构 | 适用场景 | 特点 |
|------|------|----------|------|
| **Alpaca** | instruction/input/output | 单轮指令任务 | 简单直观，适合问答、翻译 |
| **ShareGPT** | conversations数组 | 多轮对话 | 保留完整对话历史 |
| **OpenAI** | messages数组（system/user/assistant） | 通用格式 | API 兼容，支持系统提示 |

### Alpaca格式

```json
{
  "instruction": "将以下英文翻译成中文",
  "input": "Hello, how are you?",
  "output": "你好，你好吗？"
}
```

### ShareGPT格式

```json
{
  "conversations": [
    {"from": "human", "value": "你好，请介绍一下自己"},
    {"from": "gpt", "value": "你好！我是一个AI助手..."},
    {"from": "human", "value": "你能做什么？"},
    {"from": "gpt", "value": "我可以帮助你回答问题..."}
  ]
}
```

### OpenAI格式

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
  ]
}
```

---

## 🔧 数据处理流程

### 完整流程图

```
原始数据
    │
    ▼
┌─────────────┐
│  数据清洗    │ → 去除噪声、修复格式错误、规范化空白字符
└─────────────┘
    │
    ▼
┌─────────────┐
│  PII脱敏    │ → 匿名化个人隐私信息（邮箱、电话、身份证）
└─────────────┘
    │
    ▼
┌─────────────┐
│  质量过滤    │ → 过滤低质量样本（长度、重复、语言检查）
└─────────────┘
    │
    ▼
┌─────────────┐
│  去重处理    │ → 精确哈希、近重复检测与语义候选复核
└─────────────┘
    │
    ▼
┌─────────────┐
│  格式转换    │ → 转为目标训练格式（Alpaca/ShareGPT/OpenAI）
└─────────────┘
    │
    ▼
┌─────────────┐
│  数据增强    │ → 使用 GPT-4 生成、回译、同义词替换
└─────────────┘
    │
    ▼
┌─────────────┐
│  混合策略    │ → 按回归评估选择通用/领域采样配比
└─────────────┘
    │
    ▼
  训练数据集
```

### 数据构建最佳实践

| 实践 | 说明 |
|------|------|
| **GPT-4 生成** | 利用强 LLM 生成、过滤高质量数据，提升数据质量与多样性 |
| **人工审核** | 关键数据需专家审核，确保准确性 |
| **多样性检查** | 通过嵌入聚类分析，确保数据覆盖广泛 |
| **混合通用数据** | 比较多个通用数据比例，检查通用能力回退与领域收益 |

### 数据清洗

清洗规则要按数据类型分流：代码缩进、Markdown 换行、数学比较符号和工具 JSON 都有语义。统一删除 `<...>` 或把所有空白变成空格，可能把正确样本清坏；HTML 只在确认字段是 HTML 时用解析器处理。

```python
import unicodedata

def clean_text(text: str) -> str:
    # NFC 保留字符语义；不合并换行、不删除尖括号
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n"))
    return "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc")

def validate_sample(sample):
    # 短答案、复制任务与数字输出均可能正确，不按字符数机械删除
    return (isinstance(sample.get("instruction"), str)
            and isinstance(sample.get("output"), str)
            and bool(sample["output"].strip()))
```

保存原始文本、清洗版本与变更理由；抽样比较前后 diff，发现公式或代码被损坏时能追溯并回退规则。

### PII脱敏

下面正则只是候选识别示例，不能覆盖姓名、地址、上下文组合标识，也会误报普通数字。生产中需加标注抽检、实体识别和一致替换；留意同一人物跨轮指代被破坏、评估数据泄漏与日志中的原始值。

```python
import re

class PIIAnonymizer:
    """个人隐私信息脱敏"""
    
    PATTERNS = {
        'email': r'\b[\w.-]+@[\w.-]+\.\w+\b',
        'phone': r'\b1[3-9]\d{9}\b',
        'id_card': r'\b\d{17}[\dXx]\b',
        'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }
    
    def anonymize(self, text: str) -> str:
        for pii_type, pattern in self.PATTERNS.items():
            text = re.sub(pattern, f'[{pii_type.upper()}]', text)
        return text
```

---

## 📈 数据质量评估

### 嵌入空间分析

聚类只是覆盖诊断，不代表任务难度或答案正确性。以下英文编码器示例未针对中文数据验证；实际选择需以目标语言的相似样本抽检为准，且 `n_clusters` 不能超过样本数。

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

def analyze_diversity(texts: list, n_clusters: int = 10):
    """分析数据集多样性"""
    # 生成嵌入
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    
    # 聚类分析
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)
    
    # 计算多样性指标
    cluster_sizes = np.bincount(labels)
    # 示例描述量可为负，不能解释为 [0,1] 的质量分
    diversity_score = 1 - (cluster_sizes.std() / cluster_sizes.mean())
    
    return {
        "diversity_score": diversity_score,
        "cluster_distribution": cluster_sizes.tolist(),
        "largest_cluster_ratio": cluster_sizes.max() / len(texts)
    }
```

### 超级过滤技术

这里是语义近重复筛查示例，不是名为 Superfiltering 的具体论文算法。全量相似度矩阵需要 O(N²) 存储，只适合小样本；大数据先用哈希/分桶/近邻索引找候选。相似指令的不同答案可能是冲突标签，应人工处理而非仅留第一条。

```python
def super_filter(samples: list, threshold: float = 0.9) -> list:
    """超级过滤：去除高度相似的样本"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [s['instruction'] + s.get('input', '') for s in samples]
    embeddings = model.encode(texts)
    
    # 计算相似度矩阵
    sim_matrix = cosine_similarity(embeddings)
    
    # 去重
    keep_indices = []
    for i in range(len(samples)):
        is_duplicate = False
        for j in keep_indices:
            if sim_matrix[i][j] > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)
    
    return [samples[i] for i in keep_indices]
```

---

## 🗄️ Parquet格式优化

> 来源：[Parquet范式：大语言模型训练数据格式优化](https://dd-ff.blog.csdn.net/article/details/154654277)

### 为什么使用Parquet？

Parquet 的列式布局、压缩和列裁剪，适合按字段扫描、统计和流式加载；实际收益取决于压缩算法、行组大小、字段分布与读取模式。不能从一个基准推导“空间固定减少 87%”或“查询固定快 34.8 倍”。[Apache Parquet 概述](https://parquet.apache.org/docs/overview/)

若训练总是顺序读完整 messages，测全字段吞吐；若只统计元数据，测列裁剪。数据规模大时分片写入，下面的 pandas 全量加载只是小文件转换示例。

### 转换示例

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def json_to_parquet(json_path: str, parquet_path: str):
    """将JSON数据转换为Parquet格式"""
    # 读取JSON
    df = pd.read_json(json_path, lines=True)
    
    # 转换为Parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(
        table, 
        parquet_path,
        compression='snappy',  # 压缩算法
        row_group_size=10000   # 行组大小
    )

def read_parquet_efficiently(parquet_path: str, columns: list = None):
    """高效读取Parquet（列裁剪）"""
    return pq.read_table(
        parquet_path,
        columns=columns  # 只读取需要的列
    ).to_pandas()
```

---

## 🏷️ 特殊Token与模板

> 来源：[深入探秘LLM的"暗语"：特殊Token与LlamaFactory的模板魔法](https://dd-ff.blog.csdn.net/article/details/152328698)

::: danger 关键警告
训练与推理模板不一致是需要优先排查的故障，但没有通用的“90%”归因比例。[Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating)说明了模型控制 token 与模板的一致性要求。
:::

### 常见特殊Token

| Token | 作用 | 示例 |
|-------|------|------|
| `<s>` / `<bos>` | 序列开始 | 标记输入起点 |
| `</s>` / `<eos>` | 序列结束 | 标记输出终点 |
| `[INST]` / `[/INST]` | 指令边界 | Llama格式 |
| `<|im_start|>` / `<|im_end|>` | 消息边界 | ChatML格式 |

### ChatML模板示例

```
<|im_start|>system
你是一个有帮助的AI助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的吗？<|im_end|>
```

### 模板匹配检查

```python
# 示例：保存实际 token ID 作为回归样本，而非只比特殊 token 集合
messages = [{"role": "user", "content": "输出 JSON: {\"ok\": true}"}]
ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print(ids)
print(tokenizer.decode(ids, skip_special_tokens=False))
```

训练时包含完整 assistant 回答，推理时只包含待续写的 assistant 前缀，二者不应逐字相同。核对的是相同会话前缀的 token 序列、角色顺序、终止符及生成起点；集合相同不能发现顺序错乱或重复 BOS/EOS。

## 数据集交付与失败诊断

1. 先按来源文档、用户/会话或时间建立分组，再去重和切分，避免同一事实改写分散到训练与测试。
2. 合成数据记录生成模型、prompt 与种子；针对答案正确性审核，不把流畅度当作事实证据。
3. 统计每轮处理的保留率、任务占比、长度/token 分布、截断率与重复率。过滤后某个语言或难例骤减，要检查阈值偏置。
4. 输出 manifest：数据版本、来源许可、清洗规则、分片哈希、采样比例和留出策略，支持删除与重建。

验收从分层抽样开始：人审错误率、模板一致性、有效回答标签比例与跨切分重复量均有报告。若训练指标突然大幅改善但新问题不变，优先查污染；若代码任务退化，优先查空白/标签清洗，再调模型。

## 🔗 相关阅读

## 🔗 相关阅读

- [训练微调概述](/llms/training/) - 了解完整训练流程
- [SFT监督微调](/llms/training/sft) - 如何使用准备好的数据
- [LoRA高效微调](/llms/training/lora) - 低资源训练方案

> **相关文章**：
> - [垃圾进，垃圾出：打造高质量微调数据集](https://dd-ff.blog.csdn.net/article/details/152254276)
> - [Parquet范式：训练数据格式优化](https://dd-ff.blog.csdn.net/article/details/154654277)
> - [深入探秘LLM的"暗语"：特殊Token与模板](https://dd-ff.blog.csdn.net/article/details/152328698)

> **外部资源**：
> - [Hugging Face Datasets](https://huggingface.co/docs/datasets)
> - [Apache Parquet](https://parquet.apache.org/)
