---
title: 文档切分策略
description: RAG 系统中的文档切分技术详解 - 15种实战策略
---

# 文档切分策略

> 掌握文档切分的核心技术，选择合适的切分策略优化RAG效果

## 🎯 核心概念

### 为什么需要文档切分？

RAG系统的第一步就是将长文档切分成小块，这样做的原因：

- **Token 限制**：LLM 上下文窗口有限（如 GPT-4 的 128K tokens）
- **检索精度**：小块更容易匹配用户查询的具体信息
- **计算效率**：减少向量化和检索的计算开销
- **语义聚焦**：避免一个chunk包含过多无关主题

### 切分的核心挑战

::: warning 关键平衡
**粒度平衡**：太大检索不准，太小丢失上下文  
**语义完整**：避免在句子中间切断，保持逻辑连贯  
**重叠处理**：边界信息的保留与冗余的权衡
:::

---

## 🔬 文档智能预处理（切分前的关键步骤）

> 基于[《超越纯文本：解锁高级RAG中复杂文档预处理的艺术》](https://dd-ff.blog.csdn.net/article/details/152045489)

::: tip 核心洞察
**RAG系统的上限不是由LLM决定的，而是由数据准备的质量决定的。** 原始文档（PDF、PPT、扫描件）在原生状态下并非机器可读，预处理流水线是将混乱转化为结构化知识的幕后英雄。
:::

### 文档智能六阶段流水线

| 阶段 | 技术 | 作用 | 关键工具 |
|------|------|------|----------|
| **1. 布局分析** | LayoutLM、DocLayout-YOLO | 识别标题、段落、表格、图片等结构元素 | LayoutLMv3、GNN |
| **2. OCR识别** | 光学字符识别 | 将图像中的文字转为可编辑文本 | PaddleOCR、Tesseract |
| **3. 表格识别** | 表格结构识别 | 从视觉网格重建结构化数据 | TableTransformer |
| **4. 公式识别** | 数学/化学公式解析 | 将公式图像转为LaTeX等格式 | Pix2Tex、Mathpix |
| **5. 图像分析** | VLM视觉语言模型 | 为图表生成文本描述 | GPT-4V、LLaVA |
| **6. 阅读顺序** | 逻辑流检测 | 确定多栏文档的正确阅读顺序 | 规则/ML混合 |

### 为什么布局分析至关重要？

```python
# 反面案例：双栏PDF直接提取文本
# 结果：两栏句子被胡乱拼接，毫无逻辑

# 正确做法：先进行布局分析
class DocumentIntelligencePipeline:
    """文档智能处理流水线"""
    
    def process(self, document):
        # 1. 布局分析：识别结构元素
        layout = self.layout_analyzer.detect(document)
        # 输出：标题、段落、表格、图片的边界框和类型
        
        # 2. 按元素类型分别处理
        for element in layout.elements:
            if element.type == 'table':
                content = self.table_parser.extract(element)
            elif element.type == 'formula':
                content = self.formula_recognizer.parse(element)
            elif element.type == 'image':
                content = self.vlm.describe(element)
            else:
                content = self.ocr.extract(element)
            
            element.content = content
        
        # 3. 按阅读顺序重组
        ordered_content = self.reading_order_detector.sort(layout)
        return ordered_content
```

### LayoutLM：布局感知的核心技术

LayoutLM通过**多模态嵌入**同时建模文本和布局信息：

1. **文本嵌入**：类BERT嵌入捕捉语义
2. **1D位置嵌入**：单词在序列中的顺序
3. **2D位置嵌入**：每个token的边界框坐标(x₀,y₀,x₁,y₁) — *突破性特征*
4. **图像嵌入**：字体样式、颜色等视觉特征

> **关键规则**：视觉上相邻的词语通常在语义上相关

---

## 📋 15种分块策略详解

> 基于[《解锁RAG效能：15种分块策略秘籍》](https://dd-ff.blog.csdn.net/article/details/149529161)的实战总结

### 1️⃣ 逐行分块法（Line-by-Line Chunking）

**核心原理**：以换行符为分割点，每行独立成块

**适用场景**：
- 聊天记录、对话转录文本
- 问答式数据（客服对话、访谈实录）
- 结构化的列表数据

**实战示例**：
```python
def line_chunking(text):
    return text.strip().split('\n')

# 示例
dialogue = """Alice: 嗨Bob，今天下午3点有空通话吗？
Bob: 可以，想讨论项目进展吗？
Alice: 是的，还要聊聊客户会议的事
Bob: 没问题！3点见"""

chunks = line_chunking(dialogue)
# 结果：4个独立的对话块
```

**优势**：保留对话天然独立性，支持细粒度检索  
**注意**：过短的行可能导致上下文缺失

### 2️⃣ 固定尺寸分块法（Fixed-Size Chunking）

**核心原理**：按固定字符数/Token数分割，无视语义边界

**适用场景**：
- 杂乱无章的非结构化文本
- OCR识别结果、网页爬取数据
- 老旧扫描文档的数字化内容

**实战示例**：
```python
def fixed_size_chunking(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
```

**优势**：实现数据块标准化，便于统一处理  
**风险**：可能割裂完整语义单元，建议仅在万不得已时使用

### 3️⃣ 滑动窗口分块法（Sliding Window Chunking）

**核心原理**：固定窗口大小，按步长滑动切分，保留重叠区域

**适用场景**：
- 连续性强的技术文档
- 小说、散文等文学作品
- 需要保持上下文连贯的长文本

**关键参数**：
- `window_size`：窗口大小（如512 tokens）
- `step_size`：滑动步长（如400 tokens，重叠112 tokens）

### 4️⃣ 句子级分块法（Sentence-Based Chunking）

**核心原理**：以句号、问号、感叹号等句子边界为切分点

**优势**：保持语义完整性，适合问答系统  
**实现**：结合NLP库（如spaCy、NLTK）进行句子分割

### 5️⃣ 段落分块法（Paragraph Chunking）

**核心原理**：以段落（双换行符`\n\n`）为切分单位

**适用场景**：
- 结构化文档（论文、报告）
- 博客文章、新闻稿
- 技术文档

### 6️⃣ 递归分块法（Recursive Chunking）

**核心原理**：从大单元开始，逐层分割过大的分块，直至符合尺寸要求

**算法流程**：
1. 先按段落分割
2. 段落过大 → 按句子分割  
3. 句子仍过大 → 按固定长度分割

**实战示例**：
```python
def recursive_chunking(text, max_size=500):
    # 第1层：段落分割
    paragraphs = text.split('\n\n')
    
    chunks = []
    for para in paragraphs:
        if len(para) <= max_size:
            chunks.append(para)
        else:
            # 第2层：句子分割
            sentences = para.split('。')
            for sent in sentences:
                if len(sent) <= max_size:
                    chunks.append(sent)
                else:
                    # 第3层：固定长度分割
                    chunks.extend(fixed_size_chunking(sent, max_size))
    return chunks
```

### 7️⃣ 表格分块法（Table Chunking）

**核心原理**：将表格作为独立分块（整体或逐行分割）

**适用场景**：
- 财务报告、销售数据
- 科学论文的实验结果
- 包含结构化数据的混合文档

**处理策略**：
- **整表保留**：小表格整体作为一个chunk
- **逐行分割**：大表格按行切分，保留表头
- **语义分组**：按业务逻辑将相关行组合

### 8️⃣ 章节/标题分块法（Section-Based Chunking）

**核心原理**：按文档的层级结构（H1、H2、H3标题）进行切分

**优势**：保留文档逻辑结构，便于理解上下文关系

### 9️⃣ 语义分块法（Semantic Chunking）

**核心原理**：基于语义相似度，将相关内容聚合为一个chunk

**实现方式**：
- 计算句子间的embedding相似度
- 设定相似度阈值进行聚类
- 动态调整chunk边界

**计算开销**：较大，但效果最佳

### 🔟 Token分块法（Token-Based Chunking）

**核心原理**：按模型的Token数量进行精确切分

**重要性**：直接对应模型输入限制，是最精确的切分方式

```python
import tiktoken

def token_chunking(text, model="gpt-3.5-turbo", max_tokens=500):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks
```

---

## ⚙️ 分块策略选择指南

### 按文档类型选择

| 文档类型 | 推荐策略 | 理由 |
|----------|----------|------|
| **对话记录** | 逐行分块 | 保持问答独立性 |
| **技术文档** | 章节分块 + 递归 | 结合结构与大小控制 |
| **学术论文** | 段落分块 | 逻辑单元完整 |
| **小说散文** | 滑动窗口 | 保持情节连贯 |
| **表格数据** | 表格专用 | 保持结构化信息 |
| **OCR文本** | 固定尺寸 | 结构混乱时的兜底方案 |

### 按应用场景选择

| 应用场景 | 策略组合 | 参数建议 |
|----------|----------|----------|
| **问答系统** | 句子级 + Token控制 | 每chunk 1-3个完整句子 |
| **文档摘要** | 段落级 + 重叠窗口 | chunk_size=800, overlap=100 |
| **代码检索** | 函数级 + 递归 | 按函数/类边界切分 |
| **多语言文档** | 语义分块 | 考虑语言特征 |

---

## 🛠️ 实战最佳实践

### LangChain 实现

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 中文优化的递归分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    keep_separator=True  # 保留分隔符
)

chunks = splitter.split_text(document)
```

### 混合策略实现

```python
def hybrid_chunking(text, doc_type="general"):
    if doc_type == "dialogue":
        return line_chunking(text)
    elif doc_type == "structured":
        return section_based_chunking(text)
    else:
        # 默认递归策略
        return recursive_chunking(text, max_size=800)
```

### 元数据保留

```python
def chunking_with_metadata(text, source_file, page_num):
    chunks = recursive_chunking(text)
    
    metadata_chunks = []
    for i, chunk in enumerate(chunks):
        metadata_chunks.append({
            "content": chunk,
            "source": source_file,
            "page": page_num,
            "chunk_id": f"{source_file}_{page_num}_{i}",
            "chunk_size": len(chunk)
        })
    
    return metadata_chunks
```

---

## ⚠️ 常见问题与解决

### 问题1：上下文丢失

**现象**：切分后的chunk缺乏完整信息  
**解决**：
- 增加overlap（推荐20-30%）
- 使用父子分段：小chunk检索，大chunk生成
- 保留关键元数据（标题、章节信息）

### 问题2：检索不精准

**现象**：返回的chunk与查询不匹配  
**解决**：
- 减小chunk_size（500-800 tokens）
- 优化分隔符选择
- 结合重排序（Rerank）

### 问题3：计算开销大

**现象**：语义切分速度慢  
**解决**：
- 预计算文档embedding
- 使用缓存机制
- 混合策略：重要文档用语义分块，其他用递归分块

---

## 🔗 相关阅读

- [RAG范式演进](/llms/rag/paradigms) - 了解RAG技术发展脉络
- [Embedding技术详解](/llms/rag/embedding) - 理解向量化原理
- [检索策略优化](/llms/rag/retrieval) - 切分后的检索技巧  
- [RAG评估方法](/llms/rag/evaluation) - 评估切分效果

> **相关文章**：
> - [解锁RAG效能：15种分块策略秘籍（附实战案例）](https://dd-ff.blog.csdn.net/article/details/149529161)
> - [超越纯文本：解锁高级RAG中复杂文档预处理的艺术](https://dd-ff.blog.csdn.net/article/details/152045489)
> - [从"拆文档"到"通语义"：RAG+知识图谱如何破解大模型"失忆+幻觉"难题？](https://dd-ff.blog.csdn.net/article/details/149354855)

> **外部资源**：
> - [LlamaIndex文档切分指南](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
> - [LangChain文本分割器](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
