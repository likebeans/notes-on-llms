---
title: 视觉编码器
description: ViT、CLIP 与视觉表征的数学原理
---

# 视觉编码器：从像素到语义

> 视觉编码器是多模态模型的"眼睛"，将连续像素转化为结构化特征向量，供后续语言模型处理。

---

## Vision Transformer (ViT)

ViT 的出现标志着计算机视觉从 CNN 向 Transformer 的彻底转型，极大削弱了对归纳偏置的依赖。

### 核心架构

```mermaid
flowchart LR
    IMG[输入图像\n224×224×3] --> PATCH[Patch 切分\n16×16]
    PATCH --> FLAT[展平\n768维向量]
    FLAT --> PROJ[线性投影\nE矩阵]
    PROJ --> POS[+ 位置编码]
    POS --> CLS[+ CLS Token]
    CLS --> ENC[Transformer Encoder\n×12层]
    ENC --> OUT[图像特征]
```

### Patch Embedding 数学原理

输入图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ 被划分为固定大小的 Patch（通常 $16 \times 16$）：

$$\mathbf{z}_0 = [\mathbf{x}_{cls}; \mathbf{x}_p^1\mathbf{E}; \mathbf{x}_p^2\mathbf{E}; \cdots; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos}$$

其中：

- $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是线性投影矩阵
- $\mathbf{E}_{pos}$ 是位置编码
- 对于 $224 \times 224$ 图像，产生 $14 \times 14 = 196$ 个 Patch

### 处理流程详解

| 步骤 | 输入 | 输出 | 说明 |
| :--- | :--- | :--- | :--- |
| **Patch 切分** | 224×224×3 | 196 个 16×16×3 | 网格化切割 |
| **展平** | 16×16×3 | 768 维向量 | 每个 Patch 拉平 |
| **线性投影** | 768 维 | D 维（如 768） | 可学习投影矩阵 |
| **添加位置编码** | N×D | N×D | 赋予空间感知 |
| **添加 CLS Token** | N×D | (N+1)×D | 用于分类任务 |

### 位置编码演进

由于 Transformer 本质上是置换不变的（Permutation Invariant），位置编码至关重要。

| 方案 | 原理 | 优势 | 局限 |
| :--- | :--- | :--- | :--- |
| **可学习位置编码** | 训练时学习固定位置嵌入 | 简单有效 | 固定分辨率 |
| **正弦/余弦编码** | 固定的三角函数 | 无需训练 | 外推性有限 |
| **RoPE 2D** | 旋转位置编码扩展到二维 | 支持可变分辨率 | 实现复杂 |
| **缩放平均位置嵌入** | 编码相对感受野大小 | 多尺度适应 | 计算开销 |

::: tip 缩放平均位置嵌入
RetinaViT 引入的机制：当输入分辨率变化时，通过计算 Patch 在 3D 图像金字塔中的相对位置来调整嵌入的范数，不仅保留二维位置信息，还编码相对感受野大小。
:::

### ViT 变体对比

| 模型 | 参数量 | Patch 大小 | 特点 |
| :--- | :--- | :--- | :--- |
| **ViT-B/16** | 86M | 16×16 | 基础版本 |
| **ViT-L/14** | 304M | 14×14 | CLIP 常用 |
| **ViT-H/14** | 632M | 14×14 | 大模型 |
| **ViT-G/14** | 1.8B | 14×14 | 巨型模型 |

---

## CLIP：视觉-语言对齐

**CLIP (Contrastive Language-Image Pre-training)** 是连接视觉与文本语义的基石，通过对比学习将图像和文本映射到同一共享嵌入空间。

### 架构设计

```mermaid
flowchart TB
    subgraph 图像编码器
        I[图像] --> VIT[ViT/ResNet]
        VIT --> VI[图像特征 v_I]
    end
    
    subgraph 文本编码器
        T[文本] --> TF[Transformer]
        TF --> VT[文本特征 v_T]
    end
    
    VI --> SIM[余弦相似度矩阵\nN×N]
    VT --> SIM
    SIM --> LOSS[InfoNCE Loss]
```

### InfoNCE Loss 数学原理

假设 Batch 中有 $N$ 个图像-文本对 $(I_1, T_1), \dots, (I_N, T_N)$：

**相似度计算**：
$$\text{sim}(I_i, T_j) = \frac{f_I(I_i) \cdot f_T(T_j)}{\|f_I(I_i)\| \|f_T(T_j)\|}$$

**图像到文本的损失**：
$$\mathcal{L}_{I \to T, i} = -\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j) / \tau)}$$

**文本到图像的损失**：
$$\mathcal{L}_{T \to I, i} = -\log \frac{\exp(\text{sim}(T_i, I_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(T_i, I_j) / \tau)}$$

**总损失**：
$$\mathcal{L} = \frac{1}{2N} \sum_{i=1}^N (\mathcal{L}_{I \to T, i} + \mathcal{L}_{T \to I, i})$$

其中 $\tau$ 是可学习的温度系数，调节分布尖锐程度。

#### 温度系数 τ 的深度解析

**参数化方式**：CLIP 将温度系数参数化为可学习标量 $\tau = \log(e^{\tau'})$

| 特性 | 说明 |
| :--- | :--- |
| **初始值** | $\tau \approx 0.07$（对应约14的倒数） |
| **训练过程** | 允许模型自适应调节对比学习难度 |
| **作用机制** | 动态调整logits分布的尖锐程度 |
| **稳定性** | 防止大规模训练中的梯度消失/爆炸 |

**数学意义**：

- **小 τ**：分布更尖锐 → 学习更难的负样本
- **大 τ**：分布更平滑 → 学习更容易
- **可学习**：模型在训练过程中动态调整最优值

### 训练机制解析

| 元素 | 作用 |
| :--- | :--- |
| **正样本对** | 对角线元素 $(I_i, T_i)$，最大化相似度 |
| **负样本对** | 非对角线元素 $(I_i, T_j)_{i \neq j}$，最小化相似度 |
| **温度系数 τ** | 小 τ → 分布更尖锐，学习更难的负样本 |
| **Batch Size** | 越大 → 负样本越多 → 对比学习效果越好 |

### CLIP 的革命性意义

<div class="compare-box">
  <div class="compare-item">
    <div class="compare-title">传统分类模型</div>
    <p class="compare-desc">固定类别标签（如 1000 类）<br/>无法泛化到新类别<br/>需要大量标注数据<br/>封闭词汇表</p>
  </div>
  <div class="compare-vs">VS</div>
  <div class="compare-item highlight">
    <div class="compare-title">CLIP 对比学习</div>
    <p class="compare-desc">开放词汇识别<br/>强大的 Zero-shot 能力<br/>自然语言作为监督信号<br/>任意文本描述</p>
  </div>
</div>

### Zero-shot 推理与提示工程

#### 提示工程（Prompt Engineering）

**单模板 vs 多模板集成**：

| 方法 | 示例 | ImageNet准确率 |
| :--- | :--- | :--- |
| **单模板** | "A photo of a {label}." | ~60% |
| **多模板集成（80个）** | 见下方示例 | **76.2%** |

**多模板示例**：

```python
# CLIP 提示模板集成
templates = [
    "a photo of a {}.",
    "a rendering of a {}.",
    "a cropped photo of the {}.",
    "the photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a photo of my {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    # ... 共80个模板
]

# 对每个模板计算相似度，然后平均
def zero_shot_classify(image, labels, templates):
    image_features = clip.encode_image(image)
    
    logits_per_template = []
    for template in templates:
        # 为每个类别生成提示
        texts = [template.format(label) for label in labels]
        text_features = clip.encode_text(texts)
        
        # 计算相似度
        logits = (image_features @ text_features.T)
        logits_per_template.append(logits)
    
    # 平均所有模板的logits
    final_logits = torch.stack(logits_per_template).mean(dim=0)
    probs = final_logits.softmax(dim=-1)
    
    return probs
```

**原理**：不同模板激活语言模型中对同一概念的不同表述，平均后更鲁棒。

#### Zero-shot 分类流程

```python
# 基础 Zero-shot 分类
image_features = clip.encode_image(image)
text_features = clip.encode_text(["a dog", "a cat", "a bird"])

# 计算相似度
similarity = (image_features @ text_features.T).softmax(dim=-1)
# 输出: [0.95, 0.03, 0.02] → 预测为 "a dog"
```

---

## CLIP 后续演进：对比学习的优化之路

### ALIGN (Google, 2021)：规模暴力

**核心思想**：数据规模 > 数据质量（在足够大时）

| 特性 | ALIGN | CLIP |
| :--- | :--- | :--- |
| **数据规模** | 18亿对 | 4亿对 |
| **数据质量** | 未清洗Alt-text | 基础过滤 |
| **清洗策略** | 几乎无 | CLIP过滤 |
| **结论** | 规模足够大，噪声学习仍有效 | 需要一定过滤 |

**关键发现**：

- 在10亿+规模时，简单双塔架构也能从噪声数据中学到SOTA表征
- 证明了"数据规模定律"在多模态领域同样适用

### SigLIP (2023)：损失函数革命

**问题**：Softmax在分布式训练中的通信瓶颈

```python
# Softmax需要全局归一化（All-Reduce）
loss_i2t = -log(exp(sim(i,t_i)) / sum_j exp(sim(i,t_j)))  # 需要所有GPU的sim
```

**SigLIP方案**：使用Sigmoid替代Softmax

$$\mathcal{L} = -\frac{1}{N^2} \sum_{i,j} \log \sigma(y_{ij} \cdot z_{ij})$$

其中 $y_{ij} = 1$ 如果 $i = j$，否则 $y_{ij} = -1$。

**优势**：

- ✅ **消除全局通信**：每个样本对独立计算损失
- ✅ **支持极大Batch**：32k+（传统Softmax难以达到）
- ✅ **训练更稳定**：避免指数运算的数值问题
- ✅ **负样本利用更高效**：所有配对都参与训练

**性能对比**：

| 模型 | Batch Size | ImageNet准确率 |
| :--- | :--- | :--- |
| CLIP | 32K | 76.2% |
| SigLIP | 32K | **78.1%** |

### CoCa (Google, 2022)：理解+生成统一

**核心创新**：解耦解码器架构

```mermaid
flowchart TB
    subgraph "CoCa架构"
        IMG[图像] --> VIT[ViT编码器]
        VIT --> POOL[池化特征]
        
        TXT[文本] --> UNI[单模态文本层]
        UNI --> MULTI[多模态文本层]
        
        POOL --> CONTRA[对比学习头]
        UNI --> CONTRA
        
        POOL --> CROSS[Cross-Attention]
        MULTI --> CROSS
        CROSS --> GEN[生成头]
    end
```

**双流设计**：

| 模块 | 输入 | 任务 | 损失 |
| :--- | :--- | :--- | :--- |
| **单模态文本层** | 仅文本 | 对比学习 | Contrastive Loss |
| **多模态文本层** | 文本+图像 | 文本生成 | Captioning Loss |

**训练目标**：

$$\mathcal{L}_{total} = \mathcal{L}_{contrastive} + \mathcal{L}_{captioning}$$

**效果**：

- ImageNet零样本：**86.3%**（超越CLIP的76.2%）
- 同时具备理解和生成能力
- 一次前向传播计算两种损失

---

## 实践建议

### 选择视觉编码器

| 场景 | 推荐 | 理由 |
| :--- | :--- | :--- |
| **通用理解** | CLIP ViT-L/14 | 平衡效果与效率 |
| **细粒度识别** | ViT-H/14 或更大 | 更多参数捕获细节 |
| **实时应用** | ViT-B/16 | 速度优先 |
| **多语言** | SigLIP | 更好的多语言支持 |

### 常见问题

::: warning 分辨率陷阱
ViT 对分辨率敏感。如果推理分辨率与训练不同，需要插值位置编码或使用支持动态分辨率的方案（如 AnyRes）。
:::

---

## 参考资源

| 论文 | 主题 |
| :--- | :--- |
| [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) | ViT 原始论文 |
| [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020) | CLIP |
| [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | SigLIP |
