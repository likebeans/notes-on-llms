---
title: 训练微调知识体系
description: 大模型训练与微调技术全景 - 从预训练到对齐
---

# 训练微调知识体系

> 从数据到部署，掌握大模型训练的完整链路

## 🗺️ 训练流程全景图

> 来源：[一个大模型落地的技术详解](https://dd-ff.blog.csdn.net/article/details/150265751)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          大模型训练四阶段流程                                 │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│    预训练       │    监督微调      │    人类对齐      │    部署优化             │
│  (Pre-training) │     (SFT)       │  (RLHF/DPO)     │  (Serving)              │
├─────────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│                 │                 │                 │                         │
│  海量文本       │   指令数据集     │   偏好数据集    │   量化/蒸馏/剪枝        │
│      ↓          │       ↓         │       ↓         │        ↓                │
│  自监督学习     │   任务适配       │   价值对齐      │   高效推理              │
│      ↓          │       ↓         │       ↓         │        ↓                │
│  基座模型       │   指令模型       │   对话模型      │   生产部署              │
│  (Base)         │  (Instruct)     │   (Chat)        │                         │
│                 │                 │                 │                         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
```


---

## 🎯 三大模型版本

> 来源：[大型语言模型（LLM）的版本化、对齐机制与核心概念解析](https://dd-ff.blog.csdn.net/article/details/152255344)

| 版本 | 训练方式 | 特点 | 适用场景 |
|------|----------|------|----------|
| **Base（基座）** | 预训练 | 知识储备丰富，无指令遵循能力 | 知识检索、RAG、续写 |
| **Instruct（指令）** | SFT | 遵循指令，执行任务 | 结构化任务、API调用 |
| **Chat（对话）** | RLHF/DPO | 自然对话，安全对齐 | 聊天机器人、助手 |

::: warning 对齐税（Alignment Tax）
经过RLHF对齐的模型可能在某些任务（如RAG检索）上表现不如基座模型，这被称为"对齐税"。选择模型版本需根据具体场景。
:::

---

## 🔧 核心技术栈

### 训练方法分类

| 方法 | 参数更新 | 资源需求 | 适用场景 |
|------|----------|----------|----------|
| **全量微调（FFT）** | 100% | 极高 | 资源充足，追求最佳效果 |
| **LoRA/QLoRA** | <1% | 低 | 消费级硬件，快速迭代 |
| **Adapter** | <5% | 中 | 多任务适配 |
| **Prefix Tuning** | <1% | 低 | 特定任务优化 |

### 对齐技术对比

| 技术 | 原理 | 复杂度 | 效果 |
|------|------|--------|------|
| **SFT** | 监督学习 | 低 | 基础指令遵循 |
| **RLHF (PPO)** | 强化学习+奖励模型 | 高 | 最佳对齐效果 |
| **DPO** | 直接偏好优化 | 中 | 简化版RLHF |
| **ORPO** | 无参考模型 | 低 | 新兴方法 |

---

## 📊 训练数据要点

> 来源：[垃圾进，垃圾出：打造高质量LLM微调数据集的终极指南](https://dd-ff.blog.csdn.net/article/details/152254276)

::: tip 核心原则
**数据质量 > 数据数量**。高质量的1000条数据，往往胜过低质量的10万条。
:::

### 数据格式

| 格式 | 结构 | 适用场景 |
|------|------|----------|
| **Alpaca** | instruction/input/output | 单轮指令任务 |
| **ShareGPT** | conversations数组 | 多轮对话 |
| **OpenAI** | messages数组 | 通用格式 |

### 数据处理流程

```
原始数据 → 清洗去噪 → PII脱敏 → 质量过滤 → 格式转换 → 数据增强
```

---

## 📚 我的训练微调文章

### 🎓 训练全流程

| 文章 | 简介 |
|------|------|
| [一个大模型落地的技术详解](https://dd-ff.blog.csdn.net/article/details/150265751) | 预训练→微调→蒸馏→强化学习全流程 |
| [Genesis-LLM全流程开源项目解析](https://dd-ff.blog.csdn.net/article/details/155355144) | 从数据处理到部署的完整工具链 |
| [从"扩充书库"到"教授技能"](https://dd-ff.blog.csdn.net/article/details/152267590) | CPT/SFT路线图与"健忘症"解药 |

### 📝 数据与预处理

| 文章 | 简介 |
|------|------|
| [垃圾进，垃圾出：打造高质量微调数据集](https://dd-ff.blog.csdn.net/article/details/152254276) | 数据清洗、格式、质量评估 |
| [Parquet范式：训练数据格式优化](https://dd-ff.blog.csdn.net/article/details/154654277) | 列式存储优化，减少99.8%数据扫描 |
| [深入探秘LLM的"暗语"：特殊Token与模板](https://dd-ff.blog.csdn.net/article/details/152328698) | 模板不一致导致90%微调失败 |
| [词表构建技术深度剖析](https://dd-ff.blog.csdn.net/article/details/155357751) | BPE/Unigram、词表扩充实践 |

### ⚡ 高效微调（PEFT）

| 文章 | 简介 |
|------|------|
| [大模型微调的"省钱"秘笈：PEFT技术深度解析](https://dd-ff.blog.csdn.net/article/details/153965724) | LoRA/Adapter/Prefix Tuning对比 |

### 🎯 对齐技术

| 文章 | 简介 |
|------|------|
| [语言模型对齐技术论述：从PPO到DPO](https://dd-ff.blog.csdn.net/article/details/153269912) | RLHF三阶段流程详解 |
| [强化学习对齐指南：PPO和DPO实施与评估](https://dd-ff.blog.csdn.net/article/details/153184150) | Hugging Face TRL实战教程 |
| [微调高级推理大模型（COT）综合指南](https://dd-ff.blog.csdn.net/article/details/153210150) | 思维链训练方法 |
| [LLM的版本化、对齐机制与核心概念](https://dd-ff.blog.csdn.net/article/details/152255344) | Base/Instruct/Chat版本解析 |

### 🚀 部署与优化

| 文章 | 简介 |
|------|------|
| [压缩巨兽：大语言模型压缩的底层科学](https://dd-ff.blog.csdn.net/article/details/150932519) | 量化/剪枝/蒸馏技术 |
| [llama.cpp工作流与GGUF转换指南](https://dd-ff.blog.csdn.net/article/details/154353525) | 本地化部署实践 |
| [verl与Ray多节点RL终极指南](https://dd-ff.blog.csdn.net/article/details/154654476) | 分布式强化学习 |

---

## 🔗 章节导航

| 章节 | 内容 | 状态 |
|------|------|------|
| [数据处理](/training/data) | 数据清洗、格式、质量控制 | 📝 |
| [SFT监督微调](/training/sft) | 指令微调、模板设计 | 📝 |
| [LoRA高效微调](/training/lora) | 低秩适配、QLoRA | 📝 |
| [RLHF对齐](/training/rlhf) | PPO、奖励模型 | 📝 |
| [DPO直接偏好优化](/training/dpo) | 简化对齐方法 | 📝 |
| [模型评估](/training/eval) | 评估指标、基准测试 | 📝 |
| [部署推理](/training/serving) | 量化、推理优化 | 📝 |

---

## 🌐 外部学习资源

### 开源框架

| 框架 | 说明 |
|------|------|
| [Hugging Face TRL](https://huggingface.co/docs/trl) | RLHF/DPO训练库 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 一站式微调框架 |
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | 简化微调流程 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | 分布式训练优化 |

### 重要论文

| 论文 | 说明 |
|------|------|
| [LoRA](https://arxiv.org/abs/2106.09685) | 低秩适配原始论文 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | RLHF经典论文 |
| [DPO](https://arxiv.org/abs/2305.18290) | 直接偏好优化 |
