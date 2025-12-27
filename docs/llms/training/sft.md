---
title: SFT 监督微调
description: Supervised Fine-Tuning - 让模型学会遵循指令
---

# SFT 监督微调

> 从知识储备到任务执行的关键一步

## 🎯 核心概念

> 来源：[从“扩充书库”到“教授技能”](https://dd-ff.blog.csdn.net/article/details/152267590) | [RLHF之PPO、DPO详解](https://www.zhihu.com/tardis/zm/art/717010380)

![LLM训练三阶段](https://pic3.zhimg.com/v2-3b375dd479626f33ebc50dd7cba374fc_r.jpg)
*LLM 训练流程：预训练 → SFT → RLHF*

### 什么是SFT？

::: tip 定义
**SFT（Supervised Fine-Tuning）** 是在预训练模型基础上，使用高质量结构化标签数据（指令-输入-输出对）教导模型遵循特定任务行为和输出格式的训练方法。
:::

**底层原理**：训练模型将输入（Prompt/Instruction）映射到期望输出（Completion），实现**行为对齐**：让模型从“仅会预测下一个词”的基座模型，转变为“能理解并执行指令”的聊天助手或任务解决者。

### 领域大模型定制的两大目标

| 目标 | 说明 | 实现方式 |
|------|------|----------|
| **知识注入** | 确保模型掌握专业领域的术语和事实 | CPT（持续预训练） |
| **行为对齐** | 教会模型按照用户特定指令和格式输出 | SFT（监督微调） |

### SFT的作用

| 阶段 | 模型能力 | 训练目标 |
|------|----------|----------|
| **预训练后** | 知识储备丰富，但不会对话 | 预测下一个 Token |
| **SFT后** | 理解指令，按要求回答 | 生成符合指令的响应 |
| **RLHF后** | 输出符合人类偏好 | 最大化奖励信号 |

```
Base Model (续写能力) → SFT → Instruct Model (指令遵循) → RLHF → Chat Model (对齐)
```

### SFT vs CPT vs RAG

| 策略 | 目标 | 数据类型 | 成本 | 核心优势 |
|------|------|----------|------|----------|
| **CPT** | 注入领域知识 | 海量非结构化文本 | 高（7-100万美元） | 深度适配领域语言，修复知识结构性缺陷 |
| **SFT** | 教授指令遵循行为 | 高质量结构化问答数据 | 中低（0.5-14万美元） | 精准控制输出格式、风格和任务解决能力 |
| **RAG** | 访问实时/私有知识 | 外部文档/知识库 | 低 | 知识即时更新，高可追溯性 |

::: warning 成本对比
- 从头训练：~7800万美元（GPT-4估计）
- CPT：7-100万美元
- SFT + PEFT：成本最低，价值实现时间最短
:::

### SFT vs 强化学习

根据 OpenAI 联合创始人 John Schulman 的报告，SFT 和强化学习各有优势：

| 维度 | SFT | 强化学习（RL） |
|------|-----|---------------|
| **反馈粒度** | 针对单个 Token | 针对整体输出 |
| **表达多样性** | 受限于标注数据 | 可探索多种表达 |
| **幻觉问题** | 容易产生幻觉 | 可通过奖励函数缓解 |
| **多轮对话** | 难以建模长期目标 | 可累积奖励优化 |
| **训练难度** | 简单，类似监督学习 | 复杂，需要调参 |

---

## 📊 SFT四种模式

SFT并非单一流程，数据选择与训练目标决定模型最终形态：

### 模式对比

| 模式 | 数据来源 | 目标 | 优点 | 缺点 |
|------|----------|------|------|------|
| **通用SFT** | 开源通用数据集 | 建立基础指令遵循能力 | 广泛能力、多任务 | 领域表现一般 |
| **领域SFT** | 领域专用数据 | 适配领域任务需求 | 专业性强 | 可能遗忘通用能力 |
| **混合SFT** | 通用+领域混合 | 平衡专业性与通用性 | CF防御标准实践 | 数据配比需调优 |
| **持续SFT** | 增量领域数据 | 适应新任务 | 灵活扩展 | 需要防遗忘策略 |

### A. 通用SFT（指令微调）

通用 SFT 通常被称为**指令微调（Instruction Tuning）**，是 LLM 后训练的初始阶段：

- 目标：建立模型基础指令遵循能力和多任务处理能力
- 数据：覆盖翻译、摘要、问答等场景的广泛、多样化指令数据集
- 结果：模型从 Base Model 转化为 **Instruct Model** 或 **Chat Model**

### B. 领域SFT

领域 SFT 是“在特定领域数据集上训练模型，适配领域任务需求”：

- 数据：领域专业术语、特定格式、复杂任务规则标注（如医疗本体库、药物相互作用规则）
- 挑战：**灾难性遗忘**——领域数据通常高度集中、范围狭窄

### C. 混合SFT（推荐）

::: tip 标准实践
混合 SFT 已成为领域微调的**标准实践**（非可选模式），本质是基于“回放（Rehearsal）”的数据级保障机制。
:::

**核心原理**：在领域数据集训练过程中，混合通用指令数据，确保模型学习领域技能时，与通用能力相关的权重不被完全失活。

### D. 模型起点选择

SFT 前的核心决策——选择 **Base Model** 还是 **Instruct Model** 作为起点：

| 模型类型 | 核心优势 | 适用场景 |
|----------|----------|----------|
| **Base Model** | 灵活性最高，可开发全新专业化对话格式 | 需高度定制化输出格式/任务 |
| **Instruct Model** | 已具备对话结构、“助手”人设、多轮上下文理解 | 领域专家聊天机器人，需连贯聊天体验 |

### 灾难性遗忘问题

::: danger 核心挑战
领域SFT可能导致模型"忘记"预训练阶段学到的通用知识，这被称为**灾难性遗忘**。
:::

**解决方案**：

| 方法 | 原理 | 效果 |
|------|------|------|
| **混合数据** | 混入 5-10% 通用数据 | 简单有效 |
| **EWC 正则化** | 保护重要参数不变 | 理论扎实 |
| **LoRA 微调** | 仅更新少量参数 | 推荐方案 |
| **Replay 机制** | 重放历史数据 | 计算成本高 |

### SFT 数据质量要求

::: warning 关键洞察
研究表明：**高质量的 1,000 条数据 > 低质量的 100,000 条数据**
:::

| 维度 | 要求 |
|------|------|
| **准确性** | 响应内容正确无误 |
| **相关性** | 响应切题，符合指令 |
| **多样性** | 覆盖多种任务类型和表达方式 |
| **一致性** | 风格、格式保持统一 |
| **安全性** | 不含有害、偏见内容 |

---

## 🔧 SFT实现

### 使用Transformers

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# 1. 加载模型和分词器
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. 准备数据集
def format_instruction(sample):
    """格式化为指令格式"""
    return f"""### 指令:
{sample['instruction']}

### 输入:
{sample.get('input', '')}

### 回答:
{sample['output']}"""

def tokenize(sample):
    text = format_instruction(sample)
    return tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

dataset = load_dataset("json", data_files="train.json")
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# 3. 训练配置
training_args = TrainingArguments(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
)

# 4. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)
trainer.train()
```

### 使用TRL SFTTrainer

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# SFT配置
sft_config = SFTConfig(
    output_dir="./sft_output",
    max_seq_length=2048,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    packing=True,  # 样本打包，提升效率
)

# 创建训练器
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=format_instruction,
)

trainer.train()
```

---

## 📝 模板设计

### 常见模板格式

#### Alpaca模板
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

#### ChatML模板
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
```

#### Llama2模板
```
<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST] {assistant_message} </s>
```

### 损失掩码

::: warning 重要
SFT训练时，只应该计算**响应部分**的损失，不应该计算指令/输入部分的损失。
:::

```python
def create_labels_with_mask(input_ids, response_start_idx):
    """创建带掩码的标签"""
    labels = input_ids.clone()
    # 将指令部分的标签设为-100（忽略）
    labels[:response_start_idx] = -100
    return labels
```

---

## ⚙️ 超参数调优

### 关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **learning_rate** | 1e-5 ~ 5e-5 | 学习率，太高易过拟合 |
| **batch_size** | 根据显存调整 | 有效batch=batch×gradient_accumulation |
| **epochs** | 1-3 | SFT通常不需要太多轮次 |
| **warmup_ratio** | 0.03-0.1 | 预热比例 |
| **max_seq_length** | 2048-4096 | 最大序列长度 |

### 学习率调度

```python
from transformers import get_cosine_schedule_with_warmup

# 余弦退火调度器
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)
```

---

## 📈 评估方法

### 自动评估指标

```python
from evaluate import load

# 加载评估指标
bleu = load("bleu")
rouge = load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU分数
    bleu_score = bleu.compute(
        predictions=decoded_preds, 
        references=[[l] for l in decoded_labels]
    )
    
    # ROUGE分数
    rouge_score = rouge.compute(
        predictions=decoded_preds, 
        references=decoded_labels
    )
    
    return {
        "bleu": bleu_score["bleu"],
        "rouge-l": rouge_score["rougeL"]
    }
```

### 人工评估维度

| 维度 | 评估内容 |
|------|----------|
| **相关性** | 回答是否切题 |
| **准确性** | 内容是否正确 |
| **流畅性** | 表达是否自然 |
| **完整性** | 是否覆盖要点 |
| **安全性** | 是否有害内容 |

---

## 🔗 相关阅读

- [训练微调概述](/llms/training/) - 了解完整训练流程
- [数据处理](/llms/training/data) - 准备高质量训练数据
- [LoRA高效微调](/llms/training/lora) - 低资源SFT方案
- [RLHF对齐](/llms/training/rlhf) - SFT之后的对齐

> **相关文章**：
> - [从"扩充书库"到"教授技能"](https://dd-ff.blog.csdn.net/article/details/152267590)
> - [深入探秘LLM的"暗语"：特殊Token与模板](https://dd-ff.blog.csdn.net/article/details/152328698)
> - [Genesis-LLM全流程开源项目解析](https://dd-ff.blog.csdn.net/article/details/155355144)

> **外部资源**：
> - [Hugging Face TRL](https://huggingface.co/docs/trl/sft_trainer)
> - [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
