---
title: SFT 监督微调
description: Supervised Fine-Tuning - 让模型学会遵循指令
---

# SFT 监督微调

> 从知识储备到任务执行的关键一步

## 🎯 核心概念

> 来源：[从"扩充书库"到"教授技能"](https://dd-ff.blog.csdn.net/article/details/152267590)

### 什么是SFT？

::: tip 定义
**SFT（Supervised Fine-Tuning）** 是在预训练模型基础上，使用标注的指令-响应数据进行监督学习，使模型学会遵循人类指令、执行特定任务的训练方法。
:::

### SFT的作用

| 阶段 | 模型能力 | 训练目标 |
|------|----------|----------|
| **预训练后** | 知识储备丰富，但不会对话 | 预测下一个Token |
| **SFT后** | 理解指令，按要求回答 | 生成符合指令的响应 |

```
Base Model (续写能力) → SFT → Instruct Model (指令遵循)
```

---

## 📊 SFT四种模式

### 模式对比

| 模式 | 数据来源 | 优点 | 缺点 |
|------|----------|------|------|
| **通用SFT** | 开源通用数据集 | 广泛能力 | 领域表现一般 |
| **领域SFT** | 领域专用数据 | 专业性强 | 可能遗忘通用能力 |
| **混合SFT** | 通用+领域混合 | 平衡两者 | 数据配比需调优 |
| **持续SFT** | 增量领域数据 | 适应新任务 | 需要防遗忘策略 |

### 灾难性遗忘问题

::: danger 核心挑战
领域SFT可能导致模型"忘记"预训练阶段学到的通用知识，这被称为**灾难性遗忘**。
:::

**解决方案**：
- 混入5-10%通用数据
- 使用正则化技术（如EWC）
- 采用LoRA等参数高效方法

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

- [训练微调概述](/training/) - 了解完整训练流程
- [数据处理](/training/data) - 准备高质量训练数据
- [LoRA高效微调](/training/lora) - 低资源SFT方案
- [RLHF对齐](/training/rlhf) - SFT之后的对齐

> **相关文章**：
> - [从"扩充书库"到"教授技能"](https://dd-ff.blog.csdn.net/article/details/152267590)
> - [深入探秘LLM的"暗语"：特殊Token与模板](https://dd-ff.blog.csdn.net/article/details/152328698)
> - [Genesis-LLM全流程开源项目解析](https://dd-ff.blog.csdn.net/article/details/155355144)

> **外部资源**：
> - [Hugging Face TRL](https://huggingface.co/docs/trl/sft_trainer)
> - [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
