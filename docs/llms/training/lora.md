---
title: LoRA 高效微调
description: Low-Rank Adaptation - 用1%参数达到全量微调效果
---

# LoRA 高效微调

> 让消费级显卡也能微调大模型

## 🎯 核心概念

> 来源：[大模型微调的"省钱"秘笈：PEFT技术深度解析与实战指南](https://dd-ff.blog.csdn.net/article/details/153965724)

### 什么是LoRA？

::: tip 定义
**LoRA（Low-Rank Adaptation）** 是一种参数高效微调（PEFT）技术，通过在预训练权重旁边注入可训练的低秩矩阵，仅更新<1%的参数即可达到接近全量微调的效果。
:::

### LoRA原理

```
原始权重 W (d×k)     LoRA分解
     │                  │
     │           ┌──────┴──────┐
     │           │             │
     ▼           ▼             ▼
   冻结        B (d×r)      A (r×k)
     │           │             │
     │           └──────┬──────┘
     │                  │
     ▼                  ▼
  W·x    +           B·A·x
     │                  │
     └────────┬─────────┘
              │
              ▼
         W·x + B·A·x  (r << d, k)
```

**核心思想**：权重更新矩阵ΔW可以分解为两个低秩矩阵的乘积：ΔW = B × A

---

## 📊 PEFT方法对比

| 方法 | 参数量 | 显存需求 | 训练速度 | 效果 |
|------|--------|----------|----------|------|
| **全量微调（FFT）** | 100% | 极高 | 慢 | 最佳 |
| **LoRA** | <1% | 低 | 快 | 接近FFT |
| **QLoRA** | <1% | 极低 | 中 | 接近LoRA |
| **Adapter** | ~3% | 中 | 中 | 良好 |
| **Prefix Tuning** | <1% | 低 | 快 | 中等 |

### LoRA vs QLoRA

| 特性 | LoRA | QLoRA |
|------|------|-------|
| **基座模型精度** | FP16/BF16 | 4-bit量化 |
| **显存占用** | 中等 | 极低 |
| **训练速度** | 快 | 稍慢 |
| **效果** | 最佳 | 接近LoRA |
| **适用场景** | 有GPU资源 | 消费级显卡 |

---

## 🔧 LoRA实现

### 使用PEFT库

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载基座模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 配置LoRA
lora_config = LoraConfig(
    r=16,                          # 低秩维度
    lora_alpha=32,                 # 缩放因子
    target_modules=[               # 目标模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,             # Dropout比例
    bias="none",                   # 不训练偏置
    task_type=TaskType.CAUSAL_LM   # 任务类型
)

# 3. 应用LoRA
model = get_peft_model(model, lora_config)

# 4. 查看可训练参数
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

### QLoRA实现

```python
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 1. 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4量化
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True       # 双重量化
)

# 2. 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. 准备模型进行k-bit训练
model = prepare_model_for_kbit_training(model)

# 4. 应用LoRA
model = get_peft_model(model, lora_config)
```

---

## ⚙️ 关键超参数

### r（秩）

| r值 | 参数量 | 效果 | 建议场景 |
|-----|--------|------|----------|
| 4 | 最少 | 够用 | 简单任务 |
| 8 | 较少 | 良好 | 一般任务 |
| 16 | 中等 | 很好 | 复杂任务（推荐） |
| 32 | 较多 | 最佳 | 追求极致效果 |
| 64+ | 很多 | 饱和 | 通常不必要 |

### lora_alpha

```python
# 实际缩放因子 = lora_alpha / r
# 常用配置：lora_alpha = 2 * r

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # 缩放因子2
    # ...
)
```

### target_modules

| 模型 | 推荐目标模块 |
|------|-------------|
| **LLaMA/Qwen** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **GPT-2** | c_attn, c_proj, c_fc |
| **BLOOM** | query_key_value, dense, dense_h_to_4h, dense_4h_to_h |

```python
# 自动检测所有线性层
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

# 或手动指定
target_modules = ["q_proj", "v_proj"]  # 最小配置
target_modules = "all-linear"          # 所有线性层
```

---

## 📈 训练流程

### 完整训练脚本

```python
from transformers import TrainingArguments
from trl import SFTTrainer

# 训练参数
training_args = TrainingArguments(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,              # LoRA通常用更高学习率
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",        # 8-bit优化器节省显存
    gradient_checkpointing=True,      # 梯度检查点
)

# 创建训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=2048,
)

# 开始训练
trainer.train()

# 保存LoRA权重
model.save_pretrained("./lora_weights")
```

### 合并权重

```python
from peft import PeftModel

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, "./lora_weights")

# 合并权重
merged_model = model.merge_and_unload()

# 保存完整模型
merged_model.save_pretrained("./merged_model")
```

---

## 🎯 最佳实践

### 显存优化

```python
# 1. 使用梯度检查点
model.gradient_checkpointing_enable()

# 2. 使用8-bit优化器
training_args = TrainingArguments(
    optim="paged_adamw_8bit",
    # ...
)

# 3. 使用Flash Attention（如果支持）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"
)
```

### 多LoRA适配器

```python
from peft import PeftModel

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained("base_model")

# 加载多个LoRA适配器
model = PeftModel.from_pretrained(base_model, "lora_adapter_1")
model.load_adapter("lora_adapter_2", adapter_name="adapter2")

# 切换适配器
model.set_adapter("adapter2")

# 或合并多个适配器
model.add_weighted_adapter(
    adapters=["default", "adapter2"],
    weights=[0.7, 0.3],
    adapter_name="merged"
)
```

---

## 🔗 相关阅读

- [训练微调概述](/training/) - 了解完整训练流程
- [SFT监督微调](/training/sft) - LoRA常用于SFT
- [数据处理](/training/data) - 准备训练数据

> **相关文章**：
> - [大模型微调的"省钱"秘笈：PEFT技术深度解析](https://dd-ff.blog.csdn.net/article/details/153965724)
> - [Genesis-LLM全流程开源项目解析](https://dd-ff.blog.csdn.net/article/details/155355144)

> **外部资源**：
> - [LoRA原始论文](https://arxiv.org/abs/2106.09685)
> - [PEFT库文档](https://huggingface.co/docs/peft)
> - [QLoRA论文](https://arxiv.org/abs/2305.14314)
