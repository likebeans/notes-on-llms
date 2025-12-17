---
title: DPO 直接偏好优化
description: Direct Preference Optimization - 无需奖励模型的简化对齐
---

# DPO 直接偏好优化

> 用监督学习的方式做强化学习的事

## 🎯 核心概念

> 来源：[强化学习对齐指南：PPO和DPO实施与评估](https://dd-ff.blog.csdn.net/article/details/153184150)

### 什么是DPO？

::: tip 定义
**DPO（Direct Preference Optimization）** 是一种直接在偏好数据上优化语言模型的方法，无需训练奖励模型，将RLHF简化为类似监督学习的过程。
:::

### DPO vs RLHF

| 特性 | RLHF (PPO) | DPO |
|------|------------|-----|
| **模型数量** | 4个（策略+价值+奖励+参考） | 2个（策略+参考） |
| **训练复杂度** | 高（强化学习） | 低（监督学习） |
| **稳定性** | 需要精细调参 | 相对稳定 |
| **计算成本** | 高 | 中等 |
| **效果** | 最佳 | 接近RLHF |

---

## 🔬 DPO原理

### 核心思想

DPO的关键洞察：**奖励函数可以用策略模型和参考模型的对数概率差来表示**

```
r(x, y) = β * log[π(y|x) / π_ref(y|x)] + β * log Z(x)
```

因此可以跳过奖励模型训练，直接优化偏好：

```
L_DPO = -log σ(β * [log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)])

其中：
- y_w: 偏好的（chosen）响应
- y_l: 不偏好的（rejected）响应
- β: 温度参数
- σ: sigmoid函数
```

### 直观理解

```
DPO目标：
  ┌─────────────────────────────────────┐
  │  增加 chosen 响应的概率              │
  │  降低 rejected 响应的概率            │
  │  同时不要偏离参考模型太远             │
  └─────────────────────────────────────┘
```

---

## 🔧 DPO实现

### 使用TRL库

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("sft_model")
ref_model = AutoModelForCausalLM.from_pretrained("sft_model")
tokenizer = AutoTokenizer.from_pretrained("sft_model")

# 2. 准备偏好数据集
# 格式: {"prompt": "...", "chosen": "好回答", "rejected": "差回答"}
dataset = load_dataset("json", data_files="preference_data.json")

# 3. DPO配置
dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                          # 温度参数
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,                # DPO通常用较低学习率
    num_train_epochs=1,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
)

# 4. 创建DPO训练器
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 5. 开始训练
trainer.train()
```

### 数据格式

```json
{
  "prompt": "请解释什么是人工智能",
  "chosen": "人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统...",
  "rejected": "AI就是机器人啊"
}
```

### 结合LoRA

```python
from peft import LoraConfig, get_peft_model

# LoRA配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# DPO训练（使用LoRA）
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 使用LoRA时可以不需要显式参考模型
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)
```

---

## ⚙️ 关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **beta** | 0.1 ~ 0.5 | 温度参数，控制偏离参考模型的程度 |
| **learning_rate** | 1e-7 ~ 5e-6 | 学习率，比SFT低很多 |
| **epochs** | 1-3 | 训练轮次 |
| **max_length** | 512-1024 | 最大序列长度 |
| **max_prompt_length** | 128-256 | 最大提示长度 |

### Beta参数影响

| beta值 | 效果 |
|--------|------|
| 小 (0.01-0.1) | 更强的偏好学习，可能偏离参考模型较远 |
| 中 (0.1-0.3) | 平衡（推荐） |
| 大 (0.5-1.0) | 更保守，接近参考模型 |

---

## 📊 DPO变体

### ORPO (Odds Ratio Preference Optimization)

无需参考模型的对齐方法：

```python
from trl import ORPOTrainer, ORPOConfig

orpo_config = ORPOConfig(
    output_dir="./orpo_output",
    beta=0.1,
    # ... 其他参数
)

trainer = ORPOTrainer(
    model=model,
    # 注意：无需ref_model
    args=orpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
```

### IPO (Identity Preference Optimization)

```python
# IPO使用不同的损失函数
dpo_config = DPOConfig(
    loss_type="ipo",  # 使用IPO损失
    # ...
)
```

### 方法对比

| 方法 | 需要参考模型 | 复杂度 | 效果 |
|------|-------------|--------|------|
| **DPO** | ✅ 是 | 中 | 很好 |
| **ORPO** | ❌ 否 | 低 | 良好 |
| **IPO** | ✅ 是 | 中 | 很好 |
| **KTO** | ❌ 否 | 低 | 良好 |

---

## 🎯 最佳实践

### 数据质量

```python
def validate_preference_data(sample):
    """验证偏好数据质量"""
    # 1. chosen和rejected不能相同
    if sample["chosen"] == sample["rejected"]:
        return False
    
    # 2. 响应不能过短
    if len(sample["chosen"]) < 50 or len(sample["rejected"]) < 20:
        return False
    
    # 3. 响应需要有实质差异
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, sample["chosen"], sample["rejected"]).ratio()
    if similarity > 0.9:
        return False
    
    return True
```

### 训练监控

```python
# 关注的关键指标
# 1. rewards/chosen - chosen响应的隐式奖励
# 2. rewards/rejected - rejected响应的隐式奖励
# 3. rewards/margins - 两者差距（应该增加）
# 4. logps/chosen - chosen的对数概率
# 5. logps/rejected - rejected的对数概率
```

---

## 🔗 相关阅读

- [训练微调概述](/training/) - 了解完整训练流程
- [RLHF对齐](/training/rlhf) - 传统强化学习对齐
- [SFT监督微调](/training/sft) - DPO的前置步骤

> **相关文章**：
> - [强化学习对齐指南：PPO和DPO实施与评估](https://dd-ff.blog.csdn.net/article/details/153184150)
> - [语言模型对齐技术论述：从PPO到DPO](https://dd-ff.blog.csdn.net/article/details/153269912)

> **外部资源**：
> - [DPO原始论文](https://arxiv.org/abs/2305.18290)
> - [Hugging Face TRL DPO](https://huggingface.co/docs/trl/dpo_trainer)
> - [ORPO论文](https://arxiv.org/abs/2403.07691)
