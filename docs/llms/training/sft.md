---
title: SFT 监督微调
description: Supervised Fine-Tuning - 让模型学会遵循指令
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
techVersion: 原理与示例复核于 2026-10；训练脚本未在 GPU 实测
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
| **领域适配** | 增强领域语言与任务知识，不能保证事实记忆准确 | CPT / SFT，动态事实配合 RAG |
| **行为对齐** | 教会模型按照用户特定指令和格式输出 | SFT（监督微调） |

### SFT的作用

| 阶段 | 模型能力 | 训练目标 |
|------|----------|----------|
| **预训练后** | 知识储备丰富，但不会对话 | 预测下一个 Token |
| **SFT后** | 理解指令，按要求回答 | 生成符合指令的响应 |
| **RLHF后** | 输出符合人类偏好 | 最大化奖励信号 |

```
一种训练路线：Base → SFT → 偏好优化；Instruct/Chat 是产品命名，不对应强制独立阶段
```

### SFT vs CPT vs RAG

| 策略 | 改变什么 | 适用条件 | 主要验收点 |
| --- | --- | --- | --- |
| CPT | 继续优化语言建模分布 | 大量领域语料、基础领域表征不足 | 领域困惑度与下游任务，通用能力回归 |
| SFT | 提高目标回答的条件概率 | 有稳定任务规范与高质量示范 | 任务成功率、格式正确率、未见问题泛化 |
| RAG | 为当前请求提供外部证据 | 内容常变、需要引用或权限过滤 | 召回覆盖、引用准确性、答案忠实度 |

它们可组合使用。SFT 也能学习知识，CPT 也会改变行为；区分的是训练目标与输入数据，而不是互斥的能力。成本由模型规模、token 数、硬件、标注和服务流量决定，不使用无预算口径的美元区间比较。

### SFT vs 强化学习

SFT 用 teacher forcing 逐 token 拟合示范；RL 通过奖励与采样优化行为。SFT 可以学习多轮对话和多种正确表达，RL 也会被错误奖励诱导产生幻觉，两者都没有天然真实性保证。[InstructGPT](https://arxiv.org/abs/2203.02155)

若任务“好答案长什么样”容易示范，先用 SFT；若多种输出难写唯一标准答案、但能可靠比较或验证结果，再评估偏好优化。用 [DPO](/llms/training/dpo) 或 [RLHF](/llms/training/rlhf)前，先确认 SFT 基线和独立评估足够稳定。

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
混合通用样本是一种回放策略。是否混合、混多少应由通用回归集与领域目标共同决定，不能把某个比例作为所有任务的必选项。
:::

**核心原理**：在领域数据集训练过程中，混合通用指令数据，为通用任务持续提供梯度信号，降低遗忘风险，但不保证所有通用能力保留。

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
| **混合数据** | 通过消融确定通用数据占比 | 简单有效 |
| **EWC 正则化** | 保护重要参数不变 | 理论扎实 |
| **LoRA 微调** | 限制更新参数 | 仍需通用能力回归，不能保证不遗忘 |
| **Replay 机制** | 重放历史数据 | 计算成本高 |

### SFT 数据质量要求

::: warning 关键洞察
[LIMA](https://arxiv.org/abs/2305.11206)展示了少量精选示范在特定基座和任务上的价值；不能把它解释为固定的“1,000 胜过 100,000”定律。质量、覆盖和难度需要一起看。
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

下面展示单轮 Alpaca 样本的显式 labels 构造，重点是回答监督与 padding。为了让输入与输出拼接边界可核对，示例采用自定义模板；已有聊天基座应使用它自己的 chat template，不能直接换成 Alpaca。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
max_length = 2048

def tokenize(sample):
    prompt = (f"### 指令:\n{sample['instruction']}\n\n"
              f"### 输入:\n{sample.get('input', '')}\n\n### 回答:\n")
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(sample["output"], add_special_tokens=False)["input_ids"]
    answer_ids += [tokenizer.eos_token_id]
    # 超长样本交给上游拆分或拒绝，避免静默截掉全部回答
    if len(prompt_ids) + len(answer_ids) > max_length:
        raise ValueError("样本超长：请先统计并制定截断策略")
    ids = prompt_ids + answer_ids
    pad = max_length - len(ids)
    return {
        "input_ids": ids + [tokenizer.pad_token_id] * pad,
        "attention_mask": [1] * len(ids) + [0] * pad,
        "labels": [-100] * len(prompt_ids) + answer_ids + [-100] * pad,
    }

raw = load_dataset("json", data_files="train.json", split="train")
data = raw.map(tokenize, remove_columns=raw.column_names)
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./sft_output", max_steps=10,
                           per_device_train_batch_size=1, learning_rate=2e-5),
    train_dataset=data,
)
# 先检查一条 labels 的非 -100 部分恰为回答与终止符，再启动短跑
trainer.train()
```

这只是全量训练的教学短跑；7B 模型的完整训练状态通常超出单张消费卡容量，资源受限时按 [LoRA](/llms/training/lora)改造。需要 PyTorch、Transformers、Datasets 与模型访问权限，未进行 GPU 实测。

### 使用TRL SFTTrainer

对于包含 `prompt`/`completion` 的对话数据，使用模型内置模板可减少手写分隔符错误。以下假定 `sft_model` 为本地聊天基座，JSONL 的两列均为 messages 数组。

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("sft_model")
train = load_dataset("json", data_files="train.jsonl", split="train")
trainer = SFTTrainer(
    model="sft_model",
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir="./sft_output", max_length=2048,
        completion_only_loss=True, packing=False,
        per_device_train_batch_size=1, num_train_epochs=1,
    ),
    train_dataset=train,
)
trainer.train()
```

按 [TRL SFT 文档](https://huggingface.co/docs/trl/sft_trainer)复核接口；项目应固定依赖版本。先不启用 packing，核对角色、EOS、有效 labels 与截断统计后再比较吞吐。`assistant_only_loss=True` 是另一种 messages 监督选择，要求模板提供 assistant mask，不能对任意模板盲开。

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
指令跟随任务通常只监督 completion 或 assistant 消息；全文语言建模也有使用场景。关键是明确选择并验证 mask。padding 必须忽略，不能因为 pad 与 eos 共用 ID 就把真正的 EOS 标签一并屏蔽。
:::

```python
def create_labels_with_mask(input_ids, response_start_idx):
    """创建带掩码的标签"""
    labels = input_ids.clone()
    # 将指令部分的标签设为-100（忽略）
    labels[..., :response_start_idx] = -100
    return labels
```

---

## ⚙️ 超参数调优

### 关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **learning_rate** | 1e-5 ~ 5e-5 | 学习率，太高易过拟合 |
| **batch_size** | 根据显存调整 | 有效batch=每卡batch×累积步数×数据并行卡数；packing 时另报有效 token 数 |
| **epochs** | 1-3 | SFT通常不需要太多轮次 |
| **warmup_ratio** | 0.03-0.1 | 预热比例 |
| **max_length** | 按业务长度分布确定 | 当前 TRL SFTConfig 的长度上限；记录截断率 |

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

以下函数只接收独立 `generate()` 产生的 token ID 和对应标签；普通 Trainer 默认返回的是 logits，不能直接 `batch_decode`。BLEU/ROUGE 适合作为翻译或摘要的辅助指标，不是开放问答正确性的验收指标。

```python
from evaluate import load

# 加载评估指标
bleu = load("bleu")
rouge = load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[tokenizer.pad_token_id if t == -100 else t for t in row] for row in labels]
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

### 排障顺序与发布门槛

- **loss 很低但不回答**：先解码有效 labels，看是否只学了输入、padding，或回答全被截断。
- **复述用户、无法停止**：比较训练和推理模板、assistant 起始标记、EOS 与停止配置。
- **领域集变好但真实问题变差**：按来源/用户/时间检查泄漏，检查示范是否只有单一模板。
- **通用能力退化**：比较小学习率、早停与混合数据；用独立回归集决定比例。

一次合格实验要保留未微调基线、按来源分组的留出集、模板和 tokenizer revision；按任务成功率、格式通过率与通用回归选 checkpoint，不能只选最低训练 loss。

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
