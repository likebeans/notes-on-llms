---
title: DPO 直接偏好优化
description: Direct Preference Optimization - 无需奖励模型的简化对齐
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
reviewScope: DPO 与 GRPO 的数据来源、训练信号及对照实验边界复核；训练未运行
exampleStatus: not-run
techVersion: 原理与示例复核于 2026-10；训练脚本未在 GPU 实测
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
| **效果** | 取决于奖励与在线采样质量 | 取决于偏好对覆盖与训练设置 |

---

## 🔬 DPO原理

> 来源：[RLHF之PPO、DPO详解](https://www.zhihu.com/tardis/zm/art/717010380) | [DPO原理深度解析](https://zhuanlan.zhihu.com/p/11913305485)

![PPO vs DPO](https://pic2.zhimg.com/v2-44f397a445692fe8631990b251d10bdf_r.jpg)
*PPO 和 DPO 的区别*

### 核心思想

DPO的关键洞察：**RLHF 的优化目标存在显式解，可以将奖励函数与最优策略建立解析映射**。

### 从 PPO 到 DPO 的数学推导

**Step 1：KL 正则化奖励最大化的最优策略形式**

对固定奖励、正则系数 β>0 和参考策略，在理想优化条件下，KL 正则化目标的最优策略可以写为（不是 PPO 裁剪算法本身的闭式解）：

```text
π*(y|x) = π_ref(y|x) × exp(r(x,y) / β) / Z(x)
```

其中 `Z(x) = Σ_y π_ref(y|x) × exp(r(x,y) / β)` 是归一化的分区函数。

**Step 2：重参数化奖励函数**

将上式对数化并重排，可以得到奖励函数的形式：

```text
r(x,y) = β × log(π*(y|x) / π_ref(y|x)) + β × log Z(x)
```

**Step 3：代入 Bradley-Terry 偏好模型**

偏好数据遵循 Bradley-Terry 模型，代入重参数化后的 `r(x,y)` 并消去 `Z(x)`，得到：

```text
Δ_θ(x,y) = log(π_θ(y|x) / π_ref(y|x))

p(y_w ≻ y_l | x) = σ(β × [Δ_θ(x,y_w) − Δ_θ(x,y_l)])
```

**Step 4：最终 DPO 损失函数**

```text
L_DPO(π_θ; π_ref)
  = − E_(x,y_w,y_l)~D [log σ(β × [Δ_θ(x,y_w) − Δ_θ(x,y_l)])]

Δ_θ(x,y) = log(π_θ(y|x) / π_ref(y|x))
```

其中：
- `y_w`: 偏好的（chosen）响应
- `y_l`: 不偏好的（rejected）响应
- `β`: 从 KL 正则化目标得到的系数；实际调参还会改变梯度尺度
- `σ`: sigmoid函数

**DPO 本质**：将 RLHF 巧妙转化为类似 SFT 的监督学习，隐式学习奖励函数。

### 直观理解

```
DPO目标：
  ┌─────────────────────────────────────┐
  │  提高 chosen 相对 rejected 的优势   │
  │  优化相对参考策略的对数概率比         │
  │  同时不要偏离参考模型太远             │
  └─────────────────────────────────────┘
```

---

## ⚠️ DPO vs PPO 深度分析

[DPO 论文](https://arxiv.org/abs/2305.18290)从 KL 正则化奖励目标与 Bradley–Terry 偏好模型出发得到分类损失。数学联系并不意味着有限数据、有限模型容量和不同采样方式下训练结果相同。

### 1. Distribution Shift（分布偏移）

常规 DPO 在固定离线偏好对上训练；PPO 用当前策略采样，再由奖励模型评分。前者易受数据覆盖不足影响，后者需要更多 rollout 计算，且仍可能遇到奖励模型对新分布失准。两者都需要新问题和当前模型输出上的评估。

### 2. Reward Hacking 风险

PPO 可能利用显式奖励模型的漏洞；DPO 也可能学会偏好数据里的捷径，例如把“更长”误当作“更正确”。KL 正则只限制分布偏离，不会证明答案正确。不要用没有假设条件的“解集包含关系”推导哪个方法必然更安全。

### 3. 分区函数缺失

这个标题对应一个常见误解：同一 prompt 下，奖励差中的 `β × log Z(x)` **精确抵消**，并非 DPO 漏掉归一化。策略本身仍由归一化的语言模型定义。工程问题是数据、偏好假设和有限优化的误差，不是分区函数被消去。

### 4. Length Bias（长度偏差）

序列 log probability 是有效回答 token 的求和，其尺度与长度有关，但不能近似为“chosen 长度减 rejected 长度”，也不能断言 DPO 总偏好更短回答。若给损失加上仅由固定数据长度决定的常数，梯度为零，不会纠偏。

先按长度区间报告胜率、人工检查同内容不同长度的偏好对；再尝试平衡数据、长度受控评估或有明确论文定义的长度归一化目标。改变求和为平均会改变训练目标，不能只当作数值优化。

### 结论

已有可靠离线偏好对、需要降低 rollout 复杂度时，可把 DPO 作为基线；有可校准的奖励信号、需要持续探索当前策略输出时，再比较在线 RL。选择依据是留出业务胜率、能力回退和预算，而不是“PPO 总最好”或“DPO 只适合学术”。

---

### 偏好数据与在线奖励的选择边界

**2026-10-08 复核**：固定 `chosen/rejected` 对适合先用 DPO 做受控对照；若目标是让当前策略持续尝试新的解题轨迹，且结果能可靠评分，才进一步比较[在线 RL / GRPO](/llms/training/rlhf)。[DPO 原论文](https://arxiv.org/abs/2305.18290)的离线偏好目标与 [DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300)处理的数据来源和优化流程不同。

数据来自旧策略时，先看其回答是否仍代表当前部署错误。可以采集当前策略的新候选并重新标注，形成迭代偏好实验；不能仅因为换成“在线”就省略标注一致性、长度偏差和独立留出评估。比较两条路线时统一初始 checkpoint、评估生成预算和业务 rubric，再把偏好标注、rollout、verifier 成本分开报告。本文完成方法边界核验，未重跑两种训练。



## 🔧 DPO实现

### 使用TRL库

示例依赖 Transformers、Datasets、TRL 和可加载的本地 `sft_model`。按复核时的 [DPOTrainer 文档](https://huggingface.co/docs/trl/dpo_trainer)使用 `processing_class`；未在 GPU 实测，需锁定版本并先用小数据验证一轮。冻结参考模型须与起始 SFT 策略及其模板对应。

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
    beta=0.1,                          # 初始实验值，结合学习率做扫描
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
    processing_class=tokenizer,
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
from peft import LoraConfig

# 此处 model 为重新加载的普通 SFT 基座，尚未包装 PeftModel
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # 显式固定 SFT 参考便于审计；也可按版本配置共享/缓存策略
    args=dpo_config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    peft_config=lora_config,
)
```

`ref_model=None` 不等于算法不需要参考策略；Trainer 可能保存初始策略、切换 adapter 或使用预计算参考 log probability。若 SFT 本身也是 adapter，务必确认参考路径没有退回未做 SFT 的基座。不要先 `get_peft_model` 再重复要求 Trainer 注入同一 adapter。

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

以下方法有各自目标与假设，并非参数开关等价替换。TRL 的实验 Trainer 和接口会随版本调整，代码用于表达配置关系；运行前查看所固定版本的支持范围。

![Iterative-DPO流程](https://pic3.zhimg.com/v2-d5bf8d5dbb07200a39df63b5762b27f0_r.jpg)
*Iterative-DPO 流程*

### Iterative-DPO（迭代式DPO）

迭代式偏好优化是一类工作流；[Iterative Reasoning Preference Optimization](https://arxiv.org/abs/2404.19733)是面向推理的具体研究，不能与所有迭代 DPO 配方混同。通用设计可以是：

1. 确定候选的评价方式（人工、可验证结果或经校准的奖励模型）
2. 将数据分成 m 份
3. 对每份数据：用当前 LLM 采样 k 个回答 → RM 打分 → 选最高/最低构建 pair 对 → 训练一轮 DPO → 更新 LLM
4. 重复直到所有数据训练完成

**优势**：每轮训练后基于最新模型重新采样，缓解 DPO 的分布偏移问题。

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
    train_dataset=dataset["train"],
    processing_class=tokenizer,
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

KTO 不要求 chosen/rejected 配对，不代表不使用参考策略；见 [KTO 原论文](https://arxiv.org/abs/2402.01306)。表中其他效果形容不作为排名，应以同数据预算实验为准。

| 方法 | 需要参考模型 | 复杂度 | 效果 |
|------|-------------|--------|------|
| **DPO** | ✅ 是 | 中 | 很好 |
| **ORPO** | ❌ 否 | 低 | 良好 |
| **IPO** | ✅ 是 | 中 | 很好 |
| **KTO** | ✅ 通常使用参考策略 | 无需成对偏好，但有参考项 | 需实测 |

---

## 🎯 最佳实践

### 数据质量

```python
def validate_preference_data(sample):
    # 仅检查结构；“差异是否有价值”由标注规范与抽样审查判断
    return (
        isinstance(sample.get("prompt"), str)
        and isinstance(sample.get("chosen"), str)
        and isinstance(sample.get("rejected"), str)
        and bool(sample["chosen"].strip())
        and bool(sample["rejected"].strip())
        and sample["chosen"] != sample["rejected"]
    )
```

不要按最小字符数或 0.9 文本相似度直接丢弃样本：只有一个数字、否定词或工具参数不同的回答，可能恰是重要偏好对。逐项检查同一上下文、标签理由、长度偏差、截断后两答是否仍不同；按 prompt 来源/任务分组切分，避免同题变体跨越训练与测试。

### 训练监控

```python
# 关注的关键指标
# 1. rewards/chosen - chosen响应的隐式奖励
# 2. rewards/rejected - rejected响应的隐式奖励
# 3. rewards/margins - 训练差距增大不等于留出质量改善
# 4. logps/chosen - chosen的对数概率
# 5. logps/rejected - rejected的对数概率
```

---

### 从短跑到验收

1. 固定 SFT 基线和参考策略，抽样核对两条回答的有效 loss mask 与终止 token。
2. 在小子集比较 β 和学习率，监控 chosen/rejected log probability、margin、回答长度与重复率。
3. 用未参与训练的 prompt 生成新回答，盲评且交换展示顺序；分事实性、任务成功、风格与安全统计。
4. 若 margin 上升而胜率下降，优先检查标签捷径、分布覆盖和过拟合；若两答 log probability 同降，单看差值无法判断质量。

验收使用相对 SFT 基线的留出胜率与置信区间，并单列能力回退。算法日志中的隐式 reward 不能替代独立评价。

## 🔗 相关阅读

- [训练微调概述](/llms/training/) - 了解完整训练流程
- [RLHF对齐](/llms/training/rlhf) - 传统强化学习对齐
- [SFT监督微调](/llms/training/sft) - DPO的前置步骤

> **相关文章**：
> - [强化学习对齐指南：PPO和DPO实施与评估](https://dd-ff.blog.csdn.net/article/details/153184150)
> - [语言模型对齐技术论述：从PPO到DPO](https://dd-ff.blog.csdn.net/article/details/153269912)

> **外部资源**：
> - [DPO原始论文](https://arxiv.org/abs/2305.18290)
> - [Hugging Face TRL DPO](https://huggingface.co/docs/trl/dpo_trainer)
> - [ORPO论文](https://arxiv.org/abs/2403.07691)
