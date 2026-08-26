---
title: 训练微调面试题
description: 训练微调相关面试题精编
pageType: article
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - interviews
level: intermediate
prerequisites:
  - /llms/training/
reviewed: '2026-08-26'
techVersion: 2026-08（SFT、偏好优化与评估）
---

# 训练微调面试题

训练微调面试常见陷阱是把所有问题都回答成“加数据再训”。更好的回答方式是先判断要改的是能力、行为、格式、事实性还是部署表现，再选择数据、SFT、LoRA、DPO/RLHF 或评估方案。

## 核心问题

1. Base、Instruct、Chat 模型的训练目标和使用边界有什么不同？
2. SFT 适合解决什么问题？什么时候 SFT 不如 prompt、RAG 或产品规则？
3. LoRA/PEFT 降低了哪些成本？它不能解决哪些基础能力问题？
4. RLHF 和 DPO 的输入数据、训练流程和风险有什么差异？
5. 微调后你会如何设计离线评估、红队评估和线上回归？

## 追问角度

- 让候选人解释数据版本、训练集/验证集隔离、重复样本和评估污染。
- 追问 chosen/rejected 数据：是否有标注指南、一致性检查、长度偏差检查和安全标签。
- 追问 serving：量化、batching、KV cache、上下文长度和采样参数改变后是否需要重新评估。

## 示例回答骨架

我会先建立失败样本表：每条失败标注为知识不足、格式错误、推理错误、安全边界错误或服务配置问题。格式与风格问题优先用 SFT；领域轻量适配可以用 LoRA；多个答案都有道理但偏好稳定时再考虑 DPO/RLHF；如果失败来自检索不到证据，就不应该用微调掩盖 RAG 问题。所有训练都要绑定固定评估集、人工抽样和可回滚模型版本。

相关复习：[LLM 训练全景](/llms/training/) 与 [模型评估](/llms/training/eval)。
