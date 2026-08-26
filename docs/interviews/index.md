---
title: 面试专区
description: LLM 应用、RAG、Agent、训练微调、系统设计和代码题的准备路径。
pageType: landing
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - interviews
---

# 面试专区

面试准备的关键不是背很多名词，而是能把问题讲成“目标、方案、权衡、评估、失败案例”。这一页帮你把五类题目串成一条准备路径。

## 准备顺序

1. 先过 [术语表](/reference/glossary)：确保 Token、Embedding、RAG、Agent、LoRA、DPO 等词能说准。
2. 再准备 [RAG 面试题](/interviews/rag-questions)：检索、重排、引用、评估是应用岗高频区。
3. 接着准备 [Agent 面试题](/interviews/agent-questions)：工具调用、规划、记忆、安全和监控。
4. 如果投算法/平台岗，补 [训练微调面试题](/interviews/training-questions)。
5. 最后练 [系统设计](/interviews/system-design) 与 [代码题](/interviews/coding)，把概念落到架构和实现。

## 五个题型

| 题型 | 重点能力 | 回答结构 |
| --- | --- | --- |
| [系统设计](/interviews/system-design) | 需求澄清、架构拆分、指标与风险 | 场景 → 数据 → 检索/模型/工具 → 评估 → 运维 |
| [RAG 面试题](/interviews/rag-questions) | chunk、embedding、召回、重排、引用 | 问题 → 失败原因 → 改进方案 → 指标 |
| [Agent 面试题](/interviews/agent-questions) | 工具 schema、规划、状态、权限 | 任务边界 → 工具 → 控制流 → 失败恢复 |
| [训练微调面试题](/interviews/training-questions) | 数据、SFT/LoRA、DPO/RLHF、评估 | 是否该训练 → 数据 → 方法 → 风险 |
| [代码题](/interviews/coding) | 小函数、数据结构、工程边界 | 先说复杂度和边界，再写可测实现 |

## 面试冲刺路径

| 时间 | 任务 | 产出 |
| --- | --- | --- |
| 第 1-2 天 | 整理术语和个人项目 | 一页术语卡、两段项目介绍 |
| 第 3-7 天 | RAG/Agent 高频问答 | 每题 2 分钟口述答案 |
| 第 2 周 | 系统设计与评估 | 2-3 张架构草图、指标清单 |
| 第 3 周 | 代码题和项目复盘 | 可运行小题、失败案例库 |
| 第 4 周以后 | 模拟面试与查漏补缺 | 录音复盘、问题清单 |

## 回答时的检查清单

- 是否说明了业务目标，而不是只讲技术名词？
- 是否说清数据从哪里来、如何更新、如何清洗？
- 是否能定位错误来自检索、上下文、模型、工具还是评估？
- 是否有指标：召回、准确性、引用、延迟、成本、安全？
- 是否提到生产约束：权限、审计、回滚、监控、人机确认？

更多可复用表达见 [模板](/reference/templates)。
