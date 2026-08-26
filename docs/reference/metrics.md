---
title: 评估指标
description: RAG、生成、Agent、训练和生产监控常用指标速查。
pageType: article
module: site
updated: '2026-08-26'
contentStatus: needs-review
tags:
  - reference
  - metrics
level: intermediate
prerequisites: []
reviewed: '2026-08-26'
techVersion: 2026-08（指标速查）
---

# 评估指标

指标不是越多越好。先选能解释业务失败的指标，再补自动化评估。每个指标都要配样本，否则只是漂亮数字。

## RAG 指标

| 指标 | 看什么 | 适用场景 |
| --- | --- | --- |
| Recall@k | 正确证据是否进入前 k 个检索结果 | 调 chunk、embedding、召回策略 |
| MRR / NDCG | 正确证据排得是否靠前 | 比较 rerank、混合检索 |
| Context Precision | 放进上下文的内容是否大多相关 | 控制上下文污染 |
| Faithfulness | 答案是否被上下文支持 | 检查幻觉与引用可靠性 |
| Citation Accuracy | 引用是否对应答案中的关键断言 | 企业知识库、合规场景 |

相关阅读：[RAG 评估](/llms/rag/evaluation)。

## 生成质量指标

| 指标 | 看什么 | 注意事项 |
| --- | --- | --- |
| Exact Match | 输出是否与标准答案完全一致 | 适合短答案，不适合开放问答 |
| F1 / Rouge | 文本重叠程度 | 容易奖励“像答案”而非“对答案” |
| JSON/schema pass rate | 是否符合结构化输出约束 | 适合信息抽取、自动化管道 |
| Human preference | 人类更偏好哪个输出 | 要控制评审标准和样本顺序 |
| Refusal quality | 该拒答时是否拒答且解释充分 | 安全、医疗/法律等高风险场景 |

相关阅读：[Prompt 安全测试](/llms/prompt/security)。

## Agent 指标

| 指标 | 看什么 | 适用场景 |
| --- | --- | --- |
| Task Success Rate | 任务是否完成且结果可用 | 端到端 Agent 评估 |
| Tool Call Accuracy | 工具选择和参数是否正确 | 函数调用、MCP 工具 |
| Recovery Rate | 工具失败后能否重试或降级 | 生产工作流 |
| Human Intervention Rate | 需要人工接管的比例 | 半自动系统、审批流 |
| Cost / Latency per Task | 单任务成本和耗时 | 预算控制、体验优化 |

相关阅读：[Agent 评估与监控](/llms/agent/evaluation-monitoring)。

## 训练与微调指标

| 指标 | 看什么 | 适用场景 |
| --- | --- | --- |
| Validation Loss | 模型在验证集上的拟合情况 | 训练过程监控 |
| Win Rate | 新模型相对基线被偏好的比例 | SFT/DPO 后对比 |
| Regression Rate | 旧能力退化比例 | 发布前回归 |
| Safety Pass Rate | 安全/合规样本通过率 | 对齐和上线评估 |
| Calibration | 置信度与正确率是否匹配 | 风险决策、拒答策略 |

相关阅读：[训练评估](/llms/training/eval)。

## 生产监控指标

- 延迟：P50/P95/P99，按模型、工具、检索阶段拆分。
- 成本：输入/输出 token、检索成本、工具调用成本。
- 错误率：模型错误、工具错误、解析错误、超时。
- 安全：越权请求、提示注入、敏感数据命中。
- 用户反馈：点赞/点踩、人工修正、重复问题。

## 指标选择原则

1. 先定义失败样本，再定义指标。
2. 自动指标只能做筛查，关键发布仍要人工抽检。
3. 每次改模型、提示、检索或工具，都用同一评估集对比。
4. 把线上失败样本回流，指标才会越来越贴近真实风险。
