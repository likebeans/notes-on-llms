---
title: 开源项目
description: LLM 应用、Agent、MCP、训练和评估相关的官方或主流开源项目索引。
pageType: landing
module: site
updated: '2026-10-08'
contentStatus: needs-review
tags:
  - resources
  - repos
---

# 开源项目

看仓库时，不要只看 star。更重要的是：是否还在维护、示例是否能跑、issue 里暴露了哪些工程边界。

<SourceList :items="[
  { title: 'OpenAI Cookbook', href: 'https://github.com/openai/openai-cookbook', note: '持续更新 · OpenAI API、RAG、评估、Agent 和结构化输出样例。' },
  { title: 'OpenAI Evals', href: 'https://github.com/openai/evals', note: '2023 · 组织模型评估样本、指标和回归。' },
  { title: 'Model Context Protocol', href: 'https://github.com/modelcontextprotocol', note: '2024 · MCP 协议、SDK 与官方生态入口。' },
  { title: 'Model Context Protocol servers', href: 'https://github.com/modelcontextprotocol/servers', note: '2024 · MCP server 参考实现与社区资源入口。' },
  { title: 'Hugging Face Transformers', href: 'https://github.com/huggingface/transformers', note: '2018 · 模型加载、推理、微调和多模态模型生态。' },
  { title: 'LangChain', href: 'https://github.com/langchain-ai/langchain', note: '2022 · LLM 应用组件、链式编排和 Agent 工程模式。' },
  { title: 'LangGraph', href: 'https://github.com/langchain-ai/langgraph', note: '2024 · 用图建模 Agent 状态、控制流和多步任务。' },
  { title: 'FAISS', href: 'https://github.com/facebookresearch/faiss', note: '2017 · 向量索引、近似最近邻检索和性能权衡。' },
  { title: 'LLaVA', href: 'https://github.com/haotian-liu/LLaVA', note: '2023 · 视觉指令微调和多模态模型实践。' }
]" />

## 读仓库的顺序

1. README：确认项目定位、维护状态和最小示例。
2. examples/cookbook：找能直接跑的小样例。
3. docs：确认 API 当前版本。
4. tests：学习作者如何定义正确性。
5. issues：收集常见坑，尤其是生产部署、版本冲突和性能问题。

## 把“看源码”变成一次小验证

| 关注方向 | 在仓库中追踪的路径 | 验证产物 |
| --- | --- | --- |
| RAG | 文档读取 → 索引 → 查询 → 返回候选 | 一条有来源 ID 的检索记录 |
| Agent | 状态初始化 → 工具调度 → 检查点 → 终态 | 失败一次后恢复的执行记录 |
| MCP | 服务注册 → 初始化 → 工具发现 → 调用 | 无需模型参与的 Client 测试 |
| 训练 | 数据整理 → 模板/标签 → loss → 保存加载 | 小批次形状检查与加载后回归 |
| 多模态 | 预处理 → 编码 → 连接 → 生成 | 图像尺寸与 token/特征形状记录 |

先固定一个 tag 或 commit，再照对应文档运行最小示例。记录 Python/Node、依赖锁文件、硬件与启动参数；查看 license、模型权重许可和数据许可时分别核对，不能只看仓库首页的一个许可证。项目选择最终用自己的任务集、运维成本与退出成本比较。
