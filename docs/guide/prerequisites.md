---
title: 前置知识
description: 学习大模型应用开发需要的基础知识
---

# 前置知识

## 🎯 本篇目标

> 了解学习大模型应用开发前需要掌握的基础知识。

## 📊 知识清单

### 必备技能

| 领域 | 要求 | 优先级 | 相关模块 |
|------|------|--------|----------|
| **Python** | 熟练使用，理解面向对象、异步编程 | ⭐⭐⭐ | 所有模块 |
| **HTTP/API** | 理解 RESTful API、JSON 格式 | ⭐⭐⭐ | [Agent](/llms/agent/)、[MCP](/llms/mcp/) |
| **Git** | 基本的版本控制操作 | ⭐⭐ | 所有模块 |

### 加分技能

| 领域 | 要求 | 优先级 | 相关模块 |
|------|------|--------|----------|
| **机器学习** | 理解训练/推理、损失函数、梯度下降 | ⭐⭐ | [训练与微调](/llms/training/) |
| **NLP 基础** | 理解 Tokenization、Embedding 概念 | ⭐⭐ | [RAG](/llms/rag/)、[Prompt](/llms/prompt/) |
| **深度学习** | 了解 Transformer 架构 | ⭐ | [训练与微调](/llms/training/) |
| **Docker** | 容器化部署基础 | ⭐ | [训练与微调](/llms/training/) |

### 模块前置知识映射

| LLM 模块 | 所需前置知识 | 难度 |
|----------|-------------|------|
| [🔍 RAG 检索增强](/llms/rag/) | Python、NLP 基础（Embedding、向量检索）、数据库基础 | ⭐⭐ |
| [🤖 Agent 智能体](/llms/agent/) | Python、HTTP/API、异步编程、基础算法思维 | ⭐⭐⭐ |
| [⚙️ 训练与微调](/llms/training/) | 机器学习、深度学习、PyTorch/TensorFlow、分布式计算 | ⭐⭐⭐⭐ |
| [👁️ 多模态](/llms/multimodal/) | 计算机视觉基础、NLP 基础、Transformer 架构 | ⭐⭐⭐ |
| [✨ Prompt 工程](/llms/prompt/) | 自然语言处理直觉、逻辑思维、实验方法论 | ⭐ |
| [🔌 MCP 协议](/llms/mcp/) | HTTP/API、协议设计、数据流处理 | ⭐⭐ |

## 💻 快速补课资源

### Python

- [Python 官方教程](https://docs.python.org/zh-cn/3/tutorial/)
- 重点掌握：类型提示、异步编程、包管理

### Transformer

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### LLM 基础

- [What Is ChatGPT Doing](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)
- 理解：Token、上下文窗口、Temperature、Top-p

## ⚠️ 常见误区

- ❌ 必须精通深度学习才能开始
- ✅ 应用开发可以先用 API，边做边学原理
- ❌ 需要 GPU 才能学习
- ✅ 大部分学习可以用云 API 完成

## 📚 延伸阅读

### 指南文档

- [学习路线图](/guide/roadmap) - 查看完整的学习路径规划

### 核心技术模块

<div class="custom-card-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">

  <a href="/llms/rag/" class="custom-card" style="padding: 1rem; border: 1px solid #e1e4e8; border-radius: 6px; text-decoration: none; color: inherit;">
    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">🔍 RAG 检索增强</h3>
    <p style="margin: 0; font-size: 0.9rem; color: #586069;">检索增强生成技术，解决知识滞后和幻觉问题</p>
  </a>

  <a href="/llms/agent/" class="custom-card" style="padding: 1rem; border: 1px solid #e1e4e8; border-radius: 6px; text-decoration: none; color: inherit;">
    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">🤖 Agent 智能体</h3>
    <p style="margin: 0; font-size: 0.9rem; color: #586069;">规划、工具使用和复杂任务执行能力</p>
  </a>

  <a href="/llms/training/" class="custom-card" style="padding: 1rem; border: 1px solid #e1e4e8; border-radius: 6px; text-decoration: none; color: inherit;">
    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">⚙️ 训练与微调</h3>
    <p style="margin: 0; font-size: 0.9rem; color: #586069;">SFT、DPO、RLHF、LoRA 等模型定制技术</p>
  </a>

  <a href="/llms/multimodal/" class="custom-card" style="padding: 1rem; border: 1px solid #e1e4e8; border-radius: 6px; text-decoration: none; color: inherit;">
    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">👁️ 多模态</h3>
    <p style="margin: 0; font-size: 0.9rem; color: #586069;">视觉与语言的融合，GPT-4V、LLaVA 等</p>
  </a>

  <a href="/llms/prompt/" class="custom-card" style="padding: 1rem; border: 1px solid #e1e4e8; border-radius: 6px; text-decoration: none; color: inherit;">
    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">✨ Prompt 工程</h3>
    <p style="margin: 0; font-size: 0.9rem; color: #586069;">掌握与大模型高效沟通的艺术</p>
  </a>

  <a href="/llms/mcp/" class="custom-card" style="padding: 1rem; border: 1px solid #e1e4e8; border-radius: 6px; text-decoration: none; color: inherit;">
    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">🔌 MCP 协议</h3>
    <p style="margin: 0; font-size: 0.9rem; color: #586069;">Model Context Protocol，标准化上下文协议</p>
  </a>

</div>

### 快速开始建议

1. **零基础入门**：先学习 [Prompt 工程](/llms/prompt/)（最简单），了解如何与模型交互
2. **应用开发**：掌握 [RAG](/llms/rag/) 和 [Agent](/llms/agent/) 技术，构建实用应用
3. **深度定制**：学习 [训练与微调](/llms/training/)，打造专属模型
4. **前沿探索**：了解 [多模态](/llms/multimodal/) 和 [MCP 协议](/llms/mcp/)，拓展应用边界
