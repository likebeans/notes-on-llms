---
title: 智能体间通信
description: A2A协议 - 跨框架智能体协作的开放标准
pageType: article
module: agent
updated: '2025-12-27'
contentStatus: needs-review
tags:
  - agent
level: advanced
prerequisites:
  - /llms/prompt/
  - /llms/rag/
reviewed: '2026-08-25'
techVersion: 待复核（2026-08）
---

# 智能体间通信（A2A）

> 让不同框架的AI智能体无缝协作

## 🎯 核心概念

> 来源：[Agentic Design Patterns - Inter-Agent Communication](https://github.com/ginobefun/agentic-design-patterns-cn)

### 什么是A2A？

::: tip 定义
**Agent2Agent (A2A)** 是Google推出的开放标准协议，使基于不同框架（LangGraph、CrewAI、ADK等）构建的AI智能体能够无缝协调、任务委派和信息交换。
:::

### 为什么需要A2A？

| 问题 | A2A解决方案 |
|------|-------------|
| 不同框架智能体无法通信 | 提供通用协议 |
| 集成成本高、耗时 | 标准化接口 |
| 智能体孤立运行 | 支持协作和任务委派 |

### 行业支持

Atlassian、Box、LangChain、MongoDB、Salesforce、SAP、ServiceNow、Microsoft等均支持A2A协议。

---

## 🔄 核心参与者

```
用户 → A2A客户端(Client Agent) → A2A服务器(Remote Agent)
       代表用户请求                处理请求返回结果
```

| 角色 | 说明 |
|------|------|
| **用户** | 发起智能体协助请求 |
| **A2A客户端** | 代表用户请求操作或信息 |
| **A2A服务器** | 处理请求的远程智能体（不透明系统） |

---

## 🪪 智能体卡片（Agent Card）

智能体的数字身份文件（JSON格式）：

| 内容 | 说明 |
|------|------|
| **身份信息** | 名称、描述、版本 |
| **端点URL** | 通信地址 |
| **功能声明** | 流式传输、推送通知等 |
| **技能列表** | 智能体能做什么 |
| **认证要求** | apiKey、OAuth等 |

### 示例

```json
{
  "name": "WeatherBot",
  "description": "提供天气预报",
  "url": "http://weather.example.com/a2a",
  "capabilities": { "streaming": true },
  "skills": [
    { "id": "get_weather", "name": "获取天气" }
  ]
}
```

---

## 🔍 智能体发现

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **Well-Known URI** | 标准路径`/.well-known/agent.json` | 公共服务 |
| **托管注册中心** | 集中目录查询 | 企业环境 |
| **直接配置** | 私下嵌入共享 | 私有系统 |

---

## 📨 通信与任务

### 任务（Task）

| 属性 | 说明 |
|------|------|
| **唯一ID** | 任务标识符 |
| **状态** | submitted → working → completed |
| **contextId** | 关联多次交互 |

### 消息（Message）

| 组成 | 说明 |
|------|------|
| **attributes** | 元数据（优先级、时间等） |
| **parts** | 实际内容（文本、文件、JSON） |

### 通信协议

- **传输**：HTTP(S)
- **格式**：JSON-RPC 2.0

---

## ⚙️ 四种交互机制

| 机制 | 方法 | 适用场景 |
|------|------|----------|
| **同步请求/响应** | `sendTask` | 快速即时操作 |
| **异步轮询** | 返回taskId定期查询 | 长时间任务 |
| **流式更新(SSE)** | `sendTaskSubscribe` | 实时增量结果 |
| **推送通知(Webhooks)** | 注册URL推送 | 超长运行任务 |

---

## 🔐 安全性

| 机制 | 说明 |
|------|------|
| **双向TLS** | 加密认证连接 |
| **审计日志** | 记录所有通信 |
| **卡片声明** | 明确认证要求 |
| **凭证处理** | OAuth/API密钥通过HTTP标头 |

---

## 🔀 A2A vs MCP

| 协议 | 侧重点 | 作用 |
|------|--------|------|
| **A2A** | 智能体之间 | 协调、任务委派、协作 |
| **MCP** | 智能体与外部资源 | 构建上下文、工具交互 |

> A2A和MCP是**互补的**：MCP连接外部工具，A2A让智能体协作。

---

## 📋 应用场景

| 场景 | 说明 |
|------|------|
| **多框架协作** | ADK、LangGraph、CrewAI等智能体协作 |
| **工作流编排** | 数据收集 → 分析 → 报告生成 |
| **动态信息检索** | 主智能体调用数据获取智能体 |

---

## 💡 核心要点

### 使用场景

- 协调**两个或多个AI智能体**
- 智能体使用**不同框架**构建
- 构建**复杂模块化应用**
- 需要**动态发现**其他智能体能力

### 一句话总结

> **A2A协议通过标准化通信方式，使不同框架的AI智能体能够无缝协调、任务委派和信息共享，为构建复杂多智能体系统提供模块化和可扩展的生态。**

---

## 🔗 相关阅读

- [多智能体协作](/llms/agent/multi-agent) - A2A是实现协作的通信基础
- [工具调用](/llms/agent/tool-calling) - MCP处理工具交互

> **参考文献**：
> - [Google A2A Protocol](https://github.com/google/A2A)
> - [Agentic Design Patterns](https://github.com/ginobefun/agentic-design-patterns-cn)
