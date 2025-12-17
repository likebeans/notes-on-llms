---
title: Agent 知识体系
description: AI智能体技术全景图谱 - 从理论到生产实践
---

# Agent 知识体系

> 从演示到生产，掌握AI智能体的设计与实现

## 🗺️ Agent 知识图谱

<div class="knowledge-map">
  <div class="map-center">
    <span class="map-title">AI Agent</span>
  </div>
  <div class="map-branches">
    <div class="branch branch-1">
      <div class="branch-title">🧠 核心能力</div>
      <ul>
        <li><strong>推理</strong>：思维链、ReAct</li>
        <li><strong>规划</strong>：任务分解、目标追踪</li>
        <li><strong>记忆</strong>：短期/长期记忆</li>
        <li><strong>行动</strong>：工具调用、环境交互</li>
      </ul>
    </div>
    <div class="branch branch-2">
      <div class="branch-title">🔧 工具调用</div>
      <ul>
        <li>Function Calling</li>
        <li>MCP协议</li>
        <li>Code Interpreter</li>
        <li>Web Search</li>
      </ul>
    </div>
    <div class="branch branch-3">
      <div class="branch-title">🏗️ 架构模式</div>
      <ul>
        <li>ReAct循环</li>
        <li>Plan-and-Execute</li>
        <li>Multi-Agent协作</li>
        <li>Human-in-the-Loop</li>
      </ul>
    </div>
    <div class="branch branch-4">
      <div class="branch-title">🛠️ 开发框架</div>
      <ul>
        <li>LangGraph</li>
        <li>AutoGen</li>
        <li>OpenAI Agents</li>
        <li>CrewAI</li>
      </ul>
    </div>
  </div>
</div>

<style>
.knowledge-map {
  background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
  border-radius: 16px;
  padding: 2rem;
  margin: 2rem 0;
  color: white;
}
.map-center {
  text-align: center;
  margin-bottom: 1.5rem;
}
.map-title {
  display: inline-block;
  background: white;
  color: #3b82f6;
  font-size: 2rem;
  font-weight: bold;
  padding: 1rem 2rem;
  border-radius: 50px;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.map-branches {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}
.branch {
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 1rem;
}
.branch-title {
  font-weight: bold;
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid rgba(255,255,255,0.3);
}
.branch ul {
  margin: 0;
  padding-left: 1.2rem;
  font-size: 0.9rem;
}
.branch li {
  margin: 0.3rem 0;
}
</style>

---

## 🎯 什么是AI Agent？

::: tip 核心定义
**AI Agent（智能体）** = LLM + 工具 + 记忆 + 规划

它是一个能够**自主感知环境、做出决策、执行行动**的AI系统，通过循环迭代完成复杂任务。
:::

### Agent vs 传统LLM应用

| 特性 | 传统LLM应用 | AI Agent |
|------|-------------|----------|
| **交互模式** | 单轮问答 | 多轮迭代 |
| **决策方式** | 固定流程 | 动态规划 |
| **工具使用** | 无/有限 | 自主调用多种工具 |
| **记忆能力** | 上下文窗口内 | 短期+长期记忆 |
| **错误处理** | 需人工干预 | 自我修正 |

### Agent的核心循环

```python
# Agent核心循环：感知→思考→行动→观察
class AgentLoop:
    def run(self, task):
        while not self.is_complete(task):
            # 1. 感知：获取当前状态和上下文
            context = self.perceive()
            
            # 2. 思考：推理下一步行动
            thought = self.reason(task, context)
            
            # 3. 行动：执行工具调用
            action = self.act(thought)
            
            # 4. 观察：获取行动结果
            observation = self.observe(action)
            
            # 5. 更新记忆
            self.memory.add(thought, action, observation)
        
        return self.generate_response()
```

---

## 🔥 两大核心策略：Function Calling vs ReAct

> 来源：[解码AI智能体的大脑：Function Calling 与 ReAct 策略深度对决](https://dd-ff.blog.csdn.net/article/details/153210207)

### 核心比喻

| 策略 | 比喻 | 特点 |
|------|------|------|
| **Function Calling** | 经验丰富的**主管** | 直接解析任务、下达指令，高效但决策"黑盒" |
| **ReAct** | 缜密的**侦探** | 步步思考、观察、调整，透明但耗时较长 |

### Function Calling工作流程

```
1. 定义工具箱 → 2. 模型决策 → 3. 生成JSON指令 → 4. 外部执行 → 5. 整合反馈
```

```python
# Function Calling示例
# 用户："帮我查一下北京今天的天气"

# 模型返回的结构化调用指令
{
    "name": "get_weather",
    "arguments": {
        "city": "Beijing",
        "unit": "celsius"
    }
}

# 执行函数得到结果
{"temperature": 25, "condition": "晴"}

# 模型生成回复："北京今天天气晴朗，气温是25摄氏度。"
```

### ReAct工作流程

```
Thought → Action → Observation → Thought → Action → ... → Final Answer
```

```python
# ReAct示例
# 问题："苹果公司的CEO是谁？他出生在哪一年？"

# Thought 1: 我需要先找到苹果的现任CEO
# Action 1: Search('Apple Inc. current CEO')
# Observation 1: Tim Cook is the current CEO of Apple Inc.

# Thought 2: 好的，CEO是蒂姆·库克。现在我需要找到他的出生年份
# Action 2: Search('Tim Cook birth year')
# Observation 2: Tim Cook was born on November 1, 1960.

# Thought 3: 我已经获得了所有需要的信息
# Final Answer: 苹果公司的现任CEO是蒂姆·库克，他出生于1960年。
```

### 策略对比

| 维度 | Function Calling | ReAct |
|------|------------------|-------|
| **效率** | ⭐⭐⭐⭐⭐ 高效，通常一次调用 | ⭐⭐⭐ 多轮交互，耗时较长 |
| **透明度** | ⭐⭐ 决策过程不透明 | ⭐⭐⭐⭐⭐ 思考过程完全可见 |
| **复杂任务** | ⭐⭐⭐ 适合明确任务 | ⭐⭐⭐⭐⭐ 擅长多步推理 |
| **纠错能力** | ⭐⭐ 较弱 | ⭐⭐⭐⭐ 可根据观察调整 |
| **成本** | ⭐⭐⭐⭐⭐ Token消耗少 | ⭐⭐ Token消耗较高 |

::: warning 未来趋势
**融合才是王道**：让ReAct负责高层推理和规划，而Function Calling负责底层工具的精准调用。
:::

---

## 🏭 12-Factor Agent：生产级Agent方法论

> 来源：[从演示到生产：构建可靠AI系统的12-Factor Agent方法论](https://dd-ff.blog.csdn.net/article/details/154185674)

### "80%墙"现象

::: danger 生产力鸿沟
许多在演示环境表现出色的Agent，在迈向生产环境时步履维艰。问题根源不是"模型不行"，而是**糟糕的工程实践**。
:::

典型失败模式：
- 陷入无限循环
- 生成格式错误的工具调用
- 丢失状态跟踪
- 无法处理边界情况

### 12个核心原则

| 序号 | 原则 | 核心要点 |
|------|------|----------|
| **1** | 自然语言→工具调用 | LLM作为"路由器"，将意图转为结构化命令 |
| **2** | 掌控你的提示 | 提示词作为代码，纳入版本控制 |
| **3** | 掌控上下文窗口 | 精细控制进入模型的每条信息 |
| **4** | 工具是代码 | 工具函数要像普通代码一样测试 |
| **5** | 统一工具接口 | 标准化工具输入输出格式 |
| **6** | 使用最小上下文 | 只提供最相关的信息 |
| **7** | 模型可切换 | 支持多模型，避免供应商锁定 |
| **8** | 拥抱确定性 | 尽可能使用确定性代码 |
| **9** | 可观测性优先 | 全链路日志和追踪 |
| **10** | 失败是常态 | 设计健壮的错误处理 |
| **11** | 安全边界 | 实施权限控制和沙箱 |
| **12** | 渐进式增强 | 从简单开始，逐步添加能力 |

---

## 📚 我的Agent系列文章

### 🎯 Agent原理与架构

| 文章 | 简介 |
|------|------|
| [解码AI智能体的大脑：Function Calling与ReAct深度对决](https://dd-ff.blog.csdn.net/article/details/153210207) | 两大核心策略对比分析 |
| [12-Factor Agent方法论综合分析](https://dd-ff.blog.csdn.net/article/details/154185674) | 生产级Agent开发原则 |
| [AI不止于掉包：AI系统开发的15条实战原则](https://dd-ff.blog.csdn.net/article/details/149168126) | 从API调用到智能代理 |
| [未来的认知架构：深入剖析自主AI研究智能体](https://dd-ff.blog.csdn.net/article/details/150217636) | 深度研究智能体架构 |

### 🔧 LangGraph系列

| 文章 | 简介 |
|------|------|
| [LangGraph深度解析（一）：核心原理到生产级工作流](https://dd-ff.blog.csdn.net/article/details/151024355) | 状态中心设计、显式控制流 |
| [LangGraph深度解析（二）：函数式API的状态化工作流](https://dd-ff.blog.csdn.net/article/details/151024840) | @entrypoint和@task装饰器 |
| [LangGraph深度解析（三）：流式架构权威指南](https://dd-ff.blog.csdn.net/article/details/151106004) | 可观测、交互式智能体 |
| [LangGraph多智能体系统权威指南](https://dd-ff.blog.csdn.net/article/details/151153365) | 多Agent协作架构 |
| [LangGraph内存机制综合指南](https://dd-ff.blog.csdn.net/article/details/151118407) | 短期/长期记忆管理 |
| [LangGraph工具使用权威指南](https://dd-ff.blog.csdn.net/article/details/151148039) | 构建工具调用型Agent |
| [LangGraph人机协同综合指南](https://dd-ff.blog.csdn.net/article/details/151149262) | Human-in-the-Loop实现 |

### 🤖 AutoGen系列

| 文章 | 简介 |
|------|------|
| [AutoGen AgentChat快速入门](https://dd-ff.blog.csdn.net/article/details/149055083) | 构建智能工具调用型代理 |
| [AutoGen多智能体团队实战指南](https://dd-ff.blog.csdn.net/article/details/149090900) | Teams协作与任务执行 |
| [AutoGen人机交互指南](https://dd-ff.blog.csdn.net/article/details/149093906) | Human-in-the-Loop实现 |
| [AutoGen状态管理实战](https://dd-ff.blog.csdn.net/article/details/149097602) | 从内存到持久化 |
| [AutoGen自定义智能体开发全攻略](https://dd-ff.blog.csdn.net/article/details/149098144) | 从基础到模型集成 |

### 🌐 OpenAI Agent工具

| 文章 | 简介 |
|------|------|
| [OpenAI Agent工具全面开发者指南](https://dd-ff.blog.csdn.net/article/details/154445828) | 从RAG到Computer Use |
| [Responses API完整开发者指南](https://dd-ff.blog.csdn.net/article/details/154444088) | 下一代智能体API |
| [OpenAI Realtime API权威技术指南](https://dd-ff.blog.csdn.net/article/details/154490186) | 语音代理开发 |

### 🔌 MCP协议

| 文章 | 简介 |
|------|------|
| [FastMCP快速入门指南](https://dd-ff.blog.csdn.net/article/details/148854073) | 搭建MCP服务 |
| [FastMCP客户端深度解析](https://dd-ff.blog.csdn.net/article/details/149111605) | 构建MCP交互桥梁 |
| [如何配置Dify中的MCP服务](https://dd-ff.blog.csdn.net/article/details/148588405) | 企业MCP部署 |

---

## 🔗 章节导航

| 章节 | 内容 | 状态 |
|------|------|------|
| [工具调用](/agent/tool-calling) | Function Calling、MCP协议 | 📝 |
| [规划与推理](/agent/planning) | ReAct、Plan-and-Execute | 📝 |
| [记忆系统](/agent/memory) | 短期/长期记忆、状态管理 | 📝 |
| [多智能体](/agent/multi-agent) | 协作架构、通信机制 | 📝 |
| [安全与沙箱](/agent/safety) | 权限控制、沙箱技术 | 📝 |
| [评估方法](/agent/evaluation) | Agent性能评估 | 📝 |

---

## 🌐 外部学习资源

### 官方文档

| 资源 | 说明 |
|------|------|
| [LangGraph文档](https://langchain-ai.github.io/langgraph/) | LangChain的Agent框架 |
| [AutoGen文档](https://microsoft.github.io/autogen/) | 微软多Agent框架 |
| [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) | 官方工具调用指南 |
| [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use) | Claude工具使用 |

### 重要论文

| 论文 | 说明 |
|------|------|
| [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) | ReAct框架原始论文 |
| [Toolformer](https://arxiv.org/abs/2302.04761) | LLM自主使用工具 |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | 斯坦福AI小镇 |
