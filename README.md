# 📚 notes-on-llms

> **系统性理解大语言模型（LLM）从原理到工程实践的知识仓库**

这是一个面向 AI 工程师、研究者以及深度学习学习者的 **大语言模型全景技术参考指南**，帮助你不只是理解模型，还能**理解模型背后的系统工程逻辑**。

<p align="center">
  <a href="https://likebeans.github.io/notes-on-llms/">
    <img src="https://img.shields.io/badge/📖_在线阅读-GitHub_Pages-blue?style=for-the-badge" alt="在线阅读">
  </a>
  <a href="https://github.com/likebeans/notes-on-llms/stargazers">
    <img src="https://img.shields.io/github/stars/likebeans/notes-on-llms?style=for-the-badge&logo=github" alt="Stars">
  </a>
  <a href="https://github.com/likebeans/notes-on-llms/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
  </a>
</p>

---

## 🚀 为什么要有这个仓库？

当前关于 LLM 的内容非常碎片化：

- 🚫 只讲 Prompt / API 调用教程
- 🚫 只讲单篇论文解读
- 🚫 只讲模型用法而不讲原理
- ❓ 缺少整体视角与工程上下文

而 **notes-on-llms** 追求的是：

> ⭐ **从原理 → 架构 → 工程 → 推理 → 安全 → 多模态 全面理解 LLM 技术栈**

不仅仅是知识笔记，而是一套 **可复用且持续更新的认知框架**。

---

## 🧠 这个仓库适合谁？

| 人群 | 收获 |
|------|------|
| ✅ **AI 工程师** | 系统理解 LLM 技术栈，构建全栈认知地图 |
| ✅ **研究者** | 从"会用模型"进阶到"理解模型原理与架构" |
| ✅ **求职者** | 系统准备 LLM 相关岗位面试 |
| ✅ **学习者** | 已有深度学习基础，希望深入 LLM 领域 |

---

## 🗂 仓库结构概览

> 每个模块都是相对独立的知识单元，组合起来构成完整的大语言模型认知体系。

```
📦 notes-on-llms/docs/llms/
├── 📌 rag/              # RAG 检索增强生成：分块、Embedding、向量库、重排、评估
├── 📌 agent/            # Agent 智能体：规划(CoT/ToT/ReAct)、记忆、工具调用、多智能体
├── 📌 training/         # 训练微调：数据工程、SFT、DPO、RLHF、LoRA、推理优化
├── 📌 prompt/           # 提示工程：ICL、结构化框架、安全防御
├── 📌 multimodal/       # 多模态：视觉编码、模态连接、扩散模型、统一架构
├── 📌 mcp/              # MCP 协议：AI 工具调用标准协议
├── 📌 interviews/       # 面试专区：系统设计、RAG/Agent/训练面试题
└── 📌 reference/        # 速查手册：术语表、Checklist
```

---

## 📖 核心内容导览

| 模块 | 核心主题 | 亮点内容 |
|------|----------|----------|
| **RAG** | 检索增强生成 | 架构演进、分块策略、向量数据库选型、重排机制、评估体系 |
| **Agent** | 智能体系统 | 核心公式、CoT/ToT/ReAct、记忆系统、多智能体框架对比 |
| **Training** | 训练微调 | Transformer 架构、分布式训练、PEFT、对齐技术、推理优化 |
| **Prompt** | 提示工程 | ICL 机制、CRISPE 框架、APE 自动化、安全防御 |
| **Multimodal** | 多模态 | ViT/CLIP、模态连接器、扩散模型、Show-o 统一架构 |
| **MCP** | 工具协议 | JSON-RPC、核心原语、安全架构、生态系统 |

---

## 🔍 学习路线（推荐）

| 阶段 | 模块 | 目标 |
|------|------|------|
| **入门** | `prompt/` | 理解 ICL、CoT 等核心推理范式 |
| **进阶** | `rag/` + `agent/` | 掌握检索增强与智能体架构 |
| **深化** | `training/` | 理解训练全流程与优化技术 |
| **拓展** | `multimodal/` + `mcp/` | 探索多模态与工具协议前沿 |

📌 详细路线请访问 👉 [学习指南](https://likebeans.github.io/notes-on-llms/guide/)

---

## ✨ 本仓库三大特色

### 🎯 1. 全景式技术架构

不是零散笔记集合，而是一套从**底层原理到工程实践**的系统认知图谱。每个模块包含：
- Mermaid 流程图可视化
- 技术对比表格
- 数学公式推导
- 代码示例

### 📈 2. 工程驱动而非只读研究

读完不仅知道怎么实现，还知道：
- **为什么这么设计？**
- **不同范式之间有什么联系和权衡？**
- **在实际工程中会遇到哪些问题？**

### 🔄 3. 同步博客 + 仓库更新

仓库内容与技术博客互联互通：
- 博客提供更易读的讲解
- 仓库提供完整结构化参考
- 两者共同维护最新内容

📌 博客链接：[dd-ff.blog.csdn.net](https://dd-ff.blog.csdn.net/)

---

## 🌐 在线访问

👉 **[https://likebeans.github.io/notes-on-llms/](https://likebeans.github.io/notes-on-llms/)**

---

## 🛠️ 本地开发

```bash
# 克隆仓库
git clone https://github.com/likebeans/notes-on-llms.git
cd notes-on-llms

# 安装依赖
pnpm install

# 启动开发服务器
pnpm docs:dev

# 构建生产版本
pnpm docs:build

# 预览构建结果
pnpm docs:preview
```

---

## 🙌 如何贡献？

欢迎一起完善这个知识库：

- ✔️ 提交 Issue 反馈问题或建议
- ✔️ 提交 Pull Request 补充内容
- ✔️ 改进术语解析与图示
- ✔️ 增加工程实战案例

让这个仓库成为 LLM 技术参考的 **行业级知识地标**。

---

## ⭐ Star & Follow

如果这个仓库对你有帮助：

- 👉 ⭐ **Star** 收藏仓库
- 👉 🛎 **Watch** 关注更新
- 👉 📢 推荐给正在学习 LLM 的同学们！

让我们一起把 LLM 技术理解提升到一个新的等级 🚀

---

## 📄 License

本项目遵循 [MIT](./LICENSE) 许可协议。
