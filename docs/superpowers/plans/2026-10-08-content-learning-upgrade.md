# 内容与学习体验升级 Implementation Plan

> **For agentic workers:** Use the approved six-part scope in this conversation. Work in assigned files, preserve existing edits, and verify behavior before completion.

**Goal:** 让现有知识站更容易复现、查找与持续学习，并用官方文档和论文更新六大技术模块。

**Architecture:** 保留 VitePress 和现有视觉设计。内容更新与离线示例独立推进；共享的内容索引负责来源统计、文章筛选和中文前置标题；学习记录保存在当前浏览器，无账号或远端写入。

**Tech Stack:** VitePress / Vue / TypeScript / Vitest，Python 标准库与独立可选 MCP 依赖。

**Spec:** 用户在本任务确认的六项优化：可运行案例、长文分层、来源与核验信息、问题路线、自测与学习记录、搜索筛选，以及最新 AI 知识更新。

## Global Constraints

- 保留上一轮未提交更改；不提交、发布或发送消息。
- 所有新增知识结论用可核验的一手来源，标注核验日期和适用版本。
- 正文复核不等于代码实跑，不将旧协议示例冒充最新协议。
- UI 保持既有纸色、蓝色和中文阅读排版；支持键盘、窄屏与深色模式。
- 学习记录遇到损坏数据或禁用存储时不能使文章崩溃；不保存文章正文。

## Task 1: 可复现知识助手

- [x] 在 examples/knowledge-assistant 提供固定样本、透明检索基线、只读工具、trace 和评估。
- [x] 覆盖有答案、无答案、非法参数、引用与评测分母的行为测试。
- [x] 提供可选真实 MCP stdio 适配器与测试；教程接入实践导航。

## Task 2: 经过来源核验的知识更新与分层

- [x] Agent / MCP / Prompt：协议生命周期、工具调用、上下文、结构化输出与安全。
- [x] RAG / Training / Multimodal：检索与上下文、可验证奖励、参数高效训练、多模态模型与部署边界。
- [x] Embedding / RAG 范式增加核心摘要、按任务阅读和折叠深入内容，保留原锚点。
- [x] 六个模块各提供三个场景自测及展开答案。

## Task 3: 可信元数据

Files: docs/.vitepress/content/{sourceLinks.ts,content.data.ts,selectors.ts}; article components; tests/content/source-links.test.ts.

- [x] 先测试行内、引用式、自动链接、HTML、去重、代码与图片排除，再实现来源提取。
- [x] 前置 URL 映射到内容索引标题；展示明确复核范围，保留未实测说明。

## Task 4: 学习记录与文章检索

Files: docs/.vitepress/theme/components/learning/*; docs/.vitepress/content/discovery.ts; tests/content/{learning-records,discovery}.test.ts; docs/guide/library.md.

- [x] 测试损坏存储、只允许有效站内路径、完成/收藏切换、模块/难度/类型交集与搜索匹配。
- [x] 文章页添加完成与收藏操作；学习页显示最近阅读、已完成和收藏。
- [x] 检索页提供中文搜索、模块、难度、主线/镜像筛选；保留现有全文搜索。
- [x] 模块栏分开展示当前章节位置和实际完成数量。

## Task 5: 问题路线与整合验收

- [x] 添加按“回答不准、工具重复、延迟成本、何时微调、图表误读”组织的站内诊断路径。
- [x] 更新导航、实践入口、关于、更新日志和 README。
- [x] 运行完整 Vitest、示例单测、内容检查、生产构建、站点检查与站内链接审计。
- [x] 浏览器检查桌面/窄屏的筛选、记录持久化、长文折叠及代表性知识页面。

## Verification notes

- 77 Vitest tests and 25 offline Python tests pass.
- Final production rebuild and 115-page checks pass. All 116 generated HTML pages audited: zero internal target/anchor issues; three MCP numeric-title anchors corrected.
- Browser: combined filters return expected results; completion/bookmarks persist after refresh and update the library/module totals; test markers removed. Cross-tab updates and cancellation also verified. Desktop and 390px layout checked, no horizontal overflow or console errors. Long-form details expand and source/review metadata render.
- MCP stdio integration verified with FastMCP 2.12.5 / MCP SDK 1.16.0; external model API, GPU and production OAuth remain explicitly untested.
