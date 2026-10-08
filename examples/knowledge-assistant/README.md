# 可运行的知识助手：RAG → 工具工作流 → MCP

这个案例使用虚构的「星河公共学习中心」资料，核心仅依赖 Python 标准库，无需 API key。中文字符二元组检索 + 原文摘录是透明基线，不是向量嵌入，也不调用 LLM。Agent 阶段是固定步骤、受工具预算约束的确定性工作流；MCP 阶段接入真实 stdio 协议，同样没有模型自主规划。

## 从仓库根目录运行

需要 Python 3.10+；本次实际验证 Python 3.12.12、3.14.3。核心无需安装包：

```bash
python3 examples/knowledge-assistant/main.py demo
```

一条命令依次展示：带引用的回答、无答案拒答、非法工具参数拒绝、20 个问题的离线评估。运行只读取内置语料，不写业务文件、不读取任意路径、不联网。参数错误返回退出码 2；`evaluate` 的失败案例作为评估数据输出，命令执行成功仍返回 0。

```bash
# 单个问题和完整 JSON trace
python3 examples/knowledge-assistant/main.py ask '借阅期限是多久？' --json

# 无答案场景
python3 examples/knowledge-assistant/main.py ask '借阅押金是多少？'

# 所有样本的候选、证据、回答、分数和错误分类
python3 examples/knowledge-assistant/main.py evaluate --json

# 标准库测试，不安装 MCP 依赖
python3 -m unittest discover -s examples/knowledge-assistant/tests -v
```

## 先看清楚各层做了什么

| 层 | 真正执行的能力 | 本例没有实现 |
| --- | --- | --- |
| RAG 的检索与证据边界 | 10 篇 Markdown → 20 个片段 → 词法排序 → 带片段 ID 的原文摘录 | embedding、向量数据库、reranker、LLM 生成 |
| Agent 的运行时边界 | 固定 search → 覆盖检查 → read_evidence → 摘录；预算与结构化 trace | 模型选择工具、动态计划、反思或真实智能体 |
| MCP 协议适配 | 真实独立子进程、tools/list、tools/call、schema 和错误返回 | HTTP/OAuth、多租户、远程部署、持久任务 |

`mcp_client.py` 是确定性 Host 演示脚本。它调用远端 `ask` 工具时，服务端运行同一个固定工作流。增加 MCP 没有改变答案生成方式，也没有自动赋予模型能力。

## 文件地图

```text
knowledge_assistant.py    标准库核心：切分、检索、工具边界、回答、评估
main.py                   demo / ask / evaluate 命令
mcp_server.py              可选 FastMCP 适配器
mcp_client.py              真实 stdio 客户端兼集成验收脚本
requirements-mcp.in        可选适配器直接依赖
requirements-mcp.txt       Python 3.12 解析的依赖固定版本清单
baseline-report.json      本次运行的配置、汇总和失败样本
data/documents/*.md       10 份虚构文档
data/queries.json         20 个问题：16 个有答案、4 个无答案
tests/                    核心与 CLI 的 unittest
```

全部内容是虚构教学资料，不涉及真实读者、凭证或机构政策。片段 ID 如 `loan#p001s001` 表示文档、段落和段内分片；它只在固定语料版本内稳定，改写段落后应重标 gold evidence。评估输出包含完整语料 SHA-256。

## 透明的检索与拒答规则

1. 按空行切分段落；超过 180 个字符时硬切，并记录段号和段内字符位置。此处计量的是字符，不是 token；长段可能切断条件，作为后续改进实验。
2. 去掉代码中列明的少量问句词，提取中文相邻双字词项与小写英数词。没有中文分词模型或语义向量。
3. 词项权重为 `1 + ln((1 + 片段数)/(1 + 含该词项的片段数))`。检索分数是命中的查询词项权重之和除以全部查询词项权重之和。该算法是简单的 IDF 加权词项覆盖，不是 BM25。
4. 取前 3 个有词项交集的候选，按顺序选择能新增查询词项覆盖的片段；相同词项不会重复增加覆盖率。
5. 选中片段的查询词项联合覆盖率低于 0.60 时拒答，否则通过只读工具重新读取片段，再原样摘录。这个阈值是教学启发式，不是答案正确概率或可靠的可回答性判定。

`search` 仅接受 1–500 字符的查询与 1–5 的整数 limit；`read_evidence` 仅接受已建立索引的片段 ID。未知工具、额外字段、布尔 limit、不存在的证据 ID 会被拒绝。失败尝试也消耗工具预算；读取不完整时不返回半份答案。文档中的指令只可能被当作引文显示，工作流不会将其执行；这不代表接入 LLM 后已解决提示注入。

## 如何读 trace 和评估

每次工具尝试记录 `step/tool/arguments/status/result/error`。检索记录保留分数、匹配词项和来源；`ask --json` 输出可用于逐步复盘。默认不保存日志，若重定向到文件，文件保存与访问控制由调用者负责。

| 指标 | 分母和解释 |
| --- | --- |
| Recall@3、MRR@3、NDCG@3 | 去重后的候选 ID；仅有答案查询做宏平均，空相关集为 null |
| evidence_recall | 最终引用覆盖 gold evidence 的比例；有答案却拒答记 0 |
| citation_precision | 最终引用中 gold evidence 的占比；没有引用时为 null，并报告有效样本数 |
| answerable_success_rate | 有答案查询中，状态为 answered、所有 gold evidence 被引用且必需事实字符串均出现的比例 |
| refusal_accuracy | 4 个无答案查询中成功拒答的比例；这是无答案桶指标，不是全量准确率 |
| task_success_rate | 有答案按上述摘录标准通过、无答案正确拒答的总体比例 |

这是开发回归集，与实现共同设计，不是独立测试集。字符串和证据集合检查只验证规定事实是否被摘录；不能评估模型推理、语言质量、真正的事实一致性或生产泛化。空样本/空指标分母返回 null，不伪造 0 分或满分。修改检索、切分或门槛前保存配置，后续应用应另建锁定的测试集。

## 本次实际运行结果

2026-10-08：25 个标准库与 CLI 测试通过；MCP 2.12.5 / MCP SDK 1.16.0 的真实 stdio 集成通过，包括工具发现、检索、证据读取、回答、拒答及 7 种被拒绝的调用。

| 评估项 | 实测结果 |
| --- | --- |
| 有答案 / 无答案 | 16 / 4 |
| Recall@3 / MRR@3 / NDCG@3 | 1.0000 / 1.0000 / 1.0000 |
| 最终证据召回率 | 0.8750 |
| 引用精确率 | 1.0000（14 个有引用样本） |
| 有答案任务成功率 | 0.8750 |
| 无答案拒答准确率 | 1.0000 |
| 总任务成功率 | 0.9000（18/20） |

两个失败是 `q11：无障碍入口在哪里？` 和 `q14：周一能去看书吗？`。正确证据已被检索到，但“在哪里”“能去看书”等词项没有得到足够覆盖，启发式门槛误拒答。保留这些失败有助于区分召回、证据选择和可回答性判断；不要只把阈值调到这 20 条全过。

## 可选：运行真实 MCP 适配器

固定 FastMCP **2.12.5** 是本案例的历史教学基线，不代表当前最新版本。本次通过 [该版本源码](https://github.com/jlowin/fastmcp/tree/v2.12.5) 和实际 stdio 调用核验；升级框架时重新生成依赖清单并重跑集成。传输原理见 [官方文档](https://gofastmcp.com/clients/transports)，在线文档可能对应更新的大版本。

安装依赖需要网络，启动后的示例通过本地管道通信，不调用模型或外部业务服务。使用 uv 时加 `--no-project`，避免修改仓库原有 Python 环境：

```bash
uv run --no-project --python 3.12 \
  --with-requirements examples/knowledge-assistant/requirements-mcp.txt \
  python examples/knowledge-assistant/mcp_client.py
```

不用 uv 时，可以创建案例专用环境：

```bash
python3.12 -m venv examples/knowledge-assistant/.venv-mcp
examples/knowledge-assistant/.venv-mcp/bin/python -m pip install -r examples/knowledge-assistant/requirements-mcp.txt
examples/knowledge-assistant/.venv-mcp/bin/python examples/knowledge-assistant/mcp_client.py
```

Windows 的虚拟环境解释器路径为 `.venv-mcp\Scripts\python.exe`。本次执行环境为 macOS ARM64，未声称跨平台验收。

客户端使用 `sys.executable` 启动固定的 `mcp_server.py`，没有可由问题或文档指定的命令、服务器地址或文件路径。服务暴露 `search`、`read_evidence`、`ask` 三个只读工具；每个工具用 `request` 对象承载严格 schema，拒绝额外字段和类型强制转换，例如：

```json
{"request": {"query": "借阅期限是多久？", "limit": 3}}
```

预期最后一行：

```text
MCP stdio check passed: discovery, search, read, answer, refusal, 7 rejected calls.
```

`readOnlyHint` 只是工具提示，真正边界来自代码白名单和固定数据来源。MCP schema 也不等同于授权。调用预算只约束单次 `ask` 工作流，独立 MCP 调用尚无会话级配额。此服务没有用户身份、ACL 或远程暴露配置，不能直接当作企业服务部署。

## 失败排查与下一步

| 现象 | 检查位置 / 下一步 |
| --- | --- |
| 有证据却拒答 | 看候选 `matched_terms` 和 coverage；先确认是检索失败还是门槛误拒答 |
| 改资料后 gold ID 报错 | 对照段落/分片位置重新标注，记录新的语料 hash |
| 回答缺少某个条件 | 看条件是否被切到别的片段，再查是否进入最终 evidence |
| `ModuleNotFoundError: fastmcp` | 仅 MCP 步骤需要依赖；用上面的固定环境命令运行 |
| MCP stderr 出现 ValidationError | 集成脚本会故意测试非法调用；最终 passed 且退出码为 0 才表示预期拒绝通过 |
| Authlib 弃用提示 | 固定旧版依赖的已知提示；不影响本次 stdio 检查，升级时需重新核验 |
| 直接运行 server 看似不动 | stdio server 在等客户端初始化；运行 `mcp_client.py`，不要向 server stdout 打印业务日志 |

后续可只替换检索器为 BM25/向量/混合检索，再对照相同证据标注；接入 LLM 时保留工具校验、预算和原文证据，并另评幻觉、注入和任务完成率。不要把本例覆盖率直接复用为模型置信度。

## 实现计划与完成记录

1. 先写切分、检索、引用/拒答和参数边界测试，观察失败，再实现最小行为。
2. 加入 10 份虚构文档、20 条标注问题，按固定语料版本的片段 ID 做离线评估。
3. 输出回答与 JSON trace，验证预算、未知工具和非法参数。
4. 先运行无工具的 MCP 适配器，观察集成失败，再加入真实 stdio 发现/调用能力。
5. 运行标准库测试与 MCP 集成，记录全部结果和失败案例。

以上步骤已完成；数据规模、检索门槛和摘录规则都服务于教学，不构成上线门槛。
