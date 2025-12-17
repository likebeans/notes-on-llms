---
title: Agent 评估方法
description: Agent 性能评估 - 从指标设计到评估框架
---

# Agent 评估方法

> 科学评估AI智能体的能力与可靠性

## 🎯 核心概念

### 为什么需要Agent评估？

::: tip 评估的价值
Agent系统比传统LLM应用更复杂，需要评估的不仅是回答质量，还包括：
- **任务完成能力**：能否达成目标
- **工具使用效率**：调用是否合理
- **推理过程质量**：思考链是否正确
- **安全性**：是否产生有害行为
:::

### Agent评估的挑战

| 挑战 | 描述 |
|------|------|
| **开放式任务** | 没有唯一正确答案 |
| **多步骤过程** | 需要评估中间步骤 |
| **工具交互** | 外部API调用难以复现 |
| **非确定性** | 相同输入可能产生不同输出 |
| **成本高昂** | 完整评估需要大量API调用 |

---

## 📊 评估维度

### 四维评估框架

| 维度 | 评估内容 | 关键指标 |
|------|----------|----------|
| **任务完成** | 最终结果是否正确 | 成功率、准确率 |
| **效率** | 资源消耗是否合理 | 步骤数、Token用量、时间 |
| **过程质量** | 推理和工具使用 | 工具选择准确率、推理正确率 |
| **安全性** | 是否产生有害行为 | 越界率、错误调用率 |

### 任务完成度评估

```python
class TaskCompletionEvaluator:
    """任务完成度评估器"""
    
    def evaluate(self, task: str, result: dict, ground_truth: dict) -> dict:
        """评估任务完成情况"""
        scores = {}
        
        # 1. 目标达成度（0-1）
        scores["goal_achievement"] = self._check_goal(result, ground_truth)
        
        # 2. 答案正确性
        scores["answer_correctness"] = self._check_answer(
            result.get("answer"), 
            ground_truth.get("answer")
        )
        
        # 3. 完整性
        scores["completeness"] = self._check_completeness(
            result.get("output"),
            ground_truth.get("required_elements", [])
        )
        
        # 综合评分
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _check_goal(self, result: dict, ground_truth: dict) -> float:
        """检查目标是否达成"""
        if result.get("status") == "completed":
            # 使用LLM判断结果是否满足目标
            prompt = f"""判断以下结果是否完成了任务目标：
            
目标：{ground_truth.get('goal')}
结果：{result.get('output')}

输出0到1之间的分数，1表示完全完成。"""
            score = float(llm.generate(prompt))
            return min(max(score, 0), 1)
        return 0.0
```

### 效率评估

```python
class EfficiencyEvaluator:
    """效率评估器"""
    
    def evaluate(self, execution_trace: list) -> dict:
        """评估执行效率"""
        return {
            # 步骤效率
            "step_count": len(execution_trace),
            "redundant_steps": self._count_redundant_steps(execution_trace),
            
            # Token效率
            "total_tokens": sum(step.get("tokens", 0) for step in execution_trace),
            "tokens_per_step": self._avg_tokens_per_step(execution_trace),
            
            # 时间效率
            "total_time": sum(step.get("duration", 0) for step in execution_trace),
            "tool_call_time": self._sum_tool_time(execution_trace),
            
            # 工具效率
            "tool_calls": len([s for s in execution_trace if s.get("type") == "tool"]),
            "failed_tool_calls": len([
                s for s in execution_trace 
                if s.get("type") == "tool" and not s.get("success")
            ]),
        }
    
    def _count_redundant_steps(self, trace: list) -> int:
        """统计冗余步骤（重复的工具调用）"""
        seen = set()
        redundant = 0
        for step in trace:
            if step.get("type") == "tool":
                key = (step.get("tool"), str(step.get("args")))
                if key in seen:
                    redundant += 1
                seen.add(key)
        return redundant
```

### 过程质量评估

```python
class ProcessQualityEvaluator:
    """过程质量评估器"""
    
    def evaluate(self, execution_trace: list, task: str) -> dict:
        """评估推理和工具使用质量"""
        scores = {}
        
        # 1. 推理质量
        thoughts = [s for s in execution_trace if s.get("type") == "thought"]
        scores["reasoning_quality"] = self._evaluate_reasoning(thoughts, task)
        
        # 2. 工具选择准确性
        tool_calls = [s for s in execution_trace if s.get("type") == "tool"]
        scores["tool_selection"] = self._evaluate_tool_selection(tool_calls, task)
        
        # 3. 错误恢复能力
        scores["error_recovery"] = self._evaluate_error_recovery(execution_trace)
        
        return scores
    
    def _evaluate_reasoning(self, thoughts: list, task: str) -> float:
        """评估推理质量"""
        if not thoughts:
            return 0.0
        
        prompt = f"""评估以下推理过程的质量（0-1分）：

任务：{task}

推理过程：
{chr(10).join([t.get('content', '') for t in thoughts])}

评估标准：
- 逻辑清晰性
- 与任务相关性
- 推理步骤合理性

输出分数："""
        return float(llm.generate(prompt))
    
    def _evaluate_tool_selection(self, tool_calls: list, task: str) -> float:
        """评估工具选择是否合理"""
        if not tool_calls:
            return 1.0  # 没有工具调用不扣分
        
        correct = 0
        for call in tool_calls:
            if self._is_appropriate_tool(call, task):
                correct += 1
        
        return correct / len(tool_calls)
```

---

## 🧪 评估基准（Benchmarks）

### 常用Agent Benchmarks

| Benchmark | 评估内容 | 任务类型 |
|-----------|----------|----------|
| **AgentBench** | 通用Agent能力 | 操作系统、数据库、网页等 |
| **WebArena** | 网页操作能力 | 电商、社交、地图等网站 |
| **SWE-bench** | 代码修复能力 | GitHub Issue修复 |
| **GAIA** | 通用助手能力 | 问答、文件处理、网络搜索 |
| **ToolBench** | 工具使用能力 | API调用、工具组合 |

### 自定义评估集构建

```python
class EvaluationDataset:
    """评估数据集"""
    
    def __init__(self):
        self.test_cases = []
    
    def add_case(
        self,
        task: str,
        expected_output: str,
        required_tools: list = None,
        max_steps: int = 10,
        difficulty: str = "medium"
    ):
        """添加测试用例"""
        self.test_cases.append({
            "task": task,
            "expected_output": expected_output,
            "required_tools": required_tools or [],
            "max_steps": max_steps,
            "difficulty": difficulty,
            "metadata": {
                "created_at": time.time(),
                "category": self._categorize(task)
            }
        })
    
    def run_evaluation(self, agent, evaluator) -> dict:
        """运行评估"""
        results = []
        
        for case in self.test_cases:
            # 执行Agent
            trace = agent.run(case["task"])
            
            # 评估结果
            score = evaluator.evaluate(
                task=case["task"],
                result=trace[-1] if trace else {},
                ground_truth={"answer": case["expected_output"]}
            )
            
            results.append({
                "case": case,
                "trace": trace,
                "score": score
            })
        
        # 汇总统计
        return {
            "cases": results,
            "summary": self._summarize(results)
        }
    
    def _summarize(self, results: list) -> dict:
        """汇总评估结果"""
        scores = [r["score"]["overall"] for r in results]
        return {
            "total_cases": len(results),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "success_rate": len([s for s in scores if s > 0.8]) / len(scores),
            "by_difficulty": self._group_by_difficulty(results)
        }
```

---

## 🔄 在线评估与监控

### 生产环境监控指标

```python
class AgentMonitor:
    """Agent生产监控"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record_execution(self, execution_data: dict):
        """记录执行数据"""
        # 延迟指标
        self.metrics["latency"].append(execution_data["duration"])
        
        # 成功率
        self.metrics["success"].append(1 if execution_data["success"] else 0)
        
        # Token消耗
        self.metrics["tokens"].append(execution_data["tokens"])
        
        # 工具调用
        self.metrics["tool_calls"].append(len(execution_data["tools_used"]))
        
        # 错误追踪
        if not execution_data["success"]:
            self.metrics["errors"].append({
                "type": execution_data.get("error_type"),
                "message": execution_data.get("error_message"),
                "timestamp": time.time()
            })
    
    def get_dashboard_data(self) -> dict:
        """获取仪表盘数据"""
        return {
            "avg_latency": np.mean(self.metrics["latency"][-100:]),
            "p95_latency": np.percentile(self.metrics["latency"][-100:], 95),
            "success_rate": np.mean(self.metrics["success"][-100:]),
            "avg_tokens": np.mean(self.metrics["tokens"][-100:]),
            "error_rate": 1 - np.mean(self.metrics["success"][-100:]),
            "recent_errors": self.metrics["errors"][-10:]
        }
```

### A/B测试框架

```python
class AgentABTest:
    """Agent A/B测试"""
    
    def __init__(self, agent_a, agent_b, split_ratio: float = 0.5):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.split_ratio = split_ratio
        self.results = {"a": [], "b": []}
    
    def run(self, task: str) -> dict:
        """执行A/B测试"""
        import random
        
        # 随机分配
        use_a = random.random() < self.split_ratio
        agent = self.agent_a if use_a else self.agent_b
        group = "a" if use_a else "b"
        
        # 执行
        start = time.time()
        result = agent.run(task)
        duration = time.time() - start
        
        # 记录
        self.results[group].append({
            "task": task,
            "result": result,
            "duration": duration,
            "success": result.get("success", False)
        })
        
        return {"group": group, "result": result}
    
    def get_comparison(self) -> dict:
        """获取对比结果"""
        def stats(results):
            if not results:
                return {}
            return {
                "count": len(results),
                "success_rate": sum(1 for r in results if r["success"]) / len(results),
                "avg_duration": sum(r["duration"] for r in results) / len(results)
            }
        
        return {
            "agent_a": stats(self.results["a"]),
            "agent_b": stats(self.results["b"]),
            "winner": self._determine_winner()
        }
```

---

## 📈 评估报告生成

```python
class EvaluationReporter:
    """评估报告生成器"""
    
    def generate_report(self, evaluation_results: dict) -> str:
        """生成Markdown格式报告"""
        report = f"""# Agent 评估报告

## 概述
- 评估时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
- 测试用例数：{evaluation_results['summary']['total_cases']}
- 平均得分：{evaluation_results['summary']['avg_score']:.2%}
- 成功率：{evaluation_results['summary']['success_rate']:.2%}

## 详细指标

### 任务完成度
| 指标 | 数值 |
|------|------|
| 目标达成率 | {self._get_metric('goal_achievement'):.2%} |
| 答案正确率 | {self._get_metric('answer_correctness'):.2%} |

### 效率指标
| 指标 | 数值 |
|------|------|
| 平均步骤数 | {self._get_metric('step_count'):.1f} |
| 平均Token消耗 | {self._get_metric('total_tokens'):.0f} |
| 工具调用成功率 | {self._get_metric('tool_success_rate'):.2%} |

### 按难度分布
{self._difficulty_table(evaluation_results)}

## 失败案例分析
{self._failure_analysis(evaluation_results)}
"""
        return report
```

---

## 🔗 相关阅读

- [Agent概述](/agent/) - Agent整体架构
- [安全与沙箱](/agent/safety) - 安全性评估
- [RAG评估方法](/rag/evaluation) - 检索生成评估

> **相关文章**：
> - [12-Factor Agent方法论](https://dd-ff.blog.csdn.net/article/details/154185674)
> - [检索增强生成（RAG）系统综合评估](https://dd-ff.blog.csdn.net/article/details/152823514)

> **外部资源**：
> - [AgentBench](https://github.com/THUDM/AgentBench)
> - [WebArena](https://webarena.dev/)
> - [SWE-bench](https://www.swebench.com/)
