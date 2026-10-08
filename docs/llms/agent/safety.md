---
title: 安全与沙箱
description: Agent 安全机制 - 从风险识别到沙箱隔离
pageType: article
module: agent
updated: '2026-10-08'
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

# 安全与沙箱

> 为AI智能体构建安全的"牢笼"

## 🎯 核心概念

### AI Agent的安全挑战

> 来源：[AI智能体的牢笼：大模型沙箱技术深度解析](https://dd-ff.blog.csdn.net/article/details/151970698)

::: danger 新型安全威胁
随着AI Agent获得**自主代码执行**能力，既有安全边界需要扩展到模型生成的动作：
- 数据与代码界限模糊化
- 提示注入成为新攻击向量
- Agent可能被"越狱"执行恶意操作
:::

### 安全风险分类

| 风险类型 | 描述 | 潜在后果 |
|----------|------|----------|
| **提示注入** | 恶意输入劫持Agent行为 | 执行未授权操作 |
| **数据泄露** | 敏感信息被暴露 | 隐私/商业机密泄露 |
| **资源滥用** | 无限循环消耗资源 | 服务不可用、成本失控 |
| **系统破坏** | 恶意代码执行 | 数据损坏、系统被控制 |
| **权限提升** | 突破预设边界 | 获取更高权限 |

---

## 📖 Agentic Design Patterns 视角

> 来源：[Agentic Design Patterns - Guardrails/Safety Patterns](https://github.com/ginobefun/agentic-design-patterns-cn)

### 护栏模式概述

护栏（Guardrails）用于检测或限制风险。模型分类器和文本过滤器不是强制安全边界，授权、资源隔离和审计必须由模型之外的系统执行。

| 护栏类型 | 作用 |
|----------|------|
| **输入护栏** | 过滤恶意输入、验证请求合法性 |
| **输出护栏** | 检查响应质量、防止有害内容 |
| **执行护栏** | 限制工具调用、控制资源使用 |
| **行为护栏** | 监控智能体轨迹、检测异常行为 |

### 多层防护体系

```
用户输入 → 输入验证 → 意图分析 → 执行限制 → 输出审查 → 响应
              │           │           │           │
              ▼           ▼           ▼           ▼
           拦截恶意     防止越权     资源控制     过滤有害
```

### 与其他模式的关系

| 模式 | 安全关联 |
|------|----------|
| [人机协同](/llms/agent/human-in-the-loop) | 高危操作需人工审批 |
| [异常处理](/llms/agent/exception-handling) | 优雅降级和恢复 |
| [评估监控](/llms/agent/evaluation-monitoring) | 检测异常行为 |

---

## 🛡️ 防御策略

### 1. 输入验证与过滤

下面的正则仅是易解释的检测示例，不能可靠防止提示注入：正常技术文章也会包含 `system prompt`，恶意输入则能换语言、编码或转移到工具结果中。不要把“没有命中黑名单”当作安全证明，也不要未经说明修改用户数据。

应将检索文档、网页、附件与工具输出作为不可信数据处理，保留来源；在执行动作时按当前用户、目标资源和授权范围重新检查。发现可疑文本时可以拒绝特定动作或要求补充证据，而不是认为过滤掉几个词后就可放行。


```python
import re
from typing import Optional

class InputValidator:
    """输入验证器"""
    
    # 危险模式黑名单
    DANGEROUS_PATTERNS = [
        r"ignore previous instructions",
        r"忽略之前的指令",
        r"system prompt",
        r"<script>",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"os\.system",
        r"subprocess",
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]
    
    def validate(self, user_input: str) -> tuple[bool, Optional[str]]:
        """验证用户输入"""
        # 1. 长度检查
        if len(user_input) > 10000:
            return False, "输入过长"
        
        # 2. 危险模式检测
        for pattern in self.patterns:
            if pattern.search(user_input):
                return False, f"检测到潜在危险内容"
        
        # 3. 编码检查
        try:
            user_input.encode('utf-8')
        except UnicodeError:
            return False, "无效编码"
        
        return True, None
    
    def sanitize(self, user_input: str) -> str:
        """清理用户输入"""
        # 移除控制字符
        cleaned = ''.join(c for c in user_input if c.isprintable() or c in '\n\t')
        # 限制长度
        return cleaned[:10000]
```

### 2. 工具权限控制

该示例只演示工具级允许列表。真实授权还应验证资源归属、读写范围和参数约束；拥有 `read_file` 权限不代表能读取所有路径，较高等级也不应自动获得所有业务权限。

```python
from enum import Enum
from typing import Set

class PermissionLevel(Enum):
    READ_ONLY = 1      # 只读
    READ_WRITE = 2     # 读写
    EXECUTE = 3        # 执行
    ADMIN = 4          # 管理员

class ToolPermissionManager:
    """工具权限管理器"""
    
    def __init__(self):
        self.tool_permissions = {
            "web_search": PermissionLevel.READ_ONLY,
            "read_file": PermissionLevel.READ_ONLY,
            "write_file": PermissionLevel.READ_WRITE,
            "execute_code": PermissionLevel.EXECUTE,
            "delete_file": PermissionLevel.ADMIN,
        }
        self.user_permissions = {}
    
    def set_user_permission(self, user_id: str, level: PermissionLevel):
        """设置用户权限级别"""
        self.user_permissions[user_id] = level
    
    def can_use_tool(self, user_id: str, tool_name: str) -> bool:
        """检查用户是否可以使用工具"""
        user_level = self.user_permissions.get(user_id, PermissionLevel.READ_ONLY)
        tool_level = self.tool_permissions.get(tool_name)
        if tool_level is None:
            return False  # 未注册工具默认拒绝，包括管理员
        return user_level.value >= tool_level.value
    
    def get_allowed_tools(self, user_id: str) -> Set[str]:
        """获取用户可用的工具列表"""
        user_level = self.user_permissions.get(user_id, PermissionLevel.READ_ONLY)
        return {
            tool for tool, level in self.tool_permissions.items()
            if user_level.value >= level.value
        }
```

### 3. 速率限制与资源控制

以下为单进程教学实现，不提供多线程原子性或跨实例配额。生产环境需要共享计数与原子更新；记录 `max_memory_mb` 或事后检查耗时本身不会限制进程资源，必须交给运行环境强制执行。

```python
import time
from collections import defaultdict

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, max_calls: int = 100, period: int = 60):
        self.max_calls = max_calls
        self.period = period  # 秒
        self.calls = defaultdict(list)
    
    def allow(self, user_id: str) -> bool:
        """检查是否允许调用"""
        now = time.time()
        # 清理过期记录
        self.calls[user_id] = [
            t for t in self.calls[user_id] 
            if now - t < self.period
        ]
        # 检查是否超限
        if len(self.calls[user_id]) >= self.max_calls:
            return False
        # 记录本次调用
        self.calls[user_id].append(now)
        return True

class ResourceLimiter:
    """资源限制器"""
    
    def __init__(self):
        self.limits = {
            "max_execution_time": 30,      # 秒
            "max_memory_mb": 512,          # MB
            "max_output_size": 1_000_000,  # 字符
            "max_iterations": 50,          # 最大迭代次数
        }
    
    def check_execution_time(self, start_time: float) -> bool:
        return time.time() - start_time < self.limits["max_execution_time"]
    
    def check_output_size(self, output: str) -> bool:
        return len(output) < self.limits["max_output_size"]
```

---

## 🔒 沙箱技术

### 沙箱方案对比

| 技术 | 边界机制 | 选型时核对 |
| --- | --- | --- |
| Docker | 命名空间、cgroups、能力与系统调用策略；通常共享宿主内核 | 挂载、用户权限、daemon 访问、内核与运行时补丁 |
| gVisor | 用户态应用内核拦截并实现系统调用接口 | 系统调用兼容性、I/O 开销与运行时配置 |
| Firecracker | microVM 隔离 | 镜像生命周期、宿主配置与设备暴露 |
| WebAssembly | 运行时内存边界与受控宿主导入 | 哪些文件、网络和其他宿主能力被授予模块 |
| nsjail | 进程级命名空间、资源和 seccomp 策略 | 策略完整性与宿主内核暴露面 |

不存在脱离威胁模型的“安全性高/中”排行榜。容器的默认边界与虚拟机不同，gVisor 也不是给每个任务分配独立 Linux 内核。[Docker 安全文档](https://docs.docker.com/engine/security/)、[gVisor 架构说明](https://gvisor.dev/docs/)

### Docker沙箱实现

下面是受限演示：只挂载当前任务的脚本目录，固定容器内文件名，关闭网络，限制 CPU、内存和进程数，并在退出路径移除容器。需要本地 Docker daemon 与 `docker` Python SDK；本页未运行容器集成测试。生产部署应固定经过审查的镜像摘要、独立宿主与日志配额，不能将本例当作不可信多租户执行平台。

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import docker


class DockerSandbox:
    def __init__(self, image="python:3.11-slim", timeout=30):
        self.client = docker.from_env()
        self.image = image
        self.timeout = timeout

    def execute_code(self, code):
        if len(code.encode("utf-8")) > 100_000:
            raise ValueError("代码超过演示允许大小")
        with TemporaryDirectory() as directory:
            script = Path(directory) / "script.py"
            script.write_text(code, encoding="utf-8")
            script.chmod(0o644)
            Path(directory).chmod(0o755)
            container = None
            try:
                container = self.client.containers.run(
                    self.image, command=["python", "-I", "/code/script.py"],
                    volumes={directory: {"bind": "/code", "mode": "ro"}},
                    user="65534:65534", network_disabled=True, read_only=True,
                    cap_drop=["ALL"], security_opt=["no-new-privileges:true"],
                    mem_limit="512m", memswap_limit="512m",
                    nano_cpus=1_000_000_000, pids_limit=64,
                    tmpfs={"/tmp": "rw,noexec,nosuid,size=64m"},
                    detach=True, auto_remove=False,
                )
                result = container.wait(timeout=self.timeout)
                output = container.logs(tail=100).decode("utf-8", errors="replace")
                return {"success": result["StatusCode"] == 0,
                        "exit_code": result["StatusCode"], "output": output[:10000]}
            finally:
                if container is not None:
                    container.remove(force=True)
```

`wait(timeout=...)` 的超时只结束客户端等待，因此必须处理仍在运行的容器。示例在 `finally` 清理；若 daemon 失联，清理也可能失败，调度器还需按任务标签进行补偿清理。日志末尾行数与字符串截断只是展示限制，不能防止容器产生大量日志，实际系统需限制日志存储与读取字节数。[Docker SDK 容器接口](https://docker-py.readthedocs.io/en/stable/containers.html)

### 安全执行器整合

下面保留职责组合的伪代码，`_error`、`_execute_tool` 等由应用实现。参数校验应按工具 schema 分别编写，不能对所有代码和字符串一律套同一个关键词黑名单。拒绝、异常和超时也要写审计事件，不能只记录成功路径。

```python
class SafeToolExecutor:
    """安全的工具执行器"""
    
    def __init__(self):
        self.validator = InputValidator()
        self.permission_manager = ToolPermissionManager()
        self.rate_limiter = RateLimiter()
        self.resource_limiter = ResourceLimiter()
        self.sandbox = DockerSandbox()
        self.audit_log = []
    
    def execute(
        self, 
        user_id: str, 
        tool_name: str, 
        arguments: dict
    ) -> dict:
        """安全执行工具"""
        start_time = time.time()
        
        # 1. 速率限制
        if not self.rate_limiter.allow(user_id):
            return self._error("速率超限，请稍后重试")
        
        # 2. 权限检查
        if not self.permission_manager.can_use_tool(user_id, tool_name):
            return self._error(f"无权使用工具: {tool_name}")
        
        # 3. 输入验证
        for key, value in arguments.items():
            if isinstance(value, str):
                valid, msg = self.validator.validate(value)
                if not valid:
                    return self._error(f"参数验证失败: {msg}")
        
        # 4. 执行（高危操作使用沙箱）
        if tool_name == "execute_code":
            result = self.sandbox.execute_code(arguments.get("code", ""))
        else:
            result = self._execute_tool(tool_name, arguments)
        
        # 5. 输出检查
        if not self.resource_limiter.check_output_size(str(result)):
            result = {"output": str(result)[:10000] + "...[截断]"}
        
        # 6. 审计日志
        self._log_execution(user_id, tool_name, arguments, result, start_time)
        
        return result
    
    def _log_execution(self, user_id, tool_name, args, result, start_time):
        """记录审计日志"""
        self.audit_log.append({
            "timestamp": time.time(),
            "user_id": user_id,
            "tool": tool_name,
            "duration": time.time() - start_time,
            "success": result.get("success", True),
            "args_summary": {"keys": sorted(args)}  # 仅演示字段级摘要，非完整审计
        })
```

---

## 🔐 Human-in-the-Loop

> 来源：[精通人机协同：使用LangGraph构建交互式智能体](https://dd-ff.blog.csdn.net/article/details/151149262)

### 高危操作需人工审批

是否审批由既有授权、影响范围和可逆性决定，不只看工具名称。以下为节点片段，需配置持久化 checkpointer、稳定 `thread_id`，并实现幂等的 `perform_action`。恢复可能重跑节点，审批必须绑定未变更的具体提案；详见[人机协同](/llms/agent/human-in-the-loop)。

```python
from langgraph.types import interrupt

def execute_action(state):
    """执行操作前检查是否需要人工审批"""
    action = state["pending_action"]
    
    # 高危操作列表
    HIGH_RISK_ACTIONS = ["delete_file", "send_email", "execute_code", "make_payment"]
    
    if action["type"] in HIGH_RISK_ACTIONS:
        # 中断执行，等待人工审批
        approval = interrupt({
            "action": action,
            "message": f"即将执行高危操作: {action['type']}，是否批准？",
            "details": action["params"]
        })
        
        if not approval.get("approved"):
            return {"status": "rejected", "reason": approval.get("reason")}
    
    # 执行操作
    result = perform_action(action)
    return {"status": "completed", "result": result}
```

---

## 用越界用例验收边界

建立允许与拒绝的成对样本：读取当前任务文件应成功，读取其他租户文件应拒绝；正常联网工具访问允许域名，任意内网地址不能借模型参数绕过；工具返回“忽略规则”不能扩大权限。再演练无限循环、大量输出、子进程生成、取消与服务重启。

验收看执行层证据：实际访问了哪些资源、容器是否残留、费用是否被限制、拒绝事件能否追踪到 run。模型口头承诺遵守规则不能替代这些检查。对风险类别的整理可参考 [OWASP LLM 应用安全项目](https://owasp.org/www-project-top-10-for-large-language-model-applications/)。

## 🔗 相关阅读

- [Agent概述](/llms/agent/) - Agent整体架构
- [工具调用](/llms/agent/tool-calling) - 工具执行机制
- [多智能体](/llms/agent/multi-agent) - 多Agent安全隔离

> **相关文章**：
> - [AI智能体的牢笼：大模型沙箱技术深度解析](https://dd-ff.blog.csdn.net/article/details/151970698)
> - [精通人机协同：LangGraph交互式智能体](https://dd-ff.blog.csdn.net/article/details/151149262)
> - [12-Factor Agent方法论](https://dd-ff.blog.csdn.net/article/details/154185674)

> **外部资源**：
> - [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
> - [Docker Security](https://docs.docker.com/engine/security/)
> - [gVisor文档](https://gvisor.dev/docs/)
