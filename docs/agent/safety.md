---
title: 安全与沙箱
description: Agent 安全机制 - 从风险识别到沙箱隔离
---

# 安全与沙箱

> 为AI智能体构建安全的"牢笼"

## 🎯 核心概念

### AI Agent的安全挑战

> 来源：[AI智能体的牢笼：大模型沙箱技术深度解析](https://dd-ff.blog.csdn.net/article/details/151970698)

::: danger 新型安全威胁
随着AI Agent获得**自主代码执行**能力，传统安全模型被打破：
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

## 🛡️ 防御策略

### 1. 输入验证与过滤

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
        tool_level = self.tool_permissions.get(tool_name, PermissionLevel.ADMIN)
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

| 技术 | 隔离级别 | 性能开销 | 安全性 | 适用场景 |
|------|----------|----------|--------|----------|
| **Docker** | 容器级 | 低 | 中 | 通用隔离 |
| **gVisor** | 内核级 | 中 | 高 | 高安全需求 |
| **Firecracker** | 微虚拟机 | 低 | 高 | 多租户/Serverless |
| **WebAssembly** | 字节码级 | 极低 | 中 | 轻量级隔离 |
| **nsjail** | 命名空间 | 低 | 中高 | 进程隔离 |

### Docker沙箱实现

```python
import docker
import tempfile
import os

class DockerSandbox:
    """Docker沙箱执行环境"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.image = "python:3.11-slim"
        self.timeout = 30
        self.memory_limit = "512m"
        self.cpu_limit = 1.0
    
    def execute_code(self, code: str) -> dict:
        """在沙箱中执行代码"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            code_path = f.name
        
        try:
            # 运行容器
            container = self.client.containers.run(
                self.image,
                command=f"python /code/script.py",
                volumes={
                    os.path.dirname(code_path): {
                        'bind': '/code', 
                        'mode': 'ro'  # 只读
                    }
                },
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=int(100000 * self.cpu_limit),
                network_disabled=True,  # 禁用网络
                read_only=True,         # 只读文件系统
                detach=True,
                remove=True
            )
            
            # 等待执行完成
            result = container.wait(timeout=self.timeout)
            logs = container.logs().decode('utf-8')
            
            return {
                "success": result["StatusCode"] == 0,
                "output": logs,
                "exit_code": result["StatusCode"]
            }
            
        except docker.errors.ContainerError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"执行失败: {e}"}
        finally:
            os.unlink(code_path)
```

### 安全执行器整合

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
            "args_hash": hash(str(args))  # 不记录原始参数
        })
```

---

## 🔐 Human-in-the-Loop

> 来源：[精通人机协同：使用LangGraph构建交互式智能体](https://dd-ff.blog.csdn.net/article/details/151149262)

### 高危操作需人工审批

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

## 🔗 相关阅读

- [Agent概述](/agent/) - Agent整体架构
- [工具调用](/agent/tool-calling) - 工具执行机制
- [多智能体](/agent/multi-agent) - 多Agent安全隔离

> **相关文章**：
> - [AI智能体的牢笼：大模型沙箱技术深度解析](https://dd-ff.blog.csdn.net/article/details/151970698)
> - [精通人机协同：LangGraph交互式智能体](https://dd-ff.blog.csdn.net/article/details/151149262)
> - [12-Factor Agent方法论](https://dd-ff.blog.csdn.net/article/details/154185674)

> **外部资源**：
> - [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
> - [Docker Security](https://docs.docker.com/engine/security/)
> - [gVisor文档](https://gvisor.dev/docs/)
