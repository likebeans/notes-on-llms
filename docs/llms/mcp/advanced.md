---
title: MCP高级功能
description: 中间件、认证、代理与生产级特性
---

# MCP高级功能

> 构建生产级MCP服务——FastMCP 2.0 进阶特性

## 🏗️ FastMCP 2.0 架构

FastMCP 2.0 超越基本协议实现，提供完整的 MCP 生态工具包：

```
┌─────────────────────────────────────────────────────────────┐
│                  FastMCP 2.0 功能矩阵                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Server 能力        Client 能力        高级特性              │
│  ├─ Tools           ├─ 多传输支持       ├─ 服务器组合        │
│  ├─ Resources       ├─ 工具调用         ├─ 代理模式         │
│  ├─ Prompts         ├─ 资源读取         ├─ 中间件系统       │
│  └─ Context         └─ 提示获取         └─ 认证授权         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔐 认证与授权

### Bearer Token认证

```python
from fastmcp import FastMCP
from fastmcp.server.auth import BearerAuthProvider

mcp = FastMCP("secure-service")

# 配置认证
auth_provider = BearerAuthProvider(
    # 方式1：静态公钥
    public_key_path="./keys/public.pem",
    
    # 方式2：JWKS端点
    # jwks_url="https://auth.example.com/.well-known/jwks.json"
)

mcp.auth = auth_provider

@mcp.tool()
async def admin_action(ctx, data: str) -> str:
    """需要管理员权限的操作"""
    # 获取用户信息
    user = ctx.get_user()
    
    if "admin" not in user.roles:
        raise PermissionError("需要管理员权限")
    
    return f"操作成功: {data}"
```

### 自定义认证

```python
from fastmcp.server.auth import AuthProvider

class CustomAuthProvider(AuthProvider):
    """自定义认证提供者"""
    
    async def authenticate(self, token: str) -> dict:
        # 验证token
        user_info = await self.verify_token(token)
        return {
            "user_id": user_info["sub"],
            "roles": user_info.get("roles", []),
            "permissions": user_info.get("permissions", [])
        }
    
    async def authorize(self, user: dict, resource: str, action: str) -> bool:
        # 检查权限
        required_permission = f"{resource}:{action}"
        return required_permission in user.get("permissions", [])
```

---

## 🔄 中间件系统

### 中间件概念

中间件在请求处理前后执行自定义逻辑。

```python
from fastmcp.server.middleware import Middleware

class LoggingMiddleware(Middleware):
    """日志记录中间件"""
    
    async def process_request(self, request, context):
        context["start_time"] = time.time()
        print(f"收到请求: {request.method} {request.params}")
        return request
    
    async def process_response(self, response, context):
        duration = time.time() - context["start_time"]
        print(f"响应完成: {duration:.3f}s")
        return response

# 注册中间件
mcp.add_middleware(LoggingMiddleware())
```

### 常用中间件

```python
# 速率限制中间件
class RateLimitMiddleware(Middleware):
    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
    
    async def process_request(self, request, context):
        client_id = context.get("client_id", "default")
        now = time.time()
        
        # 清理过期记录
        self.requests[client_id] = [
            t for t in self.requests.get(client_id, [])
            if now - t < self.window
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            raise Exception("请求过于频繁，请稍后重试")
        
        self.requests[client_id].append(now)
        return request

# 错误处理中间件
class ErrorHandlerMiddleware(Middleware):
    async def process_request(self, request, context):
        try:
            return request
        except Exception as e:
            return {"error": str(e), "code": 500}
```

---

## 🔀 服务器组合

FastMCP 支持两种服务器组合方式：**静态导入**和**动态挂载**。

### 静态组合（import_server）

一次性导入，组件立即注册到主服务器：

```python
from fastmcp import FastMCP

# 子服务器
math_server = FastMCP("math")

@math_server.tool()
def add(a: int, b: int) -> int:
    return a + b

# 主服务器
main_server = FastMCP("main")

# 导入子服务器的所有组件
main_server.import_server(math_server, prefix="math")

# 工具通过 "math_add" 调用（注意是下划线）
```

### 动态组合（mount）

建立"活链接"，请求时实时转发：

```python
# 动态挂载（支持运行时更新）
main_server.mount(math_server, prefix="math")

# 挂载远程服务器
main_server.mount("http://remote-server:8000", prefix="remote")

# 不带前缀挂载
main_server.mount(api_server)
```

### 直接挂载 vs 代理挂载

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| **直接挂载** | 内存中直接访问，不执行子服务器lifespan | 无自定义lifespan时 |
| **代理挂载** | 通过Client接口通信，执行完整生命周期 | 有自定义lifespan时 |

```python
# 强制代理模式
main_server.mount(api_server, prefix="api", as_proxy=True)
```

::: warning 性能注意
动态挂载的 `list_tools()` 会受最慢子服务器影响。HTTP代理服务器可能引入300-400ms延迟。对性能敏感时考虑使用 `import_server()`。
:::

---

## 🌐 代理服务器

### 创建代理

```python
from fastmcp import FastMCP

# 创建代理，转发到后端服务
proxy = FastMCP.as_proxy(
    "backend-proxy",
    target="http://backend-service:8000"
)

# 添加认证中间件
@proxy.middleware
async def auth_middleware(request, next):
    # 验证请求
    if not validate_token(request.headers.get("Authorization")):
        raise Exception("未授权")
    return await next(request)

proxy.run()
```

### 协议桥接

```python
# stdio到HTTP桥接
mcp = FastMCP("bridge")

# stdio客户端访问
# -> 代理服务器
# -> HTTP后端服务
```

---

## 📡 传输协议

### stdio传输

```python
# 服务器端
mcp.run()  # 默认stdio

# 或显式指定
mcp.run(transport="stdio")
```

### HTTP/SSE传输

```python
# 服务器端
mcp.run(transport="sse", host="0.0.0.0", port=8000)

# 客户端连接
async with Client("http://localhost:8000/sse") as client:
    result = await client.call_tool("my_tool", {})
```

### 与ASGI框架集成

```python
from fastapi import FastAPI
from fastmcp import FastMCP

app = FastAPI()
mcp = FastMCP("api-integrated")

@mcp.tool()
def my_tool() -> str:
    return "Hello from MCP!"

# 挂载MCP到FastAPI
app.mount("/mcp", mcp.http_app())

# 其他FastAPI路由
@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## 📊 上下文与状态

### Context对象

```python
@mcp.tool()
async def context_aware_tool(ctx) -> dict:
    """使用上下文的工具"""
    
    # 记录日志
    await ctx.log("info", "工具被调用")
    
    # 报告进度
    await ctx.report_progress(0.5, "处理中...")
    
    # 获取用户信息（如果有认证）
    user = ctx.get_user()
    
    # 访问资源
    config = await ctx.read_resource("config://settings")
    
    return {"user": user, "config": config}
```

### 进度报告

```python
@mcp.tool()
async def long_running_task(ctx, items: list) -> list:
    """长时间运行的任务"""
    results = []
    total = len(items)
    
    for i, item in enumerate(items):
        # 处理项目
        result = await process_item(item)
        results.append(result)
        
        # 报告进度
        progress = (i + 1) / total
        await ctx.report_progress(
            progress, 
            f"已处理 {i+1}/{total} 项"
        )
    
    return results
```

---

## 🛡️ 错误处理

### 标准错误类型

```python
from fastmcp.exceptions import (
    ToolError,
    ResourceNotFoundError,
    ValidationError,
    AuthorizationError
)

@mcp.tool()
async def safe_tool(data: str) -> str:
    if not data:
        raise ValidationError("数据不能为空")
    
    try:
        result = await process(data)
    except ExternalServiceError as e:
        raise ToolError(f"外部服务错误: {e}")
    
    return result
```

### 优雅降级

```python
@mcp.tool()
async def resilient_tool(query: str) -> str:
    """带降级的工具"""
    
    # 尝试主服务
    try:
        return await primary_service.query(query)
    except Exception:
        pass
    
    # 降级到备用服务
    try:
        return await backup_service.query(query)
    except Exception:
        pass
    
    # 最终降级
    return "服务暂时不可用，请稍后重试"
```

---

## 📈 监控与可观测性

### 指标收集

```python
from prometheus_client import Counter, Histogram

tool_calls = Counter('mcp_tool_calls_total', 'Tool calls', ['tool_name'])
tool_duration = Histogram('mcp_tool_duration_seconds', 'Tool duration')

class MetricsMiddleware(Middleware):
    async def process_request(self, request, context):
        context["start_time"] = time.time()
        return request
    
    async def process_response(self, response, context):
        duration = time.time() - context["start_time"]
        tool_name = context.get("tool_name", "unknown")
        
        tool_calls.labels(tool_name=tool_name).inc()
        tool_duration.observe(duration)
        
        return response
```

### 分布式追踪

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@mcp.tool()
async def traced_tool(data: str) -> str:
    with tracer.start_as_current_span("traced_tool") as span:
        span.set_attribute("input.length", len(data))
        
        result = await process(data)
        
        span.set_attribute("output.length", len(result))
        return result
```

---

## 🚀 生产部署

### 部署检查清单

| 检查项 | 说明 |
|--------|------|
| **认证** | 配置 Bearer Token 或 OAuth 2.1 |
| **速率限制** | 防止滥用，保护后端服务 |
| **错误处理** | 优雅降级，避免级联失败 |
| **监控** | Prometheus 指标、分布式追踪 |
| **日志** | 结构化日志，便于问题排查 |

### Docker 部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .

# 使用 SSE 传输
CMD ["python", "server.py", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=${API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

---

## 🔍 调试与检查

### fastmcp inspect

检查服务器组件：

```bash
# 文本摘要
fastmcp inspect server.py

# FastMCP JSON格式
fastmcp inspect server.py --format fastmcp

# MCP协议格式
fastmcp inspect server.py --format mcp

# 保存到文件
fastmcp inspect server.py --format fastmcp -o manifest.json
```

### MCP Inspector

使用官方Inspector工具：

```bash
npx @modelcontextprotocol/inspector python server.py
```

---

## 🔗 相关阅读

- [MCP快速入门](/llms/mcp/quickstart) - 5分钟创建服务
- [核心概念](/llms/mcp/concepts) - Tools/Resources/Prompts
- [实战项目](/llms/mcp/practice) - 完整可运行示例
- [MCP概述](/llms/mcp/) - 协议全貌

> **外部资源**：
> - [MCP官方文档](https://modelcontextprotocol.io/)
> - [FastMCP 2.0 文档](https://gofastmcp.com/)
> - [FastMCP GitHub](https://github.com/jlowin/fastmcp)
