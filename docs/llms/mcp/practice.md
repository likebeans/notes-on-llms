---
title: MCP实战项目
description: 从零开始构建完整的MCP服务
pageType: article
module: mcp
updated: '2025-12-27'
contentStatus: needs-review
tags:
  - mcp
level: intermediate
prerequisites:
  - /llms/agent/tool-calling
reviewed: '2026-08-25'
techVersion: 待复核（2026-08）
---

# MCP实战项目

> 手把手教你构建一个完整可运行的MCP服务

## 🎯 项目一：智能任务管理器

一个完整的任务管理MCP服务，包含工具、资源和提示模板。

### 项目结构

```
task-manager-mcp/
├── server.py          # MCP服务器
├── requirements.txt   # 依赖
└── README.md
```

### 完整代码

```python
# server.py
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

# 创建MCP服务器
mcp = FastMCP(
    name="TaskManager",
    instructions="一个智能任务管理助手，可以创建、查询、更新和删除任务"
)

# ===== 数据模型 =====
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TaskInput(BaseModel):
    title: str = Field(..., description="任务标题")
    description: Optional[str] = Field(None, description="任务描述")
    priority: Priority = Field(Priority.MEDIUM, description="优先级")
    tags: list[str] = Field(default_factory=list, description="标签列表")

# 内存数据库
tasks_db: dict[int, dict] = {}
next_id = 1

# ===== 工具 =====
@mcp.tool
async def create_task(task: TaskInput, ctx: Context) -> dict:
    """创建新任务
    
    Args:
        task: 任务信息，包含标题、描述、优先级和标签
    
    Returns:
        创建的任务信息
    """
    global next_id
    
    await ctx.info(f"正在创建任务: {task.title}")
    
    new_task = {
        "id": next_id,
        "title": task.title,
        "description": task.description,
        "priority": task.priority.value,
        "tags": task.tags,
        "completed": False,
        "created_at": datetime.now().isoformat()
    }
    
    tasks_db[next_id] = new_task
    next_id += 1
    
    await ctx.info(f"任务创建成功，ID: {new_task['id']}")
    return new_task

@mcp.tool
async def list_tasks(
    priority: Optional[str] = None,
    completed: Optional[bool] = None,
    tag: Optional[str] = None,
    ctx: Context = None
) -> list[dict]:
    """列出所有任务，支持筛选
    
    Args:
        priority: 按优先级筛选 (low/medium/high)
        completed: 按完成状态筛选
        tag: 按标签筛选
    
    Returns:
        任务列表
    """
    if ctx:
        await ctx.info("正在获取任务列表...")
    
    tasks = list(tasks_db.values())
    
    if priority:
        tasks = [t for t in tasks if t["priority"] == priority]
    if completed is not None:
        tasks = [t for t in tasks if t["completed"] == completed]
    if tag:
        tasks = [t for t in tasks if tag in t["tags"]]
    
    if ctx:
        await ctx.info(f"找到 {len(tasks)} 个任务")
    
    return tasks

@mcp.tool
async def complete_task(task_id: int, ctx: Context) -> dict:
    """标记任务为完成
    
    Args:
        task_id: 任务ID
    
    Returns:
        更新后的任务信息
    """
    if task_id not in tasks_db:
        raise ValueError(f"任务 {task_id} 不存在")
    
    await ctx.info(f"正在完成任务 {task_id}...")
    
    tasks_db[task_id]["completed"] = True
    tasks_db[task_id]["completed_at"] = datetime.now().isoformat()
    
    await ctx.info(f"任务 {task_id} 已完成!")
    return tasks_db[task_id]

@mcp.tool
async def delete_task(task_id: int, ctx: Context) -> dict:
    """删除任务
    
    Args:
        task_id: 任务ID
    
    Returns:
        操作结果
    """
    if task_id not in tasks_db:
        raise ValueError(f"任务 {task_id} 不存在")
    
    await ctx.warning(f"正在删除任务 {task_id}...")
    
    deleted = tasks_db.pop(task_id)
    
    return {"message": f"任务 '{deleted['title']}' 已删除", "deleted_task": deleted}

# ===== 资源 =====
@mcp.resource("tasks://stats")
def get_stats() -> dict:
    """获取任务统计信息"""
    total = len(tasks_db)
    completed = sum(1 for t in tasks_db.values() if t["completed"])
    
    priority_counts = {}
    for task in tasks_db.values():
        p = task["priority"]
        priority_counts[p] = priority_counts.get(p, 0) + 1
    
    return {
        "total": total,
        "completed": completed,
        "pending": total - completed,
        "by_priority": priority_counts
    }

@mcp.resource("tasks://{task_id}")
def get_task_detail(task_id: str) -> dict:
    """获取单个任务详情"""
    tid = int(task_id)
    if tid not in tasks_db:
        return {"error": f"任务 {tid} 不存在"}
    return tasks_db[tid]

# ===== 提示模板 =====
@mcp.prompt
def daily_planning() -> str:
    """每日计划模板"""
    pending_tasks = [t for t in tasks_db.values() if not t["completed"]]
    high_priority = [t for t in pending_tasks if t["priority"] == "high"]
    
    tasks_summary = "\n".join([
        f"- [{t['priority'].upper()}] {t['title']}" 
        for t in pending_tasks[:10]
    ]) or "暂无待办任务"
    
    return f"""# 每日计划助手

## 当前待办任务
{tasks_summary}

## 高优先级任务数量
{len(high_priority)} 个

请帮我：
1. 分析这些任务的优先级是否合理
2. 建议今天应该完成哪些任务
3. 如果有遗漏，建议添加什么任务"""

@mcp.prompt
def task_review(days: int = 7) -> str:
    """任务回顾模板"""
    completed = [t for t in tasks_db.values() if t["completed"]]
    
    return f"""# 任务回顾

## 已完成任务 ({len(completed)} 个)
{chr(10).join([f"- {t['title']}" for t in completed[:10]]) or "暂无"}

请帮我：
1. 总结过去 {days} 天的工作成果
2. 分析工作效率
3. 提出改进建议"""

# 运行服务器
if __name__ == "__main__":
    mcp.run()
```

### 依赖文件

```txt
# requirements.txt
fastmcp>=2.0.0
pydantic>=2.0.0
```

### 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 方式1：直接运行
python server.py

# 方式2：使用FastMCP CLI
fastmcp run server.py

# 方式3：HTTP模式运行
fastmcp run server.py --transport http --port 8000
```

### 测试服务

```python
# test_client.py
import asyncio
from fastmcp import Client

async def main():
    async with Client("python server.py") as client:
        # 列出工具
        tools = await client.list_tools()
        print("可用工具:", [t.name for t in tools])
        
        # 创建任务
        result = await client.call_tool("create_task", {
            "task": {
                "title": "学习MCP协议",
                "description": "深入理解MCP的核心概念",
                "priority": "high",
                "tags": ["学习", "技术"]
            }
        })
        print("创建任务:", result)
        
        # 列出任务
        tasks = await client.call_tool("list_tasks", {})
        print("所有任务:", tasks)
        
        # 读取统计资源
        stats = await client.read_resource("tasks://stats")
        print("统计信息:", stats)

asyncio.run(main())
```

### Claude Desktop 配置

```json
{
  "mcpServers": {
    "task-manager": {
      "command": "python",
      "args": ["D:/path/to/server.py"],
      "env": {}
    }
  }
}
```

---

## 🎯 项目二：天气查询服务

一个简单的天气查询MCP服务示例。

### 完整代码

```python
# weather_server.py
from fastmcp import FastMCP, Context
import httpx

mcp = FastMCP(
    name="WeatherService",
    instructions="查询城市天气信息的助手"
)

# 模拟天气数据（实际可接入真实API）
MOCK_WEATHER = {
    "beijing": {"temp": 25, "condition": "晴", "humidity": 45},
    "shanghai": {"temp": 28, "condition": "多云", "humidity": 65},
    "guangzhou": {"temp": 32, "condition": "雷阵雨", "humidity": 80},
    "shenzhen": {"temp": 30, "condition": "晴", "humidity": 70},
}

@mcp.tool
async def get_weather(city: str, ctx: Context) -> dict:
    """获取指定城市的天气信息
    
    Args:
        city: 城市名称（拼音，如 beijing）
    
    Returns:
        天气信息
    """
    await ctx.info(f"正在查询 {city} 的天气...")
    
    city_lower = city.lower()
    if city_lower not in MOCK_WEATHER:
        available = ", ".join(MOCK_WEATHER.keys())
        return {"error": f"不支持的城市，当前支持: {available}"}
    
    weather = MOCK_WEATHER[city_lower]
    
    await ctx.info(f"查询完成: {weather['condition']}")
    
    return {
        "city": city,
        "temperature": f"{weather['temp']}°C",
        "condition": weather["condition"],
        "humidity": f"{weather['humidity']}%"
    }

@mcp.tool
async def compare_weather(cities: list[str], ctx: Context) -> list[dict]:
    """比较多个城市的天气
    
    Args:
        cities: 城市列表
    
    Returns:
        各城市天气对比
    """
    await ctx.info(f"正在比较 {len(cities)} 个城市的天气...")
    
    results = []
    for i, city in enumerate(cities):
        await ctx.report_progress(i + 1, len(cities))
        
        city_lower = city.lower()
        if city_lower in MOCK_WEATHER:
            weather = MOCK_WEATHER[city_lower]
            results.append({
                "city": city,
                "temp": weather["temp"],
                "condition": weather["condition"]
            })
    
    # 按温度排序
    results.sort(key=lambda x: x["temp"], reverse=True)
    
    return results

@mcp.resource("weather://supported-cities")
def get_supported_cities() -> list[str]:
    """获取支持的城市列表"""
    return list(MOCK_WEATHER.keys())

@mcp.prompt
def travel_weather_advice(destination: str) -> str:
    """旅行天气建议模板"""
    return f"""我计划去 {destination} 旅行。

请帮我：
1. 查询当地天气
2. 根据天气给出穿衣建议
3. 推荐适合的旅行活动
4. 提醒需要注意的天气相关事项"""

if __name__ == "__main__":
    mcp.run()
```

---

## 🎯 项目三：文件助手

一个文件操作MCP服务。

### 完整代码

```python
# file_server.py
from fastmcp import FastMCP, Context
from pathlib import Path
import json

mcp = FastMCP(
    name="FileAssistant",
    instructions="文件操作助手，可以读取、搜索和分析文件"
)

# 配置工作目录（安全起见限制在特定目录）
WORK_DIR = Path("./workspace")
WORK_DIR.mkdir(exist_ok=True)

@mcp.tool
async def read_file(filename: str, ctx: Context) -> str:
    """读取文件内容
    
    Args:
        filename: 文件名（相对于工作目录）
    
    Returns:
        文件内容
    """
    filepath = WORK_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    if not filepath.is_relative_to(WORK_DIR):
        raise PermissionError("不允许访问工作目录外的文件")
    
    await ctx.info(f"正在读取: {filename}")
    
    content = filepath.read_text(encoding="utf-8")
    
    await ctx.info(f"读取完成，共 {len(content)} 字符")
    return content

@mcp.tool
async def write_file(filename: str, content: str, ctx: Context) -> dict:
    """写入文件
    
    Args:
        filename: 文件名
        content: 文件内容
    
    Returns:
        操作结果
    """
    filepath = WORK_DIR / filename
    
    await ctx.warning(f"正在写入文件: {filename}")
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    
    await ctx.info(f"写入完成，共 {len(content)} 字符")
    
    return {
        "success": True,
        "filename": filename,
        "size": len(content)
    }

@mcp.tool
async def list_files(pattern: str = "*", ctx: Context = None) -> list[dict]:
    """列出工作目录中的文件
    
    Args:
        pattern: 文件匹配模式（如 *.txt）
    
    Returns:
        文件列表
    """
    if ctx:
        await ctx.info(f"搜索文件: {pattern}")
    
    files = []
    for path in WORK_DIR.glob(pattern):
        if path.is_file():
            files.append({
                "name": path.name,
                "size": path.stat().st_size,
                "is_dir": False
            })
        elif path.is_dir():
            files.append({
                "name": path.name,
                "items": len(list(path.iterdir())),
                "is_dir": True
            })
    
    return files

@mcp.resource("file://{filename}")
def get_file_info(filename: str) -> dict:
    """获取文件信息"""
    filepath = WORK_DIR / filename
    
    if not filepath.exists():
        return {"error": "文件不存在"}
    
    stat = filepath.stat()
    return {
        "name": filename,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "is_dir": filepath.is_dir()
    }

@mcp.prompt
def code_review(filename: str) -> str:
    """代码审查模板"""
    return f"""请审查文件 {filename} 的代码。

审查要点：
1. 代码质量和可读性
2. 潜在的bug和安全问题
3. 性能优化建议
4. 最佳实践遵循情况

请先使用 read_file 工具读取文件内容，然后进行审查。"""

if __name__ == "__main__":
    mcp.run()
```

---

## 📊 项目对比

| 项目 | 工具数 | 资源数 | 提示数 | 特点 |
|------|--------|--------|--------|------|
| **任务管理器** | 4 | 2 | 2 | CRUD操作、Pydantic模型 |
| **天气服务** | 2 | 1 | 1 | 进度报告、数据比较 |
| **文件助手** | 3 | 1 | 1 | 文件操作、安全限制 |

---

## 🚀 下一步

1. **添加持久化**：将内存数据库替换为SQLite或Redis
2. **添加认证**：使用Bearer Token保护敏感操作
3. **部署到云端**：使用FastMCP Cloud或自建HTTP服务
4. **添加中间件**：日志、缓存、速率限制

---

## 🔗 相关阅读

- [MCP快速入门](/llms/mcp/quickstart) - 基础入门
- [核心概念](/llms/mcp/concepts) - Tools/Resources/Prompts
- [高级功能](/llms/mcp/advanced) - 中间件、认证
- [MCP概述](/llms/mcp/) - 协议全貌

> **外部资源**：
> - [FastMCP 官方文档](https://gofastmcp.com/)
> - [FastMCP GitHub](https://github.com/jlowin/fastmcp)
