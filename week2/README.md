# Action Item Extractor

## 1. 项目概述（Overview）

一个将自由格式笔记转换为结构化行动项的 Web 应用。使用 FastAPI 作为后端框架，SQLite 作为数据库，并集成了 Ollama 提供的 LLM 功能进行智能提取。

主要技术栈：FastAPI、Pydantic、SQLite、Ollama、Poetry。

## 2. 本地开发环境搭建与运行（Setup & Run）

### 后端

**必要依赖：**
- Python 3.9+
- Ollama (用于 LLM 功能)
- Poetry (用于依赖管理)

**安装命令：**

激活 Conda 环境:

```
conda activate cs146s
```

在项目根目录安装依赖:

```
poetry install
```

启动后端服务:

```
poetry run uvicorn week2.app.main:app --reload --host 0.0.0.0 --port 8888
```

或者使用预定义的脚本:

```
poetry run dev
```

### 前端

本项目包含一个简单的 HTML 前端页面，无需额外构建步骤。前端会自动挂载到 FastAPI 应用中，通过静态文件服务提供访问。

访问地址: http://localhost:8888 (或启动时显示的端口)

### 整体运行说明

前后端一体化部署，前端页面直接嵌入后端服务中。无需单独启动前端服务，只需启动 FastAPI 后端即可。

## 3. API 接口与功能（API Endpoints）

### 行动项相关接口 (/action-items)

```
POST /action-items/extract

```

功能：使用启发式方法从文本中提取行动项

请求体：`{"text": "笔记内容", "save_note": true}`

响应：包含提取的行动项列表的 JSON 对象

```
POST /action-items/extract-llm

```

功能：使用 LLM 从文本中提取行动项

请求体：`{"text": "笔记内容", "save_note": true}`

响应：包含提取的行动项列表的 JSON 对象

```
GET /action-items

```

功能：列出所有行动项，可选按笔记 ID 过滤

查询参数：`?note_id=1`

响应：行动项列表

```
POST /action-items/{action_item_id}/done

```

功能：标记行动项为完成/未完成

请求体：`{"done": true}`

响应：标记结果

### 笔记相关接口 (/notes)

```
POST /notes

```

功能：创建新笔记

请求体：`{"content": "笔记内容"}`

响应：创建的笔记对象

```
GET /notes

```

功能：列出所有笔记

响应：笔记列表

```
GET /notes/{note_id}

```

功能：获取特定笔记

响应：笔记对象

### 根路径

```
GET /

```

功能：返回前端页面

## 4. 运行测试套件（Running Tests）

### 执行测试

使用 pytest 运行测试套件：

```
# 运行所有测试
poetry run pytest

# 运行所有测试并显示覆盖率
poetry run pytest --cov=week2.app

# 运行特定测试文件
poetry run pytest tests/test_extract.py
```

### 测试配置

测试套件位于 `tests/` 目录

主要测试文件包括：
- `test_extract.py`：测试行动项提取功能
- `test_extract_llm.py`：测试 LLM 行动项提取功能

### 测试覆盖范围

- 行动项提取算法的准确性
- LLM 提取功能的正确性
- API 端点的响应
- 边界情况处理（如空输入、无效输入等）

注意：运行测试前需要确保依赖已安装（`poetry install`）。部分测试可能需要 Ollama 服务运行以测试 LLM 相关功能。
