# Week 2 Write-up
Tip: To preview this markdown file
- On Mac, press `Command (⌘) + Shift + V`
- On Windows/Linux, press `Ctrl + Shift + V`

## INSTRUCTIONS

Fill out all of the `TODO`s in this file.

## SUBMISSION DETAILS

Name: **TODO** \
SUNet ID: **TODO** \
Citations: **TODO**

This assignment took me about **TODO** hours to do. 


## YOUR RESPONSES
For each exercise, please include what prompts you used to generate the answer, in addition to the location of the generated response. Make sure to clearly add comments in your code documenting which parts are generated.

### Exercise 1: Scaffold a New Feature
Prompt: 
```
你是一个严谨的 Python 工程师，正在为一个现有系统添加 LLM 支持。

当前extract.py中已存在一个函数 extract_action_items。 请仔细阅读其实现，仅基于其代码以及其引用的其他函数确定以下信息：
- 它实现一个什么样的功能
- 它接受哪些参数（名称、类型、是否有默认值）
- 它返回什么类型的值（包括嵌套结构、字段名、是否可能为 None 或空）
- 在无结果或出错时的行为（例如返回空列表、空字典、抛异常等）

现在，请实现一个新函数 `extract_action_items_llm`，要求：
- 函数签名（参数名、类型注解、默认值）必须与 `extract_action_items` 完全一致
- 返回值在类型、结构和边界行为上必须与 `extract_action_items` 完全一致
- 调用方应能无缝替换 `extract_action_items` 为 `extract_action_items_llm` 而无需修改任何代码

实现方式：
1. 使用 `ollama.chat` 调用本地运行的 `qwen3:4b` 模型执行实际的提取任务
2. 为了确保模型输出可被可靠解析：
   - 根据你从 `extract_action_items` 中观察到的**返回值结构**，定义对应的 Pydantic 模型（用于约束 Ollama 的 JSON 输出）
   - 如果返回的是列表，则需额外定义一个包装类（如 `ResultWrapper`），因为 Ollama 的 `format` 参数不支持顶层数组
   - 在 prompt 中明确指示模型：“只输出符合上述结构的 JSON，不要任何解释或额外文本”
3. 将 Ollama 返回的内容解析为该结构，并转换成与原函数**完全相同的返回形式**
4. 如果在调用 LLM、解析响应或结构转换过程中发生任何异常，必须：
   - 捕获异常
   - 记录日志（使用 `logging`）
   - 返回与原函数在同等错误/空情况下的相同结果（例如原函数返回 `[]`，你也返回 `[]`；原函数返回 `{}`，你也返回 `{}`）

重要：
- **不要对输入/输出做任何先验假设**
- **所有结构信息必须来自对 `extract_action_items` 实现的分析**
- 只输出 `extract_action_items_llm` 函数及其所需的内部辅助类（如 Pydantic 模型），不要修改原有函数

假设以下导入已可用：
from pydantic import BaseModel
from ollama import chat
import logging
``` 

Generated Code Snippets:
```python
from pydantic import BaseModel
from typing import List
import logging
from ollama import chat

# ./app/services/extract.py:93~162
```

### Exercise 2: Add Unit Tests
Prompt: 
```
你是一个专业的 Python 测试工程师，正在为函数 `extract_action_items_llm` 编写单元测试。

该函数位于 `week2/app/services/extract.py`，其行为必须与同文件中的 `extract_action_items` 完全一致。  
请先观察 `extract_action_items` 的实现，确定其：
- 输入参数（通常是一个字符串）
- 返回值类型和结构（例如列表、字典、嵌套形式等）
- 在空输入、无效输入或无匹配时的返回行为

基于这些观察，使用 `pytest` 为 `extract_action_items_llm` 编写测试用例，要求：

1. **测试用例应覆盖以下典型输入场景**：
   - 空字符串（""）
   - 纯叙述性文本（无明确行动项）
   - 项目符号列表（如以 "- " 或 "* " 开头的行）
   - 关键词前缀的行（如 "TODO:", "Action:", "需要:", "请完成:" 等常见任务标记）
   - 包含多个行动项的混合文本
   - 包含特殊字符、换行、缩进的复杂笔记

2. **每个测试用例必须**：
   - 调用 `extract_action_items_llm(input_text)`
   - 验证返回值的类型与 `extract_action_items` 一致
   - 验证返回值的结构合理（例如非空输入应返回非空结果，字段名符合预期）
   - 对于相同输入，`extract_action_items_llm` 和 `extract_action_items` 的输出应具有相同的语义（不要求完全相等，但关键字段应存在）

3. **不要硬编码期望字段名**（如 "description"），而是根据 `extract_action_items` 的实际返回动态判断。  
   如果原函数在某输入下返回 `[{"task": "review code"}]`，则 LLM 版本也应包含 `"task"` 字段。

4. **包含一个对比测试**（可选但推荐）：
   ```python
   def test_extract_consistency():
       note = "TODO: 审核 PR\n- 准备下周会议"
       old_result = extract_action_items(note)
       new_result = extract_action_items_llm(note)
       # 验证两者结构兼容（例如长度一致、都有必要字段）
``` 

Generated Code Snippets:
```python
# ./tests/test_extract_llm.py 
```

### Exercise 3: Refactor Existing Code for Clarity
Prompt: 
```
你是一位资深后端架构师，正在对项目中 `./app` 目录下的 FastAPI 应用进行**非功能性重构**。  
本次重构的目标是**提升代码可维护性与健壮性，不改变任何业务逻辑或 API 行为**。

请重点围绕以下四个方面进行改进：

#### 1. **清晰定义的 API 合约（Schemas）**
- 所有请求体（Request Body）和响应体（Response Model）必须使用 Pydantic 模型显式声明。
- 避免在路由函数中直接使用 `dict` 或未类型化的数据结构。
- 为不同场景（如创建、更新、列表、详情）定义专用的 schema，避免字段泄露（例如密码、内部状态）。
- 使用 `Field`、`validator` 等机制强化输入校验。

#### 2. **数据库层清理（Database Layer）**
- 确保数据库操作（CRUD）封装在独立的服务模块或 repository 类中，**路由层不应直接拼接 SQL 或调用 ORM 查询**。
- 移除硬编码的查询逻辑，改用参数化方法。
- 统一数据库会话（Session）管理方式（推荐使用依赖注入，如 `get_db`）。
- 避免在多个文件中重复定义相同的数据访问逻辑。

#### 3. **应用生命周期与配置管理（App Lifecycle & Configuration）**
- 将配置（如数据库 URL、Ollama 地址、超时设置等）集中到 `config.py` 或使用 `pydantic-settings`。
- 使用 `lifespan`（而非已弃用的 `on_event`）管理应用启动/关闭时的资源（如连接池、缓存）。
- 确保依赖项（如数据库连接、外部服务客户端）通过 FastAPI 的依赖注入系统提供，而非全局变量。

#### 4. **统一且健壮的错误处理（Error Handling）**
- 定义自定义异常类（如 `ActionItemExtractionError`），并在适当位置抛出。
- 使用 `@app.exception_handler` 注册全局异常处理器，将内部异常转换为标准 HTTP 错误响应（含错误码、消息、可选 trace_id）。
- 避免在路由中使用裸 `try/except` 返回 ad-hoc 错误信息。
- 记录关键错误日志（使用 `logging`），但不要暴露敏感信息给客户端。

#### 通用原则：
- **保持向后兼容**：所有公开 API 的 URL、请求/响应格式不得改变。
- **小步提交**：每个修改应聚焦单一职责（如“仅重构配置”或“仅提取 DB 层”）。
- **类型安全**：尽可能添加类型注解，启用 mypy 友好性。
- **移除死代码**：删除未使用的导入、函数、配置项。

请针对 `./app` 目录中的实际代码结构（包括 `main.py`, `routers/`, `services/`, `models/`, `schemas/`, `database.py` 等）提出具体重构建议或直接输出改进后的代码片段。  
若某些模块不存在（如无 `schemas/` 目录），可建议创建。

注意：本次任务是**重构（Refactor）**，不是重写（Rewrite）——功能行为必须完全一致。
``` 

Generated/Modified Code Snippets:
```
TODO: List all modified code files with the relevant line numbers. (We anticipate there may be multiple scattered changes here – just produce as comprehensive of a list as you can.)
```


### Exercise 4: Use Agentic Mode to Automate a Small Task
Prompt: 
```
TODO
``` 

Generated Code Snippets:
```
TODO: List all modified code files with the relevant line numbers.
```


### Exercise 5: Generate a README from the Codebase
Prompt: 
```
你是一位技术文档工程师，请基于当前项目的代码库，生成一份清晰、准确、面向开发者友好的 `README.md` 文件。

请仔细分析以下目录的代码内容：
- `./app/`：FastAPI 后端应用
- `./frontend/`：前端代码
- `./test/`：测试套件

根据你的分析，生成的 `README.md` 必须包含以下章节：

---

#### 1. **项目概述（Overview）**
- 用 2–3 句话说明本项目的核心功能（例如：“一个将自由格式笔记转换为结构化行动项的 Web 应用”）。
- 提及主要技术栈（如 FastAPI、SQLite等），**仅限代码中实际使用的**。

#### 2. **本地开发环境搭建与运行（Setup & Run）**
- **后端**：
  - 列出必要的依赖
  - 安装命令（如 `pip install -r requirements.txt`）
  - 启动命令（如 `uvicorn app.main:app --reload`）
- **前端**（如果存在可运行的前端）：
  - 构建/启动命令（如 `npm install && npm run dev`）
  - 默认访问地址（如 `http://localhost:3000`）
- **整体运行说明**：是否前后端分离？是否需要同时启动？

> 注意：所有命令必须基于实际文件（如 `requirements.txt`、`package.json`）推断，不要编造。

#### 3. **API 接口与功能（API Endpoints）**
- 列出所有公开的 API 路由
- 对每个端点说明：
  - HTTP 方法（GET/POST 等）
  - 路径
  - 功能描述（如“从文本中提取行动项”）
  - 请求/响应示例（可选，但鼓励提供简单 JSON 示例）
- 如果 API 有版本控制（如 v1），请体现。

#### 4. **运行测试套件（Running Tests）**
- 说明如何执行测试
- 是否需要额外配置
- 测试覆盖哪些模块

---

#### 格式要求：
- 使用标准 Markdown
- 语言简洁、技术准确
- 不要包含“TODO”、“示例”等占位符
- 所有信息必须**源自代码库**，若某部分不存在（如无前端），则省略或注明“暂无前端”

最终输出仅为 `README.md` 的完整内容，不要包含解释、注释或额外文本。
``` 

Generated Code Snippets:
```
modern-software-dev-assignments/week2/README.md
```


## SUBMISSION INSTRUCTIONS
1. Hit a `Command (⌘) + F` (or `Ctrl + F`) to find any remaining `TODO`s in this file. If no results are found, congratulations – you've completed all required fields. 
2. Make sure you have all changes pushed to your remote repository for grading.
3. Submit via Gradescope. 