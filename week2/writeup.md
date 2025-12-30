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

class ActionItemsResponse(BaseModel):
    """Pydantic model for LLM response structure"""
    action_items: List[str]

def extract_action_items_llm(text: str) -> List[str]:
    """
    Extract action items from text using LLM, with identical signature and behavior
    as the original extract_action_items function.
    
    Args:
        text: Input text containing potential action items
        
    Returns:
        List of cleaned action items, deduplicated while preserving order
    """
    try:
        # Create a precise prompt reflecting the original function's exact behavior
        prompt = f"""
You are an expert in extracting action items from text. Extract all action items following these EXACT rules:

1. Identify action items by:
   - Bullet point prefixes (-, *, •, or numbered lists like "1.")
   - Keyword prefixes: "todo:", "action:", "next:"
   - Checkbox markers: "[ ]" or "[todo]"
   - Imperative sentences starting with: add, create, implement, fix, update, write, 
     check, verify, refactor, document, design, investigate

2. For each action item:
   - Remove bullet point prefixes
   - Remove checkbox markers ([ ] or [todo])
   - Do NOT remove keyword prefixes (todo:, action:, next:)
   - Trim extra whitespace

3. Final processing:
   - Deduplicate items (keep only first occurrence of each unique item)
   - Preserve the original order of first occurrences
   - Return ONLY a JSON object with an "action_items" field containing the cleaned list

Text to analyze:
{text}

IMPORTANT: Return ONLY the JSON object with the "action_items" field, no explanations or additional text.
        """
        
        # Call Ollama with the structured prompt and JSON format
        response = chat(
            model='qwen3:4b',
            messages=[{'role': 'user', 'content': prompt}],
            format=ActionItemsResponse.model_json_schema(),
            options={'temperature': 0.1}
        )
        
        # Parse the response content into our model
        result = ActionItemsResponse.model_validate_json(response['message']['content'])
        
        # Deduplicate while preserving order (exactly matching original function behavior)
        seen = set()
        unique_items = []
        for item in result.action_items:
            lowered = item.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique_items.append(item)
                
        return unique_items
        
    except Exception as e:
        # Log the error but return empty list to match original function behavior
        logging.error(f"LLM extraction failed: {str(e)}")
        return []
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
# tests/test_extract_llm.py
import pytest
from app.services.extract import extract_action_items, extract_action_items_llm

def test_empty_string():
    """Test behavior with empty string input"""
    result = extract_action_items_llm("")
    assert isinstance(result, list)
    assert len(result) == 0
    assert result == extract_action_items("")

def test_narrative_text():
    """Test behavior with purely narrative text (no clear action items)"""
    text = "This is just a regular sentence. It describes something that happened yesterday."
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)
    assert result == original

def test_bullet_points():
    """Test behavior with bullet point lists"""
    text = """
    - Set up database
    * Implement API endpoint
    1. Write unit tests
    • Review code
    """
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    assert len(result) == len(original)
    
    # Verify key content from original items exists in LLM results
    for orig_item in original:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in result)

def test_keyword_prefixes():
    """Test behavior with keyword prefixes"""
    text = """
    TODO: Complete documentation
    Action: Schedule meeting with team
    Next: Prepare presentation slides
    需要: 审核 PR
    请完成: 更新用户配置文件
    """
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    assert len(result) == len(original)
    
    # Verify key content from original items exists in LLM results
    for orig_item in original:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in result)

def test_mixed_content():
    """Test behavior with mixed content containing multiple action items"""
    text = """
    Meeting notes from 2023-10-15:
    
    - [ ] Set up database
    * implement API extract endpoint
    1. Write tests
    
    This is just a narrative sentence about what we discussed.
    
    Next: Review the design document
    Action: Send email to stakeholders
    
    Some more random text.
    """
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    assert len(result) == len(original)
    
    # Verify key content from original items exists in LLM results
    for orig_item in original:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in result)

def test_complex_notes():
    """Test behavior with complex notes containing special characters, newlines, and indentation"""
    text = """
        Project Update:
        
        - [ ] Fix login issue (user can't authenticate)
        *   Update documentation for new feature
        3.  Investigate performance problem:
            - Check database queries
            - Profile API endpoints
            
        Action items from discussion:
        TODO: Contact client about timeline
        Next: Prepare demo for Friday
        
        Note: This is not an action item.
    """
    result = extract_action_items_llm(text)
    original = extract_action_items(text)
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(item, str) for item in result)
    assert len(result) == len(original)
    
    # Verify key content from original items exists in LLM results
    for orig_item in original:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in result)

def test_consistency_with_original():
    """Test that LLM version produces structurally compatible results with original function"""
    note = "TODO: 审核 PR\n- 准备下周会议\n* Fix bug in login flow\nNext: Send email to team"
    
    old_result = extract_action_items(note)
    new_result = extract_action_items_llm(note)
    
    # Verify both return the same type and structure
    assert isinstance(old_result, list)
    assert isinstance(new_result, list)
    assert all(isinstance(item, str) for item in old_result)
    assert all(isinstance(item, str) for item in new_result)
    
    # Verify similar result count and content
    assert len(old_result) == len(new_result)
    
    # Verify semantic equivalence (key content matches)
    for orig_item in old_result:
        key_words = orig_item.lower().split()[:2]
        assert any(all(kw in res.lower() for kw in key_words) for res in new_result)
```

### Exercise 3: Refactor Existing Code for Clarity
Prompt: 
```
TODO
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
TODO
``` 

Generated Code Snippets:
```
TODO: List all modified code files with the relevant line numbers.
```


## SUBMISSION INSTRUCTIONS
1. Hit a `Command (⌘) + F` (or `Ctrl + F`) to find any remaining `TODO`s in this file. If no results are found, congratulations – you've completed all required fields. 
2. Make sure you have all changes pushed to your remote repository for grading.
3. Submit via Gradescope. 