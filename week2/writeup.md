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
TODO
``` 

Generated Code Snippets:
```
TODO: List all modified code files with the relevant line numbers.
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