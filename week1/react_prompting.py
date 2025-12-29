import os
import re
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
from ollama import chat

'''
ReAct Agent Implementation
一个将 推理(Reasoning) 与 行动(Action) 交错进行的通用框架。ReAct 不是取代 CoT,
而是将其扩展为“具身智能”(Embodied Reasoning) —— 让语言模型不仅能“想”，还能“做”和“看”。

🎯 ReAct 核心思想(Yao et al., 2022)
“Think, then act. Observe, then think again.”

ReAct 的输出格式为交替的 Thought / Action / Observation 序列:
- Thought: 推理，即根据输入的 prompt 生成的答案。
- Action: 行动，即根据推理结果所选择的工具。
- Observation: 观察，即根据选择的工具所返回的结果。

🧱 架构设计
我们将实现以下组件:

BaseTool:工具抽象基类
内置工具:SearchTool)模拟)、CalculateTool)安全执行)、FinishTool
ReActAgent:主推理引擎
run_react:带验证的完整 pipeline
💡 为简化,SearchTool 默认使用预定义知识库字典)可替换为真实 API)

class DuckDuckGoTool(BaseTool):
    name = "WebSearch"
    def run(self, query: str) -> str:
        # 调用 duckduckgo-search 或 requests
        return real_search(query)[:500]  # 截断
'''

# ----------------------------
# 工具抽象与实现
# ----------------------------

class BaseTool(ABC):
    name: str  # e.g., "Search", "Calculate"

    @abstractmethod
    def run(self, input_str: str) -> str:
        pass
 

class FinishTool(BaseTool):
    name = "Finish"

    def run(self, input_str: str) -> str:
        return input_str.strip()

class CalculateTool(BaseTool):
    name = "Calculate"

    def run(self, expr: str) -> str:
        try:
            # ✅ 修复：把 - 放在末尾，避免被解释为范围
            if not re.match(r"^[0-9+\-*/().\s]+$", expr):
                # 更安全写法：显式列出允许字符，- 放最后
                if not re.match(r"^[0-9+*/().\s\-]+$", expr):
                    return "Error: Invalid characters in expression."
            # 安全求值
            result = eval(expr, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

class SearchTool(BaseTool):
    name = "Search"

    def __init__(self, knowledge_base: Dict[str, str]):
        self.kb = knowledge_base

    def run(self, query: str) -> str:
        query = query.strip().lower()
        # 尝试按关键词匹配：只要 kb key 包含 query 中的任一词，就返回
        query_words = set(query.split())
        for key, value in self.kb.items():
            key_words = set(key.lower().split())
            if query_words & key_words:  # 有交集
                return value
        return "No relevant information found."

# ----------------------------
# ReAct Agent
# ----------------------------

class ReActAgent:
    def __init__(
        self,
        model: str = "qwen3:4b",
        tools: Optional[List[BaseTool]] = None,
        max_steps: int = 6,
    ):
        self.model = model
        self.max_steps = max_steps
        self.tools = tools or [
            SearchTool(self._default_knowledge_base()),
            CalculateTool(),
            FinishTool(),
        ]
        self.tool_map = {tool.name: tool for tool in self.tools}

    @staticmethod
    def _default_knowledge_base() -> Dict[str, str]:
        return {
            "albert einstein": "Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
            "paris population": "As of 2023, the population of Paris is approximately 2.1 million.",
            "mount everest height": "Mount Everest is 8,848.86 meters (29,031.7 feet) tall.",
            "water boiling point": "Water boils at 100°C (212°F) at sea level.",
        }

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse lines like: Action: Search[query] or Action: Finish[answer]
        Returns (action_name, input_str) or (None, None)
        """
        match = re.search(r"Action:\s*(\w+)\[(.*)\]", text, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None

    def _build_prompt(self, question: str, history: List[str]) -> str:
        tool_descs = "\n".join([
            f"- {tool.name}[input]: {tool.__class__.__doc__ or 'Perform ' + tool.name.lower()}"
            for tool in self.tools
        ])
        history_str = "\n".join(history)
        return f"""You are a ReAct agent that interleaves Thought, Action, and Observation.

Tools available:
{tool_descs}

Use the following format:
Thought: [your reasoning]
Action: [tool name][input]
Observation: [result from tool]
... (repeat as needed)
Thought: I now know the final answer.
Action: Finish[answer]

Question: {question}
{history_str}"""

    def run_react(self, question: str, expected_answer: Any = None) -> Dict[str, Any]:
        history: List[str] = []
        final_answer = None
        finished = False

        for step in range(self.max_steps):
            prompt = self._build_prompt(question, history)
            response = chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            output = response.message.content.strip()

            # Append model output to history
            history.append(f"Thought: {output.split('Thought:')[-1].split('Action:')[0].strip()}")
            if "Action:" in output:
                history.append("Action: " + output.split("Action:", 1)[1].split("Observation:", 1)[0].strip())

            # Parse action
            action_name, action_input = self._parse_action(output)
            if not action_name:
                observation = "Error: Failed to parse Action."
            else:
                if action_name == "Finish":
                    final_answer = action_input
                    finished = True
                    observation = ""  # No observation for Finish
                elif action_name in self.tool_map:
                    observation = self.tool_map[action_name].run(action_input)
                else:
                    observation = f"Error: Unknown action '{action_name}'."

            if action_name != "Finish":
                history.append(f"Observation: {observation}")

            if finished:
                break

        # Validation
        correct = None
        if expected_answer is not None and final_answer is not None:
            try:
                if isinstance(expected_answer, (int, float)) and final_answer.replace('.', '').isdigit():
                    correct = abs(float(final_answer) - float(expected_answer)) < 1e-6
                else:
                    correct = final_answer.strip().lower() == str(expected_answer).strip().lower()
            except:
                correct = False

        return {
            "question": question,
            "expected": expected_answer,
            "final_answer": final_answer,
            "correct": correct,
            "steps": history,
            "truncated": not finished,
        }

# ----------------------------
# 示例与验证
# ----------------------------

if __name__ == "__main__":
    load_dotenv()

    # 测试用例(带标准答案 → 验证型)
    test_cases = [
        {
            "question": "How old was Albert Einstein in 1955?",
            "expected": "76"
        },
        {
            "question": "What is the height of Mount Everest in meters?",
            "expected": "8848.86"
        }
    ]

    agent = ReActAgent(max_steps=5)

    for case in test_cases:
        print("\n" + "=" * 70)
        print(f"❓ Question: {case['question']}")
        print(f"✅ Expected: {case['expected']}")
        print("=" * 70)

        result = agent.run_react(case["question"], expected_answer=case["expected"])

        print("\n🔍 Reasoning Trace:")
        for i, step in enumerate(result["steps"], 1):
            print(f"{i}. {step}")

        print(f"\n🎯 Final Answer: {result['final_answer']}")
        if result["correct"] is not None:
            status = "✅ CORRECT" if result["correct"] else "❌ INCORRECT"
            print(f"📊 Validation: {status}")
        if result["truncated"]:
            print("⚠️  Warning: Max steps reached without Finish.")