import os
import ast
import time
import types
import signal
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

MODEL_NAME = "qwen3:4b"


# ==============================
# 安全执行模块(轻量级沙箱)
# ==============================
'''
安全执行模块(轻量级沙箱)，用于执行生成的 Python 代码。
主要功能：
1. 静态检查代码是否包含危险操作(如导入、调用危险函数、使用全局/非局部声明)。
2. 在超时和受限环境中执行代码，防止无限循环或资源耗尽。
'''

import ast
import signal
import types
from typing import Any

"""
安全执行模块(轻量级沙箱)，用于执行生成的 Python 代码。
主要功能：
1. 静态检查代码是否包含危险操作(如导入、调用危险函数、使用全局/非局部声明)。
2. 在超时和受限环境中执行代码，防止无限循环或资源耗尽。
"""
class SafeExecutor:
    # ✅ 类变量：允许的顶层模块（可安全导入）
    ALLOWED_MODULES = {
        "math", "datetime", "decimal", "fractions", "random",
        "re", "itertools", "collections", "statistics", "string"
    }

    # ✅ 类变量：允许的内置函数/常量名称
    ALLOWED_BUILTINS_NAMES = {
        "abs", "all", "any", "bin", "bool", "chr", "divmod", "enumerate",
        "filter", "float", "format", "hash", "hex", "int", "isinstance", "len", "list",
        "map", "max", "min", "next", "oct", "ord", "pow", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
        "True", "False", "None",
    }

    @staticmethod
    def _get_builtins_dict():
        """兼容 __builtins__ 是模块或字典的情况"""
        builtins = __builtins__
        if isinstance(builtins, types.ModuleType):
            return vars(builtins)
        return builtins

    @classmethod
    def _safe_import(cls, name: str, globals=None, locals=None, fromlist=(), level=0):
        """
        安全的 __import__ 钩子，仅允许白名单模块。
        """
        if level != 0:
            raise ImportError("Relative imports are not allowed.")

        top_module = name.split(".")[0]
        if top_module not in cls.ALLOWED_MODULES:
            raise ImportError(
                f"Import of '{name}' is not allowed. "
                f"Allowed modules: {sorted(cls.ALLOWED_MODULES)}"
            )
        return __import__(name, globals, locals, fromlist, level)

    @classmethod
    def check_code_safety(cls, code: str) -> bool:
        """
        静态 AST 检查：禁止危险操作，但允许 import（由运行时控制）。
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            # 允许 Import / ImportFrom —— 安全性由 _safe_import 保证
            if isinstance(node, ast.Call):
                # 禁止 eval/exec/compile
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile"):
                        return False
                # 禁止危险属性调用
                elif isinstance(node.func, ast.Attribute):
                    attr = getattr(node.func, 'attr', '')
                    if isinstance(attr, str) and attr in (
                        "system", "popen", "exec", "eval", "write", "read",
                        "__dict__", "__globals__", "__subclasses__"
                    ):
                        return False
            # 禁止全局/非局部声明（避免污染）
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                return False

        return True

    @classmethod
    def execute_with_timeout(cls, code: str, timeout: int = 3) -> Any:
        """
        在受限环境中执行代码，支持安全动态 import。
        """
        if not cls.check_code_safety(code):
            raise ValueError("Unsafe code detected during static analysis.")

        # 构建安全的 __builtins__
        builtin_vars = cls._get_builtins_dict()
        safe_builtins = {
            name: builtin_vars[name]
            for name in cls.ALLOWED_BUILTINS_NAMES
            if name in builtin_vars
        }
        # 注入受控的 __import__
        safe_builtins["__import__"] = cls._safe_import

        safe_globals = {"__builtins__": safe_builtins}
        safe_locals = {}

        # Unix 信号超时（Windows 用户可注释或替换为 threading 方案）
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out.")

        try:
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            exec(code, safe_globals, safe_locals)
            return safe_locals.get("result", None)
        except Exception as e:
            raise RuntimeError(f"Execution error: {e}")
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


# ==============================
# PAL 主类
# ==============================
'''
PAL(Program-Aided Language Models) 是将 LLM 的自然语言理解能力与精确的程序执行环境
(如 Python 解释器)结合的关键范式，特别适合数学、逻辑、符号推理等需要精确计算的任务。

📌 核心思想：
LLM 不直接输出答案，而是生成一段可执行代码(如 Python)，然后由外部解释器运行该代码得到最终结果。
这避免了 LLM 在算术、循环、符号操作中的幻觉问题。
'''

class ProgramAidedLanguageModel:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def generate_program(self, question: str) -> str:
        prompt = f"""You are a precise programming assistant.
Read the following problem and write a Python program to solve it.
- Only use basic Python and allowed modules (math, datetime, etc.)
- Store the final answer in a variable named `result`
- Do not print anything
- Do not include explanations, comments, or markdown

Problem: {question}

Program:"""
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        code = response.message.content.strip()
        # 移除可能的首尾空白和多余行
        lines = code.splitlines()
        # 去掉空行和注释（可选）
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                clean_lines.append(line)  # 保留原始缩进
        return "\n".join(clean_lines)

    def run_pal(self, question: str, expected_answer: Any = None) -> Dict[str, Any]:
        """
        Full PAL pipeline:
        1. Generate program
        2. Safely execute it
        3. (Optional) Validate against expected_answer
        """
        print("📝 Generating program...")
        program = self.generate_program(question)
        print("Generated program:\n")
        print("-" * 50 + "\n")
        print(program)
        print("\n" + "-" * 50)

        try:
            print("⚡ Executing program safely...")
            result = SafeExecutor.execute_with_timeout(program, timeout=3)
        except Exception as e:
            return {
                "question": question,
                "program": program,
                "execution_error": str(e),
                "result": None,
                "correct": False if expected_answer is not None else None,
            }

        correct = None
        if expected_answer is not None:
            # 尝试宽松比较(数值容忍浮点误差，字符串忽略大小写/空格)
            try:
                if isinstance(expected_answer, (int, float)) and isinstance(result, (int, float)):
                    correct = abs(float(result) - float(expected_answer)) < 1e-6
                else:
                    correct = str(result).strip().lower() == str(expected_answer).strip().lower()
            except:
                correct = False

        return {
            "question": question,
            "expected": expected_answer,
            "program": program,
            "result": result,
            "correct": correct,
        }


# ==============================
# Example Usage (with validation)
# ==============================

if __name__ == "__main__":
    # 示例 1：数学题(有标准答案 → 验证型)
    QUESTION_1 = "What is 123 multiplied by 456?"
    EXPECTED_1 = 123 * 456  # 56088

    # 示例 2：日期计算
    QUESTION_2 = "How many days are between January 1, 2023 and March 1, 2023?"
    from datetime import date
    d1 = date(2023, 1, 1)
    d2 = date(2023, 3, 1)
    EXPECTED_2 = (d2 - d1).days  # 59

    pal = ProgramAidedLanguageModel()

    for q, exp in [(QUESTION_1, EXPECTED_1), (QUESTION_2, EXPECTED_2)]:
        print("\n" + "="*70)
        print(f"❓ Question: {q}")
        print(f"✅ Expected: {exp}")
        print("="*70)

        output = pal.run_pal(q, expected_answer=exp)

        print(f"\n🎯 Result: {output['result']}")
        if output["correct"] is not None:
            status = "✅ CORRECT" if output["correct"] else "❌ INCORRECT"
            print(f"🔍 Validation: {status}")
        if "execution_error" in output:
            print(f"💥 Execution Error: {output['execution_error']}")

    print("\n🏁 PAL demo completed.")