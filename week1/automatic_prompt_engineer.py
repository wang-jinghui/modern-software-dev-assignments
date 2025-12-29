import os
from dotenv import load_dotenv
from ollama import chat
from typing import List, Dict, Optional
import json

load_dotenv()

MODEL_NAME = "qwen3:4b"
TEMPERATURE_GEN = 0.8   # 指令生成需多样性
TEMPERATURE_SCORE = 0.1 # 评分需确定性

'''
APE 核心思想（Zhou et al., 2022）
Goal: 自动生成高质量的 instruction（即 system/user prompt），无需人工设计。

标准的 Automatic Prompt Engineer (APE) 流程是：

1. 生成候选指令（Instruction Generation）

2. 评估这些指令（Instruction Evaluation / Scoring）

3. 选择最优指令，并用它执行目标任务（Execution with Best Instruction）
'''

class AutomaticPromptEngineer:
    def __init__(
        self,
        target_model: str = MODEL_NAME,
        inference_model: Optional[str] = None,
        num_candidates: int = 5,
    ):
        self.target_model = target_model
        self.inference_model = inference_model or target_model
        self.num_candidates = num_candidates

    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples
        )

    def generate_instructions(self, task_desc: str, examples: List[Dict[str, str]]) -> List[str]:
        demo = self._format_examples(examples)
        prompt = f"""You are an expert prompt engineer.
Given these input-output demonstrations:

{demo}

Write a clear, general instruction that would guide an AI to produce the correct output from any similar input.
Only output the instruction. No explanations.

Instruction:"""
        
        instructions = []
        for _ in range(self.num_candidates):
            resp = chat(model=self.inference_model, messages=[{"role": "user", "content": prompt}], 
                        options={"temperature": TEMPERATURE_GEN})
            instructions.append(resp.message.content.strip())
        return instructions

    def score_instruction(self, instruction: str, examples: List[Dict[str, str]]) -> float:
        demo = self._format_examples(examples)
        eval_prompt = f"""Evaluate this instruction:
"{instruction}"

Based on these demonstrations:
{demo}

Rate from 1 to 10 how well the instruction captures the task pattern (clarity, generality, alignment).

Respond ONLY: {{"score": <number>}}"""
        
        try:
            resp = chat(model=self.inference_model, messages=[{"role": "user", "content": eval_prompt}],
                        options={"temperature": TEMPERATURE_SCORE})
            score = float(json.loads(resp.message.content.strip()).get("score", 1))
            return max(1.0, min(10.0, score))
        except:
            return 1.0

    def execute_with_instruction(self, instruction: str, input_text: str) -> str:
        """✅ 第三阶段：用最优指令执行新输入"""
        resp = chat(
            model=self.target_model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
            ],
            options={"temperature": 0.0},  # deterministic
        )
        return resp.message.content.strip()

    def run_ape(
        self,
        task_description: str,
        examples: List[Dict[str, str]],
        test_input: str,  # 👈 新增：用于最终执行的输入
    ) -> Dict:
        """
        Full APE pipeline:
        1. Generate candidate instructions
        2. Score them (LLM-based, no ground truth)
        3. Select best and EXECUTE on test_input
        """
        print("Step 1: Generating candidate instructions...")
        candidates = self.generate_instructions(task_description, examples)

        print("Step 2: Scoring instructions...")
        scored = []
        for instr in candidates:
            score = self.score_instruction(instr, examples)
            scored.append({"instruction": instr, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)

        best_instr = scored[0]["instruction"]
        print("Step 3: Executing best instruction on new input...")

        final_output = self.execute_with_instruction(best_instr, test_input)

        return {
            "task": task_description,
            "test_input": test_input,
            "best_instruction": best_instr,
            "final_output": final_output,
            "all_candidates": scored,
        }


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    TASK = "Reverse the letters in a word."
    EXAMPLES = [
        {"input": "cat", "output": "tac"},
        {"input": "dog", "output": "god"},
    ]
    TEST_INPUT = "httpstatus"  # ← 新输入，用于最终执行

    ape = AutomaticPromptEngineer(num_candidates=3)
    result = ape.run_ape(TASK, EXAMPLES, TEST_INPUT)

    print("\n" + "="*60)
    print("✅ APE COMPLETE")
    print("="*60)
    print(f"Task: {result['task']}")
    print(f"Test Input: {result['test_input']}")
    print(f"\nBest Instruction:\n\"{result['best_instruction']}\"")
    print(f"\nFinal Output: {result['final_output']}")

    print("\nAll Candidates (scored):")
    for i, c in enumerate(result["all_candidates"], 1):
        print(f"  {i}. [{c['score']:.1f}/10] {c['instruction'][:60]}...")