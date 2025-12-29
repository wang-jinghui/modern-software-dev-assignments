import os
from dotenv import load_dotenv
from ollama import chat
from typing import Optional, Dict, Any

load_dotenv()

TARGET_MODEL = "qwen3:4b"
POLICY_MODEL = "qwen3:4b"  # 可替换为更小模型,如 "phi3", "gemma2:2b"

'''
DSP: Directional Stimulus Prompting
Directional Stimulus Prompting (DSP) 是一种通过外部“策略模型”(policy LM)生成引导性提示(stimulus/hint)
来调控黑盒大模型行为的前沿方法。其核心思想是：
1. 用一个小而可训练的 policy LM 生成“方向性刺激”(如关键词、指令片段、思维锚点)
2. 注入到 frozen LLM 的 prompt 中,以引导其输出朝向期望方向(如更简洁、更事实性、更创意等)。

虽然原始论文使用 RL 微调 policy LM,但在非训练、非验证型 demo中,我们可以用 Qwen3:4B 自身
(或另一个 Ollama 模型)模拟 policy LM 的角色,实现一个通用、干净、无 RL、无 ground-truth 验证的基础架构。
'''


class DirectionalStimulusPrompting:
    def __init__(
        self,
        target_model: str = TARGET_MODEL,
        policy_model: str = POLICY_MODEL,
    ):
        self.target_model = target_model
        self.policy_model = policy_model

    def generate_stimulus(
        self,
        task_input: str,
        guidance_direction: str,
    ) -> str:
        """
        Step 1: Policy LM generates a directional stimulus (hint) based on input + desired direction.
        This simulates the "trained policy" in the paper.
        """
        policy_prompt = f"""You are a hint generator for guiding large language models.
Given the following input and desired output direction, produce a short, focused stimulus (1-2 sentences) 
that will steer the model toward that goal.

Input: {task_input}
Desired direction: {guidance_direction}

Stimulus:"""

        response = chat(
            model=self.policy_model,
            messages=[{"role": "user", "content": policy_prompt}],
            options={"temperature": 0.5},
        )
        return response.message.content.strip()

    def execute_with_stimulus(
        self,
        task_input: str,
        stimulus: str,
    ) -> str:
        """
        Step 2: Frozen black-box LLM uses the stimulus to generate final output.
        """
        full_prompt = f"""{stimulus}

Now, based on the above guidance, process the following input:

Input: {task_input}

Output:"""

        response = chat(
            model=self.target_model,
            messages=[{"role": "user", "content": full_prompt}],
            options={"temperature": 0.7},
        )
        return response.message.content.strip()

    def run_dsp(
        self,
        task_input: str,
        guidance_direction: str,
    ) -> Dict[str, Any]:
        """
        Full DSP pipeline:
        1. Generate directional stimulus
        2. Execute target LLM with stimulus
        Returns both intermediate and final outputs.
        """
        print("🎯 Generating directional stimulus...")
        stimulus = self.generate_stimulus(task_input, guidance_direction)

        print("🧠 Executing target LLM with stimulus...")
        final_output = self.execute_with_stimulus(task_input, stimulus)

        return {
            "input": task_input,
            "guidance_direction": guidance_direction,
            "stimulus": stimulus,
            "output": final_output,
        }


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    INPUT_TEXT = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. 
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower 
    from 1887 to 1889 as the centerpiece of the 1889 World's Fair. 
    Although initially criticized by some of France's leading artists and intellectuals, 
    it has become a global cultural icon of France and one of the most recognizable structures in the world.
    """

    DIRECTION = "Summarize in one sentence focusing only on historical facts."

    dsp = DirectionalStimulusPrompting()

    print("🚀 Running Directional Stimulus Prompting (DSP)...\n")
    result = dsp.run_dsp(INPUT_TEXT, DIRECTION)

    print("\n" + "="*70)
    print("✅ DSP COMPLETE")
    print("="*70)
    print(f"Guidance Direction: {result['guidance_direction']}")
    print(f"\nStimulus (from policy LM):\n\"{result['stimulus']}\"")
    print(f"\nFinal Output (from target LLM):\n{result['output']}")