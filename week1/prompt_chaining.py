import os
from dotenv import load_dotenv
from ollama import chat
from typing import List, Callable, Optional

load_dotenv()

MODEL_NAME = "qwen3:4b"
TEMPERATURE = 0.5


class PromptChain:
    def __init__(self, model: str = MODEL_NAME, temperature: float = TEMPERATURE):
        self.model = model
        self.temperature = temperature

    def run_step(
        self,
        system_prompt: str,
        user_prompt: str,
        previous_output: Optional[str] = None,
    ) -> str:
        """
        Run a single LLM step.
        If previous_output is provided, it can be inserted into user_prompt via {prev}.
        """
        # Allow user_prompt to reference {prev}
        if previous_output is not None:
            user_prompt = user_prompt.format(prev=previous_output)

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": self.temperature},
        )
        return response.message.content.strip()

    def run_chain(
        self,
        steps: List[dict],
        initial_input: Optional[str] = None,
    ) -> List[str]:
        """
        Run a chain of prompts.
        Each step is a dict with keys: 'system', 'user'
        The output of step i becomes the {prev} in step i+1's user prompt.
        """
        outputs = []
        current_output = initial_input

        for i, step in enumerate(steps):
            print(f"\n[Step {i+1}]")
            sys_prompt = step["system"]
            usr_prompt = step["user"]

            current_output = self.run_step(sys_prompt, usr_prompt, current_output)
            outputs.append(current_output)

            print(f"Output: {current_output}")

        return outputs


# ==============================
# Example: Reverse word via chaining
# ==============================

if __name__ == "__main__":
    # Step 1: Extract the word (in case input is messy)
    # Step 2: Reverse the letters
    # Step 3: Output clean result

    chain_steps = [
        {
            "system": "You are a text preprocessor. Extract only the alphabetic word from the input.",
            "user": "Input: {prev}",
        },
        {
            "system": "You are a string reverser. Reverse the order of all letters in the given word.",
            "user": "Word: {prev}",
        },
        {
            "system": "You output only the final result with no extra text.",
            "user": "Final reversed word: {prev}",
        },
    ]

    # Initial input (could come from user or another system)
    initial = "httpstatus"

    print(f"Starting prompt chain with initial input: '{initial}'\n")

    pc = PromptChain()
    results = pc.run_chain(chain_steps, initial_input=initial)

    print("\n✅ Chain completed.")
    print(f"Final output: '{results[-1]}'")