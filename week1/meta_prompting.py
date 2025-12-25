import os
import re
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

"""
Key Characteristics
According to Zhang et al. (2024), the key characteristics of meta prompting can be summarized as follows:

1. Structure-oriented: Prioritizes the format and pattern of problems and solutions over specific content.

2. Syntax-focused: Uses syntax as a guiding template for the expected response or solution.

3. Abstract examples: Employs abstracted examples as frameworks, illustrating the structure of problems and solutions without focusing on specific details.

4. Versatile: Applicable across various domains, capable of providing structured responses to a wide range of problems.

5. Categorical approach: Draws from type theory to emphasize the categorization and logical arrangement of components in a prompt.
"""

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
You must solve the problem using rigorous mathematical reasoning.
Problem Statement:
• Problem: [question to be answered]
Solution Structure:
1. Begin the response with “Let's think step by step.”
2. Follow with the reasoning steps, ensuring the solution process is broken down clearly
and logically.
"""

USER_PROMPT = """
Solve this problem, then give the final answer on the last line as "Answer: <number>".

What is 7^{9876} mod 13?
"""


EXPECTED_OUTPUT = "1"


def extract_answer(output: str) -> str | None:
    """
    Extract the final answer from model output that follows the format:
        ... 
        Answer: <value>
    
    This function:
    - Searches for a line containing "Answer:" (case-insensitive)
    - Strips whitespace and ignores markdown/code blocks
    - Returns the answer string if found, otherwise None
    
    Examples:
        "Answer: 42" → "42"
        "The result is Answer: 123" → "123"
        "answer: 7" → "7"
        "Final answer: 5" → None (doesn't match 'Answer:' pattern)
    """
    # Use case-insensitive regex to find "Answer:" followed by non-whitespace
    match = re.search(r"(?i)Answer:\s*(\S+)", output)
    if match:
        return match.group(1).strip()
    return None

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="qwen3:4b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        answer = extract_answer(output_text)
        if answer == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            print(f"Output\n: {output_text}")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)