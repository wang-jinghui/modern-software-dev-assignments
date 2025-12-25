import os
import re
from collections import Counter
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

"""
Self-Consistency
Perhaps one of the more advanced techniques out there for prompt engineering is self-consistency.
Proposed by Wang et al. (2022), self-consistency aims "to replace the naive greedy decoding used 
in chain-of-thought prompting". The idea is to sample multiple, diverse reasoning paths through 
few-shot CoT, and use the generations to select the most consistent answer. This helps to boost 
the performance of CoT prompting on tasks involving arithmetic and commonsense reasoning.

正确范式：Free-form Reasoning + Robust Answer Extraction
"""

# TODO: Fill this in! Try to get as close to 100% correctness across all runs as possible.
YOUR_SYSTEM_PROMPT = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done,
there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted.
So, they must have planted 21 - 15 = 6 trees.
Answer: 6
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. 
Answer: 5
"""

USER_PROMPT = """
Solve this problem step by step. You may use any notation you like in your reasoning — 
LaTeX, equations, Chinese, English, or code.

However, for the final answer, follow this rule strictly:

🔹 On the very last line of your entire response, write exactly:
Answer: <number>

Do NOT use LaTeX, boxes, markdown, or units in this final line.
Examples:
Good: Answer: 312
Bad: Answer: $\\boxed{312}$
Bad: The answer is 312.
Bad: Answer: 312 cans
Bad:
$$
\boxed{312}
$$

This format is required for automated grading. Violations will be marked wrong.

Question:
A lighthouse emits two signals:
- Signal X flashes every 12 seconds,
- Signal Y flashes every 25 seconds.

However, due to a fault, Signal Y started 37 seconds AFTER Signal X.

At time t = 0, Signal X flashed.

What is the smallest time t > 0 (in seconds) such that:
1. Both signals flash at the same moment, AND
2. t ≤ 1701 ?

If no such t exists within the limit, output -1.
"""

EXPECTED_OUTPUT = "312"


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
    return "None"


def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt NUM_RUNS_TIMES, majority-vote on the extracted 'Answer: ...' lines.

    Prints "SUCCESS" if the majority answer equals EXPECTED_OUTPUT.
    """
    answers: list[str] = []
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="qwen3:4b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.8},
        )
        output_text = response.message.content
        final_answer = extract_answer(output_text)
        print(f"Run {idx + 1} answer: {final_answer}")
        answers.append(final_answer.strip())

    if not answers:
        print("No answers produced.")
        return False

    counts = Counter(answers)
    majority_answer, majority_count = counts.most_common(1)[0]
    print(f"Majority answer: {majority_answer} ({majority_count}/{len(answers)})")

    if majority_answer.strip() == EXPECTED_OUTPUT.strip():
        print("SUCCESS")
        return True

    # Print distribution for debugging when majority does not match expected
    print(f"Expected output: {EXPECTED_OUTPUT}")
    print("Answer distribution:")
    for answer, count in counts.most_common():
        print(f"  {answer}: {count}")
    return False


if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)


