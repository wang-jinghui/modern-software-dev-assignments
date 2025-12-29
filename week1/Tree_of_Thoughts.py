import os
from dotenv import load_dotenv
from ollama import chat
from collections import deque
from typing import List, Dict, Callable, Optional, Any
import json

load_dotenv()

MODEL_NAME = "qwen3:4b"
TEMPERATURE = 0.7  # ToT 需要更高多样性

'''
Tree of Thoughts (ToT)
For complex tasks that require exploration or strategic lookahead, 
traditional or simple prompting techniques fall short. Yao et el. (2023) 
and Long (2023) recently proposed Tree of Thoughts (ToT), a framework 
that generalizes over chain-of-thought prompting and encourages exploration 
over thoughts that serve as intermediate steps for general problem solving 
with language models.

ToT maintains a tree of thoughts, where thoughts represent coherent language 
sequences that serve as intermediate steps toward solving a problem. This 
approach enables an LM to self-evaluate the progress through intermediate 
thoughts made towards solving a problem through a deliberate reasoning process. 
The LM's ability to generate and evaluate thoughts is then combined with search 
algorithms (e.g., breadth-first search and depth-first search) to enable systematic 
exploration of thoughts with lookahead and backtracking.
'''

class ThoughtNode:
    def __init__(self, content: str, depth: int = 0, parent: Optional["ThoughtNode"] = None):
        self.content = content          # 当前 thought 文本
        self.depth = depth              # 深度（根为 0）
        self.parent = parent            # 父节点
        self.children: List["ThoughtNode"] = []
        self.metadata: Dict[str, Any] = {}  # 可存评分、状态等

    def get_full_path(self) -> List[str]:
        """返回从根到当前节点的完整 thought 路径"""
        path = []
        node = self
        while node:
            path.append(node.content)
            node = node.parent
        return list(reversed(path))

    def __repr__(self):
        return f"ThoughtNode(depth={self.depth}, content='{self.content[:50]}...')"


class TreeOfThoughts:
    def __init__(
        self,
        model: str = MODEL_NAME,
        temperature: float = TEMPERATURE,
        max_depth: int = 3,
        branching_factor: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_depth = max_depth
        self.branching_factor = branching_factor

    def generate_thoughts(self, system_prompt: str, user_prompt: str, n: int) -> List[str]:
        """生成 n 个候选 thoughts（通过多次采样）"""
        thoughts = []
        for _ in range(n):
            response = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": self.temperature},
            )
            thoughts.append(response.message.content.strip())
        return thoughts

    def evaluate_thoughts(
        self,
        eval_system_prompt: str,
        current_state: str,
        candidate_thoughts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        对每个 candidate thought 进行评估（可返回分数、是否 promising 等）
        默认实现：让 LLM 打分（1-5），但你可替换为规则/函数
        """
        evaluated = []
        for thought in candidate_thoughts:
            eval_prompt = f"""Current reasoning state:
{current_state}

Candidate next step:
{thought}

On a scale of 1 to 5, how promising is this step toward solving the problem? 
Respond ONLY with a JSON object: {{"score": <int>, "reason": "<brief>"}}"""

            try:
                response = chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": eval_system_prompt},
                        {"role": "user", "content": eval_prompt},
                    ],
                    options={"temperature": 0.1},  # 评估需确定性
                )
                result = json.loads(response.message.content.strip())
                score = int(result.get("score", 1))
                reason = result.get("reason", "")
            except Exception as e:
                print(f"⚠️ Evaluation failed for thought '{thought[:30]}...': {e}")
                score = 1
                reason = "parse error"

            evaluated.append({
                "thought": thought,
                "score": score,
                "reason": reason
            })
        return evaluated

    def run_bfs(
        self,
        initial_prompt: str,
        gen_system_prompt: str,
        eval_system_prompt: str,
    ) -> List[ThoughtNode]:
        """
        执行广度优先搜索（BFS）构建 ToT
        返回所有达到 max_depth 的叶节点
        """
        # 初始化根节点
        root = ThoughtNode(content=initial_prompt, depth=0)
        queue = deque([root])
        leaf_nodes = []

        while queue:
            node = queue.popleft()

            if node.depth >= self.max_depth:
                leaf_nodes.append(node)
                continue

            # 生成下一步 thoughts
            current_path = " → ".join(node.get_full_path())
            gen_user_prompt = f"""Given the current reasoning path:
{current_path}

Generate one concise, coherent next step in solving the problem."""

            candidates = self.generate_thoughts(
                system_prompt=gen_system_prompt,
                user_prompt=gen_user_prompt,
                n=self.branching_factor
            )

            # 评估 candidates
            evaluated = self.evaluate_thoughts(
                eval_system_prompt=eval_system_prompt,
                current_state=current_path,
                candidate_thoughts=candidates
            )

            # 创建子节点（可在此加入剪枝：如只保留 top-k）
            for item in evaluated:
                child = ThoughtNode(
                    content=item["thought"],
                    depth=node.depth + 1,
                    parent=node
                )
                child.metadata.update(item)
                node.children.append(child)
                queue.append(child)

        return leaf_nodes


# ==============================
# Example Usage: Solve a riddle or plan a sequence
# ==============================

if __name__ == "__main__":
    INITIAL_PROMPT = "How can I measure exactly 4 liters using a 3-liter jug and a 5-liter jug?"

    GEN_SYSTEM_PROMPT = """
You are a creative problem-solving assistant.
Given a partial reasoning path, propose ONE logical next step toward solving the problem.
Be specific, actionable, and concise.
"""

    EVAL_SYSTEM_PROMPT = """
You are a critical evaluator of reasoning steps.
Assess whether the proposed step meaningfully advances the solution.
"""

    tot = TreeOfThoughts(
        max_depth=2,          # 根 → step1 → step2
        branching_factor=2,   # 每步生成 2 个候选
    )

    print("🌱 Starting Tree of Thoughts (BFS)...\n")
    leaves = tot.run_bfs(
        initial_prompt=INITIAL_PROMPT,
        gen_system_prompt=GEN_SYSTEM_PROMPT,
        eval_system_prompt=EVAL_SYSTEM_PROMPT,
    )

    print(f"\n✅ ToT completed. Found {len(leaves)} leaf paths.\n")
    for i, leaf in enumerate(leaves, 1):
        print(f"\n--- Leaf Path {i} (depth={leaf.depth}) ---")
        for j, step in enumerate(leaf.get_full_path()):
            score_info = ""
            if j > 0:  # 非根节点有 metadata
                node = leaf
                for _ in range(leaf.depth - j):
                    node = node.parent
                score = node.metadata.get("score", "?")
                score_info = f" [score={score}]"
            print(f"Step {j}: {step}{score_info}")

    # 最终输出（可选）
    print(f"\n🎯 Final leaf outputs (raw):")
    for leaf in leaves:
        print(f"- {leaf.content}")