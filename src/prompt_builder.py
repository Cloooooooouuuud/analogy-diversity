# src/prompt_builder.py
import json
from pathlib import Path
from typing import List, Dict, Any

def load_prompts(condition: str, difficulty: str) -> List[Dict[str, Any]]:
    path = Path(f"data/prompt_{condition}_{difficulty}.json")
    if not path.exists():
        # fallback: use "easy" GSM8K-level examples if specific difficulty not found
        path = Path(f"data/prompt_{condition}_easy.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(condition_type: str, question_obj: Dict[str, Any],
                 difficulty: str = "easy", n_shot: int = 3) -> str:
    header = (
        "You are a reasoning tutor. You will solve math word problems. "
        "Follow the style of the examples when provided.\n\n"
    )
    examples = []
    if condition_type in ["uniform", "diverse"]:
        examples = load_prompts(condition_type, difficulty)[:n_shot]

    prompt = header
    for i, ex in enumerate(examples, start=1):
        prompt += (
            f"Example {i}:\n"
            f"Concept: {ex['concept']}\n"
            f"Analogy: {ex['analogy']}\n"
            f"Explanation: {ex['explanation']}\n"
            f"Final Answer: {ex.get('answer','')}\n\n"
        )

    q = question_obj
    prompt += (
        f"Problem: {q['question']}\n"
        "Think step by step, reasoning clearly but briefly.\n"
        "At the end, output exactly one line as:\n"
        "Final Answer: <answer>\n"
        "Answer (with explanation):"
    )
    return prompt

def build_zero_shot(question_obj: Dict[str, Any]) -> str:
    return (
        "You are a reasoning tutor. Solve the following math word problem step by step.\n\n"
        f"Problem: {question_obj['question']}\n"
        "Think step by step. Keep the reasoning concise but precise.\n"
        "At the end, output exactly one line in the form:\n"
        "Final Answer: <answer>\n"
        "Answer (with explanation):"
    )
