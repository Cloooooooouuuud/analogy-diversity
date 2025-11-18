# src/prepare_toy_math_dataset.py

import json
from pathlib import Path
from datasets import load_dataset

def load_math_dataset(split="train", num_samples=3):
    ds = load_dataset("qwedsacf/competition_math", split=split)
    selected = ds.select(range(num_samples))
    return selected

def convert_to_toy_format(probs, prefix="MATH"):
    toy = []
    for i, item in enumerate(probs):
        toy.append({
            "id": f"{prefix}_{i+1}",
            "concept": item.get("type", "math_competition"),
            "question": item["problem"].strip(),
            # Extract final answer: the solution contains LaTeX \boxed{…}
            "answer": item["solution"].split("\\boxed{")[-1].split("}")[0].strip(),
            "category": item.get("level", "unknown_level")
        })
    return toy

def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    samples = load_math_dataset(split="train", num_samples=3)
    toy = convert_to_toy_format(samples, prefix="MATH")
    out_path = Path("data/toy_math_3.json")
    save_json(toy, out_path)
    print(f"Saved {len(toy)} toy problems to {out_path}")
