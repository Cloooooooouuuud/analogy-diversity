# src/prepare_toy_umath_dataset.py
import json
from pathlib import Path
from datasets import load_dataset

def load_umath(split="test", num_samples=3):
    ds = load_dataset("toloka/u-math", split=split)
    return ds.select(range(num_samples))

def convert_to_toy_format(probs, prefix="UMATH"):
    toy = []
    for i, item in enumerate(probs):
        toy.append({
            "id": f"{prefix}_{i+1}",
            "concept": item.get("subject", "university_math"),
            "question": item["problem_statement"].strip(),   # ← 改这里
            "answer": item["golden_answer"].strip(),         # ← 改这里
            "category": item.get("subject", "u_math_text")
        })
    return toy

def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    samples = load_umath(split="test", num_samples=3)
    toy = convert_to_toy_format(samples, prefix="UMATH")
    out_path = Path("data/toy_umath_3.json")
    save_json(toy, out_path)
    print(f"Saved {len(toy)} toy problems to {out_path}")
