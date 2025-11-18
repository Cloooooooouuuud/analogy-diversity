# src/prepare_toy_dataset.py

import json
from datasets import load_dataset
from pathlib import Path

def load_gsm8k(split="test", num_samples=3):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    # select first num_samples
    selected = ds.select(range(num_samples))
    return selected

def convert_to_pipeline_format(probs):
    toy = []
    for i, item in enumerate(probs):
        toy.append({
            "id": f"GSM8K_{i+1}",
            "concept": "grade_school_math",
            "question": item["question"].strip(),
            "answer": item["answer"].strip().split()[-1],  # 简化：取最后一个 token 作为答案
            "category": "gsm8k_algebra"
        })
    return toy

def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    toy_ds = load_gsm8k(split="test", num_samples=3)
    toy = convert_to_pipeline_format(toy_ds)
    out_path = Path("data/toy_gsm8k_3.json")
    save_json(toy, out_path)
    print(f"Saved {len(toy)} toy problems to {out_path}")
