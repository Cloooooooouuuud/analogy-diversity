# src/prepare_toy_hardmath_dataset.py

import json
from pathlib import Path

def load_hardmath(path: Path, num_samples=3):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 如果 data 是 dict, 转 list
    if isinstance(data, dict):
        # 假设题目存在键 "problems" 或类似
        # 你需要查看 JSON 结构
        if "problems" in data:
            problems = data["problems"]
        else:
            # 直接转换字典 values
            problems = list(data.values())
    else:
        problems = data
    return problems[:num_samples]

def convert_to_toy(probs, prefix="HARDMATH"):
    toy = []
    for i, item in enumerate(probs):
      toy.append({
          "id": f"{prefix}_{i+1}",
          "concept": item.get("question_type", "applied_math"),
          "question": item["question"].strip(),
          "answer": item.get("answer_val", item.get("solution", "")).strip(),
          "category": item.get("answer_type", "math_expression")
      })
    return toy

def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 请确认实际 HARDMath 数据路径
    path = Path("/home/cloud/mya/analogy-diversity/data/HARDMath.json")
    samples = load_hardmath(path, num_samples=3)
    toy = convert_to_toy(samples, prefix="HARDMATH")
    out_path = Path("data/toy_hardmath_3.json")
    save_json(toy, out_path)
    print(f"Saved {len(toy)} toy problems to {out_path}")
