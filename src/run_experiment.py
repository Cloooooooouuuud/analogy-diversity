# src/run_experiment.py

import os, csv, time, torch, argparse, sys, re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

from prompt_builder import build_prompt, build_zero_shot, load_prompts
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0
SLEEP_BETWEEN_CALLS = 0.5
SEED = 42

def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

FINAL_PATTERNS = [
    r"Final Answer\s*[:：]\s*(.*)",
    r"\\boxed\{([^}]*)\}"
]

def extract_final_answer(output: str) -> str:
    for pat in FINAL_PATTERNS:
        m = re.search(pat, output, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    lines = [ln.strip() for ln in output.split("\n") if ln.strip()]
    return lines[-1] if lines else ""

def normalize_ans(ans: str) -> str:
    ans = ans.strip()
    ans = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", ans)
    ans = re.sub(r"\\[a-zA-Z]+", "", ans)
    ans = re.sub(r"[^0-9a-zA-Z\.\-\(\)/]", "", ans)
    return ans.lower()

def load_test_questions(path: Path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def init_local_model(model_name=MODEL_NAME):
    print(f"🚀 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    return tokenizer, model

def query_local(prompt: str, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].size(1)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            early_stopping=False,
            eos_token_id=None,
            pad_token_id=tokenizer.pad_token_id
        )
    gen_ids = out[0][prompt_tokens:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    output_tokens = gen_ids.size(0)
    return text, prompt_tokens, output_tokens


def run_experiment(testfile: Path, outdir: Path, logdir: Path,
                   difficulty: str, n_shot: int):

    set_seed(SEED)

    # dataset name + difficulty naming
    dataset_name = testfile.stem.replace("toy_", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = outdir / f"results_{dataset_name}_{difficulty}_{n_shot}shot.csv"
    log_path = logdir / f"log_{dataset_name}_{difficulty}_{n_shot}shot.txt"

    logdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(log_path, "w", buffering=1)
    print(f"[LOG] Logging to {log_path}")

    examples_uniform = load_prompts("uniform", difficulty)
    examples_diverse = load_prompts("diverse", difficulty)
    questions = load_test_questions(testfile)

    tokenizer, model = init_local_model()

    fieldnames = [
        "id", "concept", "category",
        "difficulty", "condition", "n_shot",
        "true_answer", "model_answer", "correct",
        "prompt_tokens", "output_tokens", "cot_length",
        "prompt_text", "model_output",
        "timestamp"
    ]

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in questions:
            gold = q.get("answer", "")

            for condition, exset in [
                ("baseline", []),
                ("uniform", examples_uniform),
                ("diverse", examples_diverse),
            ]:
                # build prompt
                if condition == "baseline":
                    prompt = build_zero_shot(q)
                else:
                    prompt = build_prompt(condition, q, difficulty, n_shot)

                output, ptoks, otoks = query_local(prompt, tokenizer, model)

                # parse final answer
                model_ans = extract_final_answer(output)
                gold_norm = normalize_ans(gold)
                pred_norm = normalize_ans(model_ans)
                correct = int(bool(gold_norm) and pred_norm == gold_norm)

                writer.writerow({
                    "id": q.get("id", ""),
                    "concept": q.get("concept", ""),
                    "category": q.get("category", ""),
                    "difficulty": difficulty,
                    "condition": condition,
                    "n_shot": n_shot,
                    "true_answer": gold,
                    "model_answer": model_ans,
                    "correct": correct,
                    "prompt_tokens": ptoks,
                    "output_tokens": otoks,
                    "cot_length": otoks,
                    "prompt_text": prompt.replace("\n", " "),
                    "model_output": output,
                    "timestamp": datetime.now().isoformat()
                })
                f.flush()

                print("\n======================")
                print(f"Condition={condition}, n_shot={n_shot}, difficulty={difficulty}")
                print(f"Gold: {gold} | Pred: {model_ans} | Correct={correct}")
                print("======================\n")

                time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\n✅ Finished! Saved to {save_path}")
    print(f"📝 Log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testfile", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--difficulty", type=str, required=True,
                        choices=["gsm8k", "math", "umath", "hardmath"])
    parser.add_argument("--nshot", type=int, default=3)
    args = parser.parse_args()

    run_experiment(
        Path(args.testfile), Path(args.outdir), Path(args.logdir),
        difficulty=args.difficulty, n_shot=args.nshot
    )
