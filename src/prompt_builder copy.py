import json
from pathlib import Path

# ---------- Data Loading ---------- #

def load_prompts(condition: str):
    """Load few-shot examples from data/prompt_uniform.json or prompt_diverse.json"""
    path = Path(f"data/prompt_{condition}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_test_questions():
    """Load test questions"""
    path = Path("data/test_questions.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Prompt Construction ---------- #

def build_prompt(examples, question_obj):
    """
    Build a full prompt including few-shot examples and one target question.
    Each example includes 'concept', 'analogy', 'explanation'.
    """
    header = (
        "You are a reasoning tutor. Each example explains an abstract concept "
        "through a real-world analogy and short reasoning chain.\n"
        "Follow the same style to explain the final concept.\n\n"
    )

    prompt = header
    for i, ex in enumerate(examples, start=1):
        prompt += (
            f"Example {i}:\n"
            f"Concept: {ex['concept']}\n"
            f"Analogy: {ex['analogy']}\n"
            f"Explanation: {ex['explanation']}\n\n"
        )

    q = question_obj
    # prompt += (
    #     f"Now it's your turn.\n"
    #     f"Concept: {q['concept']}\n"
    #     f"Question: {q['question']}\n"
    #     "Analogy: "
    # )
    prompt += (
        f"Now it's your turn.\n"
        f"Concept: {q['concept']}\n"
        f"Question: {q['question']}\n"
        f"Please write ONLY:\n"
        f"Analogy: <your analogy>\n"
        f"Explanation: <your reasoning>\n"
        f"End your answer after the explanation." 
    )
    return prompt

# def build_prompt(examples, question_obj, condition_type="uniform"):
#     # header …
#     if condition_type == "baseline":
#         prompt = ( … only problem … )
#     else:
#         # add examples … 
#         if condition_type == "uniform":
#             … use examples_uniform …
#         elif condition_type == "diverse":
#             … use examples_diverse …
#     # then problem …
#     return prompt


def build_zero_shot(question_obj):
    """
    Build a zero-shot prompt: no examples, only task description and question.
    """
    header = (
        "You are a reasoning tutor. Explain each abstract concept "
        "through a real-world analogy and step-by-step reasoning.\n\n"
    )
    q = question_obj
    prompt = (
        header
        + f"Concept: {q['concept']}\n"
        + f"Question: {q['question']}\n"
        + "Analogy: "
    )
    return prompt
