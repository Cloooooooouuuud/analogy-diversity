# src/prompting.py
from textwrap import dedent

SYSTEM_PROMPT = (
    "You are an analogy tutor. You map abstract STEM concepts to concrete everyday scenarios. "
    "Always produce concise, precise explanations."
)

def _json_rule():
    return dedent("""\
    Respond ONLY with a JSON object of the form:
    <FINAL_JSON>{"analogy": "...", "explanation": "..."}</FINAL_JSON>
    - "analogy": a vivid real-world scenario (kitchen/sports/social etc.), 1–2 sentences.
    - "explanation": connect the scenario back to the concept, 1–2 sentences.
    - Do NOT mention other concepts. Do NOT add anything outside the JSON.
    """)

def build_uniform_prompt(concept: str, examples_sports: str) -> list:
    user = dedent(f"""\
    Each example maps an abstract concept to a SPORTS scenario, then explains the mapping.
    {examples_sports}

    Now it's your turn for ONE concept only.

    Concept: {concept}

    {_json_rule()}
    """)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]

def build_diverse_prompt(concept: str, examples_mixed: str) -> list:
    user = dedent(f"""\
    Each example maps an abstract concept to DIFFERENT DOMAINS (kitchen, sports, social), then explains.
    {examples_mixed}

    Now it's your turn for ONE concept only.

    Concept: {concept}

    {_json_rule()}
    """)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]

def build_zero_prompt(concept: str) -> list:
    user = dedent(f"""\
    Map the abstract concept to ANY everyday scenario, then explain.

    Concept: {concept}

    {_json_rule()}
    """)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
