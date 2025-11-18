import json

def load_prompts(condition="uniform"):
    path = f"data/prompt_{condition}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_test_questions():
    with open("data/test_questions.json", "r", encoding="utf-8") as f:
        return json.load(f)

# Example usage:
uniform_examples = load_prompts("uniform")
diverse_examples = load_prompts("diverse")
test_questions = load_test_questions()

print(uniform_examples[0]["analogy"])
print(test_questions[0]["question"])
