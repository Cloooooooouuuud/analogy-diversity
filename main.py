# from src.prompt_builder import load_prompts, load_test_questions, build_prompt, build_zero_shot

# # Load data
# uniform_examples = load_prompts("uniform")
# diverse_examples = load_prompts("diverse")
# test_questions = load_test_questions()

# # Pick one question
# q = test_questions[0]

# # Build three prompt variants
# prompt_uniform = build_prompt(uniform_examples, q)
# prompt_diverse = build_prompt(diverse_examples, q)
# prompt_zero = build_zero_shot(q)

# print("=== Uniform Prompt ===\n")
# print(prompt_uniform)
# print("\n=== Diverse Prompt ===\n")
# print(prompt_diverse)
# print("\n=== Zero-shot Prompt ===\n")
# print(prompt_zero)

from src.run_experiment import run_experiment

if __name__ == "__main__":
    run_experiment()