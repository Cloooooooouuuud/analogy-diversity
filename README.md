# Analogy Diversity Experiments

This repo contains the code behind our study of how the diversity of analogical few-shot examples (`LowDiv` vs. `HighDiv`) influences LLM reasoning across GSM8K, MATH, uMATH, and HARDMath style problems. We manually curate human-written analogies, keep the target question constant, and vary the example structure/domain mix to quantify effects on accuracy, chain-of-thought behavior, and error modes.

## Repository tour
- `src/prompt_builder.py` – loads the uniform/diverse analogy banks and assembles prompts for any dataset/difficulty.
- `src/run_experiment.py` – end-to-end loop that builds prompts, queries an HF model locally, and logs per-question metrics.
- `src/analysis_and_quality.py` – quick pass for interpreting result CSVs (analogy usage, token length, stats, plots).
- `src/prepare_toy_*` – helpers that download a handful of GSM8K/MATH variants to create lightweight sanity-check JSONs in `data/`.
- `data/` – curated prompt pools and toy/eval question files; `results/` and `runs/` store generated CSVs, logs, and figures.

## Environment setup
1. Use Python 3.10+ with PyTorch + Transformers that support the target model (default `meta-llama/Llama-3.1-8B-Instruct`).
2. Install dependencies:
   ```bash
   pip install -r paper/requirements.txt  # or pip install torch transformers datasets pandas matplotlib scipy
   ```
3. Export a Hugging Face token with gated-model access before running experiments:
   ```bash
   export HF_TOKEN=hf_your_token_here
   ```

## Data and prompt preparation
- Human-written analogy exemplars live in `data/prompt_uniform_*.json` and `data/prompt_diverse_*.json`. Extend them to test new domains/difficulties; the loader falls back to `*_easy.json` if a difficulty-specific file is missing.
- Toy evaluation sets (3-shot slices) are under `data/toy_<dataset>_3.json`. Regenerate them by running, e.g.,
  ```bash
  python src/prepare_toy_gsm8k_dataset.py
  ```
  Similar scripts exist for `math`, `hardmath`, and `umath`.

## Running an experiment
The quickest entry point is `python main.py`, which calls `run_experiment` with whatever arguments you pass via CLI flags (see below). For reproducibility, call `src/run_experiment.py` directly:
```bash
python -m src.run_experiment \
  --testfile data/toy_gsm8k_3.json \
  --difficulty gsm8k \
  --nshot 3 \
  --outdir results \
  --logdir runs
```
This iterates over `baseline` (zero-shot), `uniform`, and `diverse` prompts for every question, queries the selected HF model locally, and writes:
- `results/results_<dataset>_<difficulty>_<nshot>shot.csv` containing per-condition metrics (accuracy, token counts, model output, prompt text).
- `runs/log_<dataset>_<difficulty>_<nshot>shot.txt` capturing the streaming console log.

## Inspecting results
- Quick sanity checks: open the CSV in `results/` to compare `correct`, `model_answer`, and `prompt_tokens` across conditions.
- Run `python src/analysis_and_quality.py` after updating `RESULT_FILE` to your CSV path to generate:
  - Console summary of analogy usage rates and token lengths per condition.
  - Welch t-test comparing uniform vs. diverse analogy adoption.
  - `results/analysis_analogy_rate.png`, a simple bar chart saved for reporting.

## Customization tips
- Swap to another local or API-served model by editing `MODEL_NAME` (and tokenizer/model kwargs) in `src/run_experiment.py`.
- Use `--nshot` to vary the number of exemplars pulled from the prompt banks; keep uniform/diverse files balanced so comparisons remain fair.
- Add new benchmarks by dropping a JSON file with fields `id`, `concept`, `category`, `question`, and `answer` into `data/` and pointing `--testfile` to it.

That’s all you need to replicate or extend the controlled analogy diversity experiments. Happy prompting!
