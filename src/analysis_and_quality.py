import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path
import re

RESULT_FILE = Path("results/results_local.csv")

# Step 1. Load results
df = pd.read_csv(RESULT_FILE)
print(f"Loaded {len(df)} rows from {RESULT_FILE}")

# Step 2. Compute output length
df["output_len"] = df["model_output"].astype(str).apply(len)

# Step 3. Detect analogy usage (rule-based)
def detect_analogy(text):
    keywords = [
        r"\blike\b", r"\bas if\b", r"\bsimilar to\b",
        r"\bimagine\b", r"\bthink of\b", r"\bjust as\b",
        r"\bmetaphor\b", r"\bcompare\b"
    ]
    pattern = re.compile("|".join(keywords), re.IGNORECASE)
    return 1 if pattern.search(text or "") else 0

df["has_analogy"] = df["model_output"].apply(detect_analogy)

# Step 4. Aggregate by condition
summary = df.groupby("condition").agg(
    avg_len=("output_len", "mean"),
    analogy_rate=("has_analogy", "mean"),
    n=("id", "count")
).reset_index()

print("\n=== Summary ===")
print(summary)

# Step 5. Significance test (uniform vs diverse)
uni = df[df["condition"] == "uniform"]["has_analogy"]
div = df[df["condition"] == "diverse"]["has_analogy"]
t_stat, p_val = ttest_ind(uni, div, equal_var=False)
print(f"\nT-test (uniform vs diverse): t={t_stat:.3f}, p={p_val:.4f}")

# Step 6. Visualization
plt.figure(figsize=(6, 4))
plt.bar(summary["condition"], summary["analogy_rate"], color=["#66c2a5", "#fc8d62", "#8da0cb"])
plt.title("Analogy usage rate by condition")
plt.ylabel("Proportion of outputs using analogy")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/analysis_analogy_rate.png", dpi=300)
plt.show()

print("\n✅ Analysis complete. Chart saved to results/analysis_analogy_rate.png")
