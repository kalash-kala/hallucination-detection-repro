"""Build the head-to-head comparison: SNNE unsupervised baselines vs our Classifier A.

Both score the SAME target (our open_text_label = greedy correctness) on the SAME
test questions for each of the 9 model/dataset pairs, so AUROCs are directly
comparable. SNNE methods are UNSUPERVISED (no training); Classifier A is a SUPERVISED
logistic regression (20-seed mean ROC-AUC).

Outputs (results/snne_baseline/):
  comparison_vs_classifier.csv   - per pair: each SNNE method AUROC + best-SNNE +
                                    Classifier A AUROC + (classifier - best_snne)
  COMPARISON_REPORT.md           - readable summary with means and framing/caveats

Usage:
    python compare.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SNNE_DIR = REPO_ROOT / "results" / "snne_baseline"
CLF_SUMMARY = (REPO_ROOT / "results" / "distractor_features_classifier"
               / "classifier_a_correct_vs_incorrect" / "summary_all_pairs.csv")

METHODS = ["num_set", "lexical_sim", "sum_eigv", "degree", "eccentricity", "luq", "snne"]
PAIR_ORDER = [
    "llama_sciq", "llama_triviaqa", "llama_math",
    "mistral_sciq", "mistral_triviaqa", "mistral_math",
    "qwen_sciq", "qwen_triviaqa", "qwen_math",
]


def load_snne():
    """pair -> {method: auroc} from per-pair metrics csvs."""
    out = {}
    for pair in PAIR_ORDER:
        p = SNNE_DIR / f"{pair}_metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        out[pair] = dict(zip(df.method, df.auroc))
    return out


def load_classifier():
    """pair -> classifier A roc_auc (20-seed mean)."""
    df = pd.read_csv(CLF_SUMMARY)
    return {f"{r.model}_{r.dataset}": r.roc_auc for r in df.itertuples()}


def main():
    snne = load_snne()
    clf = load_classifier()
    if not snne:
        logger.error("No SNNE per-pair metrics found yet. Run run_baselines.py first.")
        return

    rows = []
    for pair in PAIR_ORDER:
        if pair not in snne:
            continue
        m = snne[pair]
        best_method = max(METHODS, key=lambda k: m.get(k, float("nan")))
        best_auroc = m.get(best_method, float("nan"))
        clf_auroc = clf.get(pair, float("nan"))
        row = {"pair": pair}
        row.update({k: m.get(k, float("nan")) for k in METHODS})
        row["best_snne"] = best_auroc
        row["best_snne_method"] = best_method
        row["classifier_a"] = clf_auroc
        row["clf_minus_best_snne"] = clf_auroc - best_auroc
        rows.append(row)

    df = pd.DataFrame(rows)
    mean_row = {"pair": "MEAN"}
    for c in METHODS + ["best_snne", "classifier_a", "clf_minus_best_snne"]:
        mean_row[c] = df[c].mean()
    mean_row["best_snne_method"] = ""
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    out_csv = SNNE_DIR / "comparison_vs_classifier.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved {out_csv}")

    # ---- markdown report ----
    n_pairs = len(rows)
    md = []
    md.append("# SNNE Baselines vs Our Classifier A\n")
    md.append(f"Coverage: {n_pairs}/9 pairs. All AUROCs predict the SAME target "
              "(our `open_text_label`, greedy correctness) on the SAME test questions.\n")
    md.append("- **SNNE methods**: unsupervised, 10 generations/question, "
              "entailment/lexical diversity. No training.\n"
              "- **Classifier A**: supervised logistic regression, distractor-informed "
              "features, 20-seed mean ROC-AUC.\n")
    md.append("\n## Per-pair AUROC\n")
    cols = ["pair"] + METHODS + ["best_snne", "classifier_a", "clf_minus_best_snne"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    md.append(header)
    md.append(sep)
    for _, r in df.iterrows():
        cells = [str(r["pair"])]
        for c in METHODS + ["best_snne", "classifier_a", "clf_minus_best_snne"]:
            v = r[c]
            cells.append(f"{v:.3f}" if pd.notna(v) else "—")
        md.append("| " + " | ".join(cells) + " |")

    mean = df[df.pair == "MEAN"].iloc[0]
    md.append("\n## Summary\n")
    md.append(f"- Mean best-SNNE AUROC: **{mean['best_snne']:.3f}**")
    md.append(f"- Mean Classifier A AUROC: **{mean['classifier_a']:.3f}**")
    md.append(f"- Mean advantage (Classifier A − best SNNE): "
              f"**{mean['clf_minus_best_snne']:+.3f}**")
    md.append("\nPer-method mean AUROC across pairs:")
    for k in METHODS:
        md.append(f"- {k}: {mean[k]:.3f}")
    md.append("\n## Caveats\n")
    md.append("1. **10 generations** per question (not the paper's 20); consistent across pairs.\n"
              "2. **5 pairs** reuse existing semantic_uncertainty generations (100% test "
              "overlap); **4 pairs** (qwen_sciq + triviaqa×3) were regenerated with vLLM "
              "replicating the same 5-shot prompt, temp=1.0, max_new_tokens=15.\n"
              "3. **Unsupervised vs supervised**: SNNE methods use no labels/training; "
              "Classifier A is trained. The comparison shows what a label-free diversity "
              "baseline achieves vs our trained detector on the same target.\n")
    (SNNE_DIR / "COMPARISON_REPORT.md").write_text("\n".join(md))
    logger.info(f"Saved {SNNE_DIR / 'COMPARISON_REPORT.md'}")
    print("\n".join(md))


if __name__ == "__main__":
    main()