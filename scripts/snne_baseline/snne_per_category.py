"""Compute per-CATEGORY (IH/IL/CH/CL) within-band AUROCs for SNNE UQ methods.

Reads the per-example scores dumped by dump_snne_scores.py, joins category labels
from the LapEigvals .pt test files, and computes for each method:
  AUROC(all)     uncertainty vs incorrectness on the full test set
  AUROC(IH v CH) within High-confidence band (both confidently asserted)
  AUROC(IL v CL) within Low-confidence band

This makes SNNE directly comparable to the 2-axis combos in CATEGORY_REPORT.md,
including the hard IH-vs-CH correctness split that a pure UQ score is expected to
struggle on.

Output: results/snne_baseline/snne_per_category.csv  (+ printed summary)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCORE_DIR = REPO_ROOT / "results" / "snne_baseline" / "scores"
DIAG_DIR = REPO_ROOT / "results" / "lapeigvals_baseline" / "diags"
OUT = REPO_ROOT / "results" / "snne_baseline" / "snne_per_category.csv"

PAIRS = [f"{m}_{d}" for m in ("llama", "mistral", "qwen", "gemma") for d in ("sciq", "triviaqa", "math")]
METHODS = ["num_set", "lexical_sim", "sum_eigv", "degree", "eccentricity", "luq", "snne"]
CAT_SHORT = {"incorrect_high": "IH", "incorrect_low": "IL",
             "correct_high": "CH", "correct_low": "CL"}


def coerce_float(series):
    """Some method columns were dumped as stringified torch tensors, e.g.
    'tensor(-0.9505)'. Strip the wrapper and convert to float."""
    return (series.astype(str)
                  .str.replace(r"^tensor\((.*)\)$", r"\1", regex=True)
                  .astype(float))


def auroc(y, s):
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def load_categories(pair):
    """id -> short category, from the .pt test split."""
    model, dataset = pair.split("_", 1)
    pt = torch.load(DIAG_DIR / f"{model}_{dataset}_test.pt", weights_only=False)
    return {str(i): CAT_SHORT.get(c, "?") for i, c in zip(pt["ids"], pt["categories"])}


def process_pair(pair, rows):
    score_path = SCORE_DIR / f"{pair}.csv"
    if not score_path.exists():
        print(f"[skip] {pair}: no scores file")
        return
    df = pd.read_csv(score_path)
    df["id"] = df["id"].astype(str)
    for m in METHODS:
        df[m] = coerce_float(df[m])
    id2cat = load_categories(pair)
    df["cat"] = df["id"].map(id2cat)
    df = df.dropna(subset=["cat"])

    # label: 1 = correct (True), 0 = incorrect. uncertainty score should be high for incorrect.
    df["is_incorrect"] = 1 - df["label"].astype(int)
    cats = df["cat"].values

    for m in METHODS:
        s = df[m].values
        a_all = auroc(df["is_incorrect"].values, s)

        def band(inc, cor):
            mask = np.isin(cats, [inc, cor])
            if mask.sum() < 5:
                return float("nan")
            return auroc(df["is_incorrect"].values[mask], s[mask])

        rows.append({
            "pair": pair, "method": m,
            "auroc_all": a_all,
            "auroc_IH_vs_CH": band("IH", "CH"),
            "auroc_IL_vs_CL": band("IL", "CL"),
            "n": len(df),
        })


def main():
    rows = []
    for pair in PAIRS:
        process_pair(pair, rows)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"[written] {OUT}\n")

    # Best method per pair by AUROC(all), with its within-band breakdown
    print("=== Best SNNE method per pair (by AUROC all) ===")
    best_rows = []
    for pair in PAIRS:
        sub = df[df.pair == pair]
        if sub.empty:
            continue
        b = sub.loc[sub.auroc_all.idxmax()]
        best_rows.append(b)
        print(f"  {pair:18s} {b['method']:12s} all={b['auroc_all']:.4f} "
              f"IHvCH={b['auroc_IH_vs_CH']:.4f} ILvCL={b['auroc_IL_vs_CL']:.4f}")

    if best_rows:
        bdf = pd.DataFrame(best_rows)
        print(f"\n=== Best-per-pair SNNE mean across pairs ===")
        print(f"  all   = {bdf.auroc_all.mean():.4f}")
        print(f"  IHvCH = {bdf.auroc_IH_vs_CH.mean():.4f}")
        print(f"  ILvCL = {bdf.auroc_IL_vs_CL.mean():.4f}")

    # Also report sum_eigv and snne specifically (common headline methods)
    for m in ("sum_eigv", "snne"):
        sub = df[df.method == m]
        print(f"\n=== {m} mean across pairs ===")
        print(f"  all={sub.auroc_all.mean():.4f} IHvCH={sub.auroc_IH_vs_CH.mean():.4f} "
              f"ILvCL={sub.auroc_IL_vs_CL.mean():.4f}")


if __name__ == "__main__":
    main()
