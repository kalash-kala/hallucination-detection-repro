"""
Band-stratified correctness AUROC — the deployment-aligned alternative to the
one-vs-rest splits in Part 8.

Every contrast keeps a SINGLE axis (correctness) and singles out one confidence
band on the positive side, so the score is never asked to rank one incorrect class
above another (the confound that made the vs-rest IH column dip below 0.5):

  IH vs {CH,CL} : confident-WRONG  vs ALL correct  -> hard-error recall
  IL vs {CH,CL} : uncertain-WRONG  vs ALL correct  -> easy-error recall
  CH vs {IH,IL} : confident-CORRECT vs ALL incorrect
  CL vs {IH,IL} : uncertain-CORRECT vs ALL incorrect

Plus two equal-weight aggregates (so the high-count IL band cannot dominate):
  bal_incorrect = mean(IHvC, ILvC)   bal_correct = mean(CHvI, CLvI)

ORIENTATION (higher is always better): raw score is P(hallucination). For the two
INCORRECT targets the AUROC is used directly. For the two CORRECT targets it is
1 - roc_auc_score(y, P(wrong)) (rank-based, scale-independent) so the column reads
"how well this correct band is ranked below the errors". >0.5 beats chance on all.

Score sources are imported verbatim from per_category_ovr_auroc.py so the two
sections share identical per-example scores and coverage.

Output: results/per_category_analysis/per_category_band_auroc.csv

Usage:
  cd /home/kalashkala/recovery-gaps-experiment
  /root/miniconda3/envs/semantic_uncertainty/bin/python scripts/per_category_band_auroc.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import per_category_ovr_auroc as OVR  # reuse record builders + constants

REPO_ROOT = OVR.REPO_ROOT
OUT_DIR = OVR.OUT_DIR
PAIR_NAMES = OVR.PAIR_NAMES
CATS = OVR.CATS

INCORRECT = {"IH", "IL"}
CORRECT = {"CH", "CL"}
# (column key, positive category) — negatives are the opposite correctness class
CONTRASTS = [("IHvC", "IH"), ("ILvC", "IL"), ("CHvI", "CH"), ("CLvI", "CL")]


def band_auroc(scores: np.ndarray, cats: np.ndarray, target: str) -> float:
    neg = CORRECT if target in INCORRECT else INCORRECT
    mask = (cats == target) | np.isin(cats, list(neg))
    y = (cats[mask] == target).astype(int)
    s = scores[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    a = float(roc_auc_score(y, s))
    # correct targets: orient toward correctness (rank-based complement)
    return (1.0 - a) if target in CORRECT else a


def all_band(scores, cats) -> dict:
    scores = np.asarray(scores, dtype=float)
    cats = np.asarray(cats)
    out = {k: band_auroc(scores, cats, t) for k, t in CONTRASTS}
    out["bal_incorrect"] = float(np.nanmean([out["IHvC"], out["ILvC"]]))
    out["bal_correct"] = float(np.nanmean([out["CHvI"], out["CLvI"]]))
    return out


def main():
    print("collecting per-example scores ...")
    recs = {}
    recs.update(OVR.records_from_per_example()); print("  baselines done")
    recs.update(OVR.records_from_snne(OVR.best_snne_per_pair())); print("  SNNE done")
    recs.update(OVR.records_from_2axis()); print("  2-axis done")
    recs.update(OVR.records_from_hs_lr()); print("  hs_lr done")

    metric_cols = ["IHvC", "ILvC", "CHvI", "CLvI", "bal_incorrect", "bal_correct"]
    rows = []
    for (pair, method), (scores, cats) in recs.items():
        cats = np.asarray(cats)
        n_by = {t: int((cats == t).sum()) for t in CATS}
        rows.append({"pair": pair, "method": method, "n": len(cats),
                     **{f"n_{t}": n_by[t] for t in CATS},
                     **all_band(scores, cats)})

    rows.sort(key=lambda r: (PAIR_NAMES.index(r["pair"]) if r["pair"] in PAIR_NAMES else 99,
                             r["method"]))

    cols = (["pair", "method", "n"] + [f"n_{t}" for t in CATS] + metric_cols)
    out_path = OUT_DIR / "per_category_band_auroc.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\n[written] {out_path}  ({len(rows)} rows)")

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "—"

    means = defaultdict(lambda: defaultdict(list))
    for pair in PAIR_NAMES:
        print(f"\n=== {pair} ===")
        print(f"{'method':24s} {'IHvC':>7s} {'ILvC':>7s} {'CHvI':>7s} {'CLvI':>7s} "
              f"{'balI':>7s} {'balC':>7s}")
        for r in [r for r in rows if r["pair"] == pair]:
            print(f"{r['method']:24s} {fmt(r['IHvC']):>7s} {fmt(r['ILvC']):>7s} "
                  f"{fmt(r['CHvI']):>7s} {fmt(r['CLvI']):>7s} "
                  f"{fmt(r['bal_incorrect']):>7s} {fmt(r['bal_correct']):>7s}")
            base = r["method"].split(" (best=")[0] if r["method"].startswith("SNNE") else r["method"]
            for k in metric_cols:
                v = r[k]
                if isinstance(v, float) and not np.isnan(v):
                    means[base][k].append(v)

    print(f"\n=== MEAN across {len(PAIR_NAMES)} pairs (band-stratified correctness) ===")
    print(f"{'method':24s} {'IHvC':>7s} {'ILvC':>7s} {'CHvI':>7s} {'CLvI':>7s} "
          f"{'balI':>7s} {'balC':>7s}")
    for m in means:
        mv = {k: np.mean(means[m][k]) if means[m][k] else float("nan") for k in metric_cols}
        print(f"{m:24s} {fmt(mv['IHvC']):>7s} {fmt(mv['ILvC']):>7s} "
              f"{fmt(mv['CHvI']):>7s} {fmt(mv['CLvI']):>7s} "
              f"{fmt(mv['bal_incorrect']):>7s} {fmt(mv['bal_correct']):>7s}")


if __name__ == "__main__":
    main()
