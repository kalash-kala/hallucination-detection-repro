"""
One-vs-rest (OvR) per-category AUROC for every hallucination-detection method.

Complements the pairwise IH-vs-CH / IL-vs-CL columns in CATEGORY_REPORT.md with
the four one-vs-rest splits:

  IH vs rest : y=1 for incorrect_high, 0 for {CH, CL, IL}
  CH vs rest : y=1 for correct_high,   0 for {CL, IH, IL}
  IL vs rest : y=1 for incorrect_low,  0 for {CH, CL, IH}
  CL vs rest : y=1 for correct_low,    0 for {CH, IH, IL}

ORIENTATION (Option B — higher is always better): every method's score is
P(hallucination) (higher = more likely incorrect). For the two INCORRECT targets
(IH, IL) the OvR-AUROC uses that score directly. For the two CORRECT targets
(CH, CL) the score is oriented toward correctness — i.e. P(correct) = 1 - P(wrong)
for probabilistic methods, or the negated uncertainty for SNNE. Because AUROC is
rank-based, that flip is exactly  ovr = 1 - roc_auc_score(y, P(wrong)),
independent of the score's scale. Net effect: ALL FOUR columns read
"higher = better separation of this category from the other three"; >0.5 beats
chance everywhere. (The IH/IL columns are unchanged from the raw-P(wrong) version;
only CH/CL are mirrored around 0.5.)

Score sources (each method scored on its own aligned coverage, exactly as the
pairwise tables are):
  LapEigvals/AttnEigvals/AttnLogDet/classifier_a : per_example_scores.csv
  SNNE (best-by-all per pair)                    : snne_baseline/scores/<pair>.csv
  2-axis combos (entropy_only ... entropy+lap+hidden, fixed k=10)
  hs_lr (wide/narrow/peak_only)                  : rebuilt from saved classifiers

Output:
  results/per_category_analysis/per_category_ovr_auroc.csv

Usage:
  cd /home/kalashkala/recovery-gaps-experiment
  /root/miniconda3/envs/semantic_uncertainty/bin/python scripts/per_category_ovr_auroc.py
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
OUT_DIR = REPO_ROOT / "results" / "per_category_analysis"

sys.path.insert(0, str(SCRIPTS / "feature_analysis"))
sys.path.insert(0, str(SCRIPTS / "classifier"))

# 2-axis loaders (reused verbatim so combo scores match two_axis_results.csv)
import train_2axis as TA  # noqa: E402
# hs_lr feature builder
from build_features import (  # noqa: E402
    BASE as HS_BASE,
    build_feature_vector,
    extract_split,
    get_buckets,
)

PAIRS = [(m, d) for m in ("llama", "mistral", "qwen", "gemma") for d in ("sciq", "triviaqa", "math")]
PAIR_NAMES = [f"{m}_{d}" for m, d in PAIRS]
CATS = ["IH", "CH", "IL", "CL"]
CAT_SHORT = {"incorrect_high": "IH", "incorrect_low": "IL",
             "correct_high": "CH", "correct_low": "CL"}

SNNE_METHODS = ["num_set", "lexical_sim", "sum_eigv", "degree", "eccentricity", "luq", "snne"]


CORRECT_TARGETS = {"CH", "CL"}  # oriented toward correctness (Option B)


def ovr_auroc(scores: np.ndarray, cats: np.ndarray, target: str) -> float:
    y = (cats == target).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    a = float(roc_auc_score(y, scores))
    # For correct targets, orient toward P(correct): rank-based, so this is the
    # exact complement of the P(wrong) AUROC and is scale-independent (works for
    # SNNE's unbounded uncertainty too).
    return (1.0 - a) if target in CORRECT_TARGETS else a


def all_ovr(scores, cats) -> dict:
    scores = np.asarray(scores, dtype=float)
    cats = np.asarray(cats)
    return {f"ovr_{t}": ovr_auroc(scores, cats, t) for t in CATS}


# ─── source 1: baseline methods from per_example_scores.csv ───────────────────
def records_from_per_example() -> dict:
    """{(pair, method): (scores, cats_short)} for the 4 cached baseline methods."""
    path = OUT_DIR / "per_example_scores.csv"
    df = pd.read_csv(path)
    df["cat"] = df["category"].map(CAT_SHORT)
    methods = ["LapEigvals", "AttnEigvals", "AttnLogDet", "classifier_a"]
    out = {}
    for pair, g in df.groupby("pair"):
        g = g.dropna(subset=["cat"])
        for m in methods:
            out[(pair, m)] = (g[m].to_numpy(float), g["cat"].to_numpy())
    return out


# ─── source 2: SNNE best-by-all per pair ──────────────────────────────────────
def best_snne_per_pair() -> dict:
    df = pd.read_csv(REPO_ROOT / "results" / "snne_baseline" / "snne_per_category.csv")
    best = {}
    for pair, g in df.groupby("pair"):
        row = g.loc[g["auroc_all"].astype(float).idxmax()]
        best[pair] = row["method"]
    return best


def records_from_snne(best_methods: dict) -> dict:
    import torch
    out = {}
    score_dir = REPO_ROOT / "results" / "snne_baseline" / "scores"
    diag_dir = REPO_ROOT / "results" / "lapeigvals_baseline" / "diags"
    for pair in PAIR_NAMES:
        method = best_methods.get(pair)
        spath = score_dir / f"{pair}.csv"
        if method is None or not spath.exists():
            continue
        df = pd.read_csv(spath)
        df["id"] = df["id"].astype(str)
        s = (df[method].astype(str)
             .str.replace(r"^tensor\((.*)\)$", r"\1", regex=True).astype(float))
        model, dataset = pair.split("_", 1)
        pt = torch.load(diag_dir / f"{model}_{dataset}_test.pt", weights_only=False)
        id2cat = {str(i): CAT_SHORT.get(c, "?") for i, c in zip(pt["ids"], pt["categories"])}
        cats = df["id"].map(id2cat).to_numpy()
        mask = cats != None  # noqa: E711
        # SNNE score is an uncertainty: higher = more likely incorrect (matches P(wrong))
        out[(pair, f"SNNE (best={method})")] = (s.to_numpy()[mask], cats[mask])
    return out


# ─── source 3: 2-axis combos (fixed k=10), regenerated to dump per-example ─────
def records_from_2axis() -> dict:
    out = {}
    combos = {
        "entropy_only":       (["entropy"], False),
        "lap_only":           ([],          True),
        "hidden_only":        (["hidden"],  False),
        "lap+hidden":         (["hidden"],  True),
        "entropy+lap":        (["entropy"], True),
        "entropy+hidden":     (["entropy", "hidden"], False),
        "entropy+lap+hidden": (["entropy", "hidden"], True),
    }
    for model, dataset in PAIRS:
        pair = f"{model}_{dataset}"
        entropy = TA.load_entropy(pair)
        _, lap_ids_tr, lab_tr, cat_tr = TA.load_lap_raw(model, dataset, "train")
        lap_diags_tr, lap_ids_tr, lab_tr, cat_tr = TA.load_lap_raw(model, dataset, "train")
        lap_diags_te, lap_ids_te, lab_te, cat_te = TA.load_lap_raw(model, dataset, "test")
        hid_tr = TA.load_hidden(model, dataset, "train")
        hid_te = TA.load_hidden(model, dataset, "test")

        k = TA.TOP_K_LAP
        lap_tr = TA.lap_feat_dict(lap_diags_tr, lap_ids_tr, k)
        lap_te = TA.lap_feat_dict(lap_diags_te, lap_ids_te, k)
        static = {"entropy": (entropy, entropy), "hidden": (hid_tr, hid_te)}

        for name, (parts, has_lap) in combos.items():
            base_tr = [static[p][0] for p in parts]
            base_te = [static[p][1] for p in parts]
            tr_dicts = base_tr + ([lap_tr] if has_lap else [])
            te_dicts = base_te + ([lap_te] if has_lap else [])
            tr_ids = TA.common_ids(lab_tr, tr_dicts)
            te_ids = TA.common_ids(lab_te, te_dicts)
            if len(tr_ids) < 30 or len(te_ids) < 20:
                continue
            Xtr = TA.build_matrix(tr_ids, tr_dicts)
            Xte = TA.build_matrix(te_ids, te_dicts)
            ytr = np.array([lab_tr[i] for i in tr_ids])
            pipe = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(**TA.LR_KWARGS))])
            pipe.fit(Xtr, ytr)
            scores = pipe.predict_proba(Xte)[:, 1]
            cats = np.array([CAT_SHORT.get(cat_te[i], "?") for i in te_ids])
            out[(pair, name)] = (scores, cats)
    return out


# ─── source 4: hs_lr (wide/narrow/peak_only) rebuilt from saved classifiers ────
def records_from_hs_lr() -> dict:
    out = {}
    for model, dataset in PAIRS:
        pair = f"{model}_{dataset}"
        test_split = HS_BASE / f"ranking_experiment_{model}_{dataset}" / "splits" / "test.jsonl"
        id2cat = {s["id"]: s.get("category") for s in
                  (json.loads(l) for l in test_split.open())}
        for variant in ("wide", "narrow", "peak_only"):
            buckets = get_buckets(variant, model, dataset)
            mid, late = buckets["mid"], buckets["late"]
            layers = sorted(set(mid + late))
            clf_root = (HS_BASE / f"ranking_experiment_{model}_{dataset}"
                        / "classifier" / variant)
            stats_path = clf_root / "layer_stats.pkl"
            clf_path = clf_root / "default" / "classifier.pkl"
            if not (stats_path.exists() and clf_path.exists()):
                continue
            test = extract_split(model, dataset, "test", layers=layers)
            with stats_path.open("rb") as f:
                layer_stats = pickle.load(f)
            with clf_path.open("rb") as f:
                clf = pickle.load(f)
            X = build_feature_vector(test.greedy_hidden, test.greedy_log_probs,
                                     layer_stats, mid, late)
            proba = clf.predict_proba(X)[:, 1]
            cats = np.array([CAT_SHORT.get(id2cat.get(sid), "?") for sid in test.greedy_ids])
            keep = cats != "?"
            out[(pair, f"hs_lr ({variant})")] = (proba[keep], cats[keep])
    return out


# ─── ordering for the report ──────────────────────────────────────────────────
METHOD_ORDER = [
    "LapEigvals", "AttnEigvals", "AttnLogDet", "classifier_a",
    "hs_lr (wide)", "hs_lr (narrow)", "hs_lr (peak_only)",
    "__SNNE__",  # placeholder; real key has the method name
    "entropy_only", "lap_only", "hidden_only",
    "entropy+lap", "entropy+hidden", "lap+hidden", "entropy+lap+hidden",
]


def main():
    print("collecting per-example scores ...")
    recs = {}
    recs.update(records_from_per_example()); print("  baselines done")
    recs.update(records_from_snne(best_snne_per_pair())); print("  SNNE done")
    recs.update(records_from_2axis()); print("  2-axis done")
    recs.update(records_from_hs_lr()); print("  hs_lr done")

    rows = []
    for (pair, method), (scores, cats) in recs.items():
        n_by = {t: int((cats == t).sum()) for t in CATS}
        rows.append({"pair": pair, "method": method, "n": len(cats),
                     **{f"n_{t}": n_by[t] for t in CATS},
                     **all_ovr(scores, cats)})

    rows.sort(key=lambda r: (PAIR_NAMES.index(r["pair"]) if r["pair"] in PAIR_NAMES else 99,
                             r["method"]))

    cols = (["pair", "method", "n"] + [f"n_{t}" for t in CATS]
            + [f"ovr_{t}" for t in CATS])
    out_path = OUT_DIR / "per_category_ovr_auroc.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\n[written] {out_path}  ({len(rows)} rows)")

    # printed per-pair tables + grand means
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "—"

    by_method_means = defaultdict(lambda: defaultdict(list))
    for pair in PAIR_NAMES:
        print(f"\n=== {pair} ===")
        print(f"{'method':24s} {'IHvR':>7s} {'CHvR':>7s} {'ILvR':>7s} {'CLvR':>7s}")
        prs = [r for r in rows if r["pair"] == pair]
        for r in prs:
            print(f"{r['method']:24s} {fmt(r['ovr_IH']):>7s} {fmt(r['ovr_CH']):>7s} "
                  f"{fmt(r['ovr_IL']):>7s} {fmt(r['ovr_CL']):>7s}")
            base = r["method"].split(" (best=")[0] if r["method"].startswith("SNNE") else r["method"]
            for t in CATS:
                v = r[f"ovr_{t}"]
                if isinstance(v, float) and not np.isnan(v):
                    by_method_means[base][t].append(v)

    print(f"\n=== MEAN across {len(PAIR_NAMES)} pairs (one-vs-rest) ===")
    print(f"{'method':24s} {'IHvR':>7s} {'CHvR':>7s} {'ILvR':>7s} {'CLvR':>7s}")
    for m in by_method_means:
        means = {t: np.mean(by_method_means[m][t]) if by_method_means[m][t] else float("nan")
                 for t in CATS}
        print(f"{m:24s} {fmt(means['IH']):>7s} {fmt(means['CH']):>7s} "
              f"{fmt(means['IL']):>7s} {fmt(means['CL']):>7s}")


if __name__ == "__main__":
    main()
