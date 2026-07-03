"""SNNE-axis 2-axis combos, band-stratified (Part-9 format).

Extends the entropy/lap/hidden combos (CATEGORY_REPORT Part 9) with a 4th feature
axis: the best per-pair SNNE uncertainty score. Trains LogisticRegression on the
TRAIN split and evaluates band-stratified AUROC on the TEST split — IDENTICAL
protocol to per_category_ovr/band_auroc.records_from_2axis, so the reproduced
lap+hidden column validates against the Part-9 baseline.

SNNE feature source (NEW, train-split regenerated under Plan B):
    train: results/snne_baseline/scores_train/<pair>.csv
    test : results/snne_baseline/scores/<pair>.csv   (existing cache)
The per-pair "best" SNNE method is chosen by TRAIN AUROC only (no test peeking).

Combos:
    existing : entropy_only, lap_only, hidden_only, entropy+lap, entropy+hidden,
               lap+hidden, entropy+lap+hidden
    new      : snne_only, snne+hidden, snne+lap, lap+hidden+snne,
               entropy+lap+hidden+snne

Output:
    results/snne_combo/snne_combo_band.csv         (per-pair)
    results/snne_combo/SNNE_COMBO_REPORT.md        (means across pairs, vs lap+hidden)

Usage:
    /root/miniconda3/envs/semantic_uncertainty/bin/python scripts/feature_analysis/snne_combo_band.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS / "feature_analysis"))
sys.path.insert(0, str(SCRIPTS))

import train_2axis as TA  # entropy/lap/hidden loaders + LR_KWARGS + TOP_K_LAP
import per_category_band_auroc as BAND  # band_auroc orientation + CONTRASTS + all_band

PAIRS = TA.PAIRS                      # [(model, dataset)] x12
CAT_SHORT = TA.CAT_SHORT
SCORE_TEST_DIR = REPO_ROOT / "results" / "snne_baseline" / "scores"
SCORE_TRAIN_DIR = REPO_ROOT / "results" / "snne_baseline" / "scores_train"
OUT_DIR = REPO_ROOT / "results" / "snne_combo"
SNNE_METHODS = ["num_set", "lexical_sim", "sum_eigv", "degree", "eccentricity", "luq", "snne"]

# Part-9 reference (means across 12 pairs) for the lap+hidden sanity check.
PART9_LAPHIDDEN = {"IHvC": 0.7446, "ILvC": 0.8311, "CHvI": 0.8377, "CLvI": 0.7178}

COMBOS = {
    # name: (static_parts, has_lap, has_snne)
    "entropy_only":            (["entropy"],            False, False),
    "lap_only":                ([],                     True,  False),
    "hidden_only":             (["hidden"],             False, False),
    "entropy+lap":             (["entropy"],            True,  False),
    "entropy+hidden":          (["entropy", "hidden"],  False, False),
    "lap+hidden":              (["hidden"],             True,  False),
    "entropy+lap+hidden":      (["entropy", "hidden"],  True,  False),
    "snne_only":               (["snne"],               False, True),
    "snne+hidden":             (["snne", "hidden"],     False, True),
    "snne+lap":                (["snne"],               True,  True),
    "lap+hidden+snne":         (["snne", "hidden"],     True,  True),
    "entropy+lap+hidden+snne": (["entropy", "snne", "hidden"], True, True),
}


def _coerce(series):
    return (series.astype(str).str.replace(r"^tensor\((.*)\)$", r"\1", regex=True).astype(float))


def load_snne_scores(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    for m in SNNE_METHODS:
        df[m] = _coerce(df[m])
    return df


def best_snne_method_on_train(train_df: pd.DataFrame) -> str:
    """Pick the SNNE method with highest TRAIN AUROC(is_incorrect, score)."""
    y = 1 - train_df["label"].astype(int).to_numpy()  # 1 = incorrect
    best, best_auc = SNNE_METHODS[0], -1.0
    for m in SNNE_METHODS:
        s = train_df[m].to_numpy()
        if len(np.unique(y)) < 2 or np.isnan(s).all():
            continue
        a = roc_auc_score(y, s)
        if a > best_auc:
            best_auc, best = a, m
    return best, best_auc


def snne_feat_dict(df: pd.DataFrame, method: str) -> dict[str, np.ndarray]:
    return {i: np.array([float(v)], dtype=np.float32)
            for i, v in zip(df["id"], df[method].to_numpy())}


def process_pair(model: str, dataset: str) -> tuple[dict, dict]:
    pair = f"{model}_{dataset}"
    # ── feature sources (train + test) ──
    entropy = TA.load_entropy(pair)                       # id -> [1]  (same dict both splits)
    lap_diags_tr, lap_ids_tr, lab_tr, cat_tr = TA.load_lap_raw(model, dataset, "train")
    lap_diags_te, lap_ids_te, lab_te, cat_te = TA.load_lap_raw(model, dataset, "test")
    hid_tr = TA.load_hidden(model, dataset, "train")
    hid_te = TA.load_hidden(model, dataset, "test")
    lap_tr = TA.lap_feat_dict(lap_diags_tr, lap_ids_tr, TA.TOP_K_LAP)
    lap_te = TA.lap_feat_dict(lap_diags_te, lap_ids_te, TA.TOP_K_LAP)

    snne_tr_df = load_snne_scores(SCORE_TRAIN_DIR / f"{pair}.csv")
    snne_te_df = load_snne_scores(SCORE_TEST_DIR / f"{pair}.csv")
    snne_tr = snne_te = None
    best_method = best_auc = None
    if snne_tr_df is not None and snne_te_df is not None:
        best_method, best_auc = best_snne_method_on_train(snne_tr_df)
        snne_tr = snne_feat_dict(snne_tr_df, best_method)
        snne_te = snne_feat_dict(snne_te_df, best_method)

    static_tr = {"entropy": entropy, "hidden": hid_tr, "snne": snne_tr}
    static_te = {"entropy": entropy, "hidden": hid_te, "snne": snne_te}

    pair_rows = {}
    for name, (parts, has_lap, has_snne) in COMBOS.items():
        if has_snne and snne_tr is None:
            continue
        tr_dicts = [static_tr[p] for p in parts] + ([lap_tr] if has_lap else [])
        te_dicts = [static_te[p] for p in parts] + ([lap_te] if has_lap else [])
        tr_ids = TA.common_ids(lab_tr, tr_dicts)
        te_ids = TA.common_ids(lab_te, te_dicts)
        if len(tr_ids) < 30 or len(te_ids) < 20:
            continue
        Xtr = TA.build_matrix(tr_ids, tr_dicts)
        Xte = TA.build_matrix(te_ids, te_dicts)
        ytr = np.array([lab_tr[i] for i in tr_ids])       # 1 = incorrect
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(**TA.LR_KWARGS))])
        pipe.fit(Xtr, ytr)
        scores = pipe.predict_proba(Xte)[:, 1]            # P(incorrect)
        cats = np.array([CAT_SHORT.get(cat_te[i], "?") for i in te_ids])
        pair_rows[name] = BAND.all_band(scores, cats)
        pair_rows[name]["n_test"] = len(te_ids)
    meta = {"best_snne": best_method, "best_snne_train_auc": best_auc}
    return pair_rows, meta


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_cols = ["IHvC", "ILvC", "CHvI", "CLvI", "bal_incorrect", "bal_correct"]
    all_rows = []
    means = defaultdict(lambda: defaultdict(list))
    print(f"{'pair':16s} best_snne(train_auc)")
    for model, dataset in PAIRS:
        pair = f"{model}_{dataset}"
        pair_rows, meta = process_pair(model, dataset)
        bm = meta["best_snne"]
        print(f"{pair:16s} {bm}({meta['best_snne_train_auc']:.3f})" if bm else f"{pair:16s} [no SNNE]")
        for name, r in pair_rows.items():
            all_rows.append({"pair": pair, "method": name, **r})
            for k in metric_cols:
                if not np.isnan(r[k]):
                    means[name][k].append(r[k])

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "snne_combo_band.csv", index=False)

    # ── means across pairs ──
    order = list(COMBOS)
    lap_hidden_means = {k: np.mean(means["lap+hidden"][k]) for k in metric_cols}

    lines = []
    lines.append("# SNNE-axis combos — band-stratified AUROC (means across 12 pairs)\n")
    lines.append("Protocol: LR trained on train split, evaluated on test split; per-pair "
                 "then averaged. Same machinery as CATEGORY_REPORT Part 9, with a 4th axis = "
                 "best per-pair SNNE score (method chosen by TRAIN AUROC).\n")
    lines.append("| method | IHvC | ILvC | CHvI | CLvI | bal-I | bal-C | ΔbalI | ΔbalC |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    hdr = f"{'method':24s} {'IHvC':>7s} {'ILvC':>7s} {'CHvI':>7s} {'CLvI':>7s} {'balI':>7s} {'balC':>7s}  {'dBalI':>7s} {'dBalC':>7s}"
    print("\n=== MEAN across pairs (band-stratified) ===")
    print(hdr)
    for name in order:
        if not means[name]["IHvC"]:
            continue
        m = {k: np.mean(means[name][k]) for k in metric_cols}
        dI = m["bal_incorrect"] - lap_hidden_means["bal_incorrect"]
        dC = m["bal_correct"] - lap_hidden_means["bal_correct"]
        print(f"{name:24s} {m['IHvC']:7.4f} {m['ILvC']:7.4f} {m['CHvI']:7.4f} "
              f"{m['CLvI']:7.4f} {m['bal_incorrect']:7.4f} {m['bal_correct']:7.4f}  "
              f"{dI:+7.4f} {dC:+7.4f}")
        lines.append(f"| {name} | {m['IHvC']:.4f} | {m['ILvC']:.4f} | {m['CHvI']:.4f} | "
                     f"{m['CLvI']:.4f} | {m['bal_incorrect']:.4f} | {m['bal_correct']:.4f} | "
                     f"{dI:+.4f} | {dC:+.4f} |")

    # sanity: reproduced lap+hidden vs Part-9
    print("\n=== sanity: reproduced lap+hidden vs CATEGORY_REPORT Part-9 ===")
    for k in ["IHvC", "ILvC", "CHvI", "CLvI"]:
        print(f"  {k}: repro={lap_hidden_means[k]:.4f}  part9={PART9_LAPHIDDEN[k]:.4f}  "
              f"Δ={lap_hidden_means[k]-PART9_LAPHIDDEN[k]:+.4f}")
    lines.append("\n## Sanity: reproduced lap+hidden vs Part-9\n")
    lines.append("| contrast | reproduced | Part-9 | Δ |")
    lines.append("|---|---|---|---|")
    for k in ["IHvC", "ILvC", "CHvI", "CLvI"]:
        lines.append(f"| {k} | {lap_hidden_means[k]:.4f} | {PART9_LAPHIDDEN[k]:.4f} | "
                     f"{lap_hidden_means[k]-PART9_LAPHIDDEN[k]:+.4f} |")

    (OUT_DIR / "SNNE_COMBO_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\n[written] {OUT_DIR/'snne_combo_band.csv'}")
    print(f"[written] {OUT_DIR/'SNNE_COMBO_REPORT.md'}")


if __name__ == "__main__":
    main()