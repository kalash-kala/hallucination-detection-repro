"""
1-vs-1 pairwise AUROC for all hallucination-detection methods.

Four new cell contrasts that fix BOTH correctness and confidence simultaneously:

  IHvCH : incorrect_high vs correct_high   — same confidence band, different correctness
  IHvCL : incorrect_high vs correct_low    — cross-confidence
  ILvCH : incorrect_low  vs correct_high   — cross-confidence
  ILvCL : incorrect_low  vs correct_low    — same confidence band, different correctness

ORIENTATION: All four contrasts have an incorrect category as positive, so the
raw roc_auc_score is used directly (higher = better hallucination detection).

Score sources (identical protocol to per_category_band_auroc.py):
  LapEigvals/AttnEigvals/AttnLogDet/classifier_a : per_example_scores.csv
  SNNE (best per pair)                            : snne_baseline/scores/<pair>.csv
  2-axis combos (entropy/lap/hidden, fixed k=10)  : rebuilt from raw features
  hs_lr (wide/narrow/peak_only)                   : saved classifiers
  sink×||V||+hidden (Exp3)                        : rebuilt from value_norms + hidden
  SNNE combos (snne_only...entropy+lap+hidden+snne): rebuilt from raw features

Output:
  results/per_category_analysis/per_category_pairwise_auroc.csv

Usage:
  cd /home/kalashkala/recovery-gaps-experiment
  /root/miniconda3/envs/semantic_uncertainty/bin/python scripts/per_category_pairwise_auroc.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "feature_analysis"))
sys.path.insert(0, str(SCRIPTS / "sinkhole"))
sys.path.insert(0, str(SCRIPTS / "classifier"))
sys.path.insert(0, str(SCRIPTS / "lapeigvals_baseline"))

import per_category_ovr_auroc as OVR   # noqa: E402 — reuse all record builders
import train_2axis as TA               # noqa: E402
import snne_combo_band as SNNE_CB      # noqa: E402
import exp3_feature as E3              # noqa: E402

PAIR_NAMES = OVR.PAIR_NAMES
CATS = OVR.CATS
INCORRECT = {"IH", "IL"}
CORRECT = {"CH", "CL"}

# All four contrasts have incorrect as positive → no mirroring needed.
PAIRWISE_CONTRASTS = [
    ("IHvCH", "IH", "CH"),
    ("IHvCL", "IH", "CL"),
    ("ILvCH", "IL", "CH"),
    ("ILvCL", "IL", "CL"),
]
OUT_DIR = OVR.OUT_DIR


def pairwise_auroc(scores: np.ndarray, cats: np.ndarray, pos: str, neg: str) -> float:
    mask = np.isin(cats, [pos, neg])
    y = (cats[mask] == pos).astype(int)
    s = scores[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    a = float(roc_auc_score(y, s))
    return (1.0 - a) if pos in CORRECT else a


def all_pairwise(scores, cats) -> dict:
    scores = np.asarray(scores, dtype=float)
    cats = np.asarray(cats)
    return {col: pairwise_auroc(scores, cats, pos, neg)
            for col, pos, neg in PAIRWISE_CONTRASTS}


def records_from_sink_hidden() -> dict:
    """Per-example (scores, cats) for sink×||V||+hidden (Exp3 feature, fixed K=10)."""
    out = {}
    for model, dataset in OVR.PAIRS:
        pair = f"{model}_{dataset}"
        if not (E3.VN_DIR / f"{model}_{dataset}_test.pt").exists():
            print(f"  [skip sink+hidden] {pair}: value norms missing")
            continue
        hid_tr = TA.load_hidden(model, dataset, "train")
        hid_te = TA.load_hidden(model, dataset, "test")
        tr = E3.build_pair(model, dataset, "train", E3.K)
        te = E3.build_pair(model, dataset, "test", E3.K)
        scores, cats = E3.fit_combo(tr, te, "f3", hid_tr, hid_te)
        out[(pair, "sink×||V||+hidden")] = (scores, cats)
    return out


def records_from_sink_ablation() -> dict:
    """Per-example (scores, cats) for the remaining Exp3 sink ablation cells:
    sink-only, sink×‖V‖-only (attention feature alone, no hidden) and
    sink+hidden (raw sink score + hidden). sink×||V||+hidden is covered by
    records_from_sink_hidden() above."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_solo(tr, te, key):
        tr_ids, te_ids = tr["ids"], te["ids"]
        Xtr = np.stack([tr[key][i] for i in tr_ids])
        Xte = np.stack([te[key][i] for i in te_ids])
        ytr = np.array([tr["labels"][i] for i in tr_ids])
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(**E3.LR_KWARGS))])
        pipe.fit(Xtr, ytr)
        sc = pipe.predict_proba(Xte)[:, 1]
        cats = np.array([te["cats"][i] for i in te_ids])
        return sc, cats

    out = {}
    for model, dataset in OVR.PAIRS:
        pair = f"{model}_{dataset}"
        if not (E3.VN_DIR / f"{model}_{dataset}_test.pt").exists():
            print(f"  [skip sink ablation] {pair}: value norms missing")
            continue
        hid_tr = TA.load_hidden(model, dataset, "train")
        hid_te = TA.load_hidden(model, dataset, "test")
        tr = E3.build_pair(model, dataset, "train", E3.K)
        te = E3.build_pair(model, dataset, "test", E3.K)

        scores, cats = fit_solo(tr, te, "sink")
        out[(pair, "sink-only")] = (scores, cats)
        scores, cats = fit_solo(tr, te, "f3")
        out[(pair, "sink×‖V‖-only")] = (scores, cats)
        scores, cats = E3.fit_combo(tr, te, "sink", hid_tr, hid_te)
        out[(pair, "sink+hidden")] = (scores, cats)
    return out


def records_from_snne_combos() -> dict:
    """Per-example (scores, cats) for SNNE-specific combos only.
    The non-SNNE combos are already covered by OVR.records_from_2axis()."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    snne_only = {k: v for k, v in SNNE_CB.COMBOS.items() if v[2]}  # has_snne=True
    out = {}
    for model, dataset in OVR.PAIRS:
        pair = f"{model}_{dataset}"
        snne_tr_df = SNNE_CB.load_snne_scores(SNNE_CB.SCORE_TRAIN_DIR / f"{pair}.csv")
        snne_te_df = SNNE_CB.load_snne_scores(SNNE_CB.SCORE_TEST_DIR / f"{pair}.csv")
        if snne_tr_df is None or snne_te_df is None:
            continue
        best_method, _ = SNNE_CB.best_snne_method_on_train(snne_tr_df)
        snne_tr = SNNE_CB.snne_feat_dict(snne_tr_df, best_method)
        snne_te = SNNE_CB.snne_feat_dict(snne_te_df, best_method)

        entropy = TA.load_entropy(pair)
        lap_diags_tr, lap_ids_tr, lab_tr, cat_tr = TA.load_lap_raw(model, dataset, "train")
        lap_diags_te, lap_ids_te, lab_te, cat_te = TA.load_lap_raw(model, dataset, "test")
        hid_tr = TA.load_hidden(model, dataset, "train")
        hid_te = TA.load_hidden(model, dataset, "test")
        lap_tr = TA.lap_feat_dict(lap_diags_tr, lap_ids_tr, TA.TOP_K_LAP)
        lap_te = TA.lap_feat_dict(lap_diags_te, lap_ids_te, TA.TOP_K_LAP)

        static_tr = {"entropy": entropy, "hidden": hid_tr, "snne": snne_tr}
        static_te = {"entropy": entropy, "hidden": hid_te, "snne": snne_te}

        for name, (parts, has_lap, _) in snne_only.items():
            tr_dicts = [static_tr[p] for p in parts] + ([lap_tr] if has_lap else [])
            te_dicts = [static_te[p] for p in parts] + ([lap_te] if has_lap else [])
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
            cats = np.array([OVR.CAT_SHORT.get(cat_te[i], "?") for i in te_ids])
            out[(pair, name)] = (scores, cats)
    return out


def main():
    print("collecting per-example scores ...")
    recs = {}
    recs.update(OVR.records_from_per_example()); print("  baselines done")
    recs.update(OVR.records_from_snne(OVR.best_snne_per_pair())); print("  SNNE done")
    recs.update(OVR.records_from_2axis()); print("  2-axis done")
    recs.update(OVR.records_from_hs_lr()); print("  hs_lr done")
    recs.update(records_from_sink_hidden()); print("  sink×||V||+hidden done")
    recs.update(records_from_sink_ablation()); print("  sink ablation (sink-only/sink×‖V‖-only/sink+hidden) done")
    recs.update(records_from_snne_combos()); print("  SNNE combos done")

    metric_cols = [col for col, _, _ in PAIRWISE_CONTRASTS]
    rows = []
    for (pair, method), (scores, cats) in recs.items():
        cats = np.asarray(cats)
        n_by = {t: int((cats == t).sum()) for t in CATS}
        rows.append({"pair": pair, "method": method, "n": len(cats),
                     **{f"n_{t}": n_by[t] for t in CATS},
                     **all_pairwise(scores, cats)})

    rows.sort(key=lambda r: (PAIR_NAMES.index(r["pair"]) if r["pair"] in PAIR_NAMES else 99,
                              r["method"]))

    cols = (["pair", "method", "n"] + [f"n_{t}" for t in CATS] + metric_cols)
    out_path = OUT_DIR / "per_category_pairwise_auroc.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})
    print(f"\n[written] {out_path}  ({len(rows)} rows)")

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "—"

    means = defaultdict(lambda: defaultdict(list))
    for pair in PAIR_NAMES:
        print(f"\n=== {pair} ===")
        print(f"{'method':30s} {'IHvCH':>7s} {'IHvCL':>7s} {'ILvCH':>7s} {'ILvCL':>7s}")
        for r in [r for r in rows if r["pair"] == pair]:
            print(f"{r['method']:30s} {fmt(r['IHvCH']):>7s} {fmt(r['IHvCL']):>7s} "
                  f"{fmt(r['ILvCH']):>7s} {fmt(r['ILvCL']):>7s}")
            base = (r["method"].split(" (best=")[0]
                    if r["method"].startswith("SNNE") else r["method"])
            for k in metric_cols:
                v = r[k]
                if isinstance(v, float) and not np.isnan(v):
                    means[base][k].append(v)

    print(f"\n=== MEAN across {len(PAIR_NAMES)} pairs (1v1 pairwise) ===")
    print(f"{'method':30s} {'IHvCH':>7s} {'IHvCL':>7s} {'ILvCH':>7s} {'ILvCL':>7s}")
    for m in means:
        mv = {k: np.mean(means[m][k]) if means[m][k] else float("nan")
              for k in metric_cols}
        print(f"{m:30s} {fmt(mv['IHvCH']):>7s} {fmt(mv['IHvCL']):>7s} "
              f"{fmt(mv['ILvCH']):>7s} {fmt(mv['ILvCL']):>7s}")


if __name__ == "__main__":
    main()
