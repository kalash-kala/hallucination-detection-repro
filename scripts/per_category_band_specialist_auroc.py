"""
Band-SPECIALIST correctness AUROC — two dedicated LRs per method per pair.

Unlike per_category_pairwise_auroc.py (which slices a GENERALIST score, trained on
the full correct-vs-incorrect task, down to one confidence band), this script
TRAINS a fresh classifier whose training data is restricted to a single
confidence band, then tests within that same band:

  HI band :  fit on {IH, CH} ∩ train  (y=1 incorrect), test on {IH, CH} ∩ test
  LO band :  fit on {IL, CL} ∩ train                  , test on {IL, CL} ∩ test

Question answered: within a fixed confidence band, does a band-specialist LR
separate correctness better than the generalist does?  Comparison column
`auroc_generalist` is the matching generalist number (IHvCH for HI, ILvCL for LO)
copied from per_category_pairwise_auroc.csv; `delta = specialist - generalist`.

ORIENTATION: positive class = incorrect in both bands → raw roc_auc_score, no flip.

Per-method convention is kept identical to the overall LR experiments:
  * eigval baselines (LapEigvals/AttnEigvals/AttnLogDet): PCA→LR, CV top_k/PCA∈{None,100}
  * every combo / sink / hs_lr                          : StandardScaler→LR, no PCA
  * entropy_only, SNNE(best)                            : TRADITIONAL raw-score AUROC
                                                          (unsupervised, no training)
  * classifier_a                                        : dropped (no reloadable features)

Guard: require >= MIN_PER_CLASS of each class in BOTH the band-train and the
band-test subset, else emit NaN (protects the thin CL cells in math/gemma).

Output:
  results/per_category_analysis/per_category_band_specialist_auroc.csv

Usage:
  cd /home/kalashkala/recovery-gaps-experiment
  /root/miniconda3/envs/semantic_uncertainty/bin/python -u \
      scripts/per_category_band_specialist_auroc.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
OUT_DIR = REPO_ROOT / "results" / "per_category_analysis"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "feature_analysis"))
sys.path.insert(0, str(SCRIPTS / "sinkhole"))
sys.path.insert(0, str(SCRIPTS / "classifier"))
sys.path.insert(0, str(SCRIPTS / "lapeigvals_baseline"))

import per_category_ovr_auroc as OVR   # noqa: E402
import train_2axis as TA               # noqa: E402
import snne_combo_band as SNNE_CB      # noqa: E402
import exp3_feature as E3              # noqa: E402
from build_features import (           # noqa: E402
    BASE as HS_BASE, build_feature_vector, extract_split, get_buckets,
)
from lapeigvals_features import (      # noqa: E402
    get_attn_log_det, get_attn_eigvals_per_head_topk,
    get_laplacian_eigvals_per_head_topk,
)
from train_lapeigvals import make_pipeline, PCA_GRID, TOP_K_EIGVALS, SEED  # noqa: E402
import pickle  # noqa: E402

PAIRS = OVR.PAIRS
PAIR_NAMES = OVR.PAIR_NAMES
CAT_SHORT = OVR.CAT_SHORT
BANDS = {"HI": ("IH", "CH"), "LO": ("IL", "CL")}  # (incorrect, correct)
MIN_PER_CLASS = 20
LR_KWARGS = TA.LR_KWARGS                                  # combos / sink
HS_LR_PARAMS = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}  # hs_lr default


# ─── generic band-specialist LR on a pre-built id→vector feature dict ─────────
def specialist_lr(feat_tr: dict, feat_te: dict, lab_tr: dict,
                  cat_tr: dict, cat_te: dict, lr_kwargs=LR_KWARGS) -> dict:
    """feat_* : id -> concatenated feature vector. lab_tr : id -> 1(incorrect)/0.
    Returns {band: (auroc, n_tr_pos, n_tr_neg, n_te_pos, n_te_neg)}."""
    out = {}
    for band, (inc, cor) in BANDS.items():
        tr_ids = [i for i in feat_tr if i in lab_tr and cat_tr.get(i) in (inc, cor)]
        te_ids = [i for i in feat_te if cat_te.get(i) in (inc, cor)]
        ytr = np.array([lab_tr[i] for i in tr_ids])
        yte = np.array([1 if cat_te[i] == inc else 0 for i in te_ids])
        out[band] = _fit_or_nan(
            lambda: np.stack([feat_tr[i] for i in tr_ids]),
            lambda: np.stack([feat_te[i] for i in te_ids]),
            ytr, yte,
            fit_fn=lambda Xtr, y: Pipeline(
                [("sc", StandardScaler()),
                 ("lr", LogisticRegression(**lr_kwargs))]).fit(Xtr, y),
        )
    return out


def _counts_ok(ytr, yte) -> bool:
    def mn(y):
        return min(int((y == 1).sum()), int((y == 0).sum()))
    return mn(ytr) >= MIN_PER_CLASS and mn(yte) >= MIN_PER_CLASS


def _fit_or_nan(build_Xtr, build_Xte, ytr, yte, fit_fn):
    """Shared guard+fit; returns (auroc, n_tr_pos, n_tr_neg, n_te_pos, n_te_neg)."""
    n_tr_pos, n_tr_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    n_te_pos, n_te_neg = int((yte == 1).sum()), int((yte == 0).sum())
    if not _counts_ok(ytr, yte):
        return (float("nan"), n_tr_pos, n_tr_neg, n_te_pos, n_te_neg)
    model = fit_fn(build_Xtr(), ytr)
    proba = model.predict_proba(build_Xte())[:, 1]
    return (float(roc_auc_score(yte, proba)),
            n_tr_pos, n_tr_neg, n_te_pos, n_te_neg)


def combo_dict(part_dicts: list[dict]) -> dict:
    """id -> concat of features, over the intersection of ids present in all parts."""
    ids = set(part_dicts[0])
    for d in part_dicts[1:]:
        ids &= set(d)
    return {i: np.concatenate([d[i] for d in part_dicts]) for i in ids}


# ─── per-pair processing ─────────────────────────────────────────────────────
def process_pair(model: str, dataset: str, rows: list):
    pair = f"{model}_{dataset}"
    print(f"\n=== {pair} ===", flush=True)

    # shared raw loads
    entropy = TA.load_entropy(pair)
    lap_diags_tr, lap_ids_tr, lab_tr, cat_full_tr = TA.load_lap_raw(model, dataset, "train")
    lap_diags_te, lap_ids_te, lab_te, cat_full_te = TA.load_lap_raw(model, dataset, "test")
    hid_tr = TA.load_hidden(model, dataset, "train")
    hid_te = TA.load_hidden(model, dataset, "test")
    cat_tr = {i: CAT_SHORT.get(c, "?") for i, c in cat_full_tr.items()}
    cat_te = {i: CAT_SHORT.get(c, "?") for i, c in cat_full_te.items()}
    lap_tr = TA.lap_feat_dict(lap_diags_tr, lap_ids_tr, TA.TOP_K_LAP)
    lap_te = TA.lap_feat_dict(lap_diags_te, lap_ids_te, TA.TOP_K_LAP)

    def emit(method, res):
        for band, (a, ntp, ntn, tep, ten) in res.items():
            rows.append({"pair": pair, "method": method, "band": band,
                         "n_tr_pos": ntp, "n_tr_neg": ntn,
                         "n_te_pos": tep, "n_te_neg": ten,
                         "auroc_specialist": a})
            print(f"  {method:26s} {band}  auroc={_fmt(a)}  "
                  f"(tr {ntp}/{ntn}, te {tep}/{ten})", flush=True)

    # ── 1. StandardScaler→LR combos (non-SNNE) ──────────────────────────────
    static_tr = {"entropy": entropy, "hidden": hid_tr, "lap": lap_tr}
    static_te = {"entropy": entropy, "hidden": hid_te, "lap": lap_te}
    NONSNNE = {
        "lap_only":           ["lap"],
        "hidden_only":        ["hidden"],
        "lap+hidden":         ["hidden", "lap"],
        "entropy+lap":        ["entropy", "lap"],
        "entropy+hidden":     ["entropy", "hidden"],
        "entropy+lap+hidden": ["entropy", "hidden", "lap"],
    }
    for name, parts in NONSNNE.items():
        ft = combo_dict([static_tr[p] for p in parts])
        fe = combo_dict([static_te[p] for p in parts])
        emit(name, specialist_lr(ft, fe, lab_tr, cat_tr, cat_te))

    # ── 2. SNNE combos (best measure RE-SELECTED on band-train) ─────────────
    snne_tr_df = SNNE_CB.load_snne_scores(SNNE_CB.SCORE_TRAIN_DIR / f"{pair}.csv")
    snne_te_df = SNNE_CB.load_snne_scores(SNNE_CB.SCORE_TEST_DIR / f"{pair}.csv")
    if snne_tr_df is not None and snne_te_df is not None:
        snne_tr_df = snne_tr_df.copy(); snne_tr_df["id"] = snne_tr_df["id"].astype(str)
        snne_te_df = snne_te_df.copy(); snne_te_df["id"] = snne_te_df["id"].astype(str)
        SNNE_COMBOS = {
            "snne_only":               (["snne"], False),
            "snne+hidden":             (["snne", "hidden"], False),
            "snne+lap":                (["snne"], True),
            "lap+hidden+snne":         (["snne", "hidden"], True),
            "entropy+lap+hidden+snne": (["entropy", "snne", "hidden"], True),
        }
        for name, (parts, has_lap) in SNNE_COMBOS.items():
            res = {}
            for band, (inc, cor) in BANDS.items():
                # re-select best SNNE measure on the band-train subset only
                band_tr_ids = {i for i in cat_tr if cat_tr[i] in (inc, cor)}
                df_band = snne_tr_df[snne_tr_df["id"].isin(band_tr_ids)]
                if len(df_band) < 2 * MIN_PER_CLASS:
                    res[band] = (float("nan"), 0, 0, 0, 0)
                    continue
                best_m, _ = SNNE_CB.best_snne_method_on_train(df_band)
                snne_tr = SNNE_CB.snne_feat_dict(snne_tr_df, best_m)
                snne_te = SNNE_CB.snne_feat_dict(snne_te_df, best_m)
                st = {"entropy": entropy, "hidden": hid_tr, "lap": lap_tr, "snne": snne_tr}
                se = {"entropy": entropy, "hidden": hid_te, "lap": lap_te, "snne": snne_te}
                plist = list(parts) + (["lap"] if has_lap else [])
                ft = combo_dict([st[p] for p in plist])
                fe = combo_dict([se[p] for p in plist])
                one = specialist_lr(ft, fe, lab_tr, cat_tr, cat_te)
                res[band] = one[band]
            emit(name, res)

    # ── 3. Exp3 sink-score / value-norm ablation (2x2: attn-only vs +hidden) ─
    if (E3.VN_DIR / f"{model}_{dataset}_test.pt").exists():
        tr = E3.build_pair(model, dataset, "train", E3.K)
        te = E3.build_pair(model, dataset, "test", E3.K)
        lab_e3 = {i: int(tr["labels"][i]) for i in tr["ids"]}
        cat_e3_tr = {i: tr["cats"][i] for i in tr["ids"]}
        cat_e3_te = {i: te["cats"][i] for i in te["ids"]}

        def e3_dict(split_dict, key, with_hidden, hid):
            if with_hidden:
                return {i: np.concatenate([split_dict[key][i], hid[i]])
                        for i in split_dict["ids"] if i in hid}
            return {i: split_dict[key][i] for i in split_dict["ids"]}

        E3_METHODS = [
            ("sink-only", "sink", False),
            ("sink×‖V‖-only", "f3", False),
            ("sink+hidden", "sink", True),
            ("sink×||V||+hidden", "f3", True),
        ]
        for name, key, with_hidden in E3_METHODS:
            ft = e3_dict(tr, key, with_hidden, hid_tr)
            fe = e3_dict(te, key, with_hidden, hid_te)
            emit(name, specialist_lr(ft, fe, lab_e3, cat_e3_tr, cat_e3_te))

    # ── 4. hs_lr (wide / narrow / peak_only): StandardScaler→LR default ─────
    for variant in ("wide", "narrow", "peak_only"):
        res = _hs_lr_bands(model, dataset, variant)
        if res is not None:
            emit(f"hs_lr ({variant})", res)

    # ── 5. eigval baselines: PCA→LR with CV top_k/PCA on band-train ─────────
    emit("AttnLogDet", _eigval_bands("attn_log_det", model, dataset))
    emit("AttnEigvals", _eigval_bands("attn_eigvals", model, dataset))
    emit("LapEigvals",  _eigval_bands("lap_eigvals", model, dataset))

    # ── 6. TRADITIONAL raw-score AUROC (no training) ────────────────────────
    emit("entropy_only", _raw_score_bands(
        {i: float(v[0]) for i, v in entropy.items()}, cat_te))
    best_snne = OVR.best_snne_per_pair().get(pair)
    if best_snne is not None and snne_te_df is not None:
        raw = {i: float(v) for i, v in zip(snne_te_df["id"], snne_te_df[best_snne])}
        emit("SNNE", _raw_score_bands(raw, cat_te))


def _hs_lr_bands(model, dataset, variant):
    buckets = get_buckets(variant, model, dataset)
    mid, late = buckets["mid"], buckets["late"]
    layers = sorted(set(mid + late))
    clf_root = (HS_BASE / f"ranking_experiment_{model}_{dataset}" / "classifier" / variant)
    stats_path = clf_root / "layer_stats.pkl"
    if not stats_path.exists():
        return None
    with stats_path.open("rb") as f:
        layer_stats = pickle.load(f)
    id2cat = {}
    for split in ("train", "test"):
        sp = HS_BASE / f"ranking_experiment_{model}_{dataset}" / "splits" / f"{split}.jsonl"
        for s in (json.loads(l) for l in sp.open()):
            id2cat[s["id"]] = CAT_SHORT.get(s.get("category"), "?")
    out = {}
    tr = extract_split(model, dataset, "train", layers=layers)
    te = extract_split(model, dataset, "test", layers=layers)
    Xtr_all = build_feature_vector(tr.greedy_hidden, tr.greedy_log_probs, layer_stats, mid, late)
    Xte_all = build_feature_vector(te.greedy_hidden, te.greedy_log_probs, layer_stats, mid, late)
    cat_tr = np.array([id2cat.get(i, "?") for i in tr.greedy_ids])
    cat_te = np.array([id2cat.get(i, "?") for i in te.greedy_ids])
    for band, (inc, cor) in BANDS.items():
        m_tr = np.isin(cat_tr, [inc, cor]); m_te = np.isin(cat_te, [inc, cor])
        ytr = (cat_tr[m_tr] == inc).astype(int)
        yte = (cat_te[m_te] == inc).astype(int)
        out[band] = _fit_or_nan(
            lambda: Xtr_all[m_tr], lambda: Xte_all[m_te], ytr, yte,
            fit_fn=lambda X, y: Pipeline(
                [("sc", StandardScaler()),
                 ("lr", LogisticRegression(**HS_LR_PARAMS))]).fit(X, y))
    return out


def _eigval_bands(variant, model, dataset):
    """Band-specialist PCA→LR for one eigval baseline, CV-selecting top_k/PCA on
    the band-train subset (mirrors train_lapeigvals, restricted to the band)."""
    diag_dir = REPO_ROOT / "results" / "lapeigvals_baseline" / "diags"
    tr = torch.load(diag_dir / f"{model}_{dataset}_train.pt", weights_only=False)
    te = torch.load(diag_dir / f"{model}_{dataset}_test.pt", weights_only=False)
    key = "attn_diags" if variant != "lap_eigvals" else "lap_diags"
    dtr = [d.float() for d in tr[key]]
    dte = [d.float() for d in te[key]]
    ytr_all = tr["labels"].numpy(); yte_all = te["labels"].numpy()
    ctr = np.array([CAT_SHORT.get(c, "?") for c in tr["categories"]])
    cte = np.array([CAT_SHORT.get(c, "?") for c in te["categories"]])
    min_seq = min(d.size(-1) for d in dtr + dte)
    if variant == "attn_log_det":
        top_ks = [None]
    else:
        top_ks = [k for k in TOP_K_EIGVALS if k <= min_seq] or [min_seq]

    def build(k):
        if variant == "attn_log_det":
            return get_attn_log_det(dtr).numpy(), get_attn_log_det(dte).numpy()
        b = (get_attn_eigvals_per_head_topk if variant == "attn_eigvals"
             else get_laplacian_eigvals_per_head_topk)
        return b(dtr, top_k=k).numpy(), b(dte, top_k=k).numpy()

    feats = {k: build(k) for k in top_ks}  # cache the (Xtr_full, Xte_full) per k

    out = {}
    for band, (inc, cor) in BANDS.items():
        m_tr = np.isin(ctr, [inc, cor]); m_te = np.isin(cte, [inc, cor])
        ytr = ytr_all[m_tr]; yte = yte_all[m_te]
        n_tr_pos, n_tr_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
        n_te_pos, n_te_neg = int((yte == 1).sum()), int((yte == 0).sum())
        if not _counts_ok(ytr, yte):
            out[band] = (float("nan"), n_tr_pos, n_tr_neg, n_te_pos, n_te_neg)
            continue
        best = (-1.0, None)  # (cv_auc, (Xtr,Xte,pca))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for k, (Xtr_full, Xte_full) in feats.items():
            Xtr_b, Xte_b = Xtr_full[m_tr], Xte_full[m_te]
            for pca in PCA_GRID:
                pipe = make_pipeline(pca, Xtr_b.shape[1], Xtr_b.shape[0])
                cv = float(np.mean(cross_val_score(
                    pipe, Xtr_b, ytr, cv=skf, scoring="roc_auc", n_jobs=1)))
                if cv > best[0]:
                    best = (cv, (Xtr_b, Xte_b, pca))
        Xtr_b, Xte_b, pca = best[1]
        pipe = make_pipeline(pca, Xtr_b.shape[1], Xtr_b.shape[0]).fit(Xtr_b, ytr)
        proba = pipe.predict_proba(Xte_b)[:, 1]
        out[band] = (float(roc_auc_score(yte, proba)),
                     n_tr_pos, n_tr_neg, n_te_pos, n_te_neg)
    return out


def _raw_score_bands(score: dict, cat_te: dict):
    """Traditional unsupervised method: raw score (higher=incorrect) AUROC on the
    band-test subset. No training; n_tr_* reported as 0."""
    out = {}
    for band, (inc, cor) in BANDS.items():
        ids = [i for i in cat_te if cat_te[i] in (inc, cor) and i in score]
        y = np.array([1 if cat_te[i] == inc else 0 for i in ids])
        s = np.array([score[i] for i in ids])
        n_te_pos, n_te_neg = int((y == 1).sum()), int((y == 0).sum())
        if min(n_te_pos, n_te_neg) < MIN_PER_CLASS or len(set(y)) < 2:
            out[band] = (float("nan"), 0, 0, n_te_pos, n_te_neg)
        else:
            out[band] = (float(roc_auc_score(y, s)), 0, 0, n_te_pos, n_te_neg)
    return out


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "—"


# ─── generalist comparison (from the pairwise CSV) ───────────────────────────
def load_generalist():
    """(pair, method_base, band) -> generalist auroc (IHvCH for HI, ILvCL for LO)."""
    path = OUT_DIR / "per_category_pairwise_auroc.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    g = {}
    for _, r in df.iterrows():
        base = r["method"].split(" (best=")[0]
        g[(r["pair"], base, "HI")] = r["IHvCH"]
        g[(r["pair"], base, "LO")] = r["ILvCL"]
    return g


def main():
    rows = []
    for model, dataset in PAIRS:
        process_pair(model, dataset, rows)

    gen = load_generalist()
    for r in rows:
        base = r["method"].split(" (best=")[0]
        g = gen.get((r["pair"], base, r["band"]), float("nan"))
        r["auroc_generalist"] = g
        a = r["auroc_specialist"]
        r["delta"] = (a - g) if (isinstance(a, float) and isinstance(g, float)
                                 and not np.isnan(a) and not np.isnan(g)) else float("nan")

    cols = ["pair", "method", "band", "n_tr_pos", "n_tr_neg", "n_te_pos", "n_te_neg",
            "auroc_specialist", "auroc_generalist", "delta"]
    out_path = OUT_DIR / "per_category_band_specialist_auroc.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, ""))
                        for c in cols})
    print(f"\n[written] {out_path}  ({len(rows)} rows)", flush=True)

    # mean tables across pairs, per band
    for band in ("HI", "LO"):
        agg = defaultdict(lambda: {"spec": [], "gen": []})
        for r in rows:
            if r["band"] != band:
                continue
            a, g = r["auroc_specialist"], r["auroc_generalist"]
            if isinstance(a, float) and not np.isnan(a):
                agg[r["method"]]["spec"].append(a)
            if isinstance(g, float) and not np.isnan(g):
                agg[r["method"]]["gen"].append(g)
        lbl = "IH vs CH" if band == "HI" else "IL vs CL"
        print(f"\n=== MEAN across pairs — {band} band ({lbl}) ===")
        print(f"{'method':26s} {'specialist':>10s} {'generalist':>10s} {'delta':>8s} {'n':>4s}")
        for m in sorted(agg):
            sp = np.mean(agg[m]["spec"]) if agg[m]["spec"] else float("nan")
            ge = np.mean(agg[m]["gen"]) if agg[m]["gen"] else float("nan")
            dl = sp - ge if not (np.isnan(sp) or np.isnan(ge)) else float("nan")
            print(f"{m:26s} {_fmt(sp):>10s} {_fmt(ge):>10s} {_fmt(dl):>8s} "
                  f"{len(agg[m]['spec']):>4d}")


if __name__ == "__main__":
    main()
