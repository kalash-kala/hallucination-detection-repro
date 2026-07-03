"""
Per-category AUROC breakdown for all hallucination-detection methods.

Categories: correct_high (CH), correct_low (CL), incorrect_high (IH), incorrect_low (IL).
Label convention: y=1 = hallucination (wrong), y=0 = correct — same as Classifier A / LapEigvals.

Methods analysed:
  - LapEigvals, AttnEigvals, AttnLogDet  (spectral-attention probes)
  - Classifier A                          (distractor-feature LR probe)
  - SNNE (best method per pair)           (requires --snne flag; loads DeBERTa)

Probe training uses the same fixed train/test split used by the ranking experiment
(saved in results/lapeigvals_baseline/diags/ for LapEigvals; reconstructed from
the greedy_sidecar + splits/ for Classifier A).

Per-category metrics reported:
  - mean_score : mean predicted P(hallucination) per category
  - auroc_all  : overall AUROC across all test examples
  - auroc_ih_ch: AUROC restricted to (IH ∪ CH) — can it find high-confidence errors?
  - auroc_il_cl: AUROC restricted to (IL ∪ CL) — can it find low-confidence errors?

Outputs:
  results/per_category_analysis/per_example_scores.csv
  results/per_category_analysis/category_metrics.csv
  results/per_category_analysis/CATEGORY_REPORT.md

Usage:
  PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
  $PY per_category_analysis.py               # LapEigvals + Classifier A only
  $PY per_category_analysis.py --snne        # also compute SNNE (needs DeBERTa, ~1h)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"
OUT_DIR = REPO_ROOT / "results" / "per_category_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIAG_DIR = REPO_ROOT / "results" / "lapeigvals_baseline" / "diags"
LAP_METRICS = REPO_ROOT / "results" / "lapeigvals_baseline" / "all_pairs_metrics.csv"
SNNE_COMPARISON = REPO_ROOT / "results" / "snne_baseline" / "comparison_vs_classifier.csv"
GEN_DIR = REPO_ROOT / "results" / "snne_baseline" / "generations"

PAIRS = [(m, d) for m in ("llama", "mistral", "qwen", "gemma") for d in ("sciq", "triviaqa", "math")]
CATS = ["correct_high", "correct_low", "incorrect_low", "incorrect_high"]
CAT_SHORT = {"correct_high": "CH", "correct_low": "CL",
             "incorrect_low": "IL", "incorrect_high": "IH"}

SEED = 42
LR_KWARGS = dict(max_iter=2000, class_weight="balanced", random_state=SEED)

# ─── helpers ────────────────────────────────────────────────────────────────

def make_pipe(pca_dim, n_feat, n_samp):
    if pca_dim is not None:
        dim = min(pca_dim, n_feat, n_samp - 1)
        return Pipeline([
            ("pca", PCA(n_components=dim, svd_solver="randomized", random_state=SEED)),
            ("lr", LogisticRegression(**LR_KWARGS)),
        ])
    return LogisticRegression(**LR_KWARGS)


def safe_auroc(y, scores):
    """AUROC, or None if fewer than 2 classes."""
    if len(set(y)) < 2 or len(y) < 2:
        return None
    return float(roc_auc_score(y, scores))


def sub_auroc(y, scores, cats, keep):
    """AUROC on the subset where category is in `keep`."""
    mask = np.array([c in keep for c in cats])
    if mask.sum() < 2:
        return None
    return safe_auroc(np.array(y)[mask], np.array(scores)[mask])


# ─── LapEigvals variants ────────────────────────────────────────────────────

sys.path.insert(0, str(SCRIPTS_DIR / "lapeigvals_baseline"))
from lapeigvals_features import (  # noqa: E402
    get_attn_eigvals_per_head_topk,
    get_attn_log_det,
    get_laplacian_eigvals_per_head_topk,
)


def _load_best_configs():
    """Read all_pairs_metrics.csv → {pair: {variant: (top_k|None, pca|None)}}."""
    cfg = {}
    with LAP_METRICS.open() as f:
        for row in csv.DictReader(f):
            pair, variant, config_str = row["pair"], row["variant"], row["config"]
            top_k = pca = None
            for part in config_str.split(","):
                part = part.strip()
                if part.startswith("top_k="):
                    v = part[6:]
                    top_k = int(v) if v != "None" else None
                elif part.startswith("pca="):
                    v = part[4:]
                    pca = int(v) if v != "None" else None
            cfg.setdefault(pair, {})[variant] = (top_k, pca)
    return cfg


def score_lapeigvals(pair: str, best_cfg: dict) -> dict[str, list[float]] | None:
    """
    Refit each variant's probe on train with best config, return dict
    variant → list[float] of P(hallucination) for test examples (in diag order).
    """
    model, dataset = pair.split("_", 1)
    tr_path = DIAG_DIR / f"{model}_{dataset}_train.pt"
    te_path = DIAG_DIR / f"{model}_{dataset}_test.pt"
    if not (tr_path.exists() and te_path.exists()):
        print(f"  [skip lapeigvals] {pair}: diags missing")
        return None

    tr = torch.load(tr_path, weights_only=False)
    te = torch.load(te_path, weights_only=False)
    ytr = tr["labels"].numpy()
    yte = te["labels"].numpy()
    if len(set(ytr)) < 2:
        print(f"  [skip lapeigvals] {pair}: degenerate train labels")
        return None

    attn_tr = [d.float() for d in tr["attn_diags"]]
    attn_te = [d.float() for d in te["attn_diags"]]
    lap_tr  = [d.float() for d in tr["lap_diags"]]
    lap_te  = [d.float() for d in te["lap_diags"]]

    cfgs = best_cfg.get(pair, {})
    results = {}

    for variant, (builder_tr, builder_te) in [
        ("AttnLogDet",  (attn_tr, attn_te)),
        ("AttnEigvals", (attn_tr, attn_te)),
        ("LapEigvals",  (lap_tr,  lap_te)),
    ]:
        top_k, pca_dim = cfgs.get(variant, (None, None))
        fn = {"AttnLogDet": get_attn_log_det,
              "AttnEigvals": get_attn_eigvals_per_head_topk,
              "LapEigvals":  get_laplacian_eigvals_per_head_topk}[variant]

        if top_k is not None:
            Xtr = fn(builder_tr, top_k=top_k).float().numpy()
            Xte = fn(builder_te, top_k=top_k).float().numpy()
        else:
            Xtr = fn(builder_tr).float().numpy()
            Xte = fn(builder_te).float().numpy()

        pipe = make_pipe(pca_dim, Xtr.shape[1], Xtr.shape[0])
        pipe.fit(Xtr, ytr)
        results[variant] = pipe.predict_proba(Xte)[:, 1].tolist()

    return results  # keys: AttnLogDet, AttnEigvals, LapEigvals


def get_lapeigvals_records(pair: str, best_cfg: dict) -> list[dict]:
    """Return per-example rows with id, category, label, scores for all 3 variants."""
    model, dataset = pair.split("_", 1)
    te_path = DIAG_DIR / f"{model}_{dataset}_test.pt"
    if not te_path.exists():
        return []
    te = torch.load(te_path, weights_only=False)
    ids = te["ids"]
    cats = te["categories"]
    labels = te["labels"].tolist()

    scores = score_lapeigvals(pair, best_cfg)
    if scores is None:
        return []

    rows = []
    for i, (sid, cat, lbl) in enumerate(zip(ids, cats, labels)):
        rows.append({
            "pair": pair, "id": sid, "category": cat, "label": lbl,
            "LapEigvals":  scores["LapEigvals"][i],
            "AttnEigvals": scores["AttnEigvals"][i],
            "AttnLogDet":  scores["AttnLogDet"][i],
        })
    return rows


# ─── Classifier A ───────────────────────────────────────────────────────────

sys.path.insert(0, str(SCRIPTS_DIR / "classifier"))
sys.path.insert(0, str(SCRIPTS_DIR / "ranking"))

from utils_distractor_features import load_and_prepare_data_all  # noqa: E402

FEATURE_NAMES = [
    "f1_s_int_best", "f2_s_ext_final", "f3_delta_int",
    "f4_div_int_ext", "f6_hardest_gap", "f8_trajectory",
    "g1_len_greedy", "g2_ext_per_token", "g4_int_traj_mean", "g8_len_delta",
]


def _load_sidecar_df(model: str, dataset: str, greedy_path: Path, dist_path: Path) -> "pd.DataFrame | None":
    """Load a greedy+distractor sidecar pair into a features dataframe with id+category."""
    import pandas as pd
    sys.path.insert(0, str(SCRIPTS_DIR / "classifier"))
    from utils_distractor_features import load_jsonl, compute_features

    greedy_sidecar = {r["id"]: r for r in load_jsonl(greedy_path)}
    dist_sidecar   = {r["id"]: r.get("distractors", []) for r in load_jsonl(dist_path)}

    records = []
    for sid, greedy_row in greedy_sidecar.items():
        distractors = dist_sidecar.get(sid, [])
        if not distractors:
            continue
        features = compute_features(greedy_row, distractors, L_best=15, L_early=5)
        if features is None:
            continue
        records.append({
            "id": sid,
            "category": greedy_row.get("category"),
            "target_c_vs_i": int(greedy_row.get("open_text_label", False)),
            **features,
        })

    if not records:
        return None
    return pd.DataFrame(records)


def get_clf_a_records(pair: str) -> list[dict]:
    """
    Train Classifier A on the 980 train-split examples (if train sidecars exist),
    test on the 420 test-split examples. Falls back to 5-fold CV on the 420 test
    examples if train sidecars have not been computed yet.
    """
    model, dataset = pair.split("_", 1)
    cache_dir = DATA_DIR / f"ranking_experiment_{model}_{dataset}" / "cache"

    greedy_train = cache_dir / "greedy_sidecar_train.jsonl"
    dist_train   = cache_dir / "distractor_sidecar_train.jsonl"
    greedy_test  = cache_dir / "greedy_sidecar.jsonl"
    dist_test    = cache_dir / "distractor_sidecar.jsonl"

    have_train_sidecars = greedy_train.exists() and dist_train.exists()

    # ── test df is always needed ──────────────────────────────────────────────
    df_te = _load_sidecar_df(model, dataset, greedy_test, dist_test)
    if df_te is None:
        df_te = load_and_prepare_data_all(model, dataset, DATA_DIR, L_best=15, L_early=5)
    if df_te is None:
        print(f"  [skip clf_a] {pair}: could not load test features")
        return []

    missing = [c for c in FEATURE_NAMES if c not in df_te.columns]
    if missing:
        print(f"  [skip clf_a] {pair}: missing feature columns {missing}")
        return []

    Xte = df_te[FEATURE_NAMES].values
    yte = (1 - df_te["target_c_vs_i"].values)

    if have_train_sidecars:
        # ── proper train → test evaluation ────────────────────────────────────
        df_tr = _load_sidecar_df(model, dataset, greedy_train, dist_train)
        if df_tr is None:
            print(f"  [skip clf_a] {pair}: could not load train sidecars")
            return []
        Xtr = df_tr[FEATURE_NAMES].values
        ytr = (1 - df_tr["target_c_vs_i"].values)
        if len(set(ytr)) < 2:
            print(f"  [skip clf_a] {pair}: degenerate train labels")
            return []
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(Xtr_s, ytr)
        probas = clf.predict_proba(Xte_s)[:, 1]
        print(f"  [clf_a] trained on {len(df_tr)} train examples, testing on {len(df_te)}")
    else:
        # ── fallback: 5-fold CV on the 420 test examples ─────────────────────
        print(f"  [clf_a] no train sidecars found — running 5-fold CV on {len(df_te)} test examples")
        if len(set(yte)) < 2:
            print(f"  [skip clf_a] {pair}: degenerate labels")
            return []
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        probas = np.zeros(len(yte))
        for tr_idx, te_idx in skf.split(Xte, yte):
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xte[tr_idx])
            Xte_s = sc.transform(Xte[te_idx])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
            clf.fit(Xtr_s, yte[tr_idx])
            probas[te_idx] = clf.predict_proba(Xte_s)[:, 1]

    rows = []
    for (_, row), prob in zip(df_te.iterrows(), probas):
        rows.append({
            "pair": pair,
            "id": row["id"],
            "category": row.get("category"),
            "label": int(1 - row["target_c_vs_i"]),
            "classifier_a": float(prob),
        })
    return rows


# ─── SNNE ───────────────────────────────────────────────────────────────────

def _load_best_snne_methods() -> dict[str, str]:
    """Read comparison CSV → {pair: best_method_name}."""
    out = {}
    with SNNE_COMPARISON.open() as f:
        for row in csv.DictReader(f):
            if row["pair"] == "MEAN":
                continue
            out[row["pair"]] = row.get("best_snne_method", "num_set")
    return out


def get_snne_records(pair: str, best_method: str, deberta, rouge, tokenizer) -> list[dict]:
    """Score each test example with the best SNNE method, return per-example rows."""
    gen_path = GEN_DIR / f"{pair}.jsonl"
    if not gen_path.exists():
        print(f"  [skip snne] {pair}: no generations file")
        return []

    model, dataset = pair.split("_", 1)
    te_path = DIAG_DIR / f"{model}_{dataset}_test.pt"
    if not te_path.exists():
        print(f"  [skip snne] {pair}: no test diag for ID→category lookup")
        return []

    te = torch.load(te_path, weights_only=False)
    id_to_cat = {sid: cat for sid, cat in zip(te["ids"], te["categories"])}
    id_to_lbl = {sid: int(lbl) for sid, lbl in zip(te["ids"], te["labels"].tolist())}

    sys.path.insert(0, str(SCRIPTS_DIR / "snne_baseline"))
    from run_baselines import per_question_scores  # noqa: E402

    rows = []
    with gen_path.open() as f:
        for line in f:
            r = json.loads(line)
            sid = r["id"]
            if sid not in id_to_cat:
                continue  # skip training examples
            gens = r.get("generations", [])
            if len(gens) < 2:
                continue
            scores = per_question_scores(r["question"], gens, deberta, rouge, tokenizer)
            rows.append({
                "pair": pair,
                "id": sid,
                "category": id_to_cat[sid],
                "label": id_to_lbl[sid],
                "snne_best": float(scores[best_method]),
                "snne_method": best_method,
            })
    return rows


# ─── per-category metrics ────────────────────────────────────────────────────

def category_metrics(pair: str, method: str, ids, cats, labels, scores) -> list[dict]:
    """Compute overall + sub-group AUROCs and per-category mean scores."""
    cats = [CAT_SHORT.get(c, c) for c in cats]
    labels = list(labels)
    scores = list(scores)

    rows = []
    auc_all  = safe_auroc(labels, scores)
    auc_ihch = sub_auroc(labels, scores, cats, {"IH", "CH"})
    auc_ilcl = sub_auroc(labels, scores, cats, {"IL", "CL"})

    # Mean predicted score per category
    cat_scores: dict[str, list] = {c: [] for c in ["CH", "CL", "IL", "IH"]}
    for c, s in zip(cats, scores):
        short = CAT_SHORT.get(c, c)
        if short in cat_scores:
            cat_scores[short].append(s)

    rows.append({
        "pair": pair, "method": method,
        "auroc_all": auc_all,
        "auroc_ih_ch": auc_ihch,
        "auroc_il_cl": auc_ilcl,
        "mean_CH": np.mean(cat_scores["CH"]) if cat_scores["CH"] else None,
        "mean_CL": np.mean(cat_scores["CL"]) if cat_scores["CL"] else None,
        "mean_IL": np.mean(cat_scores["IL"]) if cat_scores["IL"] else None,
        "mean_IH": np.mean(cat_scores["IH"]) if cat_scores["IH"] else None,
        "n_CH": len(cat_scores["CH"]),
        "n_CL": len(cat_scores["CL"]),
        "n_IL": len(cat_scores["IL"]),
        "n_IH": len(cat_scores["IH"]),
    })
    return rows


# ─── main ────────────────────────────────────────────────────────────────────

def fmt(x) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snne", action="store_true",
                        help="Also compute SNNE per-example scores (loads DeBERTa, ~1h)")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Subset of pairs e.g. llama_sciq mistral_math")
    args = parser.parse_args()

    pairs = args.pairs or [f"{m}_{d}" for m, d in PAIRS]

    best_lap_cfg  = _load_best_configs()
    best_snne_map = _load_best_snne_methods()

    # ── per-example score table ──────────────────────────────────────────────
    all_pe_rows: list[dict] = []   # one row per (pair, example)
    all_metric_rows: list[dict] = []

    if args.snne:
        print("Loading DeBERTa for SNNE scoring...")
        sys.path.insert(0, str(SCRIPTS_DIR / "snne_baseline"))
        from snne_core import EntailmentDeberta  # noqa: E402
        from rouge_score import rouge_scorer
        from transformers import AutoTokenizer as _AT
        deberta = EntailmentDeberta()
        rouge   = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        tok     = _AT.from_pretrained("gpt2")
    else:
        deberta = rouge = tok = None

    for pair in pairs:
        print(f"\n=== {pair} ===")
        model, dataset = pair.split("_", 1)

        # --- LapEigvals family ---
        lap_rows = get_lapeigvals_records(pair, best_lap_cfg)
        if lap_rows:
            for method in ("LapEigvals", "AttnEigvals", "AttnLogDet"):
                ids_   = [r["id"] for r in lap_rows]
                cats_  = [r["category"] for r in lap_rows]
                labels_= [r["label"]   for r in lap_rows]
                scores_= [r[method]    for r in lap_rows]
                mrows  = category_metrics(pair, method, ids_, cats_, labels_, scores_)
                all_metric_rows.extend(mrows)
                print(f"  {method:12s}  auroc_all={fmt(mrows[0]['auroc_all'])}  "
                      f"IH_vs_CH={fmt(mrows[0]['auroc_ih_ch'])}  "
                      f"IL_vs_CL={fmt(mrows[0]['auroc_il_cl'])}")
            # accumulate per-example rows (one row per example, all 3 lap scores)
            all_pe_rows.extend(lap_rows)

        # --- Classifier A ---
        clf_rows = get_clf_a_records(pair)
        if clf_rows:
            ids_   = [r["id"] for r in clf_rows]
            cats_  = [r["category"] for r in clf_rows]
            labels_= [r["label"]    for r in clf_rows]
            scores_= [r["classifier_a"] for r in clf_rows]
            mrows  = category_metrics(pair, "classifier_a", ids_, cats_, labels_, scores_)
            all_metric_rows.extend(mrows)
            print(f"  {'classifier_a':12s}  auroc_all={fmt(mrows[0]['auroc_all'])}  "
                  f"IH_vs_CH={fmt(mrows[0]['auroc_ih_ch'])}  "
                  f"IL_vs_CL={fmt(mrows[0]['auroc_il_cl'])}")
            # merge classifier_a score into per-example rows
            clf_by_id = {r["id"]: r["classifier_a"] for r in clf_rows}
            for pe in lap_rows:
                pe["classifier_a"] = clf_by_id.get(pe["id"])

        # --- SNNE ---
        if args.snne:
            bm = best_snne_map.get(pair, "num_set")
            snne_rows = get_snne_records(pair, bm, deberta, rouge, tok)
            if snne_rows:
                ids_   = [r["id"] for r in snne_rows]
                cats_  = [r["category"] for r in snne_rows]
                labels_= [r["label"]    for r in snne_rows]
                scores_= [r["snne_best"]for r in snne_rows]
                mrows  = category_metrics(pair, f"snne_{bm}", ids_, cats_, labels_, scores_)
                all_metric_rows.extend(mrows)
                print(f"  snne_{bm:10s}  auroc_all={fmt(mrows[0]['auroc_all'])}  "
                      f"IH_vs_CH={fmt(mrows[0]['auroc_ih_ch'])}  "
                      f"IL_vs_CL={fmt(mrows[0]['auroc_il_cl'])}")
                snne_by_id = {r["id"]: r["snne_best"] for r in snne_rows}
                for pe in lap_rows:
                    pe["snne_best"] = snne_by_id.get(pe["id"])

    # ── write per-example CSV ────────────────────────────────────────────────
    pe_path = OUT_DIR / "per_example_scores.csv"
    pe_cols = ["pair", "id", "category", "label",
               "LapEigvals", "AttnEigvals", "AttnLogDet", "classifier_a"]
    if args.snne:
        pe_cols.append("snne_best")
    with pe_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pe_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_pe_rows)
    print(f"\n[written] {pe_path}")

    # ── write category metrics CSV ───────────────────────────────────────────
    cm_path = OUT_DIR / "category_metrics.csv"
    cm_cols = ["pair", "method", "auroc_all", "auroc_ih_ch", "auroc_il_cl",
               "mean_CH", "mean_CL", "mean_IL", "mean_IH",
               "n_CH", "n_CL", "n_IL", "n_IH"]
    with cm_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cm_cols)
        w.writeheader()
        for row in all_metric_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else (v if v is not None else ""))
                        for k, v in row.items()})
    print(f"[written] {cm_path}")

    # ── markdown report ───────────────────────────────────────────────────────
    methods_in_report = ["LapEigvals", "AttnEigvals", "AttnLogDet", "classifier_a"]
    if args.snne:
        methods_in_report.append("snne_best")

    lines = [
        "# Per-Category Hallucination Detection Analysis",
        "",
        "Categories: **CH** = correct_high, **CL** = correct_low, "
        "**IL** = incorrect_low, **IH** = incorrect_high.  ",
        "Scores: P(hallucination); higher = more likely wrong.  ",
        "Sub-AUROCs: IH vs CH = hard pairs (both confident); IL vs CL = easy pairs.",
        "",
    ]

    for pair in pairs:
        pair_rows = {r["method"]: r for r in all_metric_rows if r["pair"] == pair}
        if not pair_rows:
            continue
        lines.append(f"## {pair}")
        lines.append("")

        # sub-AUROC table
        lines.append("| Method | AUROC (all) | AUROC (IH vs CH) | AUROC (IL vs CL) |")
        lines.append("|--------|------------|-----------------|-----------------|")
        for m in methods_in_report:
            r = pair_rows.get(m)
            if r is None:
                continue
            lines.append(f"| {m} | {fmt(r['auroc_all'])} | {fmt(r['auroc_ih_ch'])} | {fmt(r['auroc_il_cl'])} |")
        lines.append("")

        # mean score per category table
        lines.append("| Method | mean P(wrong) CH | mean P(wrong) CL | mean P(wrong) IL | mean P(wrong) IH |")
        lines.append("|--------|-----------------|-----------------|-----------------|-----------------|")
        for m in methods_in_report:
            r = pair_rows.get(m)
            if r is None:
                continue
            lines.append(f"| {m} | {fmt(r['mean_CH'])} | {fmt(r['mean_CL'])} | {fmt(r['mean_IL'])} | {fmt(r['mean_IH'])} |")

        # n per category (from first method with data)
        first = next(iter(pair_rows.values()))
        lines.append(f"\n*n per category: CH={first['n_CH']}, CL={first['n_CL']}, "
                     f"IL={first['n_IL']}, IH={first['n_IH']}*")
        lines.append("")

    # ── cross-pair summary (mean over pairs) ─────────────────────────────────
    lines += ["## Summary: means across all 9 pairs", ""]
    lines.append("| Method | AUROC (all) | AUROC (IH vs CH) | AUROC (IL vs CL) | mean_CH | mean_CL | mean_IL | mean_IH |")
    lines.append("|--------|------------|-----------------|-----------------|---------|---------|---------|---------|")

    def col_mean_method(method, col):
        vals = [r[col] for r in all_metric_rows
                if r["method"] == method and isinstance(r.get(col), float)]
        return sum(vals) / len(vals) if vals else None

    for m in methods_in_report:
        has = any(r["method"] == m for r in all_metric_rows)
        if not has:
            continue
        lines.append(
            f"| {m} | {fmt(col_mean_method(m, 'auroc_all'))} "
            f"| {fmt(col_mean_method(m, 'auroc_ih_ch'))} "
            f"| {fmt(col_mean_method(m, 'auroc_il_cl'))} "
            f"| {fmt(col_mean_method(m, 'mean_CH'))} "
            f"| {fmt(col_mean_method(m, 'mean_CL'))} "
            f"| {fmt(col_mean_method(m, 'mean_IL'))} "
            f"| {fmt(col_mean_method(m, 'mean_IH'))} |"
        )

    md_path = OUT_DIR / "CATEGORY_REPORT.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[written] {md_path}")


if __name__ == "__main__":
    main()