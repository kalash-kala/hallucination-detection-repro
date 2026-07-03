"""
LapEigvals baseline — STEP 2: build features and train the probes.

Loads the per-example diagonals produced by extract_attention.py and trains the
three upstream supervised variants, evaluating test AUROC on our split so the
numbers are directly comparable to Classifier A and the SNNE baseline:

  AttnLogDet  (LLMCheck)  : mean(log(diag(A)))                       no top-k
  AttnEigvals             : top-k attention-diagonal values / head   top-k sweep
  LapEigvals  (headline)  : top-k Laplacian-diagonal values / head   top-k sweep

Probe = upstream pipeline (optional PCA -> LogisticRegression, class_weight
balanced, max_iter 2000, seed 42), all-layers pooling (the paper's headline
setting). To avoid test-set peeking, top-k and PCA are selected per variant by
5-fold stratified CV on the TRAIN split only, then refit on full train and
scored on test.

Outputs:
  results/lapeigvals_baseline/{model}_{dataset}_metrics.csv  (per-variant test AUROC + chosen config)
  results/lapeigvals_baseline/{model}_{dataset}_sweep.csv    (every top-k/pca config's test AUROC, for transparency)

Usage:
  PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
  $PY train_lapeigvals.py --all
  $PY train_lapeigvals.py --model llama --dataset sciq
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from lapeigvals_features import (
    get_attn_eigvals_per_head_topk,
    get_attn_log_det,
    get_laplacian_eigvals_per_head_topk,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RES_DIR = REPO_ROOT / "results" / "lapeigvals_baseline"
DIAG_DIR = RES_DIR / "diags"

PAIRS = [(m, d) for m in ("llama", "mistral", "qwen", "gemma") for d in ("sciq", "triviaqa", "math")]
TOP_K_EIGVALS = [5, 10, 25, 50, 100]
PCA_GRID = [None, 100]
SEED = 42
LR_KWARGS = dict(max_iter=2000, class_weight="balanced", random_state=SEED)


def make_pipeline(pca_dim: int | None, n_features: int, n_samples: int):
    if pca_dim is not None:
        dim = min(pca_dim, n_features, n_samples - 1)
        return Pipeline([
            ("pca", PCA(n_components=dim, svd_solver="randomized", random_state=SEED)),
            ("lr", LogisticRegression(**LR_KWARGS)),
        ])
    return LogisticRegression(**LR_KWARGS)


def cv_auc(X: np.ndarray, y: np.ndarray, pca_dim: int | None) -> float:
    pipe = make_pipeline(pca_dim, X.shape[1], X.shape[0])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=1)
    return float(np.mean(scores))


def fit_eval(Xtr, ytr, Xte, yte, pca_dim: int | None) -> float:
    pipe = make_pipeline(pca_dim, Xtr.shape[1], Xtr.shape[0])
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, proba))


def feats(builder, diags_tr, diags_te, **kw):
    Xtr = builder(diags_tr, **kw).float().numpy()
    Xte = builder(diags_te, **kw).float().numpy()
    return Xtr, Xte


def select_and_eval(variant, diags_tr, diags_te, ytr, yte, top_ks, sweep_rows, model, dataset):
    """CV-select config on train, eval on test. Returns (test_auc, chosen_cfg_str)."""
    candidates = []  # (cv_auc, test_auc, cfg_str, top_k, pca)

    if variant == "attn_log_det":
        Xtr, Xte = feats(get_attn_log_det, diags_tr, diags_te)
        for pca in PCA_GRID:
            c = cv_auc(Xtr, ytr, pca)
            t = fit_eval(Xtr, ytr, Xte, yte, pca)
            candidates.append((c, t, f"pca={pca}", None, pca))
            sweep_rows.append([model, dataset, variant, "", str(pca), f"{c:.4f}", f"{t:.4f}"])
    else:
        builder = (get_attn_eigvals_per_head_topk if variant == "attn_eigvals"
                   else get_laplacian_eigvals_per_head_topk)
        for k in top_ks:
            Xtr, Xte = feats(builder, diags_tr, diags_te, top_k=k)
            for pca in PCA_GRID:
                c = cv_auc(Xtr, ytr, pca)
                t = fit_eval(Xtr, ytr, Xte, yte, pca)
                candidates.append((c, t, f"top_k={k},pca={pca}", k, pca))
                sweep_rows.append([model, dataset, variant, str(k), str(pca), f"{c:.4f}", f"{t:.4f}"])

    best = max(candidates, key=lambda r: r[0])  # select on CV-AUROC only
    return best[1], best[2]


def diag_min_seq(*diag_lists) -> int:
    return min(d.size(-1) for lst in diag_lists for d in lst)


def process_pair(model: str, dataset: str, metrics_rows: list, sweep_rows: list) -> None:
    tr_path = DIAG_DIR / f"{model}_{dataset}_train.pt"
    te_path = DIAG_DIR / f"{model}_{dataset}_test.pt"
    if not (tr_path.exists() and te_path.exists()):
        print(f"  [skip] {model}/{dataset}: diags missing ({tr_path.name}/{te_path.name})")
        return

    tr = torch.load(tr_path, weights_only=False)
    te = torch.load(te_path, weights_only=False)
    ytr = tr["labels"].numpy()
    yte = te["labels"].numpy()
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        print(f"  [skip] {model}/{dataset}: degenerate labels")
        return

    # cast stored fp16 diagonals back to fp32 for the feature math
    attn_tr = [d.float() for d in tr["attn_diags"]]
    attn_te = [d.float() for d in te["attn_diags"]]
    lap_tr = [d.float() for d in tr["lap_diags"]]
    lap_te = [d.float() for d in te["lap_diags"]]

    min_seq = diag_min_seq(attn_tr, attn_te, lap_tr, lap_te)
    top_ks = [k for k in TOP_K_EIGVALS if k <= min_seq]
    if not top_ks:
        top_ks = [min_seq]
    print(f"\n=== {model}/{dataset} ===  n_train={len(ytr)} n_test={len(yte)} "
          f"min_seq={min_seq} top_ks={top_ks} wrong_rate(test)={yte.mean():.3f}")

    results = {}
    results["AttnLogDet"] = select_and_eval(
        "attn_log_det", attn_tr, attn_te, ytr, yte, top_ks, sweep_rows, model, dataset)
    results["AttnEigvals"] = select_and_eval(
        "attn_eigvals", attn_tr, attn_te, ytr, yte, top_ks, sweep_rows, model, dataset)
    results["LapEigvals"] = select_and_eval(
        "lap_eigvals", lap_tr, lap_te, ytr, yte, top_ks, sweep_rows, model, dataset)

    for variant, (auc, cfg) in results.items():
        print(f"  {variant:12s} test_auc={auc:.4f}  [{cfg}]")
        metrics_rows.append([f"{model}_{dataset}", variant, f"{auc:.6f}", cfg, len(yte)])


def write_csv(path: Path, header: list, rows: list):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[written] {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["llama", "mistral", "qwen"])
    p.add_argument("--dataset", choices=["sciq", "triviaqa", "math"])
    p.add_argument("--all", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.all:
        pairs = PAIRS
    elif args.model and args.dataset:
        pairs = [(args.model, args.dataset)]
    else:
        raise SystemExit("specify --all or both --model and --dataset")

    metrics_rows, sweep_rows = [], []
    for model, dataset in pairs:
        process_pair(model, dataset, metrics_rows, sweep_rows)
        # write per-pair metrics file too (mirrors snne_baseline layout)
        pair_rows = [r for r in metrics_rows if r[0] == f"{model}_{dataset}"]
        if pair_rows:
            write_csv(RES_DIR / f"{model}_{dataset}_metrics.csv",
                      ["pair", "variant", "auroc", "config", "n_test"], pair_rows)

    if metrics_rows:
        write_csv(RES_DIR / "all_pairs_metrics.csv",
                  ["pair", "variant", "auroc", "config", "n_test"], metrics_rows)
    if sweep_rows:
        write_csv(RES_DIR / "sweep_all_configs.csv",
                  ["pair_model", "dataset", "variant", "top_k", "pca", "cv_auroc", "test_auroc"], sweep_rows)


if __name__ == "__main__":
    main()