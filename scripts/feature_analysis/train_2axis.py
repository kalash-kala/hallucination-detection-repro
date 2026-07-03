"""
2-Axis classifier: train LR on confidence (entropy) + correctness (LapEigvals,
hidden states) features. Train→test evaluation matching CATEGORY_REPORT.md.

Label convention: y = 1 for INCORRECT (hallucination), matching the LapEigvals
baseline and Classifier A, so AUROCs are directly comparable.

Feature combos evaluated per pair:
  reference: entropy_only, lap_only, hidden_only
  combos:    lap+hidden, entropy+lap, entropy+hidden, entropy+lap+hidden

Pipeline: StandardScaler -> LogisticRegression(class_weight=balanced,
max_iter=2000, C=1.0, seed 42). No PCA (matches the 8/9 LapEigvals finding that
L2 handles the high-dim collinearity better than PCA).

Metrics (test split):
  AUROC(all)     I vs C on full test
  AUROC(IH v CH) correctness within High-confidence band
  AUROC(IL v CL) correctness within Low-confidence band

Outputs:
  results/feature_analysis/two_axis_results.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lapeigvals_baseline"))
from lapeigvals_features import get_laplacian_eigvals_per_head_topk

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"
DIAG_DIR = REPO_ROOT / "results" / "lapeigvals_baseline" / "diags"
OUT_DIR = REPO_ROOT / "results" / "feature_analysis"

PAIRS = [(m, d) for m in ("llama", "mistral", "qwen", "gemma") for d in ("sciq", "triviaqa", "math")]
TOP_K_LAP = 10                       # fixed-k value used in the headline combos
TOP_K_GRID = [5, 10, 25, 50, 100]    # CV-tuned grid, mirrors lapeigvals baseline
SEED = 42
LR_KWARGS = dict(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)
CATEGORIES = ["incorrect_high", "incorrect_low", "correct_high", "correct_low"]
CAT_SHORT = {"incorrect_high": "IH", "incorrect_low": "IL",
             "correct_high": "CH", "correct_low": "CL"}

# entropy CSV path per pair (verified: full train/test id overlap)
# [repro patch] these ship with the package under data/uncertainty_runs/
_ENTROPY_DIR = REPO_ROOT / "data" / "uncertainty_runs"
ENTROPY_CSV = {
    "llama_sciq":     _ENTROPY_DIR / "uncertainty_run_llama_sciq_combined_llm_verdict_concentration_output.csv",
    "mistral_sciq":   _ENTROPY_DIR / "uncertainty_run_mistral_sciq_combined_full_llm_verdict_concentration_output.csv",
    "qwen_sciq":      _ENTROPY_DIR / "uncertainty_run_qwen_sciq_combined_full_llm_verdict_concentration_output.csv",
    "llama_triviaqa": _ENTROPY_DIR / "uncertainty_run_llama_triviaqa_combined_50K_llm_verdict_concentration_output.csv",
    "mistral_triviaqa": _ENTROPY_DIR / "uncertainty_run_mistral_triviaqa_combined_50K_llm_verdict_concentration_output.csv",
    "qwen_triviaqa":  _ENTROPY_DIR / "uncertainty_run_qwen_triviaqa_combined_50K_llm_verdict_concentration_output.csv",
    "llama_math":     _ENTROPY_DIR / "uncertainty_run_llama_answerable_math_combined_llm_verdict_concentration_output.csv",
    "mistral_math":   _ENTROPY_DIR / "uncertainty_run_mistral_answerable_math_combined_llm_verdict_concentration_output.csv",
    "qwen_math":      _ENTROPY_DIR / "uncertainty_run_qwen_answerable_math_combined_llm_verdict_concentration_output.csv",
    "gemma_sciq":     _ENTROPY_DIR / "uncertainty_run_gemma_sciq_combined_full_llm_verdict_concentration_output.csv",
    "gemma_triviaqa": _ENTROPY_DIR / "uncertainty_run_gemma_triviaqa_combined_50K_llm_verdict_concentration_output.csv",
    "gemma_math":     _ENTROPY_DIR / "uncertainty_run_gemma_answerable_math_combined_llm_verdict_concentration_output.csv",
}


# ─── feature loaders (return dict: id -> np.ndarray) ──────────────────────────

def load_entropy(pair: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(ENTROPY_CSV[pair])
    df["id"] = df["id"].astype(str)
    out = {}
    for _, row in df.iterrows():
        e = row.get("cluster_assignment_entropy", np.nan)
        if pd.notna(e):
            out[row["id"]] = np.array([float(e)], dtype=np.float32)
    return out


def load_lap_raw(model: str, dataset: str, split: str):
    """Returns (lap_diags, ids, label_dict, cat_dict). lap_diags lets us rebuild the
    LapEigvals feature at any top_k; ids/labels/cats are stable across k."""
    pt = torch.load(DIAG_DIR / f"{model}_{dataset}_{split}.pt", weights_only=False)
    lap_diags = [d.float() for d in pt["lap_diags"]]
    ids = [str(i) for i in pt["ids"]]
    labels = pt["labels"].numpy()
    cats = pt["categories"]
    lab = {i: int(labels[k]) for k, i in enumerate(ids)}
    cat = {i: cats[k] for k, i in enumerate(ids)}
    return lap_diags, ids, lab, cat


def lap_feat_dict(lap_diags, ids, top_k: int) -> dict[str, np.ndarray]:
    """id -> top-k Laplacian-eigvals feature vector, byte-identical to the baseline."""
    X = get_laplacian_eigvals_per_head_topk(lap_diags, top_k=top_k).numpy()
    return {i: X[k] for k, i in enumerate(ids)}


def lap_min_seq(*diag_lists) -> int:
    """Min sequence length across all per-example diag tensors (caps usable top_k),
    matching lapeigvals_baseline.train_lapeigvals.diag_min_seq."""
    return min(d.size(-1) for lst in diag_lists for d in lst)


def load_hidden(model: str, dataset: str, split: str) -> dict[str, np.ndarray]:
    """Late-bucket mean hidden state (layers >=22) keyed by example id."""
    cache_dir = DATA_DIR / f"ranking_experiment_{model}_{dataset}" / "cache"
    sidecar = cache_dir / ("greedy_sidecar_train.jsonl" if split == "train" else "greedy_sidecar.jsonl")
    id2cand = {}
    with open(sidecar) as f:
        for line in f:
            rec = json.loads(line)
            id2cand[str(rec["id"])] = rec["candidate"]

    late_layers = sorted(
        int(p.stem.replace("hidden_layer", ""))
        for p in cache_dir.glob("hidden_layer*.npz")
        if int(p.stem.replace("hidden_layer", "")) >= 22
    )
    layer_data = []
    for layer in late_layers:
        idx_path = cache_dir / f"index_layer{layer}.json"
        npz_path = cache_dir / f"hidden_layer{layer}.npz"
        if not (idx_path.exists() and npz_path.exists()):
            continue
        with open(idx_path) as f:
            idx = json.load(f)
        data = np.load(npz_path)["hidden"]
        layer_data.append((layer, idx, data))

    # Supplementary recovered store (greedy answers absent from the candidate-pool
    # cache; produced by ranking/recover_greedy_hidden.py). Same vectors, same math
    # (hidden_states[L][0, -1]); verified byte-identical to the main cache.
    rec_npz = cache_dir / f"greedy_late_hidden_{split}.npz"
    rec_idx_path = cache_dir / f"greedy_late_hidden_{split}_index.json"
    rec_idx, rec_data = None, None
    if rec_npz.exists() and rec_idx_path.exists():
        rec_idx = json.loads(rec_idx_path.read_text())
        rec_data = np.load(rec_npz)

    out = {}
    for qid, cand in id2cand.items():
        key = f"{qid}|||{cand}"
        vecs = []
        for layer, idx, data in layer_data:
            row = idx.get(key)
            if row is not None:
                vecs.append(data[row])
            elif rec_idx is not None and key in rec_idx:
                vecs.append(rec_data[f"L{layer}"][rec_idx[key]])
            else:
                break
        if len(vecs) == len(layer_data) and vecs:
            out[qid] = np.mean(vecs, axis=0).astype(np.float32)
    return out


# ─── assembling a combo matrix ────────────────────────────────────────────────

def build_matrix(ids: list[str], feat_dicts: list[dict]) -> np.ndarray:
    """Concatenate features for given ids (all must be present)."""
    rows = []
    for i in ids:
        rows.append(np.concatenate([fd[i] for fd in feat_dicts]))
    return np.stack(rows)


def common_ids(label_dict, feat_dicts) -> list[str]:
    ids = set(label_dict.keys())
    for fd in feat_dicts:
        ids &= set(fd.keys())
    return sorted(ids)


def auroc(y, s):
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def cv_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """5-fold stratified CV AUROC on train, same pipeline used for the final fit.
    Used to select lap top_k WITHOUT touching test (mirrors the LapEigvals baseline)."""
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(**LR_KWARGS))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return float(np.mean(cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc", n_jobs=1)))


def category_aurocs(ids, scores, cat_dict, y):
    """AUROC(all), AUROC(IH vs CH), AUROC(IL vs CL) on the given ids."""
    cats = np.array([CAT_SHORT.get(cat_dict[i], "?") for i in ids])
    y = np.asarray(y)
    s = np.asarray(scores)

    a_all = auroc(y, s)

    def split_auroc(inc, cor):
        m = np.isin(cats, [inc, cor])
        if m.sum() < 5:
            return float("nan")
        # within this band, y already encodes incorrect=1
        return auroc(y[m], s[m])

    a_high = split_auroc("IH", "CH")
    a_low = split_auroc("IL", "CL")
    return a_all, a_high, a_low


# ─── main per-pair routine ────────────────────────────────────────────────────

def fit_eval_record(name, k_mode, lap_k, cv_auc, pair,
                    tr_ids, te_ids, tr_dicts, te_dicts,
                    lab_tr, lab_te, cat_te, rows):
    """Fit LR on train, eval per-category AUROCs on test, append one result row."""
    Xtr = build_matrix(tr_ids, tr_dicts)
    Xte = build_matrix(te_ids, te_dicts)
    ytr = np.array([lab_tr[i] for i in tr_ids])
    yte = np.array([lab_te[i] for i in te_ids])

    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(**LR_KWARGS))])
    pipe.fit(Xtr, ytr)
    scores = pipe.predict_proba(Xte)[:, 1]

    a_all, a_high, a_low = category_aurocs(te_ids, scores, cat_te, yte)
    klabel = "—" if k_mode == "none" else (f"k={lap_k}" + ("(cv)" if k_mode == "tuned" else ""))
    print(f"  {name:20s} {k_mode:5s} {klabel:9s} dim={Xtr.shape[1]:6d} "
          f"n_tr={len(tr_ids)} n_te={len(te_ids)} "
          f"| AUROC all={a_all:.3f} IHvCH={a_high:.3f} ILvCL={a_low:.3f}")
    rows.append({
        "pair": pair, "combo": name, "k_mode": k_mode,
        "lap_k": lap_k if lap_k is not None else "",
        "cv_auroc": cv_auc if cv_auc is not None else float("nan"),
        "dim": Xtr.shape[1], "n_train": len(tr_ids), "n_test": len(te_ids),
        "auroc_all": a_all, "auroc_IH_vs_CH": a_high, "auroc_IL_vs_CL": a_low,
    })


def process_pair(model: str, dataset: str, rows: list):
    pair = f"{model}_{dataset}"
    print(f"\n=== {pair} ===")

    entropy = load_entropy(pair)
    lap_diags_tr, lap_ids_tr, lab_tr, cat_tr = load_lap_raw(model, dataset, "train")
    lap_diags_te, lap_ids_te, lab_te, cat_te = load_lap_raw(model, dataset, "test")
    hid_tr = load_hidden(model, dataset, "train")
    hid_te = load_hidden(model, dataset, "test")

    min_seq = lap_min_seq(lap_diags_tr, lap_diags_te)
    top_ks = [k for k in TOP_K_GRID if k <= min_seq] or [min_seq]
    print(f"  entropy={len(entropy)} lap_tr={len(lap_ids_tr)} lap_te={len(lap_ids_te)} "
          f"hid_tr={len(hid_tr)} hid_te={len(hid_te)} min_seq={min_seq} top_ks={top_ks}")

    # lap feature dicts, built per k (ids identical across k)
    lap_tr_k = {k: lap_feat_dict(lap_diags_tr, lap_ids_tr, k) for k in set(top_ks) | {TOP_K_LAP}}
    lap_te_k = {k: lap_feat_dict(lap_diags_te, lap_ids_te, k) for k in set(top_ks) | {TOP_K_LAP}}

    # which static (non-lap) feature dicts each combo uses, and whether lap is present
    combos = {
        "entropy_only":       (["entropy"], False),
        "lap_only":           ([],          True),
        "hidden_only":        (["hidden"],  False),
        "lap+hidden":         (["hidden"],  True),
        "entropy+lap":        (["entropy"], True),
        "entropy+hidden":     (["entropy", "hidden"], False),
        "entropy+lap+hidden": (["entropy", "hidden"], True),
    }
    static = {"entropy": (entropy, entropy), "hidden": (hid_tr, hid_te)}

    for name, (parts, has_lap) in combos.items():
        base_tr = [static[p][0] for p in parts]
        base_te = [static[p][1] for p in parts]

        # ids: lap presence is k-independent, so use lap@TOP_K_LAP for membership
        tr_dicts_ref = base_tr + ([lap_tr_k[TOP_K_LAP]] if has_lap else [])
        te_dicts_ref = base_te + ([lap_te_k[TOP_K_LAP]] if has_lap else [])
        tr_ids = common_ids(lab_tr, tr_dicts_ref)
        te_ids = common_ids(lab_te, te_dicts_ref)
        if len(tr_ids) < 30 or len(te_ids) < 20:
            print(f"  [skip] {name}: too few ids tr={len(tr_ids)} te={len(te_ids)}")
            continue
        ytr = np.array([lab_tr[i] for i in tr_ids])

        if not has_lap:
            fit_eval_record(name, "none", None, None, pair, tr_ids, te_ids,
                            base_tr, base_te, lab_tr, lab_te, cat_te, rows)
            continue

        # (a) fixed k=10 — the headline value, unchanged from before
        fit_eval_record(name, "fixed", TOP_K_LAP, None, pair, tr_ids, te_ids,
                        base_tr + [lap_tr_k[TOP_K_LAP]], base_te + [lap_te_k[TOP_K_LAP]],
                        lab_tr, lab_te, cat_te, rows)

        # (b) CV-tuned k — select on 5-fold train CV only, then eval once on test
        best_k, best_cv = None, -1.0
        for k in top_ks:
            Xtr_k = build_matrix(tr_ids, base_tr + [lap_tr_k[k]])
            c = cv_auroc(Xtr_k, ytr)
            if c > best_cv:
                best_cv, best_k = c, k
        fit_eval_record(name, "tuned", best_k, best_cv, pair, tr_ids, te_ids,
                        base_tr + [lap_tr_k[best_k]], base_te + [lap_te_k[best_k]],
                        lab_tr, lab_te, cat_te, rows)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["llama", "mistral", "qwen"])
    p.add_argument("--dataset", choices=["sciq", "triviaqa", "math"])
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    pairs = PAIRS if args.all else [(args.model, args.dataset)]
    if not args.all and not (args.model and args.dataset):
        raise SystemExit("specify --all or both --model and --dataset")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, dataset in pairs:
        process_pair(model, dataset, rows)

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "two_axis_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[written] {out_path}")

    # Mean across pairs per combo (fixed-k=10 and CV-tuned-k kept separate)
    print("\n=== Mean across pairs (by combo, k_mode) ===")
    summary = (df.groupby(["combo", "k_mode"])[
        ["auroc_all", "auroc_IH_vs_CH", "auroc_IL_vs_CL"]].mean())
    print(summary.round(4).to_string())

    # Per-pair chosen k for the tuned lap combos (transparency)
    tuned = df[df.k_mode == "tuned"]
    if not tuned.empty:
        print("\n=== CV-tuned lap top_k chosen per pair/combo ===")
        for _, r in tuned.sort_values(["combo", "pair"]).iterrows():
            print(f"  {r['combo']:20s} {r['pair']:18s} k={int(r['lap_k']):>3d} "
                  f"cv={r['cv_auroc']:.4f} | test all={r['auroc_all']:.4f} "
                  f"IHvCH={r['auroc_IH_vs_CH']:.4f} ILvCL={r['auroc_IL_vs_CL']:.4f}")


if __name__ == "__main__":
    main()
