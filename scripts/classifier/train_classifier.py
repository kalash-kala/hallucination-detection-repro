"""
Train the bucketed hidden-state error detector for all 9 (model, dataset) pairs.

For each pair × bucket variant × LR variant, fits a LogisticRegression:
  Bucket variants:
    - wide:       all layers in mid/late region (model-specific)
    - narrow:     peak ± 2 layers within region (pair-specific, Approach 2 peaks)
    - peak_only:  single peak layer per region (pair-specific, Approach 2 peaks)

  LR variants:
    - default:    C=1.0  (sklearn default)
    - strong_reg: C=0.1  (stronger L2)

Both LR variants use class_weight="balanced". Artifacts saved under:
  ranking_experiment_{model}_{dataset}/classifier/{bucket_variant}/{lr_variant}/

Peak layers are selected via Approach 2 (incorrect-only AUC):
  roc_auc_score(gt_vs_reference_win, gt_score) on validation samples where
  open_text_label=False, using the internal scorer from per_sample_layer*.jsonl.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_features import (
    BASE,
    BUCKET_VARIANT_NAMES,
    build_feature_vector,
    compute_layer_stats,
    extract_split,
    get_buckets,
)

PAIRS = [
    ("llama",   "sciq"),
    ("llama",   "triviaqa"),
    ("llama",   "math"),
    ("mistral", "sciq"),
    ("mistral", "triviaqa"),
    ("mistral", "math"),
    ("qwen",    "sciq"),
    ("qwen",    "triviaqa"),
    ("qwen",    "math"),
    ("gemma",   "sciq"),
    ("gemma",   "triviaqa"),
    ("gemma",   "math"),
]

LR_VARIANTS = {
    "default":    {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
    "strong_reg": {"C": 0.1, "class_weight": "balanced", "max_iter": 1000},
}


def _eval(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "auc_roc":             float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
        "accuracy":            float(accuracy_score(y_true, y_pred)),
        "precision_incorrect": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_incorrect":    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_incorrect":        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "n":                   int(len(y_true)),
        "n_wrong":             int(int(y_true.sum())),
    }
    return metrics


def train_one_pair(model: str, dataset: str, bucket_variant: str, log_lines: list) -> dict:
    print(f"\n=== {model}/{dataset}  [bucket={bucket_variant}] ===")
    log_lines.append(f"=== {model}/{dataset}  [bucket={bucket_variant}] ===")

    buckets = get_buckets(bucket_variant, model, dataset)
    mid  = buckets["mid"]
    late = buckets["late"]
    layers = sorted(set(mid + late))

    t0 = time.time()
    print("  extracting train features...")
    train = extract_split(model, dataset, "train", layers=layers)
    print("  extracting test features...")
    test  = extract_split(model, dataset, "test",  layers=layers)
    print(f"  extraction took {time.time() - t0:.1f}s")
    log_lines.append(f"  extraction took {time.time() - t0:.1f}s")

    layer_stats = compute_layer_stats(train.pair_hidden)
    hidden_dim = next(iter(train.pair_hidden.values())).shape[1]

    X_train      = build_feature_vector(train.pair_hidden, train.pair_log_probs,
                                        layer_stats, mid, late)
    y_train      = train.pair_y
    X_test_pair  = build_feature_vector(test.pair_hidden, test.pair_log_probs,
                                        layer_stats, mid, late)
    y_test_pair  = test.pair_y
    X_test_greedy = build_feature_vector(test.greedy_hidden, test.greedy_log_probs,
                                         layer_stats, mid, late)
    y_test_greedy = test.greedy_y

    print(f"  shapes: X_train={X_train.shape} X_test_pair={X_test_pair.shape} "
          f"X_test_greedy={X_test_greedy.shape}")
    print(f"  class balance: train wrong={y_train.mean():.3f} "
          f"test_pair wrong={y_test_pair.mean():.3f} "
          f"test_greedy wrong={y_test_greedy.mean():.3f}")
    log_lines.append(
        f"  X_train={X_train.shape} X_test_pair={X_test_pair.shape} "
        f"X_test_greedy={X_test_greedy.shape}"
    )

    clf_root = BASE / f"ranking_experiment_{model}_{dataset}" / "classifier" / bucket_variant
    clf_root.mkdir(parents=True, exist_ok=True)

    with (clf_root / "layer_stats.pkl").open("wb") as f:
        pickle.dump(layer_stats, f)

    config = {
        "model":          model,
        "dataset":        dataset,
        "bucket_variant": bucket_variant,
        "n_layers":       len(layers),
        "hidden_dim":     hidden_dim,
        "mid_bucket":     mid,
        "late_bucket":    late,
        "label_map":      {"0": "not_wrong", "1": "wrong"},
        "n_train_pair":   int(len(y_train)),
        "n_test_pair":    int(len(y_test_pair)),
        "n_test_greedy":  int(len(y_test_greedy)),
        "n_train_greedy_gt_fallback": int(train.greedy_used_gt_fallback.sum()),
        "n_test_greedy_gt_fallback":  int(test.greedy_used_gt_fallback.sum()),
    }
    (clf_root / "config.json").write_text(json.dumps(config, indent=2))

    summary = {"model": model, "dataset": dataset, "bucket_variant": bucket_variant, "lr_variants": {}}

    for vname, params in LR_VARIANTS.items():
        print(f"  training lr_variant '{vname}' (C={params['C']})...")
        log_lines.append(f"  lr_variant {vname}  params={params}")

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**params)),
        ])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        log_lines.append(f"    fit took {time.time() - t0:.1f}s")

        proba_pair   = pipe.predict_proba(X_test_pair)[:, 1]
        proba_greedy = pipe.predict_proba(X_test_greedy)[:, 1]

        full_metrics   = _eval(y_test_pair,   proba_pair)
        greedy_metrics = _eval(y_test_greedy, proba_greedy)

        eval_metrics = {
            "full":        full_metrics,
            "greedy_only": greedy_metrics,
            "lr_variant":  vname,
            "params":      {k: v for k, v in params.items()},
        }

        vdir = clf_root / vname
        vdir.mkdir(parents=True, exist_ok=True)
        with (vdir / "classifier.pkl").open("wb") as f:
            pickle.dump(pipe, f)
        (vdir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2))

        summary["lr_variants"][vname] = eval_metrics
        print(f"    full AUC={full_metrics['auc_roc']:.4f}  "
              f"greedy-only AUC={greedy_metrics['auc_roc']:.4f}  "
              f"greedy acc={greedy_metrics['accuracy']:.4f}")
        log_lines.append(
            f"    full_auc={full_metrics['auc_roc']:.4f} "
            f"greedy_auc={greedy_metrics['auc_roc']:.4f}"
        )

    (clf_root / "training_log.txt").write_text("\n".join(log_lines) + "\n")
    return summary


def main():
    all_summary = []
    log_lines = []
    for bucket_variant in BUCKET_VARIANT_NAMES:
        print(f"\n{'='*60}")
        print(f"BUCKET VARIANT: {bucket_variant}")
        print(f"{'='*60}")
        for model, dataset in PAIRS:
            s = train_one_pair(model, dataset, bucket_variant, log_lines)
            all_summary.append(s)

    rows = []
    for s in all_summary:
        for vname, m in s["lr_variants"].items():
            rows.append({
                "model":               s["model"],
                "dataset":             s["dataset"],
                "bucket_variant":      s["bucket_variant"],
                "lr_variant":          vname,
                "C":                   m["params"]["C"],
                "n_train":             m["full"]["n"],
                "n_test_pair":         m["full"]["n"],
                "n_test_greedy":       m["greedy_only"]["n"],
                "auc_full":            m["full"]["auc_roc"],
                "auc_greedy_only":     m["greedy_only"]["auc_roc"],
                "accuracy_greedy":     m["greedy_only"]["accuracy"],
                "precision_greedy":    m["greedy_only"]["precision_incorrect"],
                "recall_greedy":       m["greedy_only"]["recall_incorrect"],
                "f1_greedy":           m["greedy_only"]["f1_incorrect"],
                "accuracy_full":       m["full"]["accuracy"],
                "f1_full":             m["full"]["f1_incorrect"],
            })

    import csv
    summary_path = BASE / "classifier_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[summary] {summary_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
