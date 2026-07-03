"""
Compute Approach 2 peak layers using TRAINING data only.

Approach 2 (incorrect-only AUC):
  roc_auc_score(gt_vs_reference_win, gt_score) on training samples where
  open_text_label=False (greedy was wrong), using s_int from the per-layer probe.

  gt_score          = s_int of ground truth candidate at layer L
  reference         = hardest distractor (highest s_ext distractor in candidate_pool)
  gt_vs_reference_win = 1 if gt_score > s_int(reference), else 0

  s_int is computed by: sigmoid(probe.decision_function(scaler.transform(hidden)))
  probes are already trained on the training split — we apply them to training
  hidden states to get s_int, then compute AUC over incorrect-greedy samples.

Outputs:
  - Prints per-layer AUC tables for mid and late regions
  - Writes updated hook_layer.txt
  - Prints the Python dict to paste into build_features.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import joblib
from scipy.special import expit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")  # suppress sklearn version warnings

BASE = Path(__file__).parent.parent.parent / "recovery-gaps-data" / "data"

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

# Wide layer regions per model (same as BUCKET_VARIANTS["wide"])
REGIONS = {
    "llama":   {"mid": list(range(11, 22)), "late": list(range(22, 33))},
    "mistral": {"mid": list(range(11, 22)), "late": list(range(22, 33))},
    "qwen":    {"mid": list(range(10, 19)), "late": list(range(19, 29))},
    "gemma":   {"mid": list(range(17, 33)), "late": list(range(33, 49))},
}


def _clip_narrow(peak: int, region: list[int]) -> list[int]:
    """Return peak±2 clipped to region boundaries."""
    lo, hi = min(region), max(region)
    return [l for l in range(peak - 2, peak + 3) if lo <= l <= hi]


def compute_pair_peaks(model: str, dataset: str) -> dict:
    exp_dir = BASE / f"ranking_experiment_{model}_{dataset}"
    probe_dir = exp_dir / "probe"

    # Load training split for open_text_label and candidate_pool
    train_samples = {
        s["id"]: s
        for s in [json.loads(l) for l in (exp_dir / "splits" / "train.jsonl").open()]
    }

    # Load s_ext scores for all candidates (to find hardest distractor)
    score_map: dict[tuple, float] = {}
    for l in (exp_dir / "cache" / "scores_layer1.jsonl").open():
        r = json.loads(l)
        score_map[(r["id"], r["candidate"])] = r["s_ext"]

    # Build set of incorrect training sample ids
    incorrect_ids = {sid for sid, s in train_samples.items() if not s["open_text_label"]}
    print(f"  {model}/{dataset}: {len(incorrect_ids)} incorrect training samples")

    # For each incorrect sample: find GT candidate and hardest distractor
    gt_cands: dict[str, str] = {}    # sample_id -> GT candidate string
    hd_cands: dict[str, str] = {}    # sample_id -> hardest distractor string

    for sid in incorrect_ids:
        s = train_samples[sid]
        gt_norm = s["ground_truth"].strip().lower()

        # GT: pick alias with highest s_ext (mirrors build_features logic)
        aliases = s.get("gt_aliases") or [s["ground_truth"]]
        best_alias = max(aliases, key=lambda a: score_map.get((sid, a), -1e9))
        gt_cands[sid] = best_alias

        # Hardest distractor: highest s_ext among non-GT candidates
        distractors = [(c, score_map.get((sid, c), -1e9))
                       for c in s["candidate_pool"]
                       if c.strip().lower() != gt_norm]
        distractors.sort(key=lambda t: t[1], reverse=True)
        hd_cands[sid] = distractors[0][0]

    regions = REGIONS[model]
    all_layers = sorted(set(regions["mid"] + regions["late"]))

    layer_aucs: dict[int, float] = {}

    for L in all_layers:
        H = np.load(exp_dir / "cache" / f"hidden_layer{L}.npz")["hidden"]
        idx = json.loads((exp_dir / "cache" / f"index_layer{L}.json").read_text())
        scaler = joblib.load(probe_dir / f"scaler_layer{L}.joblib")
        probe  = joblib.load(probe_dir / f"probe_layer{L}.joblib")

        # Compute s_int for GT and hardest distractor for each incorrect sample
        gt_keys = [f"{sid}|||{gt_cands[sid]}"  for sid in incorrect_ids]
        hd_keys = [f"{sid}|||{hd_cands[sid]}"  for sid in incorrect_ids]

        # Skip if any key missing from index
        valid_mask = [gk in idx and hk in idx for gk, hk in zip(gt_keys, hd_keys)]
        valid_sids   = [sid for sid, v in zip(incorrect_ids, valid_mask) if v]
        valid_gt_keys = [gk for gk, v in zip(gt_keys, valid_mask) if v]
        valid_hd_keys = [hk for hk, v in zip(hd_keys, valid_mask) if v]

        if len(valid_sids) < 10:
            layer_aucs[L] = float("nan")
            continue

        X_gt = np.stack([H[idx[k]] for k in valid_gt_keys]).astype(np.float32)
        X_hd = np.stack([H[idx[k]] for k in valid_hd_keys]).astype(np.float32)

        s_gt = expit(probe.decision_function(scaler.transform(X_gt)))
        s_hd = expit(probe.decision_function(scaler.transform(X_hd)))

        gt_vs_ref_win = (s_gt > s_hd).astype(int)

        if len(set(gt_vs_ref_win)) < 2:
            layer_aucs[L] = float("nan")
            continue

        layer_aucs[L] = float(roc_auc_score(gt_vs_ref_win, s_gt))

    # Find peak per region
    peaks = {}
    for region_name, region_layers in regions.items():
        valid = {L: auc for L, auc in layer_aucs.items()
                 if L in region_layers and not np.isnan(auc)}
        if not valid:
            peaks[region_name] = {"peak": region_layers[len(region_layers)//2],
                                  "auc": float("nan")}
        else:
            peak_L = max(valid, key=valid.__getitem__)
            peaks[region_name] = {"peak": peak_L, "auc": valid[peak_L],
                                   "all_aucs": {L: valid[L] for L in sorted(valid)}}

    return {
        "model": model, "dataset": dataset,
        "peaks": peaks, "layer_aucs": layer_aucs,
    }


def main():
    results = []
    for model, dataset in PAIRS:
        print(f"\n=== {model}/{dataset} ===")
        r = compute_pair_peaks(model, dataset)
        results.append(r)

        for region_name, pdata in r["peaks"].items():
            print(f"  {region_name}: peak_layer={pdata['peak']}  auc={pdata['auc']:.4f}")
            if "all_aucs" in pdata:
                aucs_str = "  ".join(f"L{L}={v:.3f}" for L, v in pdata["all_aucs"].items())
                print(f"    [{aucs_str}]")

    # ── Build and print bucket variant configs ──
    print("\n" + "=" * 70)
    print("BUCKET VARIANT CONFIGS (paste into build_features.py)")
    print("=" * 70)

    narrow: dict[str, dict] = {}
    peak_only: dict[str, dict] = {}

    for r in results:
        key = f"{r['model']}_{r['dataset']}"
        regions = REGIONS[r["model"]]
        mid_peak  = r["peaks"]["mid"]["peak"]
        late_peak = r["peaks"]["late"]["peak"]
        narrow[key] = {
            "mid":  _clip_narrow(mid_peak,  regions["mid"]),
            "late": _clip_narrow(late_peak, regions["late"]),
        }
        peak_only[key] = {
            "mid":  [mid_peak],
            "late": [late_peak],
        }

    print('\n"narrow": {')
    for key, v in narrow.items():
        print(f'    "{key}": {{"mid": {v["mid"]}, "late": {v["late"]}}},')
    print("},")

    print('\n"peak_only": {')
    for key, v in peak_only.items():
        print(f'    "{key}": {{"mid": {v["mid"]}, "late": {v["late"]}}},')
    print("},")

    # ── Machine-readable dump so build_features.get_buckets() can pick up pairs
    #    that are not hardcoded in BUCKET_VARIANTS (e.g. gemma) without manual paste. ──
    peaks_json = Path(__file__).parent / "computed_peaks.json"
    existing = {}
    if peaks_json.exists():
        existing = json.loads(peaks_json.read_text())
    existing.setdefault("narrow", {}).update(narrow)
    existing.setdefault("peak_only", {}).update(peak_only)
    peaks_json.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"\n[written] {peaks_json}")

    # ── Write hook_layer.txt ──
    out_path = Path(__file__).parent.parent.parent / "hook_layer_train.txt"
    header = ("model,dataset,"
              "mid_peak_layer,mid_peak_auc,"
              "late_peak_layer,late_peak_auc,"
              "narrow_mid_layers,narrow_late_layers")
    rows = [header]
    for r in results:
        mid  = r["peaks"]["mid"]
        late = r["peaks"]["late"]
        key  = f"{r['model']}_{r['dataset']}"
        rows.append(
            f"{r['model']},{r['dataset']},"
            f"{mid['peak']},{mid['auc']:.4f},"
            f"{late['peak']},{late['auc']:.4f},"
            f"\"{narrow[key]['mid']}\",\"{narrow[key]['late']}\""
        )
    out_path.write_text("\n".join(rows) + "\n")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
