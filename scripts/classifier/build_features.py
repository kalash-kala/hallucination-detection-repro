"""
Feature extraction from existing ranking-experiment cache.

For each (model, dataset) pair, builds:
  - Paired examples (mirrors probe_dataset.py logic): 2 examples per sample
      label 0 = "not wrong" = GT row (greedy itself when correct)
      label 1 = "wrong"     = greedy row (when greedy wrong) OR hardest distractor (when greedy correct)
  - Greedy-only examples: 1 row per sample = greedy hidden state
      label 0 = greedy correct, label 1 = greedy wrong

Hidden states are gathered for every layer in mid_bucket + late_bucket.
log_prob is the layer-independent s_ext (read once from scores_layer1.jsonl).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.parent.parent / "recovery-gaps-data" / "data"

# Original wide buckets (kept for backward compatibility)
BUCKETS = {
    "llama":   {"mid": list(range(11, 22)), "late": list(range(22, 33))},  # 32 layers
    "mistral": {"mid": list(range(11, 22)), "late": list(range(22, 33))},  # 32 layers
    "qwen":    {"mid": list(range(10, 19)), "late": list(range(19, 29))},  # 28 layers
    "gemma":   {"mid": list(range(17, 33)), "late": list(range(33, 49))},  # 48 layers
}

# Three bucket variants for testing layer selection strategies.
# Peak layers identified via Approach 2 (incorrect-only AUC):
#   roc_auc_score(gt_vs_reference_win, gt_score) on validation samples where
#   open_text_label=False (greedy was wrong), using the internal scorer.
# Narrow = peak ± 2 layers, clipped to region boundaries.
BUCKET_VARIANTS = {
    # Wide: all layers in region — model-specific, dataset-agnostic
    "wide": {
        "llama":   {"mid": list(range(11, 22)), "late": list(range(22, 33))},
        "mistral": {"mid": list(range(11, 22)), "late": list(range(22, 33))},
        "qwen":    {"mid": list(range(10, 19)), "late": list(range(19, 29))},
        "gemma":   {"mid": list(range(17, 33)), "late": list(range(33, 49))},
    },
    # Narrow: peak ± 2 layers within region — pair-specific
    # Peaks computed via Approach 2 (incorrect-only AUC) on TRAINING data only.
    # See compute_peak_layers.py for methodology.
    "narrow": {
        "llama_sciq":       {"mid": [13, 14, 15, 16, 17], "late": [27, 28, 29, 30, 31]},
        "llama_triviaqa":   {"mid": [11, 12, 13, 14, 15], "late": [30, 31, 32]},
        "llama_math":       {"mid": [19, 20, 21],          "late": [25, 26, 27, 28, 29]},
        "mistral_sciq":     {"mid": [16, 17, 18, 19, 20], "late": [30, 31, 32]},
        "mistral_triviaqa": {"mid": [16, 17, 18, 19, 20], "late": [29, 30, 31, 32]},
        "mistral_math":     {"mid": [18, 19, 20, 21],     "late": [28, 29, 30, 31, 32]},
        "qwen_sciq":        {"mid": [12, 13, 14, 15, 16], "late": [19, 20, 21]},
        "qwen_triviaqa":    {"mid": [14, 15, 16, 17, 18], "late": [19, 20, 21, 22]},
        "qwen_math":        {"mid": [12, 13, 14, 15, 16], "late": [19, 20, 21, 22]},
    },
    # Peak-only: single peak layer per region — pair-specific
    "peak_only": {
        "llama_sciq":       {"mid": [15], "late": [29]},
        "llama_triviaqa":   {"mid": [13], "late": [32]},
        "llama_math":       {"mid": [21], "late": [27]},
        "mistral_sciq":     {"mid": [18], "late": [32]},
        "mistral_triviaqa": {"mid": [18], "late": [31]},
        "mistral_math":     {"mid": [20], "late": [30]},
        "qwen_sciq":        {"mid": [14], "late": [19]},
        "qwen_triviaqa":    {"mid": [16], "late": [20]},
        "qwen_math":        {"mid": [14], "late": [20]},
    },
}

BUCKET_VARIANT_NAMES = ["wide", "narrow", "peak_only"]


_COMPUTED_PEAKS_PATH = Path(__file__).parent / "computed_peaks.json"


def get_buckets(bucket_variant: str, model: str, dataset: str) -> dict:
    """Return {"mid": [...], "late": [...]} for the given variant and pair.

    For narrow/peak_only pairs not hardcoded above (e.g. gemma), fall back to the
    peaks emitted by compute_peak_layers.py into computed_peaks.json."""
    v = BUCKET_VARIANTS[bucket_variant]
    if bucket_variant == "wide":
        return v[model]
    key = f"{model}_{dataset}"
    if key in v:
        return v[key]
    if _COMPUTED_PEAKS_PATH.exists():
        computed = json.loads(_COMPUTED_PEAKS_PATH.read_text())
        if key in computed.get(bucket_variant, {}):
            return computed[bucket_variant][key]
    raise KeyError(
        f"no bucket config for {key} variant={bucket_variant}; "
        f"run compute_peak_layers.py to populate {_COMPUTED_PEAKS_PATH.name}"
    )


@dataclass
class SplitFeatures:
    # Paired: 2 rows per sample (positive + negative)
    pair_hidden:    dict          # {layer_idx: ndarray (N_pair, D)}
    pair_log_probs: np.ndarray    # (N_pair,)
    pair_y:         np.ndarray    # (N_pair,)  0=not_wrong, 1=wrong

    # Greedy-only: 1 row per sample
    greedy_hidden:    dict        # {layer_idx: ndarray (N_samp, D)}
    greedy_log_probs: np.ndarray  # (N_samp,)
    greedy_y:         np.ndarray  # (N_samp,)  0=greedy_correct, 1=greedy_wrong
    greedy_ids:       list        # sample ids
    greedy_used_gt_fallback: np.ndarray  # (N_samp,) bool — True if GT used as proxy
    greedy_fallback_kinds:   list        # (N_samp,) ∈ {exact, loose, gt_fallback, hardest_distractor_fallback}


def _exp_dir(model: str, dataset: str) -> Path:
    return BASE / f"ranking_experiment_{model}_{dataset}"


def _read_split(model: str, dataset: str, split: str) -> list[dict]:
    path = _exp_dir(model, dataset) / "splits" / f"{split}.jsonl"
    return [json.loads(line) for line in path.open()]


def _read_score_map(model: str, dataset: str) -> dict:
    """Read scores_layer1.jsonl → {(sample_id, candidate): {s_ext, is_gt, is_greedy}}.
    s_ext is layer-independent, so any layer file works."""
    path = _exp_dir(model, dataset) / "cache" / "scores_layer1.jsonl"
    m = {}
    for line in path.open():
        r = json.loads(line)
        m[(r["id"], r["candidate"])] = r
    return m


def _read_index(model: str, dataset: str, layer: int) -> dict:
    path = _exp_dir(model, dataset) / "cache" / f"index_layer{layer}.json"
    return json.loads(path.read_text())


def _read_hidden(model: str, dataset: str, layer: int) -> np.ndarray:
    path = _exp_dir(model, dataset) / "cache" / f"hidden_layer{layer}.npz"
    return np.load(path)["hidden"]


def _pick_positive_candidate(sample: dict, score_map: dict) -> str:
    """For multi-alias datasets pick the GT alias with highest s_ext.
    Mirrors probe_dataset.py logic."""
    aliases = sample.get("gt_aliases") or [sample["ground_truth"]]
    if len(aliases) == 1:
        return aliases[0]
    best = aliases[0]
    best_s = score_map.get((sample["id"], best), {}).get("s_ext", -float("inf"))
    for a in aliases[1:]:
        s = score_map.get((sample["id"], a), {}).get("s_ext", -float("inf"))
        if s > best_s:
            best_s, best = s, a
    return best


def _hardest_distractor(sample: dict, score_map: dict) -> str:
    gt_norm = sample["ground_truth"].strip().lower()
    distractors = [c for c in sample["candidate_pool"] if c.strip().lower() != gt_norm]
    scored = [(c, score_map[(sample["id"], c)]["s_ext"]) for c in distractors]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[0][0]


def _pick_negative_candidate(sample: dict, score_map: dict, index: dict) -> str:
    """Mirror probe_dataset.pick_negative with the same out-of-pool fallback."""
    if sample["open_text_label"]:
        return _hardest_distractor(sample, score_map)
    # greedy is wrong → use greedy. Fallback to hardest distractor if greedy not in index.
    greedy = sample["greedy_prediction"]
    if f"{sample['id']}|||{greedy}" in index:
        return greedy
    return _hardest_distractor(sample, score_map)


_QUOTE_CHARS = " \t\n\"'`"


def _norm_loose(s: str) -> str:
    """Aggressive normalization: lowercase + strip whitespace + strip leading/trailing quote-like chars."""
    return s.strip().strip(_QUOTE_CHARS).lower()


def _find_greedy_candidate(sample: dict, score_map: dict, index: dict) -> tuple[str, str]:
    """Find the candidate string representing the greedy answer.

    Returns (candidate_string, fallback_kind) where fallback_kind ∈
      {"exact", "loose", "gt_fallback", "hardest_distractor_fallback"}.

    Match strategy:
      1. Case-insensitive match in candidate_pool ("exact").
      2. Quote-stripped case-insensitive match ("loose")  — handles cases like
         greedy="'Shut it'" vs pool entry "Shut it".
      3. Direct index lookup.
      4. For correct samples: GT alias (semantically equivalent per LLM judge).
      5. For incorrect samples: hardest distractor (mirrors probe_dataset.pick_negative
         fallback when greedy is out-of-pool). Both greedy and hardest_distractor are
         wrong answers, so the label remains 1.
    """
    gp = sample["greedy_prediction"]
    gp_strict = gp.strip().lower()
    gp_loose  = _norm_loose(gp)

    # Step 1: case-insensitive match
    for c in sample["candidate_pool"]:
        if c.strip().lower() == gp_strict:
            return c, "exact"

    # Step 2: loose (quote-stripped) match
    for c in sample["candidate_pool"]:
        if _norm_loose(c) == gp_loose:
            return c, "loose"

    # Step 3: direct index lookup
    direct_key = f"{sample['id']}|||{gp}"
    if direct_key in index:
        return gp, "exact"

    # Step 4/5: semantic fallback
    if sample["open_text_label"]:
        return _pick_positive_candidate(sample, score_map), "gt_fallback"
    return _hardest_distractor(sample, score_map), "hardest_distractor_fallback"


def extract_split(
    model: str, dataset: str, split: str, layers: list[int] | None = None
) -> SplitFeatures:
    """Extract all features for one split (train or test) of one (model, dataset) pair.

    layers: explicit list of layer indices to load. Defaults to all layers in
            BUCKETS[model]["mid"] + BUCKETS[model]["late"] for backward compatibility.
    """
    samples = _read_split(model, dataset, split)
    score_map = _read_score_map(model, dataset)

    if layers is None:
        layers = sorted(set(BUCKETS[model]["mid"] + BUCKETS[model]["late"]))
    else:
        layers = sorted(set(layers))

    # Load one index to determine candidate lookups (index is layer-independent in this pipeline)
    index_ref = _read_index(model, dataset, layers[0])

    # Resolve per-sample positive/negative/greedy candidate strings + log_probs
    pair_rows = []   # list of dicts: {key, log_prob, label}
    greedy_rows = [] # list of dicts: {key, log_prob, label, sample_id}
    n_skipped_pair = 0
    n_skipped_greedy = 0

    for s in samples:
        # ── paired examples ──
        pos_cand = _pick_positive_candidate(s, score_map)
        key_pos = f"{s['id']}|||{pos_cand}"
        if key_pos not in index_ref:
            n_skipped_pair += 1
            continue

        neg_cand = _pick_negative_candidate(s, score_map, index_ref)
        key_neg = f"{s['id']}|||{neg_cand}"
        if key_neg not in index_ref:
            n_skipped_pair += 1
            continue

        pos_s_ext = score_map[(s["id"], pos_cand)]["s_ext"]
        neg_s_ext = score_map[(s["id"], neg_cand)]["s_ext"]

        pair_rows.append({"key": key_pos, "log_prob": pos_s_ext, "label": 0})
        pair_rows.append({"key": key_neg, "log_prob": neg_s_ext, "label": 1})

        # ── greedy-only example ──
        greedy_cand, fallback_kind = _find_greedy_candidate(s, score_map, index_ref)
        greedy_key = f"{s['id']}|||{greedy_cand}"
        if greedy_key not in index_ref:
            n_skipped_greedy += 1
            continue
        greedy_s_ext = score_map[(s["id"], greedy_cand)]["s_ext"]
        greedy_label = 0 if s["open_text_label"] else 1
        greedy_rows.append({
            "key": greedy_key,
            "log_prob": greedy_s_ext,
            "label": greedy_label,
            "sample_id": s["id"],
            "fallback_kind": fallback_kind,
        })

    if n_skipped_pair or n_skipped_greedy:
        print(f"    [{split}] skipped {n_skipped_pair} pair / {n_skipped_greedy} greedy "
              f"(out of {len(samples)} samples)")

    # ── load hidden states per layer using each layer's own index ──
    pair_hidden = {}
    greedy_hidden = {}
    for L in layers:
        idx = _read_index(model, dataset, L)
        H = _read_hidden(model, dataset, L)
        pair_hidden[L] = np.stack(
            [H[idx[r["key"]]] for r in pair_rows], axis=0
        ).astype(np.float32)
        greedy_hidden[L] = np.stack(
            [H[idx[r["key"]]] for r in greedy_rows], axis=0
        ).astype(np.float32)

    return SplitFeatures(
        pair_hidden    = pair_hidden,
        pair_log_probs = np.array([r["log_prob"] for r in pair_rows], dtype=np.float32),
        pair_y         = np.array([r["label"]    for r in pair_rows], dtype=np.int64),
        greedy_hidden    = greedy_hidden,
        greedy_log_probs = np.array([r["log_prob"]  for r in greedy_rows], dtype=np.float32),
        greedy_y         = np.array([r["label"]     for r in greedy_rows], dtype=np.int64),
        greedy_ids       = [r["sample_id"] for r in greedy_rows],
        greedy_used_gt_fallback = np.array(
            [r["fallback_kind"] == "gt_fallback" for r in greedy_rows], dtype=bool
        ),
        greedy_fallback_kinds = [r["fallback_kind"] for r in greedy_rows],
    )


def build_feature_vector(
    hidden_by_layer: dict,
    log_probs: np.ndarray,
    layer_stats: dict,
    mid_bucket: list,
    late_bucket: list,
) -> np.ndarray:
    """Per-layer z-score → bucket-mean → concat → (N, 2D+1)."""
    N = next(iter(hidden_by_layer.values())).shape[0]
    D = next(iter(hidden_by_layer.values())).shape[1]

    def _bucket_mean(bucket: list) -> np.ndarray:
        acc = np.zeros((N, D), dtype=np.float32)
        for L in bucket:
            mu, sigma = layer_stats[L]
            acc += (hidden_by_layer[L] - mu) / sigma
        return acc / len(bucket)

    h_mid  = _bucket_mean(mid_bucket)
    h_late = _bucket_mean(late_bucket)
    return np.concatenate([h_mid, h_late, log_probs.reshape(-1, 1)], axis=1)


def compute_layer_stats(hidden_by_layer: dict, eps: float = 1e-6) -> dict:
    """Per-layer (mu, sigma) over the rows in this hidden dict."""
    stats = {}
    for L, H in hidden_by_layer.items():
        mu = H.mean(axis=0).astype(np.float32)
        sigma = H.std(axis=0).astype(np.float32) + eps
        stats[L] = (mu, sigma)
    return stats
