"""
Utility functions for computing distractor-based features for classifier training.

Features computed:
  f1: s_int_greedy(L_best)              - absolute internal probe score at best layer
  f2: s_ext_lmhead_greedy               - external LM head score (model confidence)
  f3: Δ_int(L_best)                    - greedy score minus mean distractor score (internal)
  f4: div_int_ext                       - divergence between internal and external deltas
  f5: var_s_int_dist(L_best)           - variance of distractor internal scores
  f6: max_s_int_dist - s_int_greedy     - gap between hardest distractor and greedy
  f7: vote_rate_internal                - fraction of distractors that greedy beats
  f8: s_int_greedy(L_best) - s_int_greedy(L_early) - internal probe trajectory

New features (added after feature-importance analysis, see
FEATURE_IMPORTANCE_ANALYSIS.md):
  g1: len_greedy                        - greedy answer length (n_answer_tokens)
  g2: ext_per_token                     - s_ext_final / n_answer_tokens (length-normed conf.)
  g4: int_traj_mean                     - mean s_int over ALL layers (robust internal conf.)
  g8: len_delta                         - len_greedy - mean(len_distractors)

Note: f5 (distractor variance) and f7 (vote rate) are computed but EXCLUDED from
the production feature set (FEATURE_NAMES) - drop-ablation showed they are
near dead-weight and removing them slightly improves both classifiers.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def auto_select_layers(model: str, dataset: str, model_layers: dict, split_data=None):
    """
    Auto-select L_best and L_early for a model/dataset pair.

    L_best: layer with highest mean divergence on training split (incorrect_high samples)
    L_early: fixed layer 5 for all models

    Args:
        model: model name
        dataset: dataset name
        model_layers: dict mapping model -> num_layers
        split_data: optional pre-computed data for more accurate selection

    Returns:
        (L_best, L_early, divergence_per_layer) tuple
    """
    L_early = 5
    n_layers = model_layers.get(model, 32)

    if split_data is None:
        # Fallback: use middle-to-late layer as best
        L_best = max(n_layers // 2, 15)
        logger.warning(
            f"[{model}/{dataset}] No split data provided; using heuristic L_best={L_best}"
        )
        return L_best, L_early, {}

    # Compute mean divergence per layer on incorrect_high samples in training split
    divergence_per_layer = {}
    for L in range(1, n_layers + 1):
        divs = [
            row.get(f'div_int_ext_L{L}')
            for row in split_data
            if row.get('category') == 'incorrect_high' and row.get(f'div_int_ext_L{L}') is not None
        ]
        if divs:
            divergence_per_layer[L] = float(np.mean(np.abs(divs)))

    if divergence_per_layer:
        L_best = max(divergence_per_layer.keys(), key=lambda k: divergence_per_layer[k])
        logger.info(
            f"[{model}/{dataset}] Selected L_best={L_best} "
            f"(mean |divergence|={divergence_per_layer[L_best]:.4f})"
        )
    else:
        L_best = max(n_layers // 2, 15)
        logger.warning(
            f"[{model}/{dataset}] No divergence data found; using heuristic L_best={L_best}"
        )

    return L_best, L_early, divergence_per_layer


def compute_features(greedy_row: dict, distractor_rows: list[dict], L_best: int, L_early: int):
    """
    Compute 8 features for one sample.

    Args:
        greedy_row: sidecar entry for greedy prediction
        distractor_rows: list of distractor entries
        L_best: best layer for internal probe
        L_early: early layer for trajectory

    Returns:
        dict with 8 features, or None if computation fails
    """
    try:
        L_best_str = str(L_best)
        L_early_str = str(L_early)

        # f1: s_int_greedy(L_best)
        s_int_greedy_best = greedy_row.get('s_int', {}).get(L_best_str)
        if s_int_greedy_best is None:
            return None
        f1 = float(s_int_greedy_best)

        # f2: s_ext_lmhead_greedy (final layer, layer-independent)
        # Get final layer dynamically from the keys of s_ext_L
        s_ext_L_dict = greedy_row.get('s_ext_L', {})
        if not s_ext_L_dict:
            return None
        final_layer = str(max(int(k) for k in s_ext_L_dict.keys()))
        s_ext_greedy = s_ext_L_dict.get(final_layer)
        if s_ext_greedy is None:
            return None
        f2 = float(s_ext_greedy)

        # Distractor scores: mean and variance
        dist_s_int_best = []
        dist_s_int_early = []
        for d in distractor_rows:
            s_int = d.get('s_int', {}).get(L_best_str)
            if s_int is not None:
                dist_s_int_best.append(float(s_int))

            s_int_early = d.get('s_int', {}).get(L_early_str)
            if s_int_early is not None:
                dist_s_int_early.append(float(s_int_early))

        if not dist_s_int_best:
            return None

        mean_s_int_dist_best = float(np.mean(dist_s_int_best))
        max_s_int_dist_best = float(np.max(dist_s_int_best))
        var_s_int_dist_best = float(np.var(dist_s_int_best))

        # f3: Δ_int(L_best) = s_int_greedy - mean(s_int_dist)
        f3 = f1 - mean_s_int_dist_best

        # f4: div_int_ext = Δ_int - Δ_ext_lmhead (OPTION A: probability space via exp(s_ext))
        # Δ_ext = s_ext_greedy - mean(s_ext_dist), both converted to probability space
        # Get external scores from final layer of each distractor
        dist_s_ext_final = []
        for d in distractor_rows:
            d_s_ext_L = d.get('s_ext_L', {})
            if d_s_ext_L:
                d_final_layer = str(max(int(k) for k in d_s_ext_L.keys()))
                d_s_ext = d_s_ext_L.get(d_final_layer)
                if d_s_ext is not None:
                    dist_s_ext_final.append(float(d_s_ext))

        if dist_s_ext_final:
            # Convert log probabilities to probability space: exp(s_ext)
            # This ensures both deltas are in [0,1] and comparable
            s_ext_greedy_prob = float(np.exp(f2))
            dist_s_ext_probs = [float(np.exp(s)) for s in dist_s_ext_final]
            mean_s_ext_dist_prob = float(np.mean(dist_s_ext_probs))

            # Now both deltas are in probability space [0, 1]
            delta_ext = s_ext_greedy_prob - mean_s_ext_dist_prob
            f4 = f3 - delta_ext
        else:
            # Fallback: no external scores available, use internal delta only
            f4 = f3

        # f5: var_s_int_dist(L_best)
        f5 = var_s_int_dist_best

        # f6: max_s_int_dist - s_int_greedy
        f6 = max_s_int_dist_best - f1

        # f7: vote_rate_internal = fraction of distractors greedy beats
        n_beaten = sum(1 for d_score in dist_s_int_best if f1 > d_score)
        f7 = float(n_beaten) / len(dist_s_int_best)

        # f8: trajectory = s_int_greedy(L_best) - s_int_greedy(L_early)
        s_int_greedy_early = greedy_row.get('s_int', {}).get(L_early_str)
        if s_int_greedy_early is None:
            f8 = 0.0  # fallback
        else:
            f8 = f1 - float(s_int_greedy_early)

        # ── New features (validated via feature-importance analysis) ──────────
        # g1: greedy answer length (n_answer_tokens). Strongest single new signal,
        #     especially for IH-vs-IL: confidently-wrong answers are shorter.
        len_greedy = greedy_row.get('n_answer_tokens')
        if not len_greedy or len_greedy <= 0:
            return None
        g1 = float(len_greedy)

        # g2: length-normalized external confidence (de-confounds f2, which mixes
        #     per-token confidence with answer length since s_ext is a log-prob sum).
        g2 = f2 / g1

        # g4: mean internal probe score over ALL layers (more robust than the
        #     single-layer f1).
        s_int_greedy_all = greedy_row.get('s_int', {})
        traj_vals = [float(v) for v in s_int_greedy_all.values() if v is not None]
        g4 = float(np.mean(traj_vals)) if traj_vals else f1

        # g8: greedy length minus mean distractor length (length-confound proxy).
        dist_lens = [
            float(d.get('n_answer_tokens'))
            for d in distractor_rows
            if d.get('n_answer_tokens')
        ]
        g8 = g1 - float(np.mean(dist_lens)) if dist_lens else 0.0

        return {
            'f1_s_int_best': f1,
            'f2_s_ext_final': f2,
            'f3_delta_int': f3,
            'f4_div_int_ext': f4,
            'f5_var_dist': f5,
            'f6_hardest_gap': f6,
            'f7_vote_rate': f7,
            'f8_trajectory': f8,
            'g1_len_greedy': g1,
            'g2_ext_per_token': g2,
            'g4_int_traj_mean': g4,
            'g8_len_delta': g8,
        }

    except Exception as e:
        logger.warning(f"Feature computation failed: {e}")
        return None


def load_and_prepare_data_all(model: str, dataset: str, base_dir: Path, L_best: int, L_early: int):
    """
    Load all samples (correct + incorrect) with computed features.

    Returns:
        pd.DataFrame with columns: [id, category, open_text_label, f1..f8, target]
        target = 1 if open_text_label==True (correct), else 0
    """
    exp_dir = base_dir / f"ranking_experiment_{model}_{dataset}"
    cache = exp_dir / "cache"

    greedy_path = cache / "greedy_sidecar.jsonl"
    dist_path = cache / "distractor_sidecar.jsonl"

    if not greedy_path.exists() or not dist_path.exists():
        logger.warning(f"Missing sidecars for {model}/{dataset}")
        return None

    greedy_sidecar = {r['id']: r for r in load_jsonl(greedy_path)}
    dist_sidecar = load_jsonl(dist_path)
    dist_by_id = {r['id']: r.get('distractors', []) for r in dist_sidecar}

    records = []
    for sid, greedy_row in greedy_sidecar.items():
        if sid not in dist_by_id:
            continue

        distractors = dist_by_id[sid]
        if not distractors:
            continue

        features = compute_features(greedy_row, distractors, L_best, L_early)
        if features is None:
            continue

        record = {
            'id': sid,
            'category': greedy_row.get('category'),
            'open_text_label': greedy_row.get('open_text_label'),
            **features,
        }

        # Target for C vs I: 1 if correct, 0 if incorrect
        record['target_c_vs_i'] = int(greedy_row.get('open_text_label', False))

        records.append(record)

    if not records:
        logger.warning(f"No valid samples for {model}/{dataset}")
        return None

    df = pd.DataFrame(records)
    logger.info(
        f"[{model}/{dataset}] Loaded {len(df)} all samples. "
        f"Correct: {(df['target_c_vs_i']==1).sum()}, Incorrect: {(df['target_c_vs_i']==0).sum()}"
    )
    return df


def load_and_prepare_data_incorrect_only(model: str, dataset: str, base_dir: Path, L_best: int, L_early: int):
    """
    Load only incorrect samples with computed features.

    Returns:
        pd.DataFrame with columns: [id, category, open_text_label, f1..f8, target]
        target = 1 if category=='incorrect_high', else 0
    """
    df_all = load_and_prepare_data_all(model, dataset, base_dir, L_best, L_early)
    if df_all is None:
        return None

    df_incorrect = df_all[df_all['target_c_vs_i'] == 0].copy()

    if df_incorrect.empty:
        logger.warning(f"No incorrect samples for {model}/{dataset}")
        return None

    # Target for IH vs IL: 1 if incorrect_high, 0 if incorrect_low
    df_incorrect['target_ih_vs_il'] = (df_incorrect['category'] == 'incorrect_high').astype(int)

    logger.info(
        f"[{model}/{dataset}] Loaded {len(df_incorrect)} incorrect samples. "
        f"High: {(df_incorrect['target_ih_vs_il']==1).sum()}, Low: {(df_incorrect['target_ih_vs_il']==0).sum()}"
    )
    return df_incorrect
