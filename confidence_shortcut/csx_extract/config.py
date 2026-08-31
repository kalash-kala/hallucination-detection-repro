"""Extraction constants, and the layer-bucket rule.

Everything here that reproduces published behaviour is derived from the existing
pipeline rather than re-chosen, and the derivation is asserted in tests so a
drift shows up as a failure instead of a slightly different number.
"""

from __future__ import annotations

EXTRACTOR_VERSION = "csx_extract/1"

# Stored top-k width per head. TOP_K_GRID in the probe maxes at 50, so keeping 50
# is lossless for every spectral family while cutting [L,H,S] to [L,H,50] -- the
# difference between ~60 GB and ~300 MB for one vqav2 pair.
EXTRACT_TOP_K = 50
SINK_K = 10

# s_ext = sum(log P(answer tokens)) / n_tokens**ALPHA, i.e. the mean answer-token
# log-probability at ALPHA=1.0. From scripts/ranking/config.py:29 and
# external_scorer.py:56; it is the +1 column of every hs_* feature vector.
S_EXT_ALPHA = 1.0

# Peak-layer selection (full_natural/02_reduce_hidden.py).
PEAK_LR = dict(C=1.0, class_weight="balanced", max_iter=500)
VAL_FRACTION = 0.2
SEED = 42

# Micro-batching. Phase 2 materialises [L,H,S,S] attention, which is quadratic in
# S, so it goes one row at a time; phase 1 is linear and can batch.
PHASE1_BATCH = 8
PHASE2_BATCH = 1


# ── layer buckets ────────────────────────────────────────────────────────────
# The published BUCKET_VARIANTS["wide"] table (classifier/build_features.py:37)
# hardcodes mid/late layer ranges for the four qa8 backbones only. New pairs need
# the same regions at other depths, so the rule is recovered as fractions of the
# block count:
#
#     mid  = [round(0.35 L) .. round(0.6875 L) - 1]
#     late = [round(0.6875 L) .. L]
#
# This reproduces all four published configs EXACTLY -- llama/mistral (32),
# qwen (28) and gemma (48) -- which is what makes it a recovered rule rather than
# a fresh guess. tests/test_buckets.py pins that.
MID_FRAC = 0.35
LATE_FRAC = 0.6875


def wide_buckets(n_layers: int) -> dict[str, list[int]]:
    """Mid/late layer regions for a model with `n_layers` transformer blocks.

    Layers are 1-indexed to match `hidden_states[L]`, where index 0 is the
    embedding output and is never selectable.
    """
    if n_layers < 4:
        raise ValueError(f"n_layers={n_layers} is too small to split into regions")
    mid_lo = round(MID_FRAC * n_layers)
    late_lo = round(LATE_FRAC * n_layers)
    return {
        "mid": list(range(mid_lo, late_lo)),
        "late": list(range(late_lo, n_layers + 1)),
    }


def narrow_buckets(peaks: dict[str, int], wide: dict[str, list[int]]) -> dict:
    """peak +/- 2, clipped to the region (02_reduce_hidden.py)."""
    return {r: [L for L in wide[r] if peaks[r] - 2 <= L <= peaks[r] + 2]
            for r in wide}


def peak_only_buckets(peaks: dict[str, int], wide: dict[str, list[int]]) -> dict:
    return {r: [peaks[r]] for r in wide}


def buckets_for(scheme: str, peaks: dict[str, int],
                wide: dict[str, list[int]]) -> dict[str, list[int]]:
    if scheme == "hs_wide":
        return wide
    if scheme == "hs_narrow":
        return narrow_buckets(peaks, wide)
    if scheme == "hs_peak_only":
        return peak_only_buckets(peaks, wide)
    raise KeyError(f"unknown hs scheme {scheme!r}")
