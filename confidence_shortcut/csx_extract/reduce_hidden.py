"""Reduce per-layer hidden shards to the three pooled hs_* matrices, then delete them.

This mirrors `full_natural/02_reduce_hidden.py` step for step, because the eight
qa8 pairs' features were produced by it and new pairs must land in the same
space.

  1. Peak layers, on TRAIN rows only: per-layer AUROC of StandardScaler ->
     LogisticRegression, fit on an 80/20 category-stratified internal split of
     train and scored on the held-out 20%. Peak per region, mid and late.
  2. Per scheme, X = [mean(mid bucket), mean(late bucket), s_ext], with each
     layer z-scored using TRAIN statistics only. Width 2*D+1.
  3. Delete the raw shards.

Two things are easy to get wrong and are called out where they happen: the
z-score and the peak search must both be fit on train rows only (otherwise test
information leaks into the feature definition), and the reduction is genuinely
destructive -- changing the pooling scheme later means re-running the GPU pass.

There is no train/test split in the store: arms are built later, by csx_probe,
and different arms use different splits. So "train rows" here means the first
70% of the row table in pool order, which is the fraction the published natural
arm uses (10,500 of 15,000). It is recorded in peaks.json so the choice is
visible rather than implied.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from csx_common import registry
from csx_common.store_schema import HS_SCHEMES

from . import config, writer

TRAIN_FRACTION = 0.70


def _train_mask(rows, seed: int = config.SEED) -> np.ndarray:
    """Category-stratified train rows. Stratifying matters because the peak
    search is a correctness classifier and the four cells are far from
    balanced."""
    rng = np.random.default_rng(seed)
    cats = rows["category"].to_numpy()
    mask = np.zeros(len(rows), dtype=bool)
    for c in np.unique(cats):
        idx = np.flatnonzero(cats == c)
        take = rng.permutation(len(idx))[:int(round(len(idx) * TRAIN_FRACTION))]
        mask[idx[take]] = True
    return mask


def _peak_layers(pair_key: str, segment: str, n_layers: int, wide: dict,
                 y: np.ndarray, tr: np.ndarray,
                 seed: int = config.SEED) -> tuple[dict, dict]:
    """Per-layer validation AUROC, and the argmax layer within each region."""
    rng = np.random.default_rng(seed)
    y_tr = y[tr]
    val = np.zeros(int(tr.sum()), dtype=bool)
    # 80/20 inside train, stratified on the label being predicted.
    for c in np.unique(y_tr):
        idx = np.flatnonzero(y_tr == c)
        take = rng.permutation(len(idx))[:int(round(len(idx) * config.VAL_FRACTION))]
        val[idx[take]] = True

    needed = sorted(set(wide["mid"]) | set(wide["late"]))
    auc: dict[int, float] = {}
    for L in needed:
        H = writer.load_shard(pair_key, segment, L)[tr].astype(np.float32)
        clf = Pipeline([("sc", StandardScaler()),
                        ("lr", LogisticRegression(**config.PEAK_LR))])
        clf.fit(H[~val], y_tr[~val])
        auc[L] = float(roc_auc_score(y_tr[val], clf.decision_function(H[val])))
        del H
    peaks = {r: max(wide[r], key=lambda L: auc[L]) for r in wide}
    return peaks, auc


def run(pair_key: str, *, keep_raw: bool = False, allow_partial: bool = False,
        verbose: bool = True) -> dict:
    pair = registry.get(pair_key)
    rows = writer.read_rows(pair_key)
    meta = writer.load_meta(pair_key)

    # Reduction is destructive and its output carries no memory of how many rows
    # produced it. Reducing a 20-row smoke pass would yield a perfectly
    # well-formed entry that csx_probe would fit probes on, and the only trace
    # would be a small n in a table nobody re-reads. So it is refused here.
    if meta.partial_limit and not allow_partial:
        raise RuntimeError(
            f"{pair_key}: phase 1 last ran under --limit {meta.partial_limit}, so "
            f"only {meta.n_rows} of {meta.n_kept} rows have hidden states. "
            f"Reducing that would produce a valid-looking entry with a silently "
            f"tiny n. Re-run stages 20 and 21 in full, or pass --allow-partial "
            f"if a smoke reduction is genuinely what you want.")

    n_layers = meta.layers
    if n_layers is None:
        raise RuntimeError(f"{pair_key}: meta.json has no model.layers")

    y = rows["category"].isin(registry.I_CATS).to_numpy().astype(int)
    tr = _train_mask(rows)
    wide = config.wide_buckets(n_layers)
    s_ext = rows["s_ext"].to_numpy().astype(np.float32).reshape(-1, 1)

    peaks_out: dict = {
        "train_fraction": TRAIN_FRACTION,
        "seed": config.SEED,
        "wide": wide,
        "segments": {},
    }

    for segment in pair.segments:
        peaks, auc = _peak_layers(pair_key, segment, n_layers, wide, y, tr)
        if verbose:
            print(f"  [{pair_key}/{segment}] peaks {peaks}", flush=True)

        # Per-layer z-score statistics, TRAIN rows only. Computed once per layer
        # and reused across schemes, since the schemes differ only in which
        # layers they average.
        stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for L in sorted(set(wide["mid"]) | set(wide["late"])):
            H = writer.load_shard(pair_key, segment, L).astype(np.float32)
            mu = H[tr].mean(axis=0)
            sd = H[tr].std(axis=0) + 1e-6
            stats[L] = (mu, sd)
            del H

        # All three schemes share one npz per segment, matching the store
        # contract. They differ only in which layers each region averages.
        arrays, used = {}, {}
        for scheme in HS_SCHEMES:
            buckets = config.buckets_for(scheme, peaks, wide)
            parts = []
            for region in ("mid", "late"):
                lls = buckets[region]
                accum = None
                for L in lls:
                    mu, sd = stats[L]
                    z = (writer.load_shard(pair_key, segment, L).astype(np.float32)
                         - mu) / sd
                    accum = z if accum is None else accum + z
                parts.append((accum / len(lls)).astype(np.float16))
            X = np.concatenate(parts + [s_ext.astype(np.float16)], axis=1)
            if not np.isfinite(X.astype(np.float32)).all():
                raise RuntimeError(
                    f"{pair_key}/{segment}/{scheme}: non-finite feature values")
            arrays[scheme] = X
            used[scheme] = buckets

        writer.save_hs(pair_key, segment, arrays)
        peaks_out["segments"][segment] = {
            "peaks": peaks,
            "buckets": used,
            "layer_auc": {str(k): v for k, v in auc.items()},
        }

    writer.save_peaks(pair_key, peaks_out)

    freed = 0
    if not keep_raw:
        freed = writer.drop_shards(pair_key)
        if verbose:
            print(f"  [{pair_key}] raw shards deleted ({freed / 1e9:.1f} GB)",
                  flush=True)

    return {"peaks": peaks_out, "raw_bytes_freed": int(freed),
            "raw_deleted": not keep_raw}
