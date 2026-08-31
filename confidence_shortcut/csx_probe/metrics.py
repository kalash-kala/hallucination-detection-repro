"""AUROC, the contrast cells, and the bootstrap.

One AUROC implementation, used everywhere. The orientation convention is global
and has no exceptions: **larger score => more incorrect**, with no per-head sign
flips. That is worth stating plainly because the most useful observation in the
whole study depends on it -- the confidence head (`sep`) scoring ~0.000 on `IHvCL`
is the *finding*, not a bug to be corrected by flipping a sign. A codebase that
flips signs to keep every number above 0.5 cannot express that result at all.

`CHvI` and `CLvI` are encoded as `(pos = I, neg = CH/CL)`. That is identically the
1-AUROC convention, with one fewer sign flip available to get wrong.

**A missing cell is `NaN`, never 0.5.** When a contrast has fewer than
`min_per_class` rows on either side, the AUROC is undefined, and 0.5 is a
*measurement* meaning "no signal". Substituting one for the other would let a cell
that was never computed pull a median toward chance and look like evidence of
nothing rather than absence of evidence.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from csx_probe import config

N_BOOT: int = config.frozen()["bootstrap"]["row_level_n"]
CI: tuple[float, float] = tuple(config.frozen()["bootstrap"]["ci"])


def safe_auc(y, s) -> float:
    """AUROC with the degenerate cases returning NaN rather than raising.

    NaN scores are dropped first: a family can legitimately produce one (an
    all-zero PCA component, say), and dropping the row is honest where imputing
    a value would not be.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def boot(y, s, *, n: int = N_BOOT, seed: int = config.SEED
         ) -> tuple[float, float, float]:
    """Percentile bootstrap CI over rows: `(mean, lo, hi)`.

    Resamples rows, not scores, so the class balance varies exactly as it would
    under a resampled study. Draws that come back single-class yield NaN and are
    dropped rather than counted as 0.5.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    n_rows = len(y)
    vals: list[float] = []
    for _ in range(n):
        idx = rng.integers(0, n_rows, n_rows)
        v = safe_auc(y[idx], s[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"),) * 3
    a = np.asarray(vals)
    lo, hi = np.quantile(a, CI[0] / 100.0), np.quantile(a, CI[1] / 100.0)
    return float(a.mean()), float(lo), float(hi)


def cell(score, categories, contrast: str, *, n_boot: int = N_BOOT,
         min_per_class: int = config.MIN_PER_CLASS) -> dict:
    """One contrast cell: AUROC, its CI, and the counts behind it."""
    if contrast not in config.CONTRASTS:
        raise ValueError(f"unknown contrast {contrast!r}")
    pos, neg = config.CONTRASTS[contrast]
    cats = np.asarray(categories)
    m = np.isin(cats, list(pos) + list(neg))
    y = np.isin(cats[m], list(pos)).astype(int)
    s = np.asarray(score, dtype=float)[m]

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if min(n_pos, n_neg) < min_per_class:
        return {"AUROC": np.nan, "AUROC_lo": np.nan, "AUROC_hi": np.nan,
                "n": int(m.sum()), "n_pos": n_pos}
    a = safe_auc(y, s)
    _, lo, hi = boot(y, s, n=n_boot)
    return {"AUROC": a, "AUROC_lo": lo, "AUROC_hi": hi,
            "n": int(m.sum()), "n_pos": n_pos}


def derived(cells: dict[str, float]) -> dict[str, float]:
    """Summaries over the six ADMISSIBLE cells only.

    The two definitional cells (`IHvCL`, `ILvCH`) are deliberately excluded: they
    are decided by the band alone, so including them would let the confidence
    channel inflate a summary that is supposed to describe correctness.

    `cell_min` is the headline. A family is only as good as its worst admissible
    cell -- a high mean over cells that includes one near-chance cell describes a
    probe that fails somewhere specific, and the mean hides where.
    """
    v = np.asarray([cells.get(c, np.nan) for c in config.ADMISSIBLE], dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return {"cell_min": np.nan, "cell_mean": np.nan, "cell_spread": np.nan}
    return {"cell_min": float(v.min()), "cell_mean": float(v.mean()),
            "cell_spread": float(v.max() - v.min())}


def arm_stats(categories, entropy, *, tau: float = np.nan) -> dict:
    """Composition of one arm split, for the `arm_stats` table."""
    cats = np.asarray(categories)
    counts = {c: int((cats == c).sum()) for c in config.CATS}
    n = int(len(cats))
    n_inc = sum(counts[c] for c in config.I_CATS)
    return {"n": n, **counts,
            "pct_incorrect": (100.0 * n_inc / n) if n else np.nan,
            "tau": float(tau),
            "entropy_IvC": safe_auc(np.isin(cats, config.I_CATS).astype(int),
                                    np.asarray(entropy, dtype=float))}
