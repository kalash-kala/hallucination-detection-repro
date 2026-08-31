"""M21 -- the SEP binarisation: a scalar uncertainty metric -> a HI/LO band.

One-dimensional 2-means (Jenks natural breaks) by exhaustive scan: try 100
candidate cuts, keep the one minimising the within-group sum of squares. This is
the rule that produced the stored `DSE_threshold` for every pair, so it is
verifiable rather than merely plausible -- `reproduces_tau` is the gate.

**The generalisation that Part C requires.** The published grid is
`linspace(1e-10, x.max(), 100)`, which quietly assumes the metric is
non-negative. Two of the eight metrics are *negated* quantities:

    lexical_sim   max = -0.289
    snne          max = -2.462

For those the published grid runs from +1e-10 *down* to a negative maximum, so
every candidate cut sits above every observation. Measured on all 9 VLM pairs,
that puts **100.0%** of rows in one band -- a silent, plausible-looking failure
rather than an error. The grid must therefore start at the observed minimum.

**Where that differs from the published grid, stated precisely.** The two agree
exactly when `x.min()` is 0, since the grids then coincide; that covers `dse`,
`eccentricity` and `luq`, and is why the stored `tau` is reproduced exactly for
every pair. When `x.min() > 0` the grids place their 100 points differently over
the same support, so the chosen cut can shift by one grid step: measured, that
moves 0 rows for `num_set` and 0.1-0.9% of rows for `sum_eigv` and `degree`.
This is a real deviation from the published procedure, not a no-op, and it is
recorded per row (`legacy_threshold`, `legacy_frac_HI`) rather than asserted
away -- the min-start grid is defensible because it never spends resolution
below the smallest observation, but the write-up must own the difference.

Orientation is deliberately NOT handled here. `band_of` maps low score -> HI,
following the project-wide convention that `H` is *low* entropy, i.e. high
confidence. A metric whose scale runs the other way must be negated by its
caller (which is exactly why `lexical_sim` and `snne` are stored negated), so
that this module never has to guess a sign.
"""

from __future__ import annotations

import numpy as np

# The published scan width. Not a tunable: the stored thresholds are only
# reproducible at exactly this resolution.
N_CUTS: int = 100


def within_group_ss(x: np.ndarray, cut: float) -> float:
    """Total within-group sum of squares for the split at `cut`.

    `inf` for a degenerate split so that `argmin` can never select one; a cut
    that empties a side has zero WSS and would otherwise always win.
    """
    lo = x[x <= cut]
    hi = x[x > cut]
    if lo.size == 0 or hi.size == 0:
        return float("inf")
    return float(((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum())


def candidate_cuts(x: np.ndarray, n_cuts: int = N_CUTS) -> np.ndarray:
    """The scan grid: `n_cuts` points from the observed min to the observed max.

    Starting at `x.min()` rather than `1e-10` is the negated-metric fix
    documented in the module docstring.
    """
    return np.linspace(float(x.min()), float(x.max()), n_cuts)


def legacy_split(x, n_cuts: int = N_CUTS) -> float:
    """The published `linspace(1e-10, max)` rule, kept so the deviation from it
    is measurable rather than merely described.

    Degenerate by construction on a negated metric -- that is the point.
    """
    x = np.asarray(x, dtype=float)
    grid = np.linspace(1e-10, float(x.max()), n_cuts)
    wss = np.array([within_group_ss(x, c) for c in grid])
    if not np.isfinite(wss).any():
        # Every cut sits outside the data: the negated-metric failure. Report
        # the boundary the published code would have returned.
        return float(grid[0])
    return float(grid[int(np.argmin(wss))])


def best_split(x, n_cuts: int = N_CUTS) -> float:
    """The 1-D 2-means threshold for `x`.

    Ties resolve to the lower cut: `argmin` returns the first minimum and the
    grid is ascending, so this is by construction rather than by chance.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("best_split: no finite values")
    if x.min() == x.max():
        raise ValueError(f"best_split: constant input at {x.min()!r}; "
                         "no split separates it into two bands")
    grid = candidate_cuts(x, n_cuts)
    wss = np.array([within_group_ss(x, c) for c in grid])
    if not np.isfinite(wss).any():
        raise ValueError("best_split: every candidate cut is degenerate")
    return float(grid[int(np.argmin(wss))])


def band_of(x, cut: float) -> np.ndarray:
    """Score -> band label, low score = `HI` (high confidence).

    Matches `arms.common.BAND_OF`'s orientation, where `H` is LOW entropy. The
    boundary belongs to `HI` (`<=`), which is what reproduces the stored
    `category` assignment.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x <= cut, "HI", "LO")


def reproduces_tau(x, tau: float, *, atol: float = 1e-9,
                   n_cuts: int = N_CUTS) -> bool:
    """The M21 gate: does the scan recover this pair's stored `DSE_threshold`?

    Verified True for all 9 VLM pairs. A False here means the binarisation has
    drifted from the one that produced the stored bands, so every band-routed
    number downstream would be describing a different partition than the labels
    it is compared against.
    """
    return bool(np.isclose(best_split(x, n_cuts), float(tau), atol=atol))
