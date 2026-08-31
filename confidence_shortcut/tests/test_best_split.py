"""M21's binarisation, on synthetic data only -- no store, no GPU.

The negated-metric tests are the point of this file. `lexical_sim` and `snne`
are stored negated, and the published `linspace(1e-10, max)` grid degenerates
silently on them rather than raising, so the failure has to be pinned by a test
that would pass under the broken version if it were written carelessly.
"""

from __future__ import annotations

import numpy as np
import pytest

from csx_probe.routing import best_split as bs


def _two_clusters(lo_at, hi_at, n=200, sd=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(lo_at, sd, n), rng.normal(hi_at, sd, n)])


def test_split_lands_between_two_separated_clusters():
    x = _two_clusters(1.0, 5.0)
    cut = bs.best_split(x)
    assert 1.0 < cut < 5.0
    # and it actually separates them
    assert (x[x <= cut].max()) < (x[x > cut].min())


def test_grid_spans_min_to_max():
    x = np.array([2.0, 3.0, 9.0])
    g = bs.candidate_cuts(x)
    assert g[0] == pytest.approx(2.0)
    assert g[-1] == pytest.approx(9.0)
    assert len(g) == bs.N_CUTS


def test_negated_metric_still_splits():
    """The regression that matters: an all-negative score.

    Under the published `linspace(1e-10, max)` grid every candidate cut would
    sit above every observation, so one band would take ~100% of rows.
    """
    x = _two_clusters(-3.3, -2.5)
    cut = bs.best_split(x)
    assert -3.3 < cut < -2.5
    band = bs.band_of(x, cut)
    frac_hi = (band == "HI").mean()
    assert 0.3 < frac_hi < 0.7, f"degenerate split: {frac_hi:.3f} in HI"


def test_legacy_grid_would_have_failed_on_negated_input():
    """Pins *why* the min-start fix is needed, not just that ours works."""
    x = _two_clusters(-3.3, -2.5)
    legacy = np.linspace(1e-10, float(x.max()), bs.N_CUTS)
    # The legacy grid is descending and entirely above the data.
    assert legacy[0] > legacy[-1]
    assert (x <= legacy.min()).all()


def test_min_start_matches_legacy_when_min_is_zero():
    """The grids coincide exactly when the data starts at 0.

    This -- not "any non-negative score" -- is the precise condition, and it is
    what makes the stored `tau` reproducible: entropy's minimum is 0.
    """
    x = np.concatenate([_two_clusters(1.0, 5.0), [0.0]])
    assert bs.best_split(x) == pytest.approx(bs.legacy_split(x), abs=1e-9)


def test_min_start_differs_by_at_most_one_grid_step_when_min_positive():
    """Strictly positive data: a real, bounded deviation, not a no-op.

    Measured on the cohort this moves 0-0.9% of rows for `sum_eigv`/`degree`.
    The test pins the size of the disagreement so a larger one is caught.
    """
    x = _two_clusters(1.0, 5.0)
    assert x.min() > 0
    ours, legacy = bs.best_split(x), bs.legacy_split(x)
    step = (x.max() - x.min()) / (bs.N_CUTS - 1)
    assert ours != pytest.approx(legacy, abs=1e-12)
    assert abs(ours - legacy) <= step


def test_legacy_split_is_degenerate_on_negated_input():
    """The published rule's failure, pinned as behaviour we can point at."""
    x = _two_clusters(-3.3, -2.5)
    frac_hi = (bs.band_of(x, bs.legacy_split(x)) == "HI").mean()
    assert frac_hi == 1.0, "expected the legacy grid to collapse to one band"


def test_degenerate_cut_is_infinite_not_zero():
    """A cut that empties a side must never win by having zero WSS."""
    x = np.array([1.0, 2.0, 3.0])
    assert bs.within_group_ss(x, 0.0) == float("inf")
    assert bs.within_group_ss(x, 99.0) == float("inf")


def test_band_orientation_low_score_is_HI():
    """`H` is LOW entropy = high confidence, matching arms.common.BAND_OF."""
    b = bs.band_of([0.0, 10.0], 5.0)
    assert list(b) == ["HI", "LO"]


def test_boundary_belongs_to_HI():
    assert list(bs.band_of([5.0], 5.0)) == ["HI"]


def test_constant_input_raises():
    with pytest.raises(ValueError, match="constant"):
        bs.best_split(np.full(10, 2.0))


def test_reproduces_tau_gate():
    x = _two_clusters(1.0, 5.0)
    tau = bs.best_split(x)
    assert bs.reproduces_tau(x, tau)
    assert not bs.reproduces_tau(x, tau + 0.5)


def test_ties_resolve_to_the_lower_cut():
    """Symmetric data: argmin over an ascending grid takes the first minimum."""
    x = np.array([0.0, 0.0, 1.0, 1.0])
    assert bs.best_split(x) <= 0.5
