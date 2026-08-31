"""Orientation, the NaN policy, and the derived summaries.

The orientation convention is the thing most likely to be "fixed" into wrongness
by a well-meaning edit, because the correct behaviour looks like a bug: the
confidence head scores ~0.0 on `IHvCL`, and the instinct is to flip a sign so it
reads above 0.5. Doing that would delete the study's central observation, so it is
pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from csx_probe import config, metrics, probes

CATS = np.array(["IH"] * 30 + ["CH"] * 30 + ["IL"] * 30 + ["CL"] * 30)


def test_larger_score_means_more_incorrect():
    """The global convention, with no per-head exceptions."""
    score = np.where(np.isin(CATS, config.I_CATS), 1.0, 0.0)
    assert metrics.cell(score, CATS, "IvC")["AUROC"] == 1.0


def test_chvi_is_the_one_minus_auroc_convention():
    """`CHvI` is encoded as (pos = I, neg = CH), which is identically 1-AUROC
    with one fewer sign flip available to get wrong."""
    rng = np.random.default_rng(0)
    score = rng.normal(size=len(CATS))
    m = np.isin(CATS, ("IH", "IL", "CH"))
    direct = metrics.safe_auc(np.isin(CATS[m], ("CH",)).astype(int), score[m])
    assert metrics.cell(score, CATS, "CHvI")["AUROC"] == pytest.approx(1 - direct)


def test_a_perfect_confidence_axis_scores_zero_on_the_shortcut_cell():
    """IH is high-confidence-incorrect, CL is low-confidence-correct. A pure
    confidence axis ranks CL above IH, so IH-vs-CL comes out at 0.000.

    That is the finding, not a bug: it is the confidence channel being caught
    predicting the *opposite* of correctness on the one cell that isolates it.
    """
    conf = np.where(np.isin(CATS, config.L_CATS), 1.0, 0.0)
    assert metrics.cell(conf, CATS, "IHvCL")["AUROC"] == 0.0


def test_undefined_cells_are_nan_never_half():
    """0.5 is a measurement meaning 'no signal'. Substituting it for a cell that
    was never computable would let absence of evidence pull a median toward
    chance and read as evidence of nothing."""
    cats = np.array(["IH"] * 3 + ["CL"] * 50)
    out = metrics.cell(np.zeros(len(cats)), cats, "IHvCL")
    assert np.isnan(out["AUROC"])


def test_derived_ignores_the_definitional_cells():
    """`IHvCL` and `ILvCH` are decided by the band alone; including them would
    let the confidence channel inflate a correctness summary."""
    cells = {c: 0.8 for c in config.ADMISSIBLE}
    cells.update({"IHvCL": 0.0, "ILvCH": 1.0})
    d = metrics.derived(cells)
    # Compared separately rather than chained: `cell_min` comes straight off an
    # input value while `cell_mean` goes through a sum, so the two differ in the
    # last ULP and an exact chained equality would fail on arithmetic, not on
    # the property being tested.
    assert d["cell_min"] == pytest.approx(0.8)
    assert d["cell_mean"] == pytest.approx(0.8)
    assert d["cell_spread"] == pytest.approx(0.0)


def test_cell_min_is_driven_by_the_worst_cell():
    cells = {c: 0.8 for c in config.ADMISSIBLE}
    cells[config.ADMISSIBLE[0]] = 0.51
    assert metrics.derived(cells)["cell_min"] == pytest.approx(0.51)


def test_bootstrap_is_seeded():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200)
    assert metrics.boot(y, s, n=50) == metrics.boot(y, s, n=50)


def test_heads_target_opposite_axes():
    """`g` is correctness, `sep` is the band. Both oriented so larger => more
    incorrect / less confident, which is what lets them be compared directly."""
    assert probes.target(CATS, "g").tolist() == np.isin(
        CATS, config.I_CATS).astype(int).tolist()
    assert probes.target(CATS, "sep").tolist() == np.isin(
        CATS, config.L_CATS).astype(int).tolist()
    with pytest.raises(ValueError, match="not a fitted head"):
        probes.target(CATS, "entropy_only")


def test_hs_ignores_pca_dim():
    """The hs vector is already pooled; the published path sends it to
    StandardScaler regardless of pca_dim, which is why it is recorded as None."""
    from sklearn.preprocessing import StandardScaler
    X = np.random.default_rng(0).normal(size=(50, 8))
    assert isinstance(probes.make_transform(X, "hs", 100), StandardScaler)
    assert isinstance(probes.make_transform(X, "spectral", None), StandardScaler)


def test_pca_is_capped_by_rows_and_features():
    X = np.random.default_rng(0).normal(size=(20, 8))
    assert probes.make_transform(X, "spectral", 100).n_components == 8
