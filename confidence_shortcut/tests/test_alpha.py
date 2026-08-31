"""The alpha ladder and the rotation geometry.

The nesting property and the placebo's construction are the two things that make
the rotation result mean anything, and both are invisible in the output if they
are wrong -- a non-nested ladder still produces a monotone-looking trend, and a
stratum-matched placebo still produces a null.
"""

from __future__ import annotations

import numpy as np
import pytest

from csx_probe.arms import alpha as al, build as ab
from csx_probe.experiments import alpha_rotation as ar
from conftest import make_entry


def _ladder():
    e = make_entry()
    arms = ab.build_all(e)
    lad, counts = al.build_ladder(e, arms["dse_natural"], arms["dse_matched2"])
    return e, arms, lad, counts


def test_ladder_is_nested():
    """Each rung must be a subset of the one below it. Independently drawn rungs
    would mix the rotation being measured with resampling noise at every step."""
    _, _, lad, _ = _ladder()
    al.assert_nested(lad)
    order = sorted(lad, reverse=True)
    for hi, lo in zip(order, order[1:]):
        assert set(lad[hi].tolist()) <= set(lad[lo].tolist())


def test_nesting_violation_is_detected():
    _, _, lad, _ = _ladder()
    bad = dict(lad)
    bad[1.0] = np.array(sorted(set(range(10_000)) - set(bad[0.0].tolist()))[:5])
    with pytest.raises(AssertionError, match="not nested"):
        al.assert_nested(bad)


def test_alpha_zero_is_natural_and_one_is_matched2_sized():
    e, arms, lad, _ = _ladder()
    assert len(lad[0.0]) == len(arms["dse_natural"].train)
    # a=1 reproduces matched2's per-cell counts, so its size matches
    assert len(lad[1.0]) == len(arms["dse_matched2"].train)


def test_ladder_sizes_decrease_monotonically():
    _, _, lad, _ = _ladder()
    sizes = [len(lad[a]) for a in sorted(lad)]
    assert sizes == sorted(sizes, reverse=True)


def test_placebo_matches_size_and_band_class_composition():
    """The control must differ from its rung ONLY in which rows it took."""
    e, arms, lad, counts = _ladder()
    nat = arms["dse_natural"]
    for a in (0.0, 1.0):
        p = al.build_placebo(e, nat, counts[a], draw=0)
        assert len(p) == len(lad[a]), f"a={a}: placebo size differs"
        for cell, k in counts[a].items():
            band, cls = cell
            cats = e.categories[p]
            got = sum(1 for c in cats
                      if (("HI" if c in ("IH", "CH") else "LO") == band
                          and ("I" if c.startswith("I") else "C") == cls))
            assert got == k, f"a={a} {cell}: {got} != {k}"


def test_placebo_leaves_the_entropy_leak_intact():
    """Drawn uniformly within (band, class), NOT within stratum.

    A stratum-matched placebo would itself be entropy-matched -- another
    treatment arm rather than a control -- and the null it produced would be the
    very effect under test.
    """
    e, arms, lad, counts = _ladder()
    from csx_probe import config
    from csx_probe.metrics import safe_auc

    p = al.build_placebo(e, arms["dse_natural"], counts[1.0], draw=0)
    c, ent = e.categories[p], e.entropy[p]
    a_placebo = safe_auc(np.isin(c, config.I_CATS).astype(int), ent)

    c2 = e.categories[lad[1.0]]
    a_rung = safe_auc(np.isin(c2, config.I_CATS).astype(int),
                      e.entropy[lad[1.0]])
    # The rung has entropy matched out; its placebo must not.
    assert abs(a_rung - 0.5) < abs(a_placebo - 0.5)


def test_placebo_draws_are_distinct_but_reproducible():
    e, arms, _, counts = _ladder()
    nat = arms["dse_natural"]
    p0 = al.build_placebo(e, nat, counts[1.0], draw=0)
    p1 = al.build_placebo(e, nat, counts[1.0], draw=1)
    assert not np.array_equal(p0, p1)
    assert np.array_equal(p0, al.build_placebo(e, nat, counts[1.0], draw=0))


def test_placebo_at_alpha0_is_degenerate():
    """At a=0 every placebo draw returns the WHOLE natural pool.

    `build_placebo` takes `min(k, len(avail))` rows without replacement, and at
    a=0 the target `k` already equals the full natural per-cell count -- so the
    sample is the pool, identically for every draw. This is not a bug, but it is
    load-bearing for how `verdict` must be described: with `p0` constant the
    400-element outer null is a pure location shift of `p1`, so the test reduces
    to `theta(1) > p95(placebo(a=1))`.

    Pinned because the degeneracy is invisible in the output -- the null still
    has spread (all of it from the a=1 rung) and still looks like a null.
    """
    e, arms, lad, counts = _ladder()
    nat = arms["dse_natural"]
    draws = [al.build_placebo(e, nat, counts[0.0], draw=i) for i in range(5)]
    for d in draws[1:]:
        assert np.array_equal(draws[0], d), "a=0 placebo draws should be identical"
    # and identical to the real a=0 arm, which is the natural arm itself
    assert np.array_equal(np.sort(draws[0]), np.sort(lad[0.0]))


def test_alpha0_degeneracy_collapses_the_null_to_a_location_shift():
    """The algebra the corrected `verdict` docstring claims, on synthetic input.

    Guards the equivalence itself, so that if a future cohort makes a=0
    non-degenerate the two forms visibly stop agreeing.
    """
    rng = np.random.default_rng(0)
    p1 = rng.normal(50, 3, 20)
    p0 = np.full(20, 40.0)          # degenerate, as on this cohort
    theta1, theta0 = 55.0, 40.0

    outer = np.percentile(np.subtract.outer(p1, p0).ravel(), ar.PASS_PCT)
    assert (theta1 - theta0 > outer) == (theta1 > np.percentile(p1, ar.PASS_PCT))

    # non-degenerate a=0: the forms are no longer interchangeable
    p0v = rng.normal(40, 3, 20)
    outer_v = np.percentile(np.subtract.outer(p1, p0v).ravel(), ar.PASS_PCT)
    assert not np.isclose(outer_v, np.percentile(p1, ar.PASS_PCT) - p0v.mean())


# ── parallelism ──────────────────────────────────────────────────────────────

def _synth_fit_inputs(n=240, d=40, seed=3):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    cat = np.array(["IH", "CH", "IL", "CL"])[rng.integers(0, 4, n)]
    return Z, cat


def test_parallel_fits_match_serial_fits():
    """`n_jobs` must be a scheduling choice, never a numerical one.

    Each fit is single-threaded and self-contained, so the worker count cannot
    reach the result. If this ever fails, something stateful has been captured
    into `_fit_one` and every bootstrap CI is suspect.
    """
    from joblib import Parallel, delayed

    Z, cat = _synth_fit_inputs()
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, len(Z), len(Z)) for _ in range(8)]

    serial = [ar._fit_one(Z, cat, "hs_wide", 1e-2, "g", i) for i in idxs]
    par = Parallel(n_jobs=4, backend="loky", inner_max_num_threads=1)(
        delayed(ar._fit_one)(Z, cat, "hs_wide", 1e-2, "g", i) for i in idxs)

    for s, p in zip(serial, par):
        np.testing.assert_allclose(s, p, rtol=0, atol=1e-9)


def test_fit_one_targets_differ_between_g_and_sep():
    """`refboot` fits the BAND label, every other row fits correctness. Collapsing
    the two would make the reference bootstrap measure the wrong thing while
    still producing a plausible spread."""
    Z, cat = _synth_fit_inputs()
    idx = np.arange(len(Z))
    w_g = ar._fit_one(Z, cat, "hs_wide", 1e-2, "g", idx)
    w_sep = ar._fit_one(Z, cat, "hs_wide", 1e-2, "sep", idx)
    assert not np.allclose(w_g, w_sep)


# ── geometry ─────────────────────────────────────────────────────────────────

def test_cos_sigma_never_materialises_sigma():
    """`u'Sigma v` computed as `(Zc u).(Zc v)` must equal the explicit form.

    At 39,200 features the explicit Sigma would be a 12 GB dense matrix, so the
    identity is what makes the experiment possible at all.
    """
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(80, 12))
    Zc = Z - Z.mean(0, keepdims=True)
    u, v = rng.normal(size=12), rng.normal(size=12)
    S = Zc.T @ Zc
    want = (u @ S @ v) / np.sqrt((u @ S @ u) * (v @ S @ v))
    got, _ = ar.cos_pair(u, v, Zc)
    assert got == pytest.approx(want)


def test_cosines_are_signed_not_absolute():
    """A negative cosine is a real finding -- both heads are oriented
    larger-is-worse, so anti-alignment means something."""
    rng = np.random.default_rng(1)
    Zc = rng.normal(size=(50, 6))
    u = rng.normal(size=6)
    cs, ce = ar.cos_pair(u, -u, Zc)
    assert cs < 0 and ce < 0
    assert ce == pytest.approx(-1.0)


def test_theta_clips_out_of_range_cosines():
    assert ar.theta(1.0 + 1e-12) == pytest.approx(0.0)
    assert ar.theta(-1.0 - 1e-12) == pytest.approx(180.0)
    assert ar.theta(0.0) == pytest.approx(90.0)


def test_null_is_the_outer_differences_not_the_paired_ones():
    """20x20 = 400 outer differences, not 20 paired ones.

    The general form is the right one -- the two rungs' draws are independent, so
    pairing by index would impose a correspondence that does not exist. This test
    uses inputs where BOTH rungs vary, which is the case the outer form is for.

    On the actual cohort a=0 is degenerate, so the choice makes no numerical
    difference there; see `test_placebo_at_alpha0_is_degenerate`.
    """
    p0 = np.arange(20, dtype=float)
    p1 = np.arange(20, dtype=float) + 5
    null = np.subtract.outer(p1, p0).ravel()
    assert null.size == 400
    paired = p1 - p0
    assert null.std() > paired.std()
