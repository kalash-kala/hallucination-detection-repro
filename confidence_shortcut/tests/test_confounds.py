"""The confound controls, on synthetic rows.

A control that is subtly wrong is worse than no control: it still produces a
ladder, and the ladder still reads plausibly. Each property here is one way that
can happen silently.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from csx_probe import config
from csx_probe.arms import build as ab, confound, gates
from conftest import make_entry


def _built(draws=3, **kw):
    e = make_entry(**kw)
    arms = ab.build_all(e)
    return e, arms, confound.build_all(e, arms, draws=draws)


# ── placebo: composition matched, leak intact ────────────────────────────────

def test_placebo_matches_target_composition_exactly():
    e, arms, ctrl = _built()
    for tgt in confound.TARGETS:
        want = Counter(e.categories[arms[tgt].train])
        for d in range(3):
            got = Counter(e.categories[ctrl[f"pl_{tgt}_d{d:02d}"].train])
            assert got == want, f"{tgt}/d{d}: {dict(got)} != {dict(want)}"


def test_placebo_leaves_the_entropy_leak_open():
    """The point of the control. A placebo that were also entropy-matched would
    be a second treatment arm, and `d_matching` would read ~0 for the wrong
    reason."""
    from csx_probe.metrics import safe_auc
    e, arms, ctrl = _built()
    p = ctrl["pl_dse_matched2_d00"].train
    r = arms["dse_matched2"].train
    a_pl = safe_auc(np.isin(e.categories[p], config.I_CATS).astype(int),
                    e.entropy[p])
    a_real = safe_auc(np.isin(e.categories[r], config.I_CATS).astype(int),
                      e.entropy[r])
    assert abs(a_real - 0.5) < abs(a_pl - 0.5)


def test_placebo_has_no_test_split():
    """It is scored on the real arms' test columns; giving it one of its own
    would compare two probes on two different populations."""
    _, _, ctrl = _built()
    assert len(ctrl["pl_dse_matched_d00"].test) == 0


# ── size-only: natural's skew, the target's n ────────────────────────────────

def test_sizeonly_hits_the_target_n_exactly_on_both_splits():
    e, arms, ctrl = _built()
    for tgt in confound.TARGETS:
        a = ctrl[f"ns_{tgt}_d00"]
        for split in ("train", "test"):
            assert a.n(split) == arms[tgt].n(split), f"{tgt}/{split}"


def test_sizeonly_preserves_natural_proportions():
    """Largest-remainder, not a uniform draw: a uniform draw would hold the
    proportions only in expectation and let them wobble between draws, putting
    composition noise back into a control that exists to hold it fixed."""
    e, arms, ctrl = _built()
    nat = Counter(e.categories[arms["dse_natural"].train])
    tot = sum(nat.values())
    got = Counter(e.categories[ctrl["ns_dse_matched2_d00"].train])
    n = sum(got.values())
    for c in config.CATS:
        assert abs(got[c] / n - nat[c] / tot) < 0.01, c


def test_sizeonly_is_NOT_cell_balanced():
    """The distinction from the placebo. If these came out balanced they would
    already close the band-level shortcut and `d_size` would absorb an effect
    that belongs to `d_composition`."""
    e, _, ctrl = _built()
    got = Counter(e.categories[ctrl["ns_dse_matched2_d00"].train])
    assert len(set(got.values())) > 1


def test_sizeonly_quota_is_exact_under_largest_remainder():
    pool = Counter({"IH": 3, "CH": 3, "IL": 3, "CL": 1})
    for n in (1, 5, 7, 10):
        q = confound.quota(pool, n)
        assert sum(q.values()) == n, (n, q)


def test_sizeonly_quota_never_exceeds_the_pool():
    pool = Counter({"IH": 2, "CH": 100, "IL": 100, "CL": 100})
    q = confound.quota(pool, 300)
    assert q["IH"] <= 2


# ── determinism and leak-freedom ─────────────────────────────────────────────

def test_sizeonly_seed_uses_crc32_not_hash():
    """`hash()` is randomised per process unless PYTHONHASHSEED is pinned, so it
    would make these arms silently unreproducible between runs."""
    a = confound.sizeonly_seed("p", "dse_matched", "train", 0)
    assert a == confound.sizeonly_seed("p", "dse_matched", "train", 0)
    assert a != confound.sizeonly_seed("p", "dse_matched", "train", 1)
    assert a != confound.sizeonly_seed("q", "dse_matched", "train", 0)
    assert a != confound.sizeonly_seed("p", "dse_matched", "test", 0)


def test_controls_are_reproducible():
    e, arms, c1 = _built()
    c2 = confound.build_all(e, arms, draws=3)
    for k in c1:
        assert np.array_equal(c1[k].train, c2[k].train), k
        assert np.array_equal(c1[k].test, c2[k].test), k


def test_draws_actually_differ():
    _, _, ctrl = _built()
    for pre in ("pl_dse_matched2_d", "ns_dse_matched2_d"):
        assert not np.array_equal(ctrl[f"{pre}00"].train,
                                  ctrl[f"{pre}01"].train), pre


def test_no_control_leaks_into_any_real_test_split():
    """Structural -- every control row comes from natural's own pools -- but
    asserted rather than argued."""
    e, arms, ctrl = _built()
    assert gates.check_no_leakage({**arms, **ctrl}) == []


def test_arm_names_parse_back_to_target_and_draw():
    """`_d` appears inside `_dse_` as well as in the draw suffix, so a left
    split silently yields `se_matched2` as the target name."""
    from csx_probe.experiments import confounds as cf
    assert cf._parse("ns_dse_matched2_d07") == ("dse_matched2", 7)
    assert cf._parse("pl_dse_balanced2_d00") == ("dse_balanced2", 0)
    assert cf._parse("pl_dse_matched_d19") == ("dse_matched", 19)
