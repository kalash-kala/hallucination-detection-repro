"""The arm invariants, on synthetic rows.

These are the properties every downstream number depends on, and each one fails
*quietly* -- a leaking arm or a half-closed confidence channel still produces
perfectly plausible AUROCs. So they are asserted on data built here, with no
store and no GPU involved.

The matched2 case is the sharp one: it must land on 0.500 **exactly**, not
approximately. `matched` is allowed its documented ~5e-3 drift, and the test
pins that difference rather than papering over it -- the two arms existing
side by side is only meaningful if the gap between them is real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csx_probe import config
from csx_probe.arms import build as ab, common, gates
from csx_probe.metrics import safe_auc
from csx_probe.store.read import Entry
from conftest import make_entry


# ── containment and leakage ──────────────────────────────────────────────────

def test_every_arm_is_contained_in_natural():
    e = make_entry()
    arms = ab.build_all(e)
    assert gates.check_containment(arms) == []


def test_no_train_test_leakage_across_any_ordered_arm_pair():
    """The property the whole transfer grid rests on. `balanced` -- the arm
    balanced2 replaced -- violated exactly this."""
    e = make_entry()
    arms = ab.build_all(e)
    assert gates.check_no_leakage(arms) == []


def test_leakage_is_actually_detected():
    """A gate that cannot fail is not a gate."""
    e = make_entry()
    arms = ab.build_all(e)
    bad = dict(arms)
    nat = bad["dse_natural"]
    bad["dse_balanced2"] = ab.Arm(
        e.pair, "dse_balanced2", nat.test[:20].copy(), nat.test[:20].copy())
    assert gates.check_no_leakage(bad)


# ── the confidence channel ───────────────────────────────────────────────────

def test_matched2_kills_entropy_within_band_exactly():
    e = make_entry()
    arms = ab.build_all(e)
    assert gates.check_entropy_dead(
        e, arms["dse_matched2"], tol=1e-9) == []


def test_matched2_kills_pooled_entropy_too():
    """Every within-band AUROC can sit at 0.500 while the POOLED one runs to
    0.689, because the band mixture reopens the channel. A single ratio across
    both bands is what closes it, and this is the test that would catch a
    per-band optimisation sneaking back in."""
    e = make_entry()
    arms = ab.build_all(e)
    m2 = arms["dse_matched2"]
    for split in ("train", "test"):
        sel = m2.rows(split)
        c, ent = e.categories[sel], e.entropy[sel]
        a = safe_auc(np.isin(c, config.I_CATS).astype(int), ent)
        assert abs(a - 0.5) < 1e-9, f"{split}: pooled entropy AUROC {a}"


def test_matched_uses_rounded_key_and_matched2_uses_raw():
    """The one-line difference between the two arms, pinned."""
    e = 0.1 + 0.2                      # 0.30000000000000004
    assert common.stratum_key_raw(e) != 0.3
    assert common.stratum_key_rounded(e) == 0.3
    assert common.stratum_key_raw(-0.0) == 0.0


# ── balanced2 ────────────────────────────────────────────────────────────────

def test_balanced2_cells_are_equal():
    e = make_entry()
    arms = ab.build_all(e)
    for split in ("train", "test"):
        c = e.categories[arms["dse_balanced2"].rows(split)]
        counts = {k: int((c == k).sum()) for k in config.CATS}
        assert len(set(counts.values())) == 1, counts


def test_balanced2_falls_back_for_small_cells_but_stays_equal():
    """A VLM pair whose smallest cell cannot fill the published quota still gets
    an arm -- with equal cells, which is what the arm actually means."""
    e = make_entry(n_per_cat=(60, 400, 400, 400))
    arms = ab.build_all(e)
    b2 = arms["dse_balanced2"]
    c = e.categories[b2.train]
    counts = {k: int((c == k).sum()) for k in config.CATS}
    assert len(set(counts.values())) == 1
    assert counts["IH"] < ab.BALANCED2_QUOTA["train"]
    assert b2.note, "a reduced quota must be recorded, not silent"


def test_parity_pair_refuses_to_shrink_the_quota():
    """`qa8` must reproduce the published draw or fail loudly; a quietly smaller
    arm there would break the 1e-4 parity gate while looking fine."""
    e = make_entry(n_per_cat=(60, 400, 400, 400), pair="llama_sciq")
    with pytest.raises(ab.ArmError, match="parity pair"):
        ab.build_all(e)


# ── determinism ──────────────────────────────────────────────────────────────

def test_arms_are_reproducible():
    e = make_entry()
    a1 = ab.build_all(e)
    a2 = ab.build_all(e)
    for k in a1:
        assert np.array_equal(a1[k].train, a2[k].train)
        assert np.array_equal(a1[k].test, a2[k].test)


def test_new_pairs_do_not_share_one_rng_stream():
    """A shared stream would make each pair's arm depend on how many pairs were
    drawn before it, so adding a VLM pair would silently change every pair after
    it. The parity pairs keep that behaviour; new ones must not inherit it."""
    assert common.pair_rng("a").random() != common.pair_rng("b").random()
    assert common.pair_rng("a").random() == common.pair_rng("a").random()


def test_band_orientation_is_checked_not_assumed():
    """H must be the LOW-entropy band. Inverting it silently swaps the meaning of
    every band conclusion while every number stays plausible."""
    e = make_entry()
    flipped = e.rows.copy()
    flipped["entropy"] = 1.0 - flipped["entropy"]
    bad = Entry(pair=e.pair, meta=e.meta, rows=flipped)
    with pytest.raises(ValueError, match="orientation"):
        ab.build_all(bad)
