"""SR-xAUC: the commutation trap, the §17.3 guard, and QA parity.

The parity test is the important one. Before any VLM number from this code is
believed, the same code has to reproduce the *published QA* numbers from the
published QA inputs — that is the strongest available evidence that our Part C is
the same experiment rather than a plausible-looking neighbour of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from csx_report.tables import srxauc

REF_ROOT = Path("/data/kalashkala/dse_data/results/sr_xauc")
SRC = Path("/home/kalashkala/recovery-gaps-experiment/dse_results/"
           "routed_vs_generalist_fixedC/per_pair_long.csv")

# 58_sr_xauc.py's roster, reproduced so the fixture selects the same rows.
SCORERS = ["entropy_only", "sep", "generalist", "generalist_cm",
           "spec1_z", "spec1_z_cm"]
ROUTED = {"spec1_z", "spec1_z_cm"}
ROUTERS = ["oracle", "sampled", "greedy"]


def _load_published() -> pd.DataFrame:
    d = pd.read_csv(SRC)
    d = d[d["contrast"].isin(list(srxauc.ATOMIC) + ["IvC"])].copy()
    unrouted = set(SCORERS) - ROUTED
    d = d[(d["scorer"].isin(unrouted))
          | (d["scorer"].isin(ROUTED) & d["router"].isin(ROUTERS))]
    d["method"] = np.where(d["scorer"].isin(ROUTED),
                           d["scorer"] + "@" + d["router"], d["scorer"])
    return d


# ── the commutation trap ─────────────────────────────────────────────────────

def test_srxauc_min_does_not_commute_with_median():
    """min-then-median (§5) must come out BELOW median-then-min (§8).

    Built so the two genuinely differ: each unit fails in a *different* cell, so
    medianing first lets every unit be rescued by the others and the commuted
    order reports a number no single unit ever achieved.
    """
    rows = []
    fail_cell = dict(zip(["p0", "p1", "p2"], srxauc.ATOMIC[:3]))
    for pair, bad in fail_cell.items():
        for contrast in srxauc.ATOMIC:
            rows.append({"family": "hs_wide", "method": "m", "pair": pair,
                         "train_arm": "dse_natural", "test_arm": "dse_natural",
                         "contrast": contrast,
                         "AUROC": 0.20 if contrast == bad else 0.90})
    res, _ = srxauc.compute(pd.DataFrame(rows))
    r = res[res["variant"] == "primary"].iloc[0]

    assert r["sr_xauc"] == pytest.approx(0.20)             # every unit's worst
    assert r["approx_min_of_medians"] == pytest.approx(0.90)
    assert r["sr_xauc"] < r["approx_min_of_medians"], (
        "we must report the smaller, honest quantity as SR-xAUC")


def test_non_all_segments_are_excluded_from_the_metric():
    """A VLM pair's `image`/`text` rows must not leak into the `all`-only metric.

    Regression for a real bug found in the store: stage 04 was run without
    `--segments all` for 6/9 VLM pairs and computed `image`/`text` too. Those
    extra rows share `(family, method, pair)` with the `all` row -- the exact
    groupby key `per_unit_min` uses -- so without a `segment` filter, `idxmin`
    would pick across segments arbitrarily instead of scoring `all` alone.
    Here the `image`/`text` rows are seeded with a much lower AUROC than
    `all`; if they leaked in, `sr_xauc` would come out at their value instead
    of the true `all`-only worst cell.
    """
    rows = []
    for seg, base in (("all", 0.80), ("image", 0.05), ("text", 0.05)):
        for contrast in srxauc.ATOMIC:
            rows.append({"family": "hs_wide", "method": "m", "pair": "p0",
                         "segment": seg, "train_arm": "dse_natural",
                         "test_arm": "dse_natural", "contrast": contrast,
                         "AUROC": base})
    res, _ = srxauc.compute(pd.DataFrame(rows))
    r = res[res["variant"] == "primary"].iloc[0]
    assert r["sr_xauc"] == pytest.approx(0.80), (
        "image/text rows leaked into the all-only SR-xAUC computation")


def test_approx_is_never_called_srxauc():
    """The commuted quantity keeps its own name in the schema."""
    src = (Path(__file__).resolve().parents[1]
           / "csx_report" / "tables" / "srxauc.py").read_text()
    assert "approx_min_of_medians" in src
    # the docstring must say it is not SR-xAUC
    assert "not** SR-xAUC" in src or "never called SR-xAUC" in src


# ── the §17.3 guard ──────────────────────────────────────────────────────────

def test_entropy_only_is_excluded_from_ranked_comparisons():
    rows = []
    for meth, auc in (("entropy_only", 0.0), ("generalist", 0.61)):
        for pair in ("p0", "p1"):
            for contrast in srxauc.ATOMIC:
                rows.append({"family": "hs_wide", "method": meth, "pair": pair,
                             "train_arm": "dse_natural",
                             "test_arm": "dse_natural",
                             "contrast": contrast, "AUROC": auc})
    res, _ = srxauc.compute(pd.DataFrame(rows))
    assert res.loc[res["method"] == "entropy_only", "unranked"].all()
    assert not res.loc[res["method"] == "generalist", "unranked"].any()
    assert "entropy_only" not in set(srxauc.rankable(res)["method"])
    assert any("by construction" in f for f in srxauc.footnotes(res)), (
        "the definitional-zero footnote is mandatory, not optional")


def test_asymmetry_sign_convention():
    """`asym = IHvCL − ILvCH`, in the report's convention and no other."""
    rows = []
    for pair in ("p0", "p1"):
        for contrast, auc in (("IHvCL", 0.30), ("ILvCH", 0.80)):
            rows.append({"family": "hs_wide", "method": "m", "pair": pair,
                         "train_arm": "dse_natural", "test_arm": "dse_natural",
                         "contrast": contrast, "AUROC": auc})
    got = srxauc.asymmetry(pd.DataFrame(rows))
    assert got["asym"].iloc[0] == pytest.approx(0.30 - 0.80)


# ── QA parity: the M18 gate ──────────────────────────────────────────────────

@pytest.mark.skipif(not (SRC.exists() and (REF_ROOT / "sr_xauc.csv").exists()),
                    reason="published QA reference artifacts are not on this box")
def test_matches_published_qa_srxauc():
    """Our SR-xAUC must reproduce `58_sr_xauc.py`'s published table.

    Same source CSV, same roster. Every scalar column, the bootstrap CIs (same
    seed and generator, so these are exact), and the limiting-cell labels.
    """
    res, atomic = srxauc.compute(_load_published())
    ref = pd.read_csv(REF_ROOT / "sr_xauc.csv")
    m = res.merge(ref, on=["variant", "family", "method"],
                  suffixes=("_ours", "_ref"))
    assert len(m) == len(ref) == len(res)

    for col in ("sr_xauc", "ci_lo", "ci_hi", "sr_xauc_star", "worst_unit",
                "best_unit", "n_units", "G_stability",
                "approx_min_of_medians", "pooled_IvC_natural"):
        np.testing.assert_allclose(
            m[f"{col}_ours"].to_numpy(dtype=float),
            m[f"{col}_ref"].to_numpy(dtype=float),
            rtol=0, atol=1e-6, err_msg=f"{col} diverges from the published table")

    assert (m["limiting_modal_ours"] == m["limiting_modal_ref"]).all()
    assert (m["limiting_n_units_ours"] == m["limiting_n_units_ref"]).all()


@pytest.mark.skipif(not (SRC.exists() and (REF_ROOT / "sr_xauc_atomic.csv").exists()),
                    reason="published QA reference artifacts are not on this box")
def test_matches_published_qa_atomic_breakdown():
    _res, atomic = srxauc.compute(_load_published())
    ref = pd.read_csv(REF_ROOT / "sr_xauc_atomic.csv")
    m = atomic.merge(ref, on=["family", "method", "test_arm", "contrast"],
                     suffixes=("_ours", "_ref"))
    assert len(m) == len(ref)
    np.testing.assert_allclose(m["median_AUROC_ours"].to_numpy(dtype=float),
                               m["median_AUROC_ref"].to_numpy(dtype=float),
                               rtol=0, atol=1e-6)
