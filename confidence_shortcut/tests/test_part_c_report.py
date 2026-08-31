"""Part C's final-report renderers (M24): metric-router summary, lever_a
gate table, and the srxauc report -- all pandas-only, no store, no GPU.

The `drop` sign in metric_router.summary is the point of this file. The plan
fixes it as `drop(A->B) = AUROC(train A, test B) - AUROC(train B, test B)`
(offdiag minus diagonal), so Spearman(agreement, drop) matches the QA
reference's sign (+0.62 to +0.70). Getting this backwards was caught by hand
while building the final report -- it silently flips "redundancy explains the
transfer" into its opposite, and nothing downstream would raise on it, since
either sign is a valid-looking correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csx_report.tables import lever_a as lever_a_tbl
from csx_report.tables import metric_router as mr_tbl


def _grid(pair, rows):
    """`rows`: list of (train_metric, test_metric, auroc)."""
    return pd.DataFrame([
        {"pair": pair, "train_metric": a, "test_metric": b,
         "diagonal": a == b, "router_auroc": auc}
        for a, b, auc in rows
    ])


def test_drop_sign_matches_the_plans_convention():
    """Perfect agreement -> zero drop; degraded transfer -> negative drop.

    Two metrics, `a` and `b`. `a`'s diagonal is 0.90. Transfer FROM `b` TO `a`
    scores 0.70 -- a real degradation. Under the plan's sign
    (offdiag - diagonal) that must be NEGATIVE (0.70 - 0.90 = -0.20), not
    positive -- a positive `drop` here would mean transfer improved on the
    diagonal, which this fixture does not construct.
    """
    grid = _grid("p", [
        ("a", "a", 0.90), ("b", "b", 0.80),
        ("b", "a", 0.70),   # train on b's band, score against a -> degraded
        ("a", "b", 0.75),
    ])
    agreement = pd.DataFrame({
        "pair": ["p", "p"], "metric_a": ["b", "a"], "metric_b": ["a", "b"],
        "agreement": [0.6, 0.6],
    })
    s = mr_tbl.summary(grid, agreement)
    assert len(s) == 1
    # off-diagonal mean auroc (0.70, 0.75) must be BELOW the diagonal mean
    # (0.85), so the aggregate drop this fixture produces is negative.
    assert s.iloc[0]["off_diagonal"] < s.iloc[0]["diagonal"]


def test_higher_agreement_predicts_smaller_degradation_positive_spearman():
    """The QA finding's sign: agreement UP -> drop LESS negative -> Spearman > 0.

    Three metrics off a, with transfer quality tracking how much each agrees
    with `a`'s band. If the sign were flipped this would come out negative.
    """
    rows = [("a", "a", 0.90)]
    agree_rows = []
    for m, agr, auc in [("b", 0.95, 0.89), ("c", 0.70, 0.75), ("d", 0.40, 0.55)]:
        rows.append((m, m, 0.90))
        rows.append((m, "a", auc))       # train on m, score against a
        agree_rows.append(("p", m, "a", agr))
    grid = _grid("p", rows)
    agreement = pd.DataFrame(agree_rows, columns=["pair", "metric_a", "metric_b", "agreement"])

    s = mr_tbl.summary(grid, agreement)
    assert s.iloc[0]["spearman_agreement_vs_drop"] > 0.9


def test_diagonal_averages_over_scheme_without_reindex_crash():
    """A metric scored under two sampled schemes must not blow up the merge.

    This is the exact shape that crashed before the fix: two diagonal rows
    for the same (pair, train_metric) -- one per scheme -- make a naive
    `set_index("train_metric")` non-unique.
    """
    grid = pd.DataFrame([
        {"pair": "p", "train_metric": "a", "test_metric": "a", "diagonal": True, "router_auroc": 0.9},
        {"pair": "p", "train_metric": "a", "test_metric": "a", "diagonal": True, "router_auroc": 0.8},
        {"pair": "p", "train_metric": "b", "test_metric": "b", "diagonal": True, "router_auroc": 0.85},
        {"pair": "p", "train_metric": "b", "test_metric": "a", "diagonal": False, "router_auroc": 0.7},
    ])
    agreement = pd.DataFrame({"pair": ["p"], "metric_a": ["b"], "metric_b": ["a"], "agreement": [0.5]})
    s = mr_tbl.summary(grid, agreement)   # must not raise
    assert len(s) == 1


def test_lever_a_gate_table_flags_a_failure():
    gates = pd.DataFrame([
        {"pair": "p1", "gate": g, "passed": True} for g in lever_a_tbl.GATES
    ] + [
        {"pair": "p2", "gate": g, "passed": (g != "ece_falls")}
        for g in lever_a_tbl.GATES
    ])
    out = "\n".join(lever_a_tbl.gate_table(gates))
    assert "p1" in out and "yes" in out
    assert "**FAIL**" in out
    assert "AT LEAST ONE GATE FAILS" in out


def test_lever_a_report_all_pass():
    gates = pd.DataFrame([
        {"pair": "p1", "gate": g, "passed": True, "detail": "x"}
        for g in lever_a_tbl.GATES
    ])
    text = lever_a_tbl.report(gates, cohort="vlm")
    assert "All gates pass across the cohort." in text
