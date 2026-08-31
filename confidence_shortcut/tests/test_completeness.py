"""The report's completeness gate: partial groups are refused, not annotated."""

from __future__ import annotations

import pandas as pd

from csx_report import completeness


def _rows(pair: str, cells: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"pair": pair, "family": f, "segment": s, "AUROC": 0.7}
                         for f, s in cells])


FULL = [("hs_wide", "all"), ("hs_wide", "image"), ("sink", "all")]


def test_uniform_group_is_complete():
    df = pd.concat([_rows("a", FULL), _rows("b", FULL)], ignore_index=True)
    cov = completeness.check(df, ["a", "b"], [])
    assert cov.complete
    assert completeness.banner(cov) == ""


def test_missing_pair_makes_the_group_incomplete():
    df = _rows("a", FULL)
    cov = completeness.check(df, ["a"], ["b"])
    assert not cov.complete
    assert "b" in cov.reasons()[0]


def test_ragged_coverage_is_caught_even_though_every_pair_is_present():
    """The silent case: a smoke-run pair with real numbers over one cell.

    It appears in the header count and contributes to some rows and not others,
    so the median's denominator varies by row with nothing on the page saying so.
    """
    df = pd.concat([_rows("a", FULL),
                    _rows("smoke", [("hs_wide", "all")])], ignore_index=True)
    cov = completeness.check(df, ["a", "smoke"], [])
    assert not cov.complete
    assert set(cov.ragged) == {"smoke"}
    assert cov.ragged["smoke"] == ["hs_wide/image", "sink/all"]


def test_reference_is_the_group_union_not_a_fixed_expectation():
    """Text pairs have no image segment; a fixed list would flag them all."""
    df = pd.concat([_rows("t1", [("hs_wide", "all")]),
                    _rows("t2", [("hs_wide", "all")])], ignore_index=True)
    cov = completeness.check(df, ["t1", "t2"], [])
    assert cov.complete
    assert cov.reference == ["hs_wide/all"]


def test_banner_names_every_shortfall():
    df = pd.concat([_rows("a", FULL),
                    _rows("smoke", [("hs_wide", "all")])], ignore_index=True)
    cov = completeness.check(df, ["a", "smoke"], ["gone"])
    b = completeness.banner(cov)
    assert "PARTIAL GROUP" in b
    assert "gone" in b and "smoke" in b
    assert b.startswith(">")


def test_a_complete_group_is_unaffected_by_pairs_outside_it():
    """Coverage is judged within the group, not against the whole store."""
    df = pd.concat([_rows("a", FULL), _rows("b", FULL),
                    _rows("other", FULL + [("lapeigvals", "text")])],
                   ignore_index=True)
    cov = completeness.check(df, ["a", "b"], [])
    assert cov.complete
