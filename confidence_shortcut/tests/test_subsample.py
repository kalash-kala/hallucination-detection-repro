"""Subsampling must be deterministic, proportional, and honest about its cap.

The rule is load-bearing in a way that is easy to miss: every pair sitting at a
comparable `n` is what stops a median over an arbitrary grouping from being
confounded by sample size. If this drifted, LLM-vs-VLM comparisons would quietly
become comparisons of pool size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csx_common import registry
from csx_extract import subsample


def _rows(counts: dict[str, int]) -> pd.DataFrame:
    ids, cats = [], []
    i = 0
    # Interleave cells so "pool order" is not already grouped by category.
    while any(counts.values()):
        for c in registry.CATS:
            if counts[c] > 0:
                ids.append(f"train::{i}")
                cats.append(c)
                counts[c] -= 1
                i += 1
    return pd.DataFrame({"id": ids, "category": cats})


def test_keeps_everything_under_the_cap():
    rows = _rows({"IH": 10, "CH": 40, "IL": 20, "CL": 30})
    s = subsample.choose(rows, "llama_sciq", n_target=1000)
    assert s.n_kept == 100 and not s.applied


def test_none_target_keeps_everything():
    rows = _rows({"IH": 10, "CH": 40, "IL": 20, "CL": 30})
    s = subsample.choose(rows, "llama_sciq", n_target=None)
    assert s.n_kept == 100 and not s.applied


def test_hits_target_exactly():
    rows = _rows({"IH": 100, "CH": 700, "IL": 150, "CL": 50})
    s = subsample.choose(rows, "llama_triviaqa", n_target=300)
    assert s.n_kept == 300
    assert sum(s.quotas.values()) == 300


def test_preserves_cell_proportions():
    """Band structure and prevalence must survive the cut -- the whole study
    conditions on these four cells."""
    counts = {"IH": 100, "CH": 700, "IL": 150, "CL": 50}
    rows = _rows(dict(counts))
    s = subsample.choose(rows, "llama_triviaqa", n_target=200)
    got = rows[rows["id"].isin(s.ids)]["category"].value_counts()
    for c, n in counts.items():
        assert abs(got[c] / 200 - n / 1000) < 0.01, c


def test_deterministic_for_a_seed():
    rows = _rows({"IH": 100, "CH": 700, "IL": 150, "CL": 50})
    a = subsample.choose(rows, "llama_triviaqa", n_target=300, seed=42)
    b = subsample.choose(rows, "llama_triviaqa", n_target=300, seed=42)
    assert a.ids == b.ids


def test_different_seed_gives_different_rows():
    rows = _rows({"IH": 100, "CH": 700, "IL": 150, "CL": 50})
    a = subsample.choose(rows, "llama_triviaqa", n_target=300, seed=42)
    b = subsample.choose(rows, "llama_triviaqa", n_target=300, seed=7)
    assert a.ids != b.ids
    assert a.quotas == b.quotas          # the apportionment is seed-independent


def test_output_is_in_pool_order():
    """Kept ids come back in the pool's order, not grouped by cell, so the
    extraction reads roughly sequentially."""
    rows = _rows({"IH": 40, "CH": 40, "IL": 40, "CL": 40})
    s = subsample.choose(rows, "llama_triviaqa", n_target=80)
    pos = {i: n for n, i in enumerate(rows["id"])}
    assert [pos[i] for i in s.ids] == sorted(pos[i] for i in s.ids)


def test_thin_cell_is_not_over_drawn():
    """A cell smaller than its proportional share is taken whole, and the
    shortfall goes to the others rather than erroring."""
    rows = _rows({"IH": 5, "CH": 700, "IL": 150, "CL": 145})
    s = subsample.choose(rows, "llama_triviaqa", n_target=500)
    assert s.quotas["IH"] <= 5
    assert s.n_kept == 500


def test_largest_remainder_sums_exactly():
    for total in (1, 7, 99, 250, 999):
        q = subsample.largest_remainder({"IH": 100, "CH": 700, "IL": 150, "CL": 50},
                                        total)
        assert sum(q.values()) == total, total


def test_largest_remainder_is_deterministic_under_ties():
    counts = {"IH": 25, "CH": 25, "IL": 25, "CL": 25}
    a = subsample.largest_remainder(counts, 50)
    b = subsample.largest_remainder(dict(reversed(list(counts.items()))), 50)
    assert a == b


def test_largest_remainder_returns_pool_when_target_exceeds_it():
    counts = {"IH": 3, "CH": 4}
    assert subsample.largest_remainder(counts, 100) == counts


def test_meta_records_the_decision():
    rows = _rows({"IH": 100, "CH": 700, "IL": 150, "CL": 50})
    m = subsample.choose(rows, "llama_triviaqa", n_target=300).to_meta()
    assert m["applied"] and m["n_kept"] == 300 and m["n_pool"] == 1000
    assert m["stratified_on"] == list(registry.CATS)


@pytest.mark.parametrize("pair,expected", [
    ("llama_sciq", None),          # sciq keeps its whole pool, matching qa8
    ("llama_triviaqa", 15000),     # the published natural-arm size
    ("llama_nq", 15000),
    ("qwen25vl_vqav2", 15000),
    ("qwen25vl_okvqa", None),
    ("qwen25vl_advqa", None),
])
def test_dataset_caps_match_the_published_convention(pair, expected):
    assert registry.get(pair).dataset.n_target == expected
