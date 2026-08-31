"""M23's grid, on synthetic data only -- no store, no GPU.

The leak-freedom tests are the point of this file. The entire metric-router
grid is legal only because the DSE train/test partition is reused verbatim, and
a violation of that would not raise anywhere: it would produce a *better*-looking
off-diagonal, which is the direction that flatters the result. So the invariant
is tested directly, and `assert_leak_free` is tested for its ability to FAIL as
well as to pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from csx_probe.arms.build import Arm
from csx_probe.experiments import metric_router as mr


# ── the stratum scalar `matched2` keys on ───────────────────────────────────

def test_discrete_metric_keeps_its_raw_values():
    """`dse` is discrete and must be matched on the raw float, as before.

    Binning it would change `matched2` for the one metric whose arm we can
    check against the published pipeline.
    """
    x = np.repeat(np.arange(7, dtype=float), 40)
    out = mr._stratum_scalar(x, "dse")
    np.testing.assert_array_equal(out, x)


def test_continuous_metric_is_binned_to_global_quantiles():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    out = mr._stratum_scalar(x, "eccentricity")
    assert not np.array_equal(out, x)
    assert len(np.unique(out)) <= mr.N_QUANTILE_BINS
    # equal-COUNT, not equal-width: a heavily skewed metric must still spread
    counts = pd.Series(out).value_counts().to_numpy()
    assert counts.max() / counts.min() < 3, "bins are wildly unbalanced"


def test_binning_is_monotone_in_the_underlying_metric():
    """Bins must preserve order, or a stratum stops meaning 'similar score'."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=3000)
    out = mr._stratum_scalar(x, "luq")
    o = np.argsort(x)
    assert np.all(np.diff(out[o]) >= 0)


def test_ties_share_a_bin():
    x = np.concatenate([np.full(500, 0.5), np.linspace(1, 2, 2000)])
    out = mr._stratum_scalar(x, "snne")
    assert len(np.unique(out[:500])) == 1


# ── the invariant the whole grid rests on ───────────────────────────────────

def _arms(train, test):
    return Arm("p", "dse_natural", np.array(train), np.array(test))


def test_leak_free_passes_on_a_shared_frozen_partition():
    """Every metric subsets ONE partition -- the legal configuration."""
    tr, te = list(range(0, 70)), list(range(70, 100))
    by_metric = {
        "dse": {"dse_natural": _arms(tr, te),
                "dse_matched2": _arms(tr[:40], te[:20])},
        "luq": {"dse_natural": _arms(tr, te),
                "dse_matched2": _arms(tr[10:50], te[5:25])},
    }
    mr.assert_leak_free(by_metric)          # must not raise


def test_leak_free_catches_a_re_split_partition():
    """The failure this exists to prevent: metric B re-split its own rows, so
    a row that trains metric A's router is scored in metric B's test set."""
    by_metric = {
        "dse": {"dse_natural": _arms(range(0, 70), range(70, 100))},
        "luq": {"dse_natural": _arms(range(30, 100), range(0, 30))},
    }
    with pytest.raises(mr.MetricRouterError, match="leak"):
        mr.assert_leak_free(by_metric)


def test_leak_free_checks_across_metrics_not_just_within():
    """A within-metric check would pass this; the cross-metric one must not."""
    by_metric = {
        "a": {"dse_natural": _arms(range(0, 50), range(50, 100))},
        "b": {"dse_natural": _arms(range(50, 100), range(0, 50))},
    }
    with pytest.raises(mr.MetricRouterError, match="leak"):
        mr.assert_leak_free(by_metric)


def test_leak_free_ignores_arms_that_could_not_be_built():
    by_metric = {"a": {"dse_natural": _arms(range(0, 5), range(5, 10)),
                       "dse_matched2": None}}
    mr.assert_leak_free(by_metric)


# ── relabelling changes exactly one coordinate ──────────────────────────────

class _FakeEntry:
    pair = "fake_pair"

    def __init__(self, cats, ids):
        self.categories = np.array(cats)
        self.ids = np.array(ids, dtype=object)


def test_relabel_preserves_correctness_and_moves_only_the_band(monkeypatch):
    """`I`/`C` must survive verbatim; only `H`/`L` may move.

    If correctness were re-derived per metric the grid would be comparing two
    different notions of "wrong answer" across the off-diagonal, and the
    transfer number would no longer mean what it claims.
    """
    ids = [f"r{i}" for i in range(200)]
    rng = np.random.default_rng(3)
    cats = rng.choice(["IH", "CH", "IL", "CL"], size=200)
    entry = _FakeEntry(cats, ids)

    # a metric whose split lands mid-range, so bands genuinely differ from DSE's
    vals = rng.normal(size=200)
    monkeypatch.setattr(mr.band_thresholds, "load_metrics",
                        lambda pair: pd.DataFrame({"id": ids, "luq": vals}))

    new, scalar = mr.relabel(entry, "luq")
    assert [c[0] for c in new] == [c[0] for c in cats], "correctness moved"
    assert set("".join(new)) <= set("ICHL")
    assert len(scalar) == 200
    # and the bands really did change, or the test proves nothing
    assert any(a[1] != b[1] for a, b in zip(new, cats))


def test_relabel_raises_when_the_roster_and_metric_table_disagree(monkeypatch):
    ids = [f"r{i}" for i in range(10)]
    entry = _FakeEntry(["IH"] * 10, ids)
    monkeypatch.setattr(mr.band_thresholds, "load_metrics",
                        lambda pair: pd.DataFrame({"id": ids[:8],
                                                   "luq": np.arange(8.0)}))
    with pytest.raises(mr.MetricRouterError, match="missing from"):
        mr.relabel(entry, "luq")


def test_relabel_rejects_an_unknown_metric(monkeypatch):
    ids = ["a", "b"]
    entry = _FakeEntry(["IH", "CL"], ids)
    monkeypatch.setattr(mr.band_thresholds, "load_metrics",
                        lambda pair: pd.DataFrame({"id": ids,
                                                   "dse": [0.0, 1.0]}))
    with pytest.raises(mr.MetricRouterError, match="absent"):
        mr.relabel(entry, "not_a_metric")


def test_the_eight_metrics_are_the_ones_m21_thresholds(monkeypatch):
    """M23's grid and M21's thresholds must cover the same metric set."""
    assert mr.METRICS == mr.band_thresholds.METRICS
    assert "lexical_sim" in mr.METRICS, "the non-NLI control must be present"
    assert len(mr.METRICS) == 8
