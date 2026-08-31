"""M21 -- one `best_split` threshold per (metric, pair), plus its band labels.

Emits atomic per-pair rows, like every other experiment here: no medians, no
cohort, and one pair's thresholds are complete and meaningful on their own.

This is what M23's metric-router grid consumes. That grid's legality rests on
reusing the DSE train/test partition **verbatim** and recomputing only the
`category` field -- so this module deliberately produces thresholds and band
labels and nothing else. It never re-splits.

The `dse` metric is included alongside the seven NLI metrics precisely because
its threshold is independently known: the stored `DSE_threshold` (`tau`). That
makes `dse` the gate row -- if the scan cannot recover a value we already have,
the other seven are not to be trusted either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from csx_common import paths
from csx_probe import results
from csx_probe.routing import best_split as bs

TABLE = "band_thresholds"

# The seven NLI metrics from M20, in `uq_metrics.parquet`.
NLI_METRICS: tuple[str, ...] = ("num_set", "lexical_sim", "sum_eigv", "degree",
                                "eccentricity", "luq", "snne")
# `dse` comes from the UQ store's `entropy`, not from uq_metrics.
DSE_METRIC: str = "dse"
METRICS: tuple[str, ...] = (DSE_METRIC,) + NLI_METRICS

# Metrics whose published grid start (1e-10) is invalid because the scale is
# negated. Recorded per row so the report can show the fix was load-bearing.
NEGATED: frozenset[str] = frozenset({"lexical_sim", "snne"})


def load_metrics(pair: str) -> pd.DataFrame:
    """All 8 metric columns for one pair, joined on row id.

    Scoped to the pair's **roster** (`rows.parquet`), not to the L2 UQ table.
    Those differ wherever a pool was subsampled: vqav2 has 34,991 CSV rows but a
    roster of `N_TARGET` = 15,000. The threshold has to be computed over exactly
    the rows the probes are fit on, or the bands it defines would not partition
    the data they label.

    `dse` is read from the UQ store rather than recomputed, so the gate compares
    the scan against a threshold this module had no hand in producing.
    """
    roster = pd.read_parquet(paths.raw_rows(pair), columns=["id"])
    if not len(roster):
        raise ValueError(f"{pair}: empty roster in {paths.raw_rows(pair)}")

    uq = pd.read_parquet(paths.uq_table())
    uq = uq[uq["pair"] == pair][["id", "entropy", "tau"]]
    if not len(uq):
        raise ValueError(f"{pair}: no rows in the UQ store")

    m_path = paths.uq_metrics(pair)
    if not m_path.exists():
        raise FileNotFoundError(
            f"{pair}: {m_path} missing; run M20 (cli_extract/28_uq_metrics.py) "
            f"for this pair first")
    met = pd.read_parquet(m_path)

    df = (roster.merge(uq, on="id", how="left", validate="one_to_one")
                .merge(met, on="id", how="left", validate="one_to_one"))
    missing_uq = int(df["tau"].isna().sum())
    if missing_uq:
        raise ValueError(f"{pair}: {missing_uq} roster rows absent from the UQ "
                         f"store; the roster and the CSV disagree")
    missing_met = int(df[list(NLI_METRICS)].isna().all(axis=1).sum())
    if missing_met:
        raise ValueError(
            f"{pair}: uq_metrics covers {len(df) - missing_met} of {len(df)} "
            f"roster rows; M20 is incomplete for this pair")
    return df.rename(columns={"entropy": DSE_METRIC})


def run_pair(pair: str, *, n_cuts: int = bs.N_CUTS) -> pd.DataFrame:
    """One row per metric: its threshold, its band split, and the gate result."""
    df = load_metrics(pair)
    tau = float(df["tau"].iloc[0])
    rows = []
    for metric in METRICS:
        if metric not in df.columns:
            raise KeyError(f"{pair}: metric {metric!r} absent from uq_metrics")
        x = df[metric].to_numpy(dtype=float)
        cut = bs.best_split(x, n_cuts)
        band = bs.band_of(x, cut)
        n_hi = int((band == "HI").sum())
        # What the published grid would have chosen, and how far the bands move.
        # Recorded so M24 can justify the deviation from data, not from prose.
        legacy_cut = bs.legacy_split(x, n_cuts)
        legacy_band = bs.band_of(x, legacy_cut)
        n_moved = int((band != legacy_band).sum())
        # Only `dse` has an independently stored threshold to check against.
        gate = (bs.reproduces_tau(x, tau, n_cuts=n_cuts)
                if metric == DSE_METRIC else None)
        rows.append({
            "pair": pair,
            "metric": metric,
            "threshold": cut,
            "n": len(x),
            "n_HI": n_hi,
            "n_LO": len(x) - n_hi,
            "frac_HI": n_hi / len(x),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "negated": metric in NEGATED,
            # The published grid would have started here. Recorded so the
            # negated-metric failure is visible in the data, not just asserted.
            "legacy_grid_valid": bool(x.max() > 0),
            "legacy_threshold": legacy_cut,
            "legacy_frac_HI": float((legacy_band == "HI").mean()),
            "n_rows_moved": n_moved,
            "frac_rows_moved": n_moved / len(x),
            "stored_tau": tau if metric == DSE_METRIC else np.nan,
            "reproduces_tau": gate,
        })
    return pd.DataFrame(rows)


def bands(pair: str, metric: str, *, n_cuts: int = bs.N_CUTS) -> pd.DataFrame:
    """`id -> band` for one metric. What M23 relabels `category` from."""
    df = load_metrics(pair)
    x = df[metric].to_numpy(dtype=float)
    return pd.DataFrame({"id": df["id"].to_numpy(),
                         "band": bs.band_of(x, bs.best_split(x, n_cuts))})


def write(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    results.write_unit(TABLE, pair, df)
    return df
