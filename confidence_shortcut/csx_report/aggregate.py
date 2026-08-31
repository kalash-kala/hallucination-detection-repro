"""The only place aggregation happens.

Everything upstream is atomic per pair, so the grouping is a **runtime argument**:
LLM vs VLM, per-dataset, per-model-size, leave-one-out -- none of it requires
refitting anything, and this module runs in seconds.

Two things it refuses to hide:

**Mixed `C` and mixed prompt templates.** Both can legitimately differ between
pairs in one group, and both change what a median over that group means. They are
surfaced as warnings attached to the result, not silently averaged over.

**The pass-rule threshold.** The pre-registered bar is "6 of 8" for `qa8`. For any
other grouping it is `ceil(0.75 * n_pairs)`, which coincides at n=8. The threshold
is returned alongside the count so a reader never has to infer the bar from the row
count -- inferring it is exactly how a 4-of-5 result gets read as if it cleared the
same bar as 6-of-8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

N_BOOT = 1000
SEED = 42
CI = (2.5, 97.5)


def hier_median_ci(values, *, n: int = N_BOOT, seed: int = SEED
                   ) -> tuple[float, float, float]:
    """Median over pairs, with a CI from resampling **pairs**.

    Resampling pairs rather than rows is what makes this a statement about the
    cohort: the row-level uncertainty is already inside each pair's own CI, and
    resampling rows here would answer a different question (how precise is this
    pair) than the one being asked (how consistent is this across pairs).
    """
    v = np.asarray([x for x in values if not (x is None or np.isnan(x))],
                   dtype=float)
    if not len(v):
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return (float(np.median(v)),
            float(np.quantile(meds, CI[0] / 100.0)),
            float(np.quantile(meds, CI[1] / 100.0)))


def pass_threshold(n_pairs: int, *, cohort: str | None = None) -> int:
    """The pre-registered bar, stated rather than implied.

    `qa8` keeps its published `6 of 8` verbatim. Everything else uses
    `ceil(0.75 * n)`, which is 6 at n=8 -- so the rule generalises without moving
    the published one.
    """
    if cohort == "qa8" and n_pairs == 8:
        return 6
    return math.ceil(0.75 * n_pairs)


@dataclass
class Grouped:
    """An aggregated table plus everything that qualifies it."""
    table: pd.DataFrame
    n_pairs: int
    pairs: list[str]
    warnings: list[str] = field(default_factory=list)


def _warn_mixed(df: pd.DataFrame) -> list[str]:
    out = []
    for col, what in (("C", "regularisation"),
                      ("prompt_template", "prompt template"),
                      ("c_mode", "C policy")):
        if col not in df.columns:
            continue
        vals = sorted(str(v) for v in df[col].dropna().unique())
        if len(vals) > 1:
            out.append(
                f"group mixes {what} ({col} in {vals}); the median is over pairs "
                f"that were not fitted identically")
    return out


def median_over_pairs(df: pd.DataFrame, *, by: list[str],
                      value: str = "AUROC",
                      cohort: str | None = None) -> Grouped:
    """Median `value` across pairs, within each `by` group."""
    pairs = sorted(df["pair"].unique())
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        med, lo, hi = hier_median_ci(g[value].to_numpy(dtype=float))
        rec = dict(zip(by, key))
        rec.update({
            f"{value}_median": med, f"{value}_lo": lo, f"{value}_hi": hi,
            "n_pairs": int(g["pair"].nunique()),
            # Kept per group, not just per cohort: a family can be missing for a
            # pair (phase 2 not done), so a group's pair count is not always the
            # cohort's.
            "n_rows": int(len(g)),
        })
        rows.append(rec)
    return Grouped(pd.DataFrame(rows), len(pairs), pairs, _warn_mixed(df))


def pass_counts(verdict: pd.DataFrame, *, by: list[str] | None = None,
                cohort: str | None = None) -> Grouped:
    """Count the per-pair `passes` flags, with the bar printed alongside."""
    by = by or ["family", "metric"]
    pairs = sorted(verdict["pair"].unique())
    rows = []
    for key, g in verdict.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        n = int(g["pair"].nunique())
        thr = pass_threshold(n, cohort=cohort)
        n_pass = int(g["passes"].astype(bool).sum())
        rec = dict(zip(by, key))
        rec.update({
            "n_pass": n_pass, "n_pairs": n, "threshold": thr,
            "meets_bar": bool(n_pass >= thr),
            "rule": f"{n_pass}/{n} >= {thr}",
            "delta_median": float(np.nanmedian(g["delta"].to_numpy(float))),
        })
        rows.append(rec)
    return Grouped(pd.DataFrame(rows), len(pairs), pairs, _warn_mixed(verdict))


def shortcut_table(per_pair: pd.DataFrame, *, head: str = "g",
                   contrast: str = "IvC",
                   cohort: str | None = None) -> Grouped:
    """The headline: one median per (family, train_arm, test_arm).

    Restricted to a single head and contrast, because a median taken across
    heads would average the correctness axis together with the confidence axis
    it is being contrasted against -- which is the one comparison the whole study
    is about.
    """
    d = per_pair[(per_pair["head"] == head) &
                 (per_pair["contrast"] == contrast)]
    return median_over_pairs(
        d, by=["family", "segment", "train_arm", "test_arm"], cohort=cohort)
