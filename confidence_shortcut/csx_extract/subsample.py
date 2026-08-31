"""Choosing which rows to extract.

Extraction is the expensive step, so pools far larger than the study needs get
capped. The rule is not invented here: `00_make_natural_splits.py` already keeps
sciq's whole pool and natural-stratified-subsamples triviaqa, and the result is
exactly 15,000 rows per triviaqa pair (10,500 train / 4,500 test, verified
against the published `natural` arm). Reusing that number means

  * the 8 parity pairs are reproduced rather than re-invented, and
  * every pair in every cohort sits at a comparable `n`, so a median over any
    grouping is not confounded by sample size.

Stratification is on the four confidence x correctness cells, so band structure
and prevalence survive the cut -- which matters because the whole study conditions
on those cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from csx_common import registry


@dataclass(frozen=True)
class Subsample:
    ids: list[str]
    n_pool: int
    n_target: int | None
    quotas: dict[str, int]
    seed: int

    @property
    def n_kept(self) -> int:
        return len(self.ids)

    @property
    def applied(self) -> bool:
        return self.n_kept < self.n_pool

    def to_meta(self) -> dict:
        return {
            "n_target": self.n_target,
            "n_pool": self.n_pool,
            "n_kept": self.n_kept,
            "seed": self.seed,
            "stratified_on": list(registry.CATS),
            "quotas": dict(self.quotas),
            "applied": self.applied,
        }


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Apportion `total` across cells in proportion to `counts`.

    Floor each share, then hand the remainder to the largest fractional parts,
    then clip to what is actually available. The same rule
    `49_build_sizeonly_arms.py` uses -- proportional rounding without it would
    over- or under-shoot the target by a few rows, which then shows up as arms
    that do not match their declared size.
    """
    pool = sum(counts.values())
    if total >= pool:
        return dict(counts)
    exact = {k: total * v / pool for k, v in counts.items()}
    base = {k: int(np.floor(v)) for k, v in exact.items()}
    # Deterministic ordering: largest fractional part first, ties by cell name,
    # so the same inputs always give the same apportionment.
    order = sorted(counts, key=lambda k: (-(exact[k] - base[k]), k))
    short = total - sum(base.values())
    for k in order:
        if short <= 0:
            break
        if base[k] < counts[k]:
            base[k] += 1
            short -= 1
    return {k: min(v, counts[k]) for k, v in base.items()}


def choose(rows: pd.DataFrame, pair_key: str, *, n_target: int | None = None,
           seed: int = 42) -> Subsample:
    """Pick the rows to extract for one pair.

    `rows` is that pair's slice of the L2 table (needs `id` and `category`).
    `n_target` defaults to the dataset's declared cap; `None` keeps everything.
    """
    p = registry.get(pair_key)
    if n_target is None:
        n_target = p.dataset.n_target

    rows = rows.reset_index(drop=True)
    counts = {c: int((rows["category"] == c).sum()) for c in registry.CATS}
    n_pool = len(rows)

    if n_target is None or n_pool <= n_target:
        return Subsample(ids=rows["id"].astype(str).tolist(), n_pool=n_pool,
                         n_target=n_target, quotas=counts, seed=seed)

    quotas = largest_remainder(counts, n_target)
    rng = np.random.default_rng(seed)
    keep: list[str] = []
    # Iterate cells in a fixed order so the draw does not depend on dict order.
    for cat in registry.CATS:
        idx = np.flatnonzero((rows["category"] == cat).to_numpy())
        k = quotas[cat]
        if k >= len(idx):
            chosen = idx
        else:
            chosen = idx[rng.choice(len(idx), k, replace=False)]
        keep.extend(rows.loc[np.sort(chosen), "id"].astype(str).tolist())

    # Return in pool order, not cell order, so the extraction reads the CSV
    # roughly sequentially and the row table is not sorted by category.
    keep_set = set(keep)
    ordered = [i for i in rows["id"].astype(str) if i in keep_set]
    return Subsample(ids=ordered, n_pool=n_pool, n_target=n_target,
                     quotas=quotas, seed=seed)
