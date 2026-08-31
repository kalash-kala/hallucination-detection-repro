"""The alpha ladder: a continuous path from `natural` (a=0) to `matched2` (a=1).

The transfer grid compares two endpoints. This builds the rungs between them, so
"does entropy-matching rotate the correctness probe off the confidence axis?" can
be asked as a *trend* rather than as a two-point difference that a single unlucky
draw could manufacture.

**The rungs are nested by construction**, and that is the point. Each cell's rows
are ordered once -- matched2-kept rows first, then the rest -- and rung `a` takes a
prefix of that fixed order:

    k(cell, a) = round((1 - a) * n_natural(cell) + a * n_matched2(cell))

Since `n_matched2 <= n_natural`, `k` shrinks as `a` grows and every rung is a
subset of the one below it. A ladder whose rungs were drawn independently would mix
the rotation being measured with resampling noise at every step.

**The placebo is what makes the result falsifiable.** A smaller training set gives
a noisier `w_g`, which sits further from any fixed reference *for free* -- so
`theta(a=1) > theta(a=0)` on its own proves nothing. Each rung therefore gets 20
controls with the identical row count and identical per-(band, class) composition,
but drawn **uniformly within (band, class) rather than within stratum**, which
leaves the entropy leak intact. The finding is a rung beating its own placebo, never
a rung beating the natural arm.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from csx_probe import config
from csx_probe.arms import common
from csx_probe.arms.build import Arm
from csx_probe.store.read import Entry

ALPHAS: tuple[float, ...] = tuple(config.frozen()["alpha"]["alphas"])
N_PLACEBO: int = config.frozen()["alpha"]["n_placebo"]
SEED_BASE: int = config.frozen()["alpha"]["placebo_seed_base"]
SEED_STRIDE: int = config.frozen()["alpha"]["placebo_seed_stride"]


def tag(a: float) -> str:
    return f"a{int(round(a * 100)):03d}"


def _cells(entry: Entry, rows: np.ndarray) -> dict[tuple, list[int]]:
    """`(band, class, stratum) -> row ids`, the unit the ladder interpolates on."""
    cells: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        cat = entry.categories[r]
        band = common.BAND_OF[cat]
        cls = "I" if cat.startswith("I") else "C"
        cells[(band, cls, common.stratum_key_raw(entry.entropy[r]))].append(int(r))
    return cells


def build_ladder(entry: Entry, natural: Arm, matched2: Arm, split: str = "train"
                 ) -> tuple[dict[float, np.ndarray], dict[float, dict]]:
    """`{alpha: row ids}` plus the per-(band, class) counts the placebos match."""
    nat_rows = natural.rows(split)
    ref = set(matched2.rows(split).tolist())

    cells = _cells(entry, nat_rows)
    rng = np.random.default_rng(config.SEED)

    ordered: dict[tuple, list[int]] = {}
    for c in sorted(cells):
        keep = [r for r in cells[c] if r in ref]
        drop = [r for r in cells[c] if r not in ref]
        rng.shuffle(keep)
        rng.shuffle(drop)
        ordered[c] = keep + drop

    n_nat = {c: len(v) for c, v in ordered.items()}
    n_ref = {c: sum(1 for r in v if r in ref) for c, v in ordered.items()}

    out: dict[float, np.ndarray] = {}
    counts: dict[float, dict] = {}
    for a in ALPHAS:
        rows: list[int] = []
        per_cell: dict[tuple, int] = defaultdict(int)
        for c, seq in ordered.items():
            k = int(round((1.0 - a) * n_nat[c] + a * n_ref[c]))
            k = max(0, min(k, n_nat[c]))
            rows.extend(seq[:k])
            per_cell[(c[0], c[1])] += k
        out[a] = np.array(sorted(rows), dtype=int)
        counts[a] = dict(per_cell)
    return out, counts


def build_placebo(entry: Entry, natural: Arm, target: dict, draw: int,
                  split: str = "train") -> np.ndarray:
    """One size- and composition-matched control for a rung.

    Drawn uniformly within (band, class) -- deliberately NOT within stratum. A
    placebo that matched strata too would also be entropy-matched, i.e. it would
    be another treatment arm rather than a control, and the null it produced
    would be the very effect under test.
    """
    pool: dict[tuple, list[int]] = defaultdict(list)
    for r in natural.rows(split):
        cat = entry.categories[r]
        pool[(common.BAND_OF[cat],
              "I" if cat.startswith("I") else "C")].append(int(r))

    rng = np.random.default_rng(SEED_BASE + SEED_STRIDE * (draw + 1))
    rows: list[int] = []
    for cell, k in sorted(target.items()):
        avail = pool[cell]
        idx = rng.choice(len(avail), size=min(k, len(avail)), replace=False)
        rows.extend(int(avail[i]) for i in idx)
    return np.array(sorted(rows), dtype=int)


def assert_nested(ladder: dict[float, np.ndarray]) -> None:
    """Each rung must be contained in the one below it."""
    order = sorted(ladder, reverse=True)
    for hi, lo in zip(order, order[1:]):
        if not set(ladder[hi].tolist()) <= set(ladder[lo].tolist()):
            raise AssertionError(
                f"alpha ladder is not nested: a={hi} is not a subset of a={lo}; "
                f"the rungs would then differ by resampling as well as by "
                f"matching, and the trend would be uninterpretable")
