"""The two controls that separate *entropy-matching* from its side effects.

Every matched arm differs from `natural` in three ways at once -- it is smaller,
it is differently composed, and it is entropy-matched. The transfer grid alone
cannot say which of those is doing the work, so each confound arm holds two of
the three fixed and varies the third.

    natural  (large, skewed, leaky)
       |
       |  <- ns_A : SIZE alone. natural's own proportions, A's row count.
       v
    ns_A
       |
       |  <- pl_A : COMPOSITION alone. A's per-cell counts, drawn uniformly
       v            inside each cell, so the entropy leak stays wide open.
    pl_A
       |
       |  <- A : the entropy-matching itself, and nothing else.
       v
    A = matched / matched2

`balanced2` is its own placebo -- it is already a uniform per-cell draw -- so
`pl_dse_balanced2` should come out statistically indistinguishable from it. It is
built anyway, as a built-in null: if that comparison shows an effect, the placebo
machinery is wrong and every other placebo row is suspect.

**Leak-freedom is structural.** Every confound row is drawn from `natural`'s own
split pools, and the family invariant already gives `A.test subset natural.test`
with `natural.train n natural.test = 0`. So a train-only placebo cannot reach any
real arm's test rows. `gates.check_no_leakage` is still run over the result
rather than trusting that argument.
"""

from __future__ import annotations

import zlib
from collections import Counter

import numpy as np

from csx_probe import config
from csx_probe.arms import alpha as alpha_arms, common
from csx_probe.arms.build import Arm
from csx_probe.store.read import Entry

# The arms a control is built to imitate. `dse_natural` is excluded because a
# control for it would be a copy of it.
TARGETS = ("dse_balanced2", "dse_matched", "dse_matched2")
N_DRAWS: int = config.frozen()["confounds"]["n_draws"]

PLACEBO_PREFIX = "pl_"
SIZEONLY_PREFIX = "ns_"


def _cell_counts(entry: Entry, rows: np.ndarray) -> dict[tuple, int]:
    """`(band, class) -> count`, the key `alpha.build_placebo` targets."""
    out: Counter = Counter()
    for r in rows:
        cat = entry.categories[r]
        out[(common.BAND_OF[cat], "I" if cat.startswith("I") else "C")] += 1
    return dict(out)


def _cat_counts(entry: Entry, rows: np.ndarray) -> Counter:
    return Counter(entry.categories[r] for r in rows)


def sizeonly_seed(pair: str, tgt: str, split: str, draw: int) -> int:
    """`zlib.crc32`, never `hash()`.

    Python randomises string hashing per process unless `PYTHONHASHSEED` is
    pinned, so `hash()` here would make these arms silently unreproducible
    between runs -- and a control that moves between runs is not a control.
    """
    key = f"{pair}|{tgt}|{split}|{draw}".encode()
    return (config.SEED + zlib.crc32(key)) % (2 ** 32)


def quota(pool: Counter, n_target: int) -> dict[str, int]:
    """Largest-remainder apportionment of `n_target` in `pool`'s proportions.

    A plain uniform draw would preserve the proportions only in expectation and
    let them wobble between draws, putting composition noise back into a control
    whose whole purpose is to hold composition fixed.
    """
    tot = sum(pool.values())
    if tot == 0:
        return {c: 0 for c in config.CATS}
    exact = {c: n_target * pool[c] / tot for c in config.CATS}
    base = {c: int(np.floor(exact[c])) for c in config.CATS}
    rem = n_target - sum(base.values())
    for c in sorted(config.CATS, key=lambda c: exact[c] - base[c],
                    reverse=True)[:rem]:
        base[c] += 1
    for c in config.CATS:                      # cannot ask for more than exists
        base[c] = min(base[c], pool[c])
    return base


def _draw_stratified(entry: Entry, pool_rows: np.ndarray, q: dict[str, int],
                     seed: int) -> np.ndarray:
    by: dict[str, list[int]] = {c: [] for c in config.CATS}
    for r in pool_rows:
        by[entry.categories[r]].append(int(r))
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for c in config.CATS:
        k = min(q.get(c, 0), len(by[c]))
        if not k:
            continue
        idx = rng.choice(len(by[c]), size=k, replace=False)
        out.extend(by[c][i] for i in idx)
    return np.array(sorted(out), dtype=int)


# ── placebo: composition matched, entropy leak intact ────────────────────────

def placebo_arms(entry: Entry, arms: dict[str, Arm], *,
                 draws: int = N_DRAWS) -> dict[str, Arm]:
    """`pl_<target>_d<NN>` -- TRAIN-only arms.

    A placebo is never a test column: it exists to be evaluated against the four
    real arms, which is what makes its row directly comparable to the real arm's
    row in the same grid. `test` is therefore empty, and `confounds.run_pair`
    supplies the test columns.
    """
    nat = arms["dse_natural"]
    out: dict[str, Arm] = {}
    for tgt in TARGETS:
        if tgt not in arms:
            continue
        target = _cell_counts(entry, arms[tgt].train)
        want = _cat_counts(entry, arms[tgt].train)
        for d in range(draws):
            rows = alpha_arms.build_placebo(entry, nat, target, d, "train")
            got = _cat_counts(entry, rows)
            if got != want:
                raise ValueError(
                    f"{entry.pair}/{tgt}/d{d}: placebo composition {dict(got)} "
                    f"!= target {dict(want)}; the control is not composition-"
                    f"matched and its comparison would be meaningless")
            out[f"{PLACEBO_PREFIX}{tgt}_d{d:02d}"] = Arm(
                pair=entry.pair, arm=f"{PLACEBO_PREFIX}{tgt}_d{d:02d}",
                train=rows, test=np.array([], dtype=int),
                note=f"composition-matched control for {tgt}, draw {d}")
    return out


# ── size-only: natural's skew, the target's n ────────────────────────────────

def sizeonly_arms(entry: Entry, arms: dict[str, Arm], *,
                  draws: int = N_DRAWS) -> dict[str, Arm]:
    """`ns_<target>_d<NN>` -- TRAIN *and* TEST arms.

    Unlike the placebo, the question here is about a probe trained *and*
    evaluated on a smaller natural population, so both splits are drawn. These
    deliberately do NOT match per-cell counts: they preserve natural's skew and
    vary only `n`, which is the one thing the placebo cannot isolate.
    """
    nat = arms["dse_natural"]
    natc = {s: _cat_counts(entry, nat.rows(s)) for s in ("train", "test")}
    out: dict[str, Arm] = {}
    for tgt in TARGETS:
        if tgt not in arms:
            continue
        q = {s: quota(natc[s], arms[tgt].n(s)) for s in ("train", "test")}
        for d in range(draws):
            name = f"{SIZEONLY_PREFIX}{tgt}_d{d:02d}"
            splits = {
                s: _draw_stratified(entry, nat.rows(s), q[s],
                                    sizeonly_seed(entry.pair, tgt, s, d))
                for s in ("train", "test")}
            out[name] = Arm(
                pair=entry.pair, arm=name, train=splits["train"],
                test=splits["test"],
                note=f"size-only control for {tgt} "
                     f"(n={arms[tgt].n('train')}/{arms[tgt].n('test')}), draw {d}")
    return out


def build_all(entry: Entry, arms: dict[str, Arm], *,
              draws: int = N_DRAWS) -> dict[str, Arm]:
    """Both control families, keyed by arm name."""
    return {**placebo_arms(entry, arms, draws=draws),
            **sizeonly_arms(entry, arms, draws=draws)}
