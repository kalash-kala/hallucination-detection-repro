"""The four arm builders.

Every arm is a pair of **row-index sets** into the L0 arrays. Nothing is copied:
category, entropy and features all live in the entry already, so an arm that
carried its own copies would be one more thing that can drift out of agreement
with the features it selects.

The family invariant, which the whole transfer grid rests on:

    for every arm A:   A.train subset natural.train   and   A.test subset natural.test

Since `natural.train n natural.test = 0`, that **forces**
`A_i.train n A_j.test = 0` for every ordered pair of arms `(i, j)`. Every cell of
the cross-arm grid is then leak-free with no rows dropped and no per-cell surgery
on the evaluation set. `gates.py` asserts it rather than trusting it -- the
original `balanced` arm violated it (8.8-14.8% of its test rows sat in
natural.train), which is precisely why `balanced2` exists.

The arms, and what each one is for:

| arm | holds fixed | why |
|---|---|---|
| `dse_natural` | nothing | the observed distribution |
| `dse_balanced2` | cell sizes | removes composition as an explanation |
| `dse_matched` | entropy within band, 1:1 | removes confidence, at 1:1 |
| `dse_matched2` | entropy within band, p:q | same gate, sample-efficiently |
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csx_probe import config
from csx_probe.arms import common
from csx_probe.store.read import Entry

TEST_FRACTION: float = config.frozen()["arms"]["test_fraction"]
BALANCED2_QUOTA: dict[str, int] = dict(
    config.frozen()["arms"]["balanced2_quota"])
MATCHED_MAX_PQ: int = config.frozen()["arms"]["matched2"]["max_pq"]
TOL: float = float(config.frozen()["arms"]["matched2"]["tol"])

ARMS: tuple[str, ...] = config.ARM_ORDER


class ArmError(Exception):
    """An arm cannot be built for this pair. The message names the shortfall."""


@dataclass(frozen=True)
class Arm:
    """One arm's row indices, plus the provenance the results contract wants."""
    pair: str
    arm: str
    train: np.ndarray          # positional row indices into the entry
    test: np.ndarray
    note: str = ""
    ratio: str = ""            # matched/matched2 only, e.g. '3:2'

    def rows(self, split: str) -> np.ndarray:
        return self.train if split == "train" else self.test

    def n(self, split: str) -> int:
        return int(len(self.rows(split)))


# ── natural ──────────────────────────────────────────────────────────────────

def natural(entry: Entry, rng: np.random.Generator) -> Arm:
    """70/30 stratified 4-way by category.

    Reproduces `01_make_dse_splits.stratified_split`, including a detail that is
    easy to normalise away: rows are assigned to test by PERMUTED position but
    appended in ORIGINAL order. Emitting them in permuted order would give the
    same set with a different row order, and any downstream consumer that is
    order-sensitive would then disagree with the published arm.
    """
    cats = entry.categories
    rows = entry.rows["row"].to_numpy()
    train: list[int] = []
    test: list[int] = []
    for cat in common.CATS:
        stratum = rows[cats == cat]
        perm = rng.permutation(len(stratum))
        n_test = int(round(len(stratum) * TEST_FRACTION))
        test_pos = set(perm[:n_test].tolist())
        for i, r in enumerate(stratum):
            (test if i in test_pos else train).append(int(r))
    return Arm(entry.pair, "dse_natural", np.array(train, dtype=int),
               np.array(test, dtype=int))


# ── balanced2 ────────────────────────────────────────────────────────────────

def balanced2_quota(entry: Entry, nat: Arm, split: str) -> int:
    """The per-cell quota this pair can actually fill.

    The published quota (245 train / 105 test) is the ORIGINAL `balanced` arm's
    size, kept so every balanced number already in the reports stays drop-in
    comparable. It was set for the `qa8` text pairs, whose pools run to
    13,565-50,000 rows.

    The VLM pairs are much smaller -- advqa is 3,000 rows -- and their smallest
    cell can sit below it (gemma3_12b_advqa has 143 CL rows in natural.train
    against a quota of 245). Three ways to respond, and only one keeps the arm
    meaning what it says:

      * fail the pair -- loses `dse_balanced2` for a third of the VLM cohort,
        and with it the composition control on exactly the pairs where
        composition is most skewed;
      * sample WITH replacement to reach 245 -- inflates n without adding
        information, and the bootstrap CI would then understate the true spread;
      * lower the quota to the largest value every cell can fill.

    The arm's purpose is *equal cells*, not the number 245, so the third is the
    only one that preserves it. The quota is therefore
    `min(published, smallest cell)`, which is the published value verbatim
    whenever the pair is large enough -- so `qa8` is untouched and its parity
    holds exactly.

    A PARITY pair is never allowed to fall back: if one of those cannot fill the
    published quota, something upstream is wrong and a silently smaller arm would
    hide it.
    """
    cats = entry.categories
    pool_rows = nat.rows(split)
    published = BALANCED2_QUOTA[split]
    smallest = min(int((cats[pool_rows] == c).sum()) for c in common.CATS)

    if smallest >= published:
        return published
    if common.is_parity_pair(entry.pair):
        raise ArmError(
            f"{entry.pair}/balanced2/{split}: smallest cell has {smallest} rows "
            f"but the published quota is {published}. This is a parity pair, so "
            f"the quota is not negotiable -- investigate the split rather than "
            f"shrinking the arm.")
    if smallest < config.MIN_PER_CLASS:
        raise ArmError(
            f"{entry.pair}/balanced2/{split}: smallest cell has {smallest} rows, "
            f"below min_per_class {config.MIN_PER_CLASS}; no usable balanced arm "
            f"exists for this pair.")
    return smallest


def balanced2(entry: Entry, nat: Arm, rng: np.random.Generator) -> Arm:
    """An equal quota per category, drawn inside the natural split."""
    cats = entry.categories
    out: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for split in ("train", "test"):
        pool_rows = nat.rows(split)
        quota = balanced2_quota(entry, nat, split)
        if quota != BALANCED2_QUOTA[split]:
            notes.append(f"{split} quota {quota} (published {BALANCED2_QUOTA[split]})")
        kept: list[int] = []
        for cat in common.CATS:
            pool = pool_rows[cats[pool_rows] == cat]
            idx = rng.choice(len(pool), quota, replace=False)
            kept.extend(int(pool[i]) for i in idx)
        out[split] = np.array(sorted(kept), dtype=int)
    return Arm(entry.pair, "dse_balanced2", out["train"], out["test"],
               note="; ".join(notes))


# ── matched / matched2 ───────────────────────────────────────────────────────

def _matched_common(entry: Entry, nat: Arm, rng: np.random.Generator, *,
                    arm: str, key, ratio_chooser) -> Arm:
    cats, ent = entry.categories, entry.entropy
    rows_all = entry.rows["row"].to_numpy()
    out: dict[str, np.ndarray] = {}
    ratios: list[str] = []

    for split in ("train", "test"):
        sel = nat.rows(split)
        by_band = {
            b: common.strata(cats[sel], ent[sel], rows_all[sel], inc, cor, key)
            for b, (inc, cor) in common.BANDS.items()
        }
        p, q = ratio_chooser(by_band)
        ratios.append(f"{p}:{q}")

        kept: list[int] = []
        for band in common.BANDS:
            by = by_band[band]
            for e, ki, kc in common.plan_ratio(by, p, q):
                pool_i, pool_c = by[e]
                for pool, k in ((pool_i, ki), (pool_c, kc)):
                    idx = rng.choice(len(pool), k, replace=False)
                    kept.extend(int(pool[j]) for j in idx)
        if not kept:
            raise ArmError(f"{entry.pair}/{arm}/{split}: no stratum survived")
        out[split] = np.array(sorted(kept), dtype=int)

    ratio = ratios[0] if ratios[0] == ratios[1] else f"{ratios[0]}/{ratios[1]}"
    return Arm(entry.pair, arm, out["train"], out["test"], ratio=ratio)


def matched(entry: Entry, nat: Arm, rng: np.random.Generator) -> Arm:
    """Strict 1:1 within each exact entropy stratum, keyed on `round(e, 12)`.

    Reproduced with its known weakness intact: within a rounded stratum the raw
    entropies still differ at the last ULP, so the raw-field AUROC lands near
    0.500 but not on it (~5e-3). `matched2` is the fix; this arm stays as
    published so the comparison between them remains meaningful.
    """
    return _matched_common(entry, nat, rng, arm="dse_matched",
                           key=common.stratum_key_rounded,
                           ratio_chooser=lambda _by: (1, 1))


def matched2(entry: Entry, nat: Arm, rng: np.random.Generator) -> Arm:
    """A single ratio `p:q` across both bands, keyed on the raw entropy float."""
    return _matched_common(entry, nat, rng, arm="dse_matched2",
                           key=common.stratum_key_raw,
                           ratio_chooser=_choose_ratio)


def _choose_ratio(by_band: dict) -> tuple[int, int]:
    """Maximise summed `n_eff` subject to **dominance** over the 1:1 plan.

    Both unconstrained objectives fail, in opposite directions:

      * maximising summed n_eff alone lets a band be traded away wholesale --
        one published pair picks 6:1 and collapses LO to ~50 effective rows,
        gutting ILvCL, ILvC and CLvI for that pair;
      * maximising the WORST band instead picks 1:2 there and drops HI below its
        `matched` value.

    So the constraint is that no band may end up worse than it is at 1:1, and the
    objective is summed n_eff within that feasible set. 1:1 always satisfies the
    constraint with equality, so a solution always exists and `matched2` is never
    worse than `matched` in any band of any pair.

    One ratio is used for BOTH bands, not one per band. Entropy is below tau for
    every H row and above it for every L row, so every cross-band comparison is
    decided by the band alone; the pooled AUROC reduces to 0.500 exactly when
    `n_IL*n_CH == n_IH*n_CL`, i.e. when `r_HI == r_LO`. Optimising per band would
    hold every within-band AUROC at 0.500 while pushing the POOLED entropy IvC
    back up to 0.585-0.689 -- reopening at the band-mixture level the very
    channel this arm exists to close.
    """
    base = {b: common.plan_neff(common.plan_ratio(by_band[b], 1, 1))
            for b in common.BANDS}
    best: tuple[float, int, int] | None = None
    for p, q in common.coprime_ratios(MATCHED_MAX_PQ):
        plans = {b: common.plan_ratio(by_band[b], p, q) for b in common.BANDS}
        if any(not pl for pl in plans.values()):
            continue                       # a band would be emptied
        ne = {b: common.plan_neff(plans[b]) for b in common.BANDS}
        if any(ne[b] < base[b] - 1e-9 for b in common.BANDS):
            continue                       # dominance: no band may regress
        v = sum(ne.values())
        if best is None or v > best[0]:
            best = (v, p, q)
    if best is None:
        raise ArmError("no admissible ratio satisfied the dominance constraint")
    return best[1], best[2]


# ── the whole family, for one pair ───────────────────────────────────────────

BUILDERS = {
    "dse_natural": None,          # built first; the others are drawn inside it
    "dse_balanced2": balanced2,
    "dse_matched": matched,
    "dse_matched2": matched2,
}


def build_all(entry: Entry, rng: np.random.Generator | None = None,
              *, arms: tuple[str, ...] = ARMS) -> dict[str, Arm]:
    """Every arm for one pair, sharing one RNG stream in a fixed arm order.

    The stream is shared across arms within a pair (as the published scripts do
    within each builder), but a NEW pair gets its own stream rather than joining
    a global one -- see `common.pair_rng` for why.
    """
    common.assert_band_orientation(entry.categories, entry.entropy)
    rng = rng if rng is not None else common.pair_rng(entry.pair)

    nat = natural(entry, rng)
    out: dict[str, Arm] = {"dse_natural": nat}
    for arm in arms:
        if arm == "dse_natural":
            continue
        fn = BUILDERS.get(arm)
        if fn is None:
            raise ArmError(f"unknown arm {arm!r}")
        out[arm] = fn(entry, nat, rng)
    return out
