"""The invariants every arm family must satisfy, checked rather than assumed.

Each of these failed at least once in the published work's history, and each one
fails **quietly** -- the numbers stay plausible, so nothing raises and the error
survives into a table. That is the whole argument for asserting them here rather
than trusting the builders.

  * **Containment.** `balanced` was drawn from the natural pool without respecting
    natural's train/test boundary, so 8.8-14.8% of its test rows sat in
    natural.train. Four of the six off-diagonal transfer cells were then part
    memorisation, and every one of them still produced a believable AUROC.

  * **No leakage.** The consequence of containment, stated directly over the grid
    that actually gets fitted.

  * **Entropy is dead within band.** This is the arm's entire purpose. `matched2`
    must land on 0.500 *exactly* -- if it only lands near 0.500, the confidence
    channel it exists to close is still partly open.

  * **Pooled entropy is dead too.** Every within-band AUROC can sit at 0.500 while
    the POOLED entropy AUROC runs to 0.689, because the band mixture reopens the
    channel. Checking only within-band would miss it entirely.
"""

from __future__ import annotations

import numpy as np

from csx_probe import config
from csx_probe.arms.build import Arm
from csx_probe.arms.common import BANDS
from csx_probe.metrics import safe_auc
from csx_probe.store.read import Entry


class GateError(AssertionError):
    """An arm family violates an invariant. The message names which."""


def check_containment(arms: dict[str, Arm]) -> list[str]:
    """`A.train subset natural.train` and `A.test subset natural.test`."""
    nat = arms.get("dse_natural")
    if nat is None:
        return ["no dse_natural arm to contain the others"]
    bad = []
    for split in ("train", "test"):
        base = set(nat.rows(split).tolist())
        for name, arm in arms.items():
            if name == "dse_natural":
                continue
            stray = set(arm.rows(split).tolist()) - base
            if stray:
                bad.append(
                    f"{arm.pair}/{name}.{split}: {len(stray)} rows are outside "
                    f"natural.{split}")
    return bad


def check_no_leakage(arms: dict[str, Arm]) -> list[str]:
    """`A_i.train n A_j.test = 0` for every ordered pair, including `i == j`."""
    bad = []
    names = list(arms)
    for i in names:
        tr = set(arms[i].train.tolist())
        for j in names:
            n = len(tr & set(arms[j].test.tolist()))
            if n:
                bad.append(f"{arms[i].pair}: {i}.train n {j}.test = {n}")
    return bad


def check_entropy_dead(entry: Entry, arm: Arm, *, tol: float,
                       pooled: bool = True) -> list[str]:
    """AUROC(entropy, incorrect vs correct) == 0.500 within each band, and pooled.

    Computed on the RAW entropy field, because that is the field every consumer
    reads. Matching on a rounded key and checking on the raw one is exactly how
    `matched` ends up at 5e-3 instead of 0.
    """
    bad = []
    cats, ent = entry.categories, entry.entropy
    for split in ("train", "test"):
        sel = arm.rows(split)
        c, e = cats[sel], ent[sel]
        for band, (inc, cor) in BANDS.items():
            m = np.isin(c, (inc, cor))
            if m.sum() == 0:
                bad.append(f"{arm.pair}/{arm.arm}.{split}/{band}: empty")
                continue
            a = safe_auc((c[m] == inc).astype(int), e[m])
            if np.isfinite(a) and abs(a - 0.5) > tol:
                bad.append(
                    f"{arm.pair}/{arm.arm}.{split}/{band}: "
                    f"AUROC(entropy, IvC) = {a:.6f} != 0.5 (tol {tol:g})")
        if pooled:
            y = np.isin(c, config.I_CATS).astype(int)
            a = safe_auc(y, e)
            if np.isfinite(a) and abs(a - 0.5) > tol:
                bad.append(
                    f"{arm.pair}/{arm.arm}.{split}: POOLED AUROC(entropy, IvC) = "
                    f"{a:.6f} != 0.5 -- the band mixture has reopened the "
                    f"confidence channel that within-band matching closed")
    return bad


def check_min_per_class(entry: Entry, arm: Arm) -> list[str]:
    """Every cell needs enough rows for its contrasts to be computable."""
    bad = []
    for split in ("train", "test"):
        c = entry.categories[arm.rows(split)]
        for cat in config.CATS:
            n = int((c == cat).sum())
            if n < config.MIN_PER_CLASS:
                bad.append(f"{arm.pair}/{arm.arm}.{split}/{cat}: {n} rows "
                           f"< min_per_class {config.MIN_PER_CLASS}")
    return bad


def check_all(entry: Entry, arms: dict[str, Arm], *, strict: bool = True
              ) -> list[str]:
    """Every gate for one pair's arm family. Returns the problems; `[]` is pass.

    Tolerances differ per arm by design: `matched2` is held to 1e-9 because its
    construction makes exactness attainable, while `matched` is allowed the 5e-3
    its rounded stratum key is known to cost. Holding `matched` to 1e-9 would
    fail every pair for a reason that is documented and intentional.
    """
    problems = check_containment(arms) + check_no_leakage(arms)
    tol_by_arm = {
        "dse_matched2": float(config.frozen()["arms"]["matched2"]["tol"]),
        "dse_matched": 5e-3,
    }
    for name, arm in arms.items():
        if name in tol_by_arm:
            problems += check_entropy_dead(entry, arm, tol=tol_by_arm[name])
        if strict:
            problems += check_min_per_class(entry, arm)
    return problems


def assert_all(entry: Entry, arms: dict[str, Arm], **kw) -> None:
    problems = check_all(entry, arms, **kw)
    if problems:
        raise GateError(f"{entry.pair}: arm gates failed:\n  "
                        + "\n  ".join(problems))
