"""Shared arm vocabulary: bands, strata, the objective, and the RNG contract.

Two things here are load-bearing and would be easy to "clean up" into wrongness.

**Band orientation.** `H` is **low** entropy, i.e. high confidence. Several
conclusions inverta if this is read the other way, so `assert_band_orientation`
checks it against the data rather than trusting the label.

**The RNG contract.** The published builders create ONE `default_rng(42)` and draw
for every pair in a fixed order, so a pair's arm depends on how many pairs were
drawn before it. That is reproduced exactly for the 8 `qa8` parity pairs -- their
published arms cannot be regenerated any other way.

It is deliberately **not** extended to new pairs. Under a shared stream, adding a
VLM pair would silently change the arms of every pair after it in the order, which
contradicts the property the rest of this design is built on: adding a pair
invalidates nothing already computed. New pairs therefore get a per-pair stream
seeded from the pair name, which is both atomic and reproducible. `qa8` keeps the
published behaviour; nothing published moves.
"""

from __future__ import annotations

import zlib
from collections import defaultdict

import numpy as np

from csx_probe import config

CATS: tuple[str, ...] = config.CATS
# band -> (incorrect category, correct category)
BANDS: dict[str, tuple[str, str]] = {"HI": ("IH", "CH"), "LO": ("IL", "CL")}
BAND_OF: dict[str, str] = {"IH": "HI", "CH": "HI", "IL": "LO", "CL": "LO"}

# The 8 pairs whose arms must reproduce the published draw, in the order the
# published scripts iterated them. This ordering IS the contract for those pairs.
QA8_ORDER: tuple[str, ...] = tuple(
    f"{m}_{d}" for m in ("llama", "mistral", "qwen", "gemma")
    for d in ("sciq", "triviaqa")
)


def is_parity_pair(pair: str) -> bool:
    return pair in QA8_ORDER


def pair_rng(pair: str, seed: int = config.SEED) -> np.random.Generator:
    """A per-pair stream, for pairs that are not on the parity path.

    `zlib.crc32` deliberately, not Python's `hash()`: string hashing is salted
    per process, so `hash()` would give a different arm on every run -- the same
    trap the published size-only builder documents.
    """
    return np.random.default_rng((seed + zlib.crc32(pair.encode())) % 2**32)


def shared_rng(seed: int = config.SEED) -> np.random.Generator:
    """The single stream the published builders share across all `qa8` pairs."""
    return np.random.default_rng(seed)


# ── strata ───────────────────────────────────────────────────────────────────

def stratum_key_raw(e: float) -> float:
    """`matched2`: the RAW float, with only `-0.0` normalised to `0.0`.

    Consumers read the raw entropy field, so the raw field is what must be
    matched. Keying on a rounded value leaves the last-ULP differences inside a
    stratum unmatched, which is exactly why `matched` drifts to ~5e-3 instead of
    landing on 0.500 exactly.
    """
    return float(e) + 0.0


def stratum_key_rounded(e: float, decimals: int = 12) -> float:
    """`matched`: `round(e, 12)`. Reproduced as-is, drift included."""
    return round(float(e), decimals) + 0.0


def strata(cats: np.ndarray, ent: np.ndarray, rows: np.ndarray,
           inc: str, cor: str, key) -> dict[float, tuple[list[int], list[int]]]:
    """`entropy -> ([incorrect row ids], [correct row ids])` for one band."""
    by: dict[float, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for c, e, r in zip(cats, ent, rows):
        if c == inc:
            by[key(e)][0].append(int(r))
        elif c == cor:
            by[key(e)][1].append(int(r))
    return dict(by)


# ── the objective matched2 maximises ─────────────────────────────────────────

def n_eff(a: int, b: int) -> float:
    """Effective sample size `1/(1/a + 1/b)`.

    This, not the raw row count, is what sets the standard error of every AUROC
    computed on the arm. Maximising rows instead would hoard majority rows that
    add almost no precision.
    """
    return 0.0 if min(a, b) == 0 else 1.0 / (1.0 / a + 1.0 / b)


def plan_ratio(by: dict[float, tuple[list[int], list[int]]], p: int, q: int
               ) -> list[tuple[float, int, int]]:
    """`(stratum, n_incorrect, n_correct)` realising the exact ratio `p:q`.

    `t = min(n_inc // q, n_cor // p)` keeps `t*q` incorrect and `t*p` correct, so
    the ratio is exact in EVERY stratum -- which is what makes the within-band
    AUROC land on 0.500 exactly rather than approximately. `t == 0` drops the
    stratum entirely.
    """
    out = []
    for e, (pi, pc) in by.items():
        t = min(len(pi) // q, len(pc) // p)
        if t:
            out.append((e, t * q, t * p))
    return out


def plan_neff(plan: list[tuple[float, int, int]]) -> float:
    return n_eff(sum(x[1] for x in plan), sum(x[2] for x in plan))


def coprime_ratios(max_pq: int) -> list[tuple[int, int]]:
    return [(p, q) for p in range(1, max_pq + 1) for q in range(1, max_pq + 1)
            if np.gcd(p, q) == 1]


# ── orientation gate ─────────────────────────────────────────────────────────

def assert_band_orientation(cats: np.ndarray, ent: np.ndarray,
                            tol: float = 0.0) -> None:
    """`H` must be the LOW-entropy band. Checked, never assumed.

    The DSE threshold splits on entropy, so every `H` row sits below every `L`
    row. If this is ever inverted upstream, the confidence head and every band
    conclusion silently swap meaning while every number still looks plausible.
    """
    cats, ent = np.asarray(cats), np.asarray(ent, dtype=float)
    hi = ent[np.isin(cats, ("IH", "CH"))]
    lo = ent[np.isin(cats, ("IL", "CL"))]
    if not len(hi) or not len(lo):
        return
    if hi.max() > lo.min() + tol:
        raise ValueError(
            f"band orientation is inverted or the bands overlap: max(H entropy)="
            f"{hi.max():.6g} > min(L entropy)={lo.min():.6g}. H must be the LOW-"
            f"entropy (high-confidence) band -- see frozen_constants.yaml:band")
