"""Putting the two band experts on one scale.

Two experts fit on different populations, in different transforms, have no
common origin: expert HI's logit of +1.2 and expert LO's logit of +1.2 do not
mean the same thing. *Some* rescaling is therefore mechanically required before
a single AUROC can be taken over both bands' rows. That is a cost of having two
heads, not a bonus the routed arm is being handed, and `proba` is carried as the
control that shows what happens when the rescaling is chosen badly.

**All four poolers are the same 2-parameter affine-per-band family.** They differ
only in where the two numbers come from:

| pooler | map | parameters from |
|---|---|---|
| `z` | `(f_b − μ_b)/σ_b` | the raw logit's distribution shape, label-blind |
| `platt` | `a_b·f_b + b_b` | fit against labels, TRAIN only |
| `platt_prior` | `a_b·f_b + (b_b − logit π_b)` | as above, minus the band's counted base rate |
| `proba` | `σ(f_b)` | *not* neutral — re-imports the band offset; the negative control |

`platt_prior` is Lever A. Its target is the prior-free quantity

    s(x) = logit P̂(I | x, band) − logit P(I | band)

and because `logit(σ(z)) == z`, the Platt sigmoid and the outer logit cancel
exactly, collapsing the whole scorer to an affine map of the raw logit. That
cancellation is why Lever A can be checked rather than argued: under **oracle**
routing every atomic cell is scored by exactly one expert, and AUROC inside a
single band is invariant to a positive affine map, so `IHvCH` and `ILvCL` must
come out **bit-identical** between `z` and `platt_prior`. Under a real router a
cell is scored by a *mixture* of two affine maps, which is not affine, so the
identity must break — and `lever_a.py` asserts that it breaks, because if it
holds under `sampled`/`greedy` then routing is not actually being applied.

`proba` is the odd one out and deliberately so. `σ` is monotone, so it changes
nothing *within* a band; across bands it preserves the expert's own intercept,
which is precisely the band base rate the pooling step is supposed to remove.

Platt is fit unregularised (`C=1e6`) and with `class_weight=None`: two parameters
on thousands of rows need no regularisation, and the experts are already fit
`class_weight='balanced'`, which re-imposes a 50/50 prior — carrying that into
the calibration would defeat the entire point of subtracting the prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from csx_probe import config

METHODS: tuple[str, ...] = config.POOLERS


class PoolingError(Exception):
    """A pooler cannot be fit or applied. The message names which band."""


@dataclass(frozen=True)
class BandMap:
    """One band's affine map `s = a·f + b`, plus the diagnostics behind it."""

    band: str
    a: float
    b: float
    pi: float = float("nan")
    logit_pi: float = float("nan")
    ece_raw: float = float("nan")
    ece_pooled: float = float("nan")
    n_train: int = 0

    def apply(self, f: np.ndarray) -> np.ndarray:
        return self.a * np.asarray(f, dtype=float) + self.b


@dataclass(frozen=True)
class Pooler:
    """The fitted pooling step: one `BandMap` per band, plus the method name."""

    method: str
    maps: dict[str, BandMap] = field(default_factory=dict)

    def apply(self, band: str, f: np.ndarray) -> np.ndarray:
        if self.method == "proba":
            # Not affine, and not meant to be: the sigmoid keeps the expert's own
            # intercept, i.e. the band's base rate. The control, by construction.
            return 1.0 / (1.0 + np.exp(-np.asarray(f, dtype=float)))
        m = self.maps.get(band)
        if m is None:
            raise PoolingError(
                f"{self.method}: no map for band {band!r}; it was not fitted "
                f"(its expert is probably None)")
        return m.apply(f)


def ece(p: np.ndarray, y: np.ndarray, bins: int | None = None) -> float:
    """Expected calibration error: occupancy-weighted |mean p − observed rate|."""
    bins = bins or config.ECE_BINS
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        e += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(e)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _fit_platt_band(f: np.ndarray, y: np.ndarray, band: str, *,
                    subtract_prior: bool) -> BandMap | None:
    """Per-band Platt scaling on TRAIN logits, optionally prior-corrected."""
    f = np.asarray(f, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if min(int(y.sum()), int((y == 0).sum())) < config.MIN_PER_CLASS:
        return None

    pl = LogisticRegression(**config.platt_kwargs()).fit(f, y)
    a = float(pl.coef_[0, 0])
    b = float(pl.intercept_[0])

    pi = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    logit_pi = float(np.log(pi / (1.0 - pi)))
    b_out = b - logit_pi if subtract_prior else b

    return BandMap(band=band, a=a, b=b_out, pi=pi, logit_pi=logit_pi,
                   ece_raw=ece(_sigmoid(f.ravel()), y),
                   ece_pooled=ece(_sigmoid(a * f.ravel() + b), y),
                   n_train=int(len(y)))


def fit(method: str, train_logits: dict[str, np.ndarray],
        train_labels: dict[str, np.ndarray] | None = None) -> Pooler:
    """Fit the pooling step from TRAIN rows only.

    `train_logits[band]` is that band's expert's raw decision function ON THE
    ROWS IT WAS FIT ON -- not on all train rows. The two poolers that read labels
    (`platt`, `platt_prior`) need `train_labels[band]` aligned to those same rows,
    so the prior is the prior of the population the expert actually learned.

    Train-only provenance for both Platt parameters *and* the prior is gate 4 of
    Lever A. Fitting either on test rows would make the calibration a
    self-fulfilling one and the prior a leak.
    """
    if method not in METHODS:
        raise PoolingError(f"unknown pooling method {method!r}; "
                           f"known: {', '.join(METHODS)}")
    if method == "proba":
        return Pooler(method=method, maps={})

    if method == "z":
        maps = {}
        for band, f in train_logits.items():
            f = np.asarray(f, dtype=float)
            if not len(f):
                continue
            mu, sd = float(f.mean()), float(f.std())
            sd = sd if sd > 1e-12 else 1.0
            maps[band] = BandMap(band=band, a=1.0 / sd, b=-mu / sd,
                                 n_train=int(len(f)))
        return Pooler(method=method, maps=maps)

    if train_labels is None:
        raise PoolingError(
            f"{method}: needs train labels -- it is fit against the true label, "
            f"not merely rescaled. That is the whole difference from `z`.")
    subtract = method == "platt_prior"
    maps = {}
    for band, f in train_logits.items():
        y = train_labels.get(band)
        if y is None or not len(np.asarray(f)):
            continue
        m = _fit_platt_band(f, y, band, subtract_prior=subtract)
        if m is not None:
            maps[band] = m
    return Pooler(method=method, maps=maps)


def pooled_scores(pooler: Pooler, logits_by_band: dict[str, np.ndarray],
                  assigned: np.ndarray) -> np.ndarray:
    """Gather one score per row from the expert its router chose.

    `logits_by_band[b]` holds band `b`'s expert's score for EVERY test row --
    both experts score everything, because a predicted band can be wrong and a
    misrouted row still needs a number. `assigned` then selects, per row, which
    of those columns survives.
    """
    assigned = np.asarray(assigned)
    n = len(assigned)
    out = np.full(n, np.nan, dtype=float)
    for band, f in logits_by_band.items():
        f = np.asarray(f, dtype=float)
        if len(f) != n:
            raise PoolingError(
                f"band {band!r} scored {len(f)} rows but the router assigned "
                f"{n}; every expert must score every test row")
        m = assigned == band
        if m.any():
            out[m] = pooler.apply(band, f[m])
    return out
