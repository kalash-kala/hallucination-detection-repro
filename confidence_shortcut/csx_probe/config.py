"""Run configuration — the frozen constants, and the `C` policy.

`C` is threaded as a plain field on a picklable dataclass, deliberately NOT
through environment variables. The published pipeline used env vars because
loky/ProcessPool workers re-import modules by reference, so a parent-side
monkeypatch never reached them. That worked, but it meant one env knob
(`RG_C_SPECTRAL`) was shared by four families whose selected `C` disagree, so any
7-family script had to be run once per distinct value with hand-managed output
tags. A dataclass that joblib pickles into the worker removes all of that: one
invocation covers every family at its own `C`.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from csx_common import paths, registry


@functools.lru_cache(maxsize=1)
def frozen() -> dict[str, Any]:
    with (paths.CONFIG_DIR / "frozen_constants.yaml").open() as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def c_policy() -> dict[str, Any]:
    with (paths.CONFIG_DIR / "c_policy.yaml").open() as fh:
        return yaml.safe_load(fh)


SEED: int = frozen()["seed"]
FAMILIES: tuple[str, ...] = tuple(frozen()["families"]["hs"] + frozen()["families"]["spectral"])
HS_FAMILIES: tuple[str, ...] = tuple(frozen()["families"]["hs"])
SPECTRAL_FAMILIES: tuple[str, ...] = tuple(frozen()["families"]["spectral"])
HEADS: tuple[str, ...] = tuple(frozen()["heads"])
ARM_ORDER: tuple[str, ...] = tuple(frozen()["arm_order"])

# ── feature derivation ───────────────────────────────────────────────────────
# The widths the spectral families are CV-selected over, and the PCA branch that
# search runs jointly with. `pca_grid` carries a null (StandardScaler, no PCA),
# which is why it is read as-is rather than coerced to ints.
TOP_K_GRID: tuple[int, ...] = tuple(frozen()["features"]["top_k_grid"])
PCA_GRID: tuple[int | None, ...] = tuple(frozen()["features"]["pca_grid"])
PCA_DIM: int = frozen()["features"]["pca_dim"]
SINK_K: int = frozen()["features"]["sink_k"]

# ── contrasts ────────────────────────────────────────────────────────────────
# Orientation is ALWAYS larger => more incorrect, with no per-head sign flips.
# CHvI/CLvI are encoded as (pos = I, neg = CH/CL): identically the 1-AUROC
# convention, with one fewer sign flip available to get wrong.
CATS: tuple[str, ...] = ("IH", "CH", "IL", "CL")
I_CATS: tuple[str, ...] = ("IH", "IL")
C_CATS: tuple[str, ...] = ("CH", "CL")
L_CATS: tuple[str, ...] = ("IL", "CL")      # L = high entropy = uncertain

CONTRASTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "IvC":   (I_CATS, C_CATS),
    "IHvC":  (("IH",), C_CATS),
    "ILvC":  (("IL",), C_CATS),
    "CHvI":  (I_CATS, ("CH",)),
    "CLvI":  (I_CATS, ("CL",)),
    "IHvCH": (("IH",), ("CH",)),
    "ILvCL": (("IL",), ("CL",)),
    "IHvCL": (("IH",), ("CL",)),
    "ILvCH": (("IL",), ("CH",)),
}
ADMISSIBLE: tuple[str, ...] = tuple(frozen()["contrasts"]["admissible"])
DEFINITIONAL: tuple[str, ...] = tuple(frozen()["contrasts"]["definitional"])
CONTRAST_ORDER: tuple[str, ...] = ("IvC", *ADMISSIBLE, *DEFINITIONAL)
SHORTCUT: str = frozen()["contrasts"]["shortcut_meter"]
MIN_PER_CLASS: int = frozen()["arms"]["min_per_class"]


# ── routing (Part C) ─────────────────────────────────────────────────────────
# HI = low entropy = confident. `BANDS[b]` is that band's (incorrect, correct)
# category pair, so a band expert's target is `cat == BANDS[b][0]`.
BANDS: dict[str, tuple[str, str]] = {
    b: tuple(v) for b, v in frozen()["routing"]["bands"].items()}
BAND_ORDER: tuple[str, ...] = tuple(BANDS)
C_POOL: tuple[str, ...] = tuple(frozen()["routing"]["c_pool"])
ROUTE_THRESHOLD: float = float(frozen()["routing"]["route_threshold"])
ROUTERS: tuple[str, ...] = tuple(frozen()["routing"]["routers"])
PRESENTED_ROUTERS: tuple[str, ...] = tuple(frozen()["routing"]["presented_routers"])
POOLERS: tuple[str, ...] = ("z", "platt", "platt_prior", "proba")
HIER_FOLDS: int = int(frozen()["routing"]["hier_folds"])
AFFINE_TOL: float = float(frozen()["routing"]["affine_tol"])
ECE_BINS: int = int(frozen()["routing"]["ece_bins"])

SRXAUC_ATOMIC: tuple[str, ...] = tuple(frozen()["srxauc"]["atomic"])
SRXAUC_TEST_ARMS: tuple[str, ...] = tuple(frozen()["srxauc"]["test_arms"])
SRXAUC_UNRANKED: tuple[str, ...] = tuple(frozen()["srxauc"]["unranked"])


def band_of(categories) -> np.ndarray:
    """Category -> band label, by the frozen (incorrect, correct) mapping.

    Never inferred from the entropy column: the band IS the entropy cut, and
    re-deriving it from a threshold here would let a rounding difference put a
    row in a different band from the one its category records.
    """
    cats = np.asarray(categories)
    out = np.empty(len(cats), dtype=object)
    for band, members in BANDS.items():
        out[np.isin(cats, list(members))] = band
    return out


def platt_kwargs() -> dict[str, Any]:
    spec = frozen()["routing"]["platt"]
    return {"C": float(spec["C"]), "class_weight": spec["class_weight"],
            "solver": spec["solver"], "max_iter": int(spec["max_iter"])}


def kind_of(family: str) -> str:
    """'hs' or 'spectral' -- picks the LR hyperparameters and the transform."""
    if family in HS_FAMILIES:
        return "hs"
    if family in SPECTRAL_FAMILIES:
        return "spectral"
    raise ValueError(f"unknown family {family!r}; known: {', '.join(FAMILIES)}")


# ── C resolution ─────────────────────────────────────────────────────────────
def c_grid() -> list[float]:
    return [float(c) for c in c_policy()["c_grid"]]


def snap_to_grid(c: float) -> float:
    """Snap a value to the nearest candidate on the log scale, ties to the lower
    (more regularised) one.

    Needed because a median over an even number of units lands *between* grid
    points whenever the middle two differ -- real observed values include
    0.00055 and 0.065. Without snapping, a pair would be fit at a `C` that no CV
    curve ever evaluated.
    """
    grid = np.asarray(c_grid(), dtype=float)
    d = np.abs(np.log10(grid) - math.log10(c))
    # argmin returns the FIRST minimum, and the grid is ascending, so an exact
    # tie resolves to the smaller C by construction.
    return float(grid[int(np.argmin(d))])


def resolve_c(family: str, pair: str,
              per_unit_best: dict[str, list[float]] | None = None) -> float:
    """The `C` this (family, pair) is fitted at.

    Modes come from configs/c_policy.yaml:
      pinned    the frozen per-family table, verbatim. Used by the 8 qa8 pairs,
                because the published numbers were fit at a single per-family C
                and the 1e-4 parity gate depends on reproducing exactly that.
      per_pair  median of that pair's own arm-units, snapped to the grid. The
                default for every new pair: it makes C atomic per pair, so
                adding a model or dataset never invalidates anything already
                computed.

    `per_unit_best` maps family -> the per-unit argmax C values for this pair,
    as produced by experiments.c_selection. Required for per_pair mode.
    """
    pol = c_policy()
    mode = pol.get("assignment", {}).get(pair, pol["default"])
    if mode == "pinned":
        return float(pol["modes"]["pinned"][family])
    if mode == "per_pair":
        if not per_unit_best or family not in per_unit_best:
            raise ValueError(
                f"{pair}/{family}: per_pair mode needs CV results; run "
                f"c_selection for this pair first (cli_probe/03_select_c.py)"
            )
        vals = per_unit_best[family]
        if not vals:
            raise ValueError(f"{pair}/{family}: no per-unit C values")
        return snap_to_grid(float(np.median(vals)))
    raise ValueError(f"unknown C mode {mode!r} for {pair!r}")


@dataclass(frozen=True)
class RunConfig:
    """Everything a worker needs, picklable so joblib can ship it into loky.

    `c_by_family` is resolved once in the parent and carried along, which is what
    lets a single invocation fit all seven families at their own `C`.
    """

    pair: str
    c_by_family: dict[str, float]
    families: tuple[str, ...] = FAMILIES
    segments: tuple[str, ...] = ("all",)
    # Recorded on every emitted row: a group whose pairs were fit under different
    # policies is a real caveat, and component 3 can only report it if the mode
    # travels with the number.
    c_mode: str = "per_pair"
    seed: int = SEED
    n_boot: int = field(default_factory=lambda: frozen()["bootstrap"]["row_level_n"])
    min_per_class: int = field(default_factory=lambda: frozen()["arms"]["min_per_class"])

    def c(self, family: str) -> float:
        return self.c_by_family[family]

    @classmethod
    def for_pair(cls, pair: str, *,
                 per_unit_best: dict[str, list[float]] | None = None,
                 families: tuple[str, ...] = FAMILIES,
                 **kw) -> "RunConfig":
        p = registry.get(pair)
        pol = c_policy()
        return cls(
            pair=pair,
            c_by_family={f: resolve_c(f, pair, per_unit_best) for f in families},
            families=families,
            segments=p.segments,
            c_mode=pol.get("assignment", {}).get(pair, pol["default"]),
            **kw,
        )


def lr_kwargs(family: str, c: float) -> dict[str, Any]:
    """LogisticRegression kwargs for a family, reproducing stage_a_common exactly.

    The hs and spectral variants genuinely differ (max_iter 1000 vs 2000, and
    only spectral sets random_state). Reproduced as-is rather than harmonised --
    harmonising would move the published numbers.
    """
    spec = frozen()["probes"][kind_of(family)]
    kw = {
        "C": float(c),
        "max_iter": spec["max_iter"],
        "solver": spec["solver"],
        "class_weight": spec["class_weight"],
    }
    if spec["random_state"] is not None:
        kw["random_state"] = spec["random_state"]
    return kw
