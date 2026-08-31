"""Named cohorts — a *report-time* grouping, deliberately kept out of the fitting path.

csx_probe never reads this module: it emits atomic per-pair rows and no `cohort`
column at all. That is what lets any grouping be medianed after the fact --
LLM vs VLM, per-dataset, per-model-size, leave-one-out -- without refitting
anything. csx_report is the only consumer.

Lives in csx_common rather than in csx_report so that a cohort definition has one
home, and so the isolation tests can keep csx_report from importing csx_probe.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import yaml

from . import paths, registry


@dataclass(frozen=True)
class Cohort:
    key: str
    description: str
    pairs: tuple[str, ...]
    frozen: bool
    pass_rule: dict

    def min_passing(self, n: int) -> int:
        """Pass-rule threshold for a group of `n` pairs.

        `qa8` uses the pre-registered literal 6-of-8, kept as a literal rather
        than as a fraction that happens to equal 6 -- so re-deriving it can never
        drift, and applying it to a different n is an error rather than a quiet
        re-interpretation of a pre-registered bar. Every other grouping uses
        ceil(0.75 * n), which coincides at n == 8.
        """
        rule = self.pass_rule
        if rule["mode"] == "literal":
            if n != rule["n_pairs"]:
                raise ValueError(
                    f"cohort {self.key!r} pins the pre-registered literal rule "
                    f"{rule['min_passing']}/{rule['n_pairs']} but was given n={n}; "
                    f"a literal bar must not be re-scaled"
                )
            return int(rule["min_passing"])
        if rule["mode"] == "fraction":
            return math.ceil(rule["fraction"] * n)
        raise ValueError(f"unknown pass_rule mode {rule['mode']!r}")

    def describe_rule(self, n: int) -> str:
        """Human-readable bar, printed beside every verdict table so the
        threshold is never inferred from the row count after the fact."""
        return f"{self.min_passing(n)}/{n}"


@functools.lru_cache(maxsize=1)
def _raw() -> dict:
    with (paths.CONFIG_DIR / "cohorts.yaml").open() as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def all_cohorts() -> dict[str, Cohort]:
    return {
        k: Cohort(
            key=k,
            description=(v.get("description") or "").strip(),
            pairs=tuple(v["pairs"]),
            frozen=bool(v.get("frozen", False)),
            pass_rule=v["pass_rule"],
        )
        for k, v in _raw()["cohorts"].items()
    }


def get(key: str) -> Cohort:
    try:
        return all_cohorts()[key]
    except KeyError:
        raise KeyError(
            f"unknown cohort {key!r}; known: {', '.join(sorted(all_cohorts()))}. "
            f"Cohorts are conveniences -- an explicit --pairs list works too."
        ) from None


def resolve(cohort: str | None = None, pairs: str | list[str] | None = None,
            *, available: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Resolve a grouping selector to (present, dropped) pair keys.

    Exactly one of `cohort` / `pairs` is expected. `available` is the set of pair
    keys that actually have results; anything named but absent is returned in
    `dropped` so the caller can report it. A shrinking denominator changes every
    median, so it must be stated rather than silently absorbed -- which is what
    `on_missing: report_and_drop` in cohorts.yaml means.
    """
    if (cohort is None) == (pairs is None):
        raise ValueError("pass exactly one of cohort= or pairs=")
    if cohort is not None:
        wanted = list(get(cohort).pairs)
    else:
        if isinstance(pairs, str):
            pairs = [p.strip() for p in pairs.split(",") if p.strip()]
        wanted = list(pairs)

    known = set(registry.pairs())
    unknown = [p for p in wanted if p not in known]
    if unknown:
        raise KeyError(f"unknown pair(s): {unknown}")

    if available is None:
        return wanted, []
    present = [p for p in wanted if p in available]
    dropped = [p for p in wanted if p not in available]
    return present, dropped
