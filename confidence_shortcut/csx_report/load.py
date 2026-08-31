"""Reading the results contract, and refusing to guess.

`csx_report` imports neither of the other packages. It reads these parquet tables
and the schema version beside them, and that is the entire coupling.

**An unknown schema version is a hard stop.** A newer `csx_probe` may have changed
what a column *means* -- `AUROC` computed under a different orientation, say --
and a best-effort read would produce a report that is confidently wrong rather
than one that failed. Adding a column is not a version bump; changing what one
means is.
"""

from __future__ import annotations

import json

import pandas as pd

from csx_common import paths

SCHEMA_VERSION = 1

REQUIRED = {
    "per_pair_long": ("pair", "family", "segment", "C", "c_mode",
                      "prompt_template", "train_arm", "test_arm", "head",
                      "contrast", "AUROC"),
    "verdict": ("pair", "family", "metric", "delta", "null_p95", "passes"),
    "rotation_long": ("pair", "family", "kind", "alpha", "draw"),
    "arm_stats": ("pair", "arm", "split", "n"),
    "c_selection": ("pair", "family", "train_arm", "best_C"),
    # The confound grids carry `target` and `draw` on top of the transfer-grid
    # schema: `target` names the real arm a control imitates, `draw` which of the
    # 20 replicate draws it is. Both are needed to median a control down to one
    # comparable number.
    "placebo_long": ("pair", "family", "segment", "target", "draw",
                     "train_arm", "test_arm", "head", "contrast", "AUROC"),
    "sizeonly_long": ("pair", "family", "segment", "target", "draw",
                      "train_arm", "test_arm", "head", "contrast", "AUROC"),
}


class ContractError(Exception):
    """The results tables cannot be read as this version of the contract."""


def check_version() -> int:
    p = paths.results_root() / "_meta.json"
    if not p.exists():
        raise ContractError(
            f"no {p}; run a csx_probe stage to completion before reporting")
    got = int(json.loads(p.read_text())["schema_version"])
    if got != SCHEMA_VERSION:
        raise ContractError(
            f"results are schema v{got}, this reporter speaks v{SCHEMA_VERSION}; "
            f"upgrade csx_report rather than reading them optimistically")
    return got


def _stale_units(name: str) -> list[str]:
    """Per-pair unit files written after the consolidated snapshot.

    The consolidated table is a cache, and it is only refreshed when a stage
    reaches the end of its batch. A stage that is killed part-way -- or one still
    running -- leaves finished units on disk that the snapshot does not contain,
    so reporting from the snapshot silently omits completed work. Cheaper to
    notice than to debug from a report whose pair count is quietly wrong.
    """
    snap = paths.results_table(name)
    unit_dir = paths.results_root() / "units" / name
    if not unit_dir.is_dir():
        return []
    cut = snap.stat().st_mtime if snap.exists() else 0.0
    return sorted(u.stem for u in unit_dir.glob("*.parquet")
                  if u.stat().st_mtime > cut)


def table(name: str, *, check: bool = True) -> pd.DataFrame:
    """One results table, validated against the columns the contract promises.

    Reads per-pair unit files directly whenever any is newer than the
    consolidated snapshot, so a report never silently reflects a stale cache.
    """
    if check:
        check_version()
    p = paths.results_table(name)
    unit_dir = paths.results_root() / "units" / name
    stale = _stale_units(name)

    if stale:
        parts = sorted(unit_dir.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(q) for q in parts], ignore_index=True)
    elif p.exists():
        df = pd.read_parquet(p)
    else:
        raise ContractError(
            f"no {name} table at {p}; the csx_probe stage that writes it has "
            f"not been consolidated")

    missing = [c for c in REQUIRED.get(name, ()) if c not in df.columns]
    if missing:
        raise ContractError(f"{name} is missing contract columns {missing}")
    if "cohort" in {c.lower() for c in df.columns}:
        raise ContractError(
            f"{name} carries a `cohort` column; atomic tables must not, or "
            f"re-grouping stops being free")
    return df


def available(name: str) -> bool:
    """Whether `table(name)` would find anything -- snapshot or loose units."""
    if paths.results_table(name).exists():
        return True
    unit_dir = paths.results_root() / "units" / name
    return unit_dir.is_dir() and any(unit_dir.glob("*.parquet"))
