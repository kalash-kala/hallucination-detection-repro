"""Writing the results contract: per-pair checkpoints, then a concatenation.

Two rules from `results_spec/RESULTS_CONTRACT.md`, both enforced here rather than
left to each experiment:

**No `cohort` column, ever.** Cohorts are chosen at report time; one baked into an
atomic table would defeat the entire point of the seam. `_reject_cohort` makes that
a hard failure rather than a convention.

**The top-level table is rebuilt from the checkpoints on disk, never from what
this process happens to hold in memory.** Rerunning one pair must not drop the
other twenty. Concatenating in-memory results would do exactly that, and the
result -- a table that is simply missing pairs -- looks completely normal.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from csx_common import paths

SCHEMA_VERSION = 1

PROVENANCE = ("pair", "family", "segment", "C", "c_mode", "prompt_template")


def _reject_cohort(df: pd.DataFrame, table: str) -> None:
    bad = [c for c in df.columns if c.lower() == "cohort"]
    if bad:
        raise ValueError(
            f"{table}: atomic tables must not carry {bad}; grouping is component "
            f"3's job and a baked-in cohort cannot be regrouped after the fact")


def write_unit(table: str, pair: str, df: pd.DataFrame) -> int:
    """Checkpoint one pair's rows for one table."""
    _reject_cohort(df, table)
    if "pair" not in df.columns:
        raise ValueError(f"{table}: every atomic row needs a `pair` column")
    stray = set(df["pair"].unique()) - {pair}
    if stray:
        raise ValueError(
            f"{table}/{pair}: checkpoint also holds rows for {sorted(stray)}; "
            f"per-pair files must stay per-pair or a rerun will not be able to "
            f"replace one pair without touching another")
    out = paths.results_units(table, pair)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    return len(df)


def read_unit(table: str, pair: str) -> pd.DataFrame | None:
    p = paths.results_units(table, pair)
    return pd.read_parquet(p) if p.exists() else None


def has_unit(table: str, pair: str) -> bool:
    return paths.results_units(table, pair).exists()


def consolidate(table: str) -> pd.DataFrame:
    """Rebuild the top-level table from every checkpoint on disk."""
    d = paths.results_units(table, "_").parent
    parts = sorted(d.glob("*.parquet")) if d.is_dir() else []
    if not parts:
        raise FileNotFoundError(
            f"{table}: no per-pair checkpoints under {d}; nothing to consolidate")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    _reject_cohort(df, table)

    out = paths.results_table(table)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    _stamp()
    return df


def _stamp() -> None:
    """Record the schema version component 3 will refuse to guess at."""
    p = paths.results_root() / "_meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
