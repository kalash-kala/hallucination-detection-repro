"""Build the sampled-generation manifest for one pair: which (id, text) forward
passes are actually needed to compute the 10-sample router / SR-xAUC features.

Every row's `n_generations` column holds 10 temperature-1 sampled answer
strings. Extracting one teacher-forced pass per slot would mean 10x the greedy
GPU cost; VQA answers repeat heavily within a row (measured 35-80% dedup across
the 9 VLM pairs, see docs/PART_C_ROUTING_PLAN.md), so this dedups WITHIN each
row -- two identical samples for the same question need one forward pass, not
two -- and writes:

  unique.parquet    one row per (id, text) actually needing extraction:
                     id, urow (0..n_unique-1), text
  manifest.parquet   the slot map: id, slot (0..9) -> urow, so aggregation
                     (mean/std/cloud over the 10 samples) can gather features
                     back into original sample order

Deliberately NOT deduped ACROSS different ids: the same answer string means a
different thing for a different question, and the teacher-forced pass conditions
on the row's own prompt (question + image), so it is a different forward pass.

The greedy answer is NOT part of this manifest -- its features already exist in
the raw store (`csx_store/raw/<pair>/hs|diag`), so `greedy_mean_std` reads it
from there at aggregation time rather than re-extracting it here.

CPU-only, no torch. Runs in seconds; validates the join before any GPU time is
spent on the actual extraction (`sampled_extract.py`).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from csx_common import paths, registry


class ManifestError(Exception):
    """The pair's sampled-generation inputs are inconsistent."""


@dataclass(frozen=True)
class BuiltManifest:
    unique: pd.DataFrame     # id, urow, text
    slots: pd.DataFrame      # id, slot, urow
    n_rows: int
    n_slots: int
    n_unique: int
    n_short: int              # slots whose text tokenises shorter than greedy
                               # (informational; the sequence-length guard is
                               # applied at extraction time, not here)

    @property
    def dedup_frac(self) -> float:
        return 1.0 - (self.n_unique / self.n_slots if self.n_slots else 1.0)


def _parse_generations(raw) -> list[str]:
    """The CSV's `n_generations` cell -> a list of exactly 10 strings.

    Stored as a Python-list literal by the upstream run, so this must
    round-trip through `ast.literal_eval` rather than `json.loads` -- the
    strings inside can themselves contain characters (quotes, unescaped
    backslashes) that are valid Python-repr but not valid JSON.
    """
    if isinstance(raw, list):
        vals = raw
    else:
        vals = ast.literal_eval(raw)
    return [str(v).strip() for v in vals]


def build(pair_key: str) -> BuiltManifest:
    pair = registry.get(pair_key)

    rows_path = paths.raw_rows(pair_key)
    if not rows_path.exists():
        raise ManifestError(
            f"{pair_key}: no {rows_path} -- run the greedy extraction "
            f"(cli_extract/20-23) first. The sampled manifest is built against "
            f"the SAME roster (post-subsample ids), not the raw CSV pool.")
    roster = pd.read_parquet(rows_path)
    ids = set(roster["id"].astype(str))

    csv_path = pair.csv_path
    if not csv_path.exists():
        raise ManifestError(f"{pair_key}: missing source CSV {csv_path}")
    df = pd.read_csv(csv_path)
    if "n_generations" not in df.columns:
        raise ManifestError(
            f"{pair_key}: {csv_path} has no `n_generations` column -- the "
            f"sampled tier needs the 10 sampled-generation strings, not just "
            f"the greedy answer.")
    df["id"] = df["id"].astype(str)
    df = df[df["id"].isin(ids)]
    missing = ids - set(df["id"])
    if missing:
        raise ManifestError(
            f"{pair_key}: {len(missing)} of {len(ids)} roster ids are absent "
            f"from {csv_path}, e.g. {sorted(missing)[:3]} -- the CSV must be "
            f"the same one the roster was built from.")

    unique_rows: list[tuple[str, int, str]] = []
    slot_rows: list[tuple[str, int, int]] = []
    n_slots = 0

    for sid, gens_raw in zip(df["id"], df["n_generations"]):
        try:
            texts = _parse_generations(gens_raw)
        except (ValueError, SyntaxError) as exc:
            raise ManifestError(
                f"{pair_key}: id {sid!r} has unparsable n_generations: {exc}")
        if len(texts) != 10:
            raise ManifestError(
                f"{pair_key}: id {sid!r} has {len(texts)} generations, "
                f"expected 10")
        n_slots += len(texts)

        # Dedup WITHIN this id only: first-seen order, so `urow` assignment is
        # deterministic and independent of run-to-run set iteration order.
        seen: dict[str, int] = {}
        for slot, text in enumerate(texts):
            if text not in seen:
                seen[text] = len(unique_rows)
                unique_rows.append((sid, seen[text], text))
            slot_rows.append((sid, slot, seen[text]))

    unique = pd.DataFrame(unique_rows, columns=["id", "urow", "text"])
    slots = pd.DataFrame(slot_rows, columns=["id", "slot", "urow"])
    # `urow` in `unique` is currently local to each id (0-based per id); make it
    # a single global row index matching `unique`'s own row order, and remap
    # `slots` to match -- this is what lets extraction address unique.parquet
    # by a flat integer instead of an (id, local_urow) pair.
    unique = unique.reset_index(drop=True)
    unique["urow"] = np.arange(len(unique), dtype="int64")
    global_of = {(sid, local): g for g, (sid, local, _)
                 in enumerate(unique_rows)}
    slots["urow"] = [global_of[(sid, u)] for sid, u in
                      zip(slots["id"], slots["urow"])]

    n_short = 0  # populated by extraction, not the manifest; kept at 0 here
    return BuiltManifest(unique=unique, slots=slots, n_rows=len(df),
                         n_slots=n_slots, n_unique=len(unique),
                         n_short=n_short)


def write(pair_key: str, built: BuiltManifest) -> None:
    d = paths.sampled_dir(pair_key)
    d.mkdir(parents=True, exist_ok=True)
    built.unique.to_parquet(paths.sampled_unique(pair_key), index=False)
    built.slots.to_parquet(paths.sampled_manifest(pair_key), index=False)
    meta = {
        "pair": pair_key,
        "n_rows": built.n_rows,
        "n_slots": built.n_slots,
        "n_unique": built.n_unique,
        "dedup_frac": round(built.dedup_frac, 4),
        "extraction": {"done": False},
    }
    paths.sampled_meta(pair_key).write_text(json.dumps(meta, indent=2))
