"""Build `rows.parquet` for one pair: which rows get extracted, and what each needs.

This is the join between the DSE run CSV (labels, bands, entropy) and the
generations folder (image_path, prompt_token_ids, the greedy answer). It runs on
CPU and touches no model, so a bad roster entry is caught in seconds rather than
after a GPU has warmed up.

Three things are gated here rather than downstream, because each one is silent if
it is wrong:

  * every extracted id must exist in BOTH sources -- a partial generations folder
    would otherwise yield a short, quietly biased pair;
  * the question string must match between CSV and generations for every row --
    that is what proves the two files describe the same example, and not two runs
    that merely share an id space;
  * every image file must exist on disk AFTER the declared path rewrite, so a
    remap that silently no-ops fails now instead of at row 12,000 of a GPU pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from csx_common import paths, registry
from csx_common.store_schema import ROW_COLUMNS

from . import subsample


class RowsError(Exception):
    """The pair's inputs are inconsistent. The message names which check failed."""


@dataclass(frozen=True)
class BuiltRows:
    rows: pd.DataFrame
    subsample_meta: dict
    n_pool: int

    @property
    def n(self) -> int:
        return len(self.rows)


def read_generations(pair: registry.Pair) -> dict[str, dict]:
    """id -> record, from combined_generations.jsonl.

    Each line is a single-key object `{id: {...}}`. Only the fields extraction
    needs are kept; the sampled `responses` block is by far the largest field and
    is dropped, which is what keeps a 546 MB file to a few hundred MB of RAM.
    """
    path = pair.generations_path
    if path is None:
        raise RowsError(f"{pair.key}: no generations folder declared in pairs.yaml")
    if not path.exists():
        raise RowsError(f"{pair.key}: missing {path}")

    keep = ("question", "image_path", "prompt_token_ids", "most_likely_answer",
            "source_split", "source_index")
    out: dict[str, dict] = {}
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RowsError(f"{pair.key}: {path}:{lineno} is not valid JSON: {exc}")
            if len(rec) != 1:
                raise RowsError(
                    f"{pair.key}: {path}:{lineno} has {len(rec)} keys, expected "
                    f"exactly one (the row id)")
            sid, body = next(iter(rec.items()))
            if sid in out:
                raise RowsError(f"{pair.key}: duplicate id {sid!r} in {path}")
            out[str(sid)] = {k: body.get(k) for k in keep}
    return out


def _greedy(rec: dict) -> str:
    mla = rec.get("most_likely_answer") or {}
    return str(mla.get("response", "") or "")


def build(pair_key: str, *, n_target: int | None = None,
          seed: int = 42) -> BuiltRows:
    """Join L2 + generations, subsample, and return the row table for one pair.

    The returned frame carries every column in ROW_COLUMNS except the ones only a
    forward pass can fill (`seq_len`, `answer_start`, `answer_end`, `s_ext`,
    and the image span); those are written by phase 1.
    """
    pair = registry.get(pair_key)

    l2 = _read_l2(pair)
    n_pool = len(l2)

    if pair.needs_generations():
        gens = read_generations(pair)
        _gate_coverage(pair, l2, gens)
        _gate_questions(pair, l2, gens)
    else:
        gens = {}

    sub = subsample.choose(l2, pair_key, n_target=n_target, seed=seed)
    kept = l2[l2["id"].isin(set(sub.ids))].reset_index(drop=True)
    # Preserve pool order, then number rows 0..n-1: `row` is the index into every
    # array phase 1 and phase 2 write, so it must be assigned once, here.
    kept = kept.sort_values("_pool_order").reset_index(drop=True)

    out = pd.DataFrame({
        "id": kept["id"].astype(str),
        "row": np.arange(len(kept), dtype="int32"),
        "category": kept["category"].astype(str),
        "entropy": kept["entropy"].astype("float64"),
        # Filled by phase 1; -1/NaN means "not extracted yet", and verify.py
        # refuses an entry whose phase1 is marked done while these are unset.
        "s_ext": np.full(len(kept), np.nan, dtype="float32"),
        "seq_len": np.full(len(kept), -1, dtype="int32"),
        "answer_start": np.full(len(kept), -1, dtype="int32"),
        "answer_end": np.full(len(kept), -1, dtype="int32"),
        "image_start": np.full(len(kept), -1, dtype="int32"),
        "image_end": np.full(len(kept), -1, dtype="int32"),
    })

    if pair.needs_generations():
        raw = [gens[i]["image_path"] for i in out["id"]]
        out["image_path"] = [pair.rewrite_path(p) for p in raw]
        _gate_images(pair, out["image_path"])
        # The generation run's own prompt, carried per row so phase 1 can gate
        # its reconstruction token-for-token rather than trusting a template.
        # It is the only ground truth for what the model actually saw.
        out["prompt_token_ids"] = [
            np.asarray(gens[i]["prompt_token_ids"], dtype="int32")
            for i in out["id"]
        ]
    else:
        out["image_path"] = ""

    out["question"] = [
        gens[i]["question"] if gens else q
        for i, q in zip(out["id"], kept["question"].astype(str))
    ]
    # The teacher-forced answer is the greedy generation. For VLM pairs it is
    # taken from the generations record, which is the same text the CSV's
    # `greedy` column holds but is the file the prompt_token_ids came from, so
    # the two halves of the sequence provably originate together.
    out["answer"] = [
        _greedy(gens[i]) if gens else a
        for i, a in zip(out["id"], kept["greedy"].astype(str))
    ]

    missing = [c for c in ROW_COLUMNS if c not in out.columns]
    if missing:
        raise RowsError(f"{pair.key}: built rows are missing {missing}")
    return BuiltRows(rows=out, subsample_meta=sub.to_meta(), n_pool=n_pool)


# ── inputs and gates ─────────────────────────────────────────────────────────

def _read_l2(pair: registry.Pair) -> pd.DataFrame:
    """That pair's slice of the L2 table, in stable pool order.

    L2 is deliberately the ONLY source of labels here. Parsing the run CSV
    directly would mean csx_extract carrying its own copy of the band-orientation
    and correctness gates -- two implementations that can disagree about which
    rows are `IH`. Stage 00 is cheap and GPU-free, so requiring it first costs
    nothing and keeps one definition of the labels.
    """
    table = paths.uq_table()
    if not table.exists():
        raise RowsError(
            f"{pair.key}: no L2 table at {table}. Run "
            f"`cli_probe/00_ingest_uq.py --run` first -- it is CPU-only and "
            f"takes a couple of minutes for every pair at once.")
    df = pd.read_parquet(table)
    df = df[df["pair"] == pair.key].reset_index(drop=True)
    if not len(df):
        raise RowsError(
            f"{pair.key}: L2 exists but holds no rows for this pair; re-run "
            f"stage 00 after adding it to pairs.yaml")
    df = df.copy()
    df["_pool_order"] = np.arange(len(df))
    return df


def _gate_coverage(pair: registry.Pair, l2: pd.DataFrame, gens: dict) -> None:
    ids = set(l2["id"].astype(str))
    missing = ids - set(gens)
    if missing:
        ex = sorted(missing)[:3]
        raise RowsError(
            f"{pair.key}: {len(missing)} of {len(ids)} CSV ids are absent from "
            f"the generations folder, e.g. {ex}. Extracting the intersection "
            f"would silently drop rows and bias the band proportions; fix the "
            f"`generations` entry in pairs.yaml instead.")


def _gate_questions(pair: registry.Pair, l2: pd.DataFrame, gens: dict) -> None:
    """The two files must describe the same examples, not merely share ids.

    Whitespace is normalised because the CSV round-trips through pandas; anything
    beyond that is a genuine mismatch.
    """
    bad = []
    for sid, q in zip(l2["id"].astype(str), l2["question"].astype(str)):
        gq = str(gens[sid].get("question") or "")
        if " ".join(q.split()) != " ".join(gq.split()):
            bad.append((sid, q, gq))
            if len(bad) >= 3:
                break
    if bad:
        lines = "\n".join(f"    {i}: csv={c!r} gen={g!r}" for i, c, g in bad)
        raise RowsError(
            f"{pair.key}: question text disagrees between the run CSV and the "
            f"generations folder:\n{lines}\n  These are different runs, or the "
            f"wrong folder is declared in pairs.yaml.")


def _gate_images(pair: registry.Pair, image_paths) -> None:
    missing = [p for p in image_paths if not Path(p).exists()]
    if missing:
        hint = ""
        if pair.path_rewrite:
            src, dst = pair.path_rewrite
            still = [p for p in missing if p.startswith(src)]
            hint = (f" The declared rewrite {src!r} -> {dst!r} did not apply to "
                    f"{len(still)} of them."
                    if still else
                    f" The rewrite {src!r} -> {dst!r} applied, but the target "
                    f"files are absent.")
        raise RowsError(
            f"{pair.key}: {len(missing)} image file(s) do not exist, "
            f"e.g. {missing[:3]}.{hint}")
