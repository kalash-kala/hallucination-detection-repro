"""Writing L0 entries, and resuming interrupted ones.

Two properties matter more than anything else here:

**A phase is marked done only after its artifacts are on disk.** `meta.json`
starts with both phases false and is flipped by `finish_phase` as the last act of
each pass, so a killed job leaves an entry that reads "not done" rather than one
that looks complete and is short. `csx-store verify` then reports it honestly.

**Writes are atomic.** Every file lands at a temporary path and is renamed, so a
crash mid-write cannot leave a truncated npz that would load with the wrong row
count and silently misalign every downstream feature.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from csx_common import paths
from csx_common.store_schema import EntryMeta, new_meta


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def init_entry(*, pair: str, model: dict, dataset: str, modality: str,
               prompt_template: str, segments: list[str], rows: pd.DataFrame,
               n_pool: int, subsample: dict | None, top_k: int, sink_k: int,
               extractor_version: str) -> None:
    """Create (or reset) an entry's meta.json and rows.parquet."""
    meta = new_meta(
        pair=pair, model=model, dataset=dataset, modality=modality,
        prompt_template=prompt_template, segments=segments, n_rows=len(rows),
        n_pool=n_pool, subsample=subsample, top_k=top_k, sink_k=sink_k,
        extractor_version=extractor_version,
    )
    write_rows(pair, rows)
    _atomic(paths.raw_meta(pair), lambda p: p.write_text(json.dumps(meta, indent=2)))


def write_rows(pair: str, rows: pd.DataFrame) -> None:
    """Persist the row table. Columns the phases fill in are updated in place by
    re-writing the whole (small) table -- it is a few MB at most."""
    _atomic(paths.raw_rows(pair), lambda p: rows.to_parquet(p, index=False))


def read_rows(pair: str) -> pd.DataFrame:
    return pd.read_parquet(paths.raw_rows(pair))


def load_meta(pair: str) -> EntryMeta:
    return EntryMeta.load(paths.raw_meta(pair))


def update_meta(pair: str, **fields) -> dict:
    path = paths.raw_meta(pair)
    meta = json.loads(path.read_text())
    meta.update(fields)
    _atomic(path, lambda p: p.write_text(json.dumps(meta, indent=2)))
    return meta


def finish_phase(pair: str, phase: str, **info) -> None:
    """Mark a phase done. Called only after every artifact is on disk."""
    path = paths.raw_meta(pair)
    meta = json.loads(path.read_text())
    meta[phase] = {"done": True, **info}
    _atomic(path, lambda p: p.write_text(json.dumps(meta, indent=2)))


def _savez(tmp, arrays: dict[str, np.ndarray]) -> None:
    # Same trap as np.save: savez_compressed appends ".npz" to a path lacking it,
    # so the ".npz.tmp" temp path must be given as an open handle instead.
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)


def save_hs(pair: str, segment: str, arrays: dict[str, np.ndarray]) -> None:
    p = paths.raw_dir(pair) / "hs" / f"{segment}.npz"
    _atomic(p, lambda t: _savez(t, arrays))


def save_peaks(pair: str, peaks: dict) -> None:
    p = paths.raw_dir(pair) / "hs" / "peaks.json"
    _atomic(p, lambda t: t.write_text(json.dumps(peaks, indent=2)))


def save_diag(pair: str, segment: str, arrays: dict[str, np.ndarray]) -> None:
    p = paths.raw_dir(pair) / "diag" / f"{segment}.npz"
    _atomic(p, lambda t: _savez(t, arrays))


# ── the raw hidden-state shards (destructively reduced) ──────────────────────
# Phase 1 writes one shard per layer per segment, phase 2 never touches them, and
# reduce deletes them. They are the only large intermediate: [N,L,D] fp32 is
# ~33 GB for gemma-3-12b's three segments at 15,000 rows, which is why the
# reduce step is destructive at all -- as full_natural/02_reduce_hidden.py does
# it. fp32 rather than fp16 because gemma-3's activations overflow fp16; see the
# dtype note in phase1_hidden._pool.

def shard_path(pair: str, segment: str, layer: int) -> Path:
    return paths.raw_dir(pair) / "_shards" / segment / f"layer{layer}.npy"


def save_shard(pair: str, segment: str, layer: int, arr: np.ndarray) -> None:
    # Written through a file handle, not a path: np.save appends ".npy" to any
    # path that lacks it, so passing the ".npy.tmp" temp path would silently
    # produce "....npy.tmp.npy" and the atomic rename would then fail.
    def write(tmp):
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
    _atomic(shard_path(pair, segment, layer), write)


def load_shard(pair: str, segment: str, layer: int) -> np.ndarray:
    return np.load(shard_path(pair, segment, layer))


def drop_shards(pair: str) -> int:
    """Delete the raw per-layer shards. Returns bytes freed."""
    d = paths.raw_dir(pair) / "_shards"
    if not d.is_dir():
        return 0
    freed = sum(f.stat().st_size for f in d.rglob("*.npy"))
    for f in d.rglob("*.npy"):
        f.unlink()
    for sub in sorted(d.glob("*"), reverse=True):
        if sub.is_dir():
            sub.rmdir()
    d.rmdir()
    return freed
