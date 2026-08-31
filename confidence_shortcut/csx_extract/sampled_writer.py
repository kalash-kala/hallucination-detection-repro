"""Atomic writers for the sampled store, mirroring `writer.py`'s pattern but
pointed at `sampled/<pair>/` instead of `raw/<pair>/`.

A separate module rather than extending `writer.py`: the sampled tree has no
`rows.parquet`/phase1/phase2 lifecycle (that is `sampled_manifest.py`'s job for
the input side, `sampled_extract.py`'s for the output side), so reusing the raw
entry's `EntryMeta` shape would carry fields that do not apply here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from csx_common import paths


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _savez(tmp, arrays: dict[str, np.ndarray]) -> None:
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)


def save_hs(pair: str, segment: str, arrays: dict[str, np.ndarray]) -> None:
    p = paths.sampled_dir(pair) / "hs" / f"{segment}.npz"
    _atomic(p, lambda t: _savez(t, arrays))


def save_diag(pair: str, segment: str, arrays: dict[str, np.ndarray]) -> None:
    p = paths.sampled_dir(pair) / "diag" / f"{segment}.npz"
    _atomic(p, lambda t: _savez(t, arrays))


def save_layer_stats(pair: str, stats: dict[str, np.ndarray]) -> None:
    """Per-bucket-layer (mu, sigma) recomputed over greedy-train rows -- kept
    for audit, not read back by anything downstream."""
    p = paths.sampled_dir(pair) / "layer_stats.npz"
    _atomic(p, lambda t: _savez(t, stats))


def load_meta(pair: str) -> dict:
    return json.loads(paths.sampled_meta(pair).read_text())


def update_meta(pair: str, **fields) -> dict:
    path = paths.sampled_meta(pair)
    meta = json.loads(path.read_text())
    meta.update(fields)
    _atomic(path, lambda p: p.write_text(json.dumps(meta, indent=2)))
    return meta


def finish_extraction(pair: str, **info) -> None:
    path = paths.sampled_meta(pair)
    meta = json.loads(path.read_text())
    meta["extraction"] = {"done": True, **info}
    _atomic(path, lambda p: p.write_text(json.dumps(meta, indent=2)))
