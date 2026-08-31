"""L1 — turning an L0 entry into one family's feature matrix.

`feature_matrix` is the single entry point every experiment uses. It dispatches
to `derive/*`, so no caller has to know that `hs_wide` is a stored vector while
`lapeigvals` is a slice of a `[n,L,H,50]` array and `attnlogdet` is not a top-k
statistic at all.

**Materialising L1 to disk is opt-in, and off by default.** The store contract
describes `features/<pair>/<family>/<segment>/X.npy`, and `materialise` writes
exactly that — but it is a cache, not a required step, because for this study it
does not pay:

  * Deriving is a slice and a reshape off bytes that are already on disk. It
    costs one contiguous read, which is the same read materialising would do.
  * It does not reduce peak memory. The transform (PCA/StandardScaler) is fit on
    the full train matrix either way, and that fit is the high-water mark.
  * At `top_k=50` it is ~2.35 GB per (pair, family, segment); across the 9 VLM
    pairs, 3 segments and the two top-k families that is ~127 GB, against ~56 GB
    for all of L0.

So it is offered for the cases where it genuinely helps — repeatedly refitting
one pair while iterating, or a machine where the store is on slow storage — and
skipped otherwise. `manifest` does **not** require it: availability is a property
of L0, so the manifest is honest whether or not anything was cached.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from csx_common import paths, registry
from csx_common.store_schema import (
    MANIFEST_COLUMNS, MANIFEST_STATUSES, FeatureMeta,
)
from csx_probe import config
from csx_probe.store import read
from csx_probe.store.derive import hidden, sink as sink_mod, spectral

BUILDER_VERSION = "1.0.0"


def feature_matrix(entry: read.Entry, family: str, segment: str = "all",
                   *, top_k: int | None = None,
                   use_cache: bool = False) -> np.ndarray:
    """`[n, dim]` float32 for one (family, segment).

    `top_k` is required for `lapeigvals` / `attn_eigvals` and must be `None` for
    every other family -- passing one where it has no meaning is a mistake worth
    surfacing, not silently ignoring, because it usually means the caller thinks
    it is selecting a width that is in fact fixed.
    """
    if family not in config.FAMILIES:
        raise ValueError(
            f"unknown family {family!r}; known: {', '.join(config.FAMILIES)}")

    if use_cache:
        cached = load_l1(entry.pair, family, segment)
        if cached is not None:
            return cached

    if family in config.HS_FAMILIES:
        _no_top_k(family, top_k)
        return hidden.build(entry, family, segment)
    if family == "sink":
        _no_top_k(family, top_k)
        return sink_mod.build(entry, segment)
    if family == "attnlogdet":
        _no_top_k(family, top_k)
        return spectral.build(entry, family, segment)
    return spectral.build(entry, family, segment, top_k=top_k)


def _no_top_k(family: str, top_k: int | None) -> None:
    if top_k is not None:
        fixed = "SINK_K at extraction" if family == "sink" else "not a top-k family"
        raise ValueError(
            f"{family}: top_k={top_k} was passed, but this family has no width "
            f"to select ({fixed}). Selecting one here would be a no-op that "
            f"reads as a decision.")


def provenance(entry: read.Entry, family: str, segment: str,
               top_k: int | None) -> dict:
    if family in config.HS_FAMILIES:
        return hidden.provenance(entry, family, segment)
    if family == "sink":
        return sink_mod.provenance(entry, segment)
    return spectral.provenance(entry, family, segment, top_k=top_k)


# ── the optional on-disk cache ───────────────────────────────────────────────

def materialise(entry: read.Entry, family: str, segment: str = "all", *,
                top_k: int | None = None, pca_dim: int | None = None,
                force: bool = False) -> dict:
    """Write `X.npy` + `ids.json` + `meta.json` for one (family, segment)."""
    d = paths.feature_dir(entry.pair, family, segment)
    if (d / "X.npy").exists() and not force:
        return json.loads((d / "meta.json").read_text())

    X = feature_matrix(entry, family, segment, top_k=top_k)
    d.mkdir(parents=True, exist_ok=True)

    # Hash the matrix, not the source: this is what a consumer actually mmaps,
    # and it is the thing whose corruption would be invisible.
    digest = hashlib.sha256(np.ascontiguousarray(X).view(np.uint8)).hexdigest()

    tmp = d / "X.npy.tmp"
    np.save(tmp, X)
    tmp.replace(d / "X.npy")
    (d / "ids.json").write_text(json.dumps([str(i) for i in entry.ids]))

    meta = FeatureMeta(
        pair=entry.pair, family=family, segment=segment,
        kind=config.kind_of(family), pca_dim=pca_dim,
        dim=int(X.shape[1]), n=int(X.shape[0]), sha256=digest,
        source=provenance(entry, family, segment, top_k),
        builder_version=BUILDER_VERSION, top_k=top_k,
        extra={"built": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    )
    (d / "meta.json").write_text(meta.to_json())
    return json.loads((d / "meta.json").read_text())


def load_l1(pair: str, family: str, segment: str = "all") -> np.ndarray | None:
    """The cached matrix, or None. Never raises on a cold cache."""
    d = paths.feature_dir(pair, family, segment)
    x = d / "X.npy"
    if not x.exists():
        return None
    return np.load(x, mmap_mode="r")


# ── manifest ─────────────────────────────────────────────────────────────────

def _status(entry: read.Entry | None, family: str, reason: str) -> str:
    if entry is None:
        return "missing_raw"
    need = "phase1" if family in config.HS_FAMILIES else "phase2"
    return "ready" if entry.has_phase(need) else "missing_raw"


def manifest(pairs: list[str] | None = None) -> pd.DataFrame:
    """One row per `(pair, family, segment)` with its availability.

    `missing_raw` is a real state, not an error. It is how a pair whose phase 2
    has not run -- or whose only artifacts are the unusable 1,400-row legacy
    sampled runs -- is represented without pretending it is buildable.
    """
    keys = pairs if pairs is not None else [p.key for p in registry.resolve()]
    rows: list[dict] = []
    for pair in keys:
        try:
            entry = read.load(pair)
            err = ""
        except read.StoreError as exc:
            entry, err = None, str(exc)

        segments = entry.segments if entry is not None else ("all",)
        for family in config.FAMILIES:
            for segment in segments:
                status = _status(entry, family, err)
                rec = {
                    "pair": pair, "family": family, "segment": segment,
                    "status": status,
                    "n": int(entry.n) if entry is not None else 0,
                    "dim": np.nan, "kind": config.kind_of(family),
                    "pca_dim": None, "source_sha256": "",
                    "built": "",
                }
                if status == "ready":
                    # Width without materialising: the shape is metadata, and
                    # reading it must not cost a 2 GB allocation.
                    try:
                        rec["dim"] = _dim_of(entry, family, segment)
                    except Exception as exc:  # noqa: BLE001
                        rec["status"], rec["dim"] = "error", np.nan
                        rec["built"] = f"{type(exc).__name__}: {exc}"
                rows.append(rec)

    df = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    bad = set(df["status"]) - set(MANIFEST_STATUSES)
    if bad:
        raise ValueError(f"manifest produced unknown statuses {bad}")
    return df


def _dim_of(entry: read.Entry, family: str, segment: str) -> float:
    """The full width, at the widest top-k for the families that have one."""
    if family in config.HS_FAMILIES:
        with np.load(paths.raw_dir(entry.pair) / "hs" / f"{segment}.npz") as z:
            return float(z[family].shape[1])
    if family == "sink":
        return float(sink_mod.dim(entry, segment))
    if family == "attnlogdet":
        return float(spectral.dim(entry, family, segment))
    return float(spectral.dim(entry, family, segment,
                              top_k=max(spectral.top_k_grid(entry, segment))))


def write_manifest(pairs: list[str] | None = None) -> pd.DataFrame:
    df = manifest(pairs)
    out = paths.manifest()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df
