"""Reading one L0 entry — the consuming half of contract 1.

This is the only module that knows the on-disk shape of what `csx_extract`
wrote. Everything downstream (derive, arms, experiments) goes through `Entry`,
so a change to the store layout lands here and nowhere else.

**No torch.** The extractor used it to build the diagonals; what it left behind
is plain `.npz`, and reading that needs numpy alone. `tests/test_isolation.py`
enforces this — csx_probe has to run on any CPU box.

Arrays are opened lazily and memory-mapped where possible. A vqav2 diag segment
is ~1.8 GB on disk, and a family build reads one key out of it; loading the whole
archive to slice one array would be the difference between a probe run that fits
in RAM and one that does not.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from csx_common import paths, registry
from csx_common.store_schema import (
    DIAG_KEYS, HS_SCHEMES, ContractError, EntryMeta,
)


class StoreError(Exception):
    """An L0 entry cannot be read. The message names what is wrong."""


@dataclass(frozen=True)
class Entry:
    """One pair's L0 entry, resolved but not yet loaded.

    `rows` is eager (a few MB); the feature arrays are not (gigabytes).
    """

    pair: str
    meta: EntryMeta
    rows: pd.DataFrame

    # ── provenance the results contract wants on every emitted row ──────────
    @property
    def model(self) -> str:
        return str(self.meta.raw["model"]["key"])

    @property
    def dataset(self) -> str:
        return str(self.meta.raw["dataset"])

    @property
    def modality(self) -> str:
        return self.meta.modality

    @property
    def prompt_template(self) -> str:
        return str(self.meta.raw["prompt_template"])

    @property
    def segments(self) -> tuple[str, ...]:
        return self.meta.segments

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def ids(self) -> np.ndarray:
        return self.rows["id"].to_numpy(dtype=object)

    @property
    def categories(self) -> np.ndarray:
        return self.rows["category"].to_numpy(dtype=object)

    @property
    def entropy(self) -> np.ndarray:
        return self.rows["entropy"].to_numpy(dtype=float)

    def has_phase(self, phase: str) -> bool:
        return self.meta.phase_done(phase)

    def available_families(self) -> tuple[str, ...]:
        """Which of the 7 families this entry can currently serve.

        A pair with phase 1 done and phase 2 not is a legitimate, usable state:
        it has the three hs_* families and none of the spectral ones. Reporting
        that honestly is what lets probing start before the expensive eager pass
        finishes.
        """
        return self.meta.available_families()

    # ── arrays ─────────────────────────────────────────────────────────────
    def hs(self, scheme: str, segment: str = "all") -> np.ndarray:
        """Pooled hidden states `[n, 2*D+1]` as float32.

        Stored fp16 to halve a 15,000-row matrix on disk; every consumer wants
        fp32, and the upcast is exact.
        """
        if scheme not in HS_SCHEMES:
            raise StoreError(
                f"{self.pair}: unknown hs scheme {scheme!r}; "
                f"known: {', '.join(HS_SCHEMES)}")
        self._require_segment(segment)
        if not self.has_phase("phase1"):
            raise StoreError(
                f"{self.pair}: phase1 is not done, so {scheme} is unavailable")
        path = paths.raw_dir(self.pair) / "hs" / f"{segment}.npz"
        with np.load(path) as z:
            if scheme not in z:
                raise StoreError(f"{self.pair}/{segment}: {path} has no {scheme}")
            X = np.asarray(z[scheme], dtype=np.float32)
        return self._check_rows(X, f"hs/{segment}/{scheme}")

    def diag(self, key: str, segment: str = "all") -> np.ndarray:
        """One reduced spectral array from `diag/<segment>.npz`.

        Returned at its stored dtype -- the callers that need fp32 upcast after
        slicing, which is meaningfully cheaper than upcasting `[n,L,H,50]` whole.
        """
        if key not in DIAG_KEYS:
            raise StoreError(
                f"{self.pair}: unknown diag key {key!r}; "
                f"known: {', '.join(DIAG_KEYS)}")
        self._require_segment(segment)
        if not self.has_phase("phase2"):
            raise StoreError(
                f"{self.pair}: phase2 is not done, so the spectral families are "
                f"unavailable (this is a valid state, not a corrupt entry)")
        path = paths.raw_dir(self.pair) / "diag" / f"{segment}.npz"
        with np.load(path) as z:
            if key not in z:
                raise StoreError(f"{self.pair}/{segment}: {path} has no {key}")
            arr = z[key]
        return self._check_rows(arr, f"diag/{segment}/{key}")

    def peaks(self, segment: str = "all") -> dict:
        """The peak layers the pooled hs_* features were built from.

        Carried into L1 provenance: a pooled vector is uninterpretable without
        knowing which layers it pooled.

        The file nests per-segment records under `segments`, alongside run-level
        keys (`seed`, `train_fraction`) that describe how the peaks were chosen.
        Reading the top level directly would silently return `{}` for every
        segment -- present, empty, and wrong -- so the nesting is required here
        rather than tolerated.
        """
        path = paths.raw_dir(self.pair) / "hs" / "peaks.json"
        if not path.exists():
            raise StoreError(f"{self.pair}: missing {path}")
        doc = json.loads(path.read_text())
        segs = doc.get("segments")
        if not isinstance(segs, dict):
            raise StoreError(
                f"{self.pair}: {path} has no `segments` block; it was written by "
                f"an extractor this reader does not understand")
        if segment not in segs:
            raise StoreError(
                f"{self.pair}: {path} has no peaks for segment {segment!r} "
                f"(has {sorted(segs)})")
        rec = segs[segment]
        return {"peaks": rec.get("peaks", {}),
                "buckets": rec.get("buckets", {}),
                "seed": doc.get("seed"),
                "train_fraction": doc.get("train_fraction")}

    # ── internals ──────────────────────────────────────────────────────────
    def _require_segment(self, segment: str) -> None:
        if segment not in self.segments:
            raise StoreError(
                f"{self.pair}: no segment {segment!r}; this is a "
                f"{self.modality} pair with {list(self.segments)}")

    def _check_rows(self, arr: np.ndarray, what: str) -> np.ndarray:
        """The leading axis must be the row axis, every time.

        Cheap, and it is the one mismatch that would otherwise line features up
        against the wrong labels silently rather than raising.
        """
        if arr.shape[0] != self.n:
            raise StoreError(
                f"{self.pair}: {what} has {arr.shape[0]} rows but rows.parquet "
                f"has {self.n}")
        return arr


@functools.lru_cache(maxsize=32)
def load(pair: str) -> Entry:
    """Load one L0 entry.

    Cached: a transfer-grid run touches the same entry once per (family,
    segment, arm), and re-reading a 15,000-row parquet each time is pure waste.
    The cache holds metadata and the row table only -- never the feature arrays.
    """
    d = paths.raw_dir(pair)
    if not d.is_dir():
        raise StoreError(
            f"{pair}: no store entry at {d}. Extraction has not run for this "
            f"pair, or it ran against a different store root.")
    try:
        meta = EntryMeta.load(paths.raw_meta(pair))
    except ContractError as exc:
        raise StoreError(f"{pair}: {exc}") from None

    rpath = paths.raw_rows(pair)
    if not rpath.exists():
        raise StoreError(f"{pair}: missing {rpath}")
    rows = pd.read_parquet(rpath)

    if meta.pair != pair:
        raise StoreError(f"{pair}: meta.json declares pair={meta.pair!r}")
    if len(rows) != meta.n_rows:
        raise StoreError(
            f"{pair}: rows.parquet has {len(rows)} rows, meta says {meta.n_rows}")
    # `row` indexes every array in the entry, so it must be exactly 0..n-1 and
    # in order -- the arrays are positional and carry no ids of their own.
    if not np.array_equal(rows["row"].to_numpy(), np.arange(len(rows))):
        raise StoreError(
            f"{pair}: rows.parquet `row` is not 0..n-1 in order; the feature "
            f"arrays are positional and would be misaligned")
    return Entry(pair=pair, meta=meta, rows=rows)


def is_usable(pair: str) -> tuple[bool, str]:
    """`(usable, reason)` without raising -- for manifests and --plan output."""
    try:
        e = load(pair)
    except StoreError as exc:
        return False, str(exc)
    if not (e.has_phase("phase1") or e.has_phase("phase2")):
        return False, "neither phase is done"
    return True, ""


def row_index(entry: Entry) -> dict[str, int]:
    """`id -> position`, for pulling an arm's rows out of a feature matrix.

    Arms are id-sets; feature matrices are positional. This is the join, and it
    is built once per entry rather than per arm.
    """
    return {str(i): int(r) for i, r in zip(entry.rows["id"], entry.rows["row"])}
