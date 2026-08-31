"""Aggregating the 10 sampled generations into one feature row per example.

`csx_extract` writes one feature row per UNIQUE `(id, answer_text)` -- the
manifest deduplicates within each id, because two identical sampled answers to
the same question are the same forward pass. This module reverses that: it
gathers each example's 10 slots back through the slot map and reduces them to a
single row aligned to the greedy roster, which is what the `_cm` scorers and the
`sampled` router consume.

**The slot map is expanded before aggregating, not after.** A row whose 10
generations collapsed to 3 unique texts still has 10 slots, and 7 of them are
repeats. Averaging the 3 unique rows would silently reweight that example toward
its rare answers -- exactly backwards, since the repeats are the *confident* mass.
So `manifest.parquet` is used to expand unique rows back to 10 slots first, and
the statistic is taken over the 10.

**`greedy_mean_std` concatenates the greedy row.** The sampled block alone
discards the one generation the rest of the pipeline is built on; the router and
the `_cm` experts want both, which is only sound because sampled extraction reuses
the greedy pass's peak layers and bucket definitions, so the two halves live in
the same feature space.

**`cloud` is a different kind of feature and deserves its own note.** The other
schemes are per-dimension statistics, so their width scales with the feature
space (`mean_std` on `hs_wide` is ~14k). `cloud` instead throws the coordinate
system away and keeps only the *shape* of the 10-point cloud: pairwise distances
and cosines, the leading centred-Gram eigenvalues, and the displacement from the
greedy vector. That is **10 numbers regardless of family or width**. In the QA
run it stayed competitive with `mean_std` at three orders of magnitude fewer
dimensions -- the reading being that the band is a property of semantic *spread*,
and these ten numbers encode spread. Whether that survives on VLM is open.

Four schemes, all row-aligned to `rows.parquet`:

    mean            per-feature mean over the 10 slots
    mean_std        mean concatenated with the across-slot standard deviation
    greedy_mean_std the greedy row concatenated with both of the above
    cloud           10 geometry scalars from the slots' Gram matrix (width-free)
"""

from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd

from csx_common import paths
from csx_probe import config
from csx_probe.store import build as store_build, read

SCHEMES: tuple[str, ...] = ("mean", "mean_std", "greedy_mean_std", "cloud")
DEFAULT_SCHEME: str = "greedy_mean_std"

# Carried verbatim from the QA run (`25_sampled_router_ovr.py`): 3 leading
# eigenvalues, so the block is 4 + 3 + 3 = 10 wide. Not a tunable -- changing it
# changes the feature and breaks comparability with the published result.
CLOUD_EIG: int = 3
CLOUD_DIM: int = 4 + CLOUD_EIG + 3

# Peak working set for the cloud pass is chunk x n_slots x n_feat x 8 bytes.
# 256 MB keeps `hs_wide` (~14k wide, 10 slots) at a few hundred rows per chunk,
# which is where the batched matmul is already at full BLAS efficiency.
_CLOUD_CHUNK_BYTES: int = 256 << 20


class SampledError(Exception):
    """The sampled tier cannot be read for this pair. The message names why."""


def is_ready(pair: str) -> bool:
    meta = paths.sampled_meta(pair)
    if not meta.exists():
        return False
    try:
        return bool(json.loads(meta.read_text())
                    .get("extraction", {}).get("done"))
    except (ValueError, OSError):
        return False


@functools.lru_cache(maxsize=8)
def slot_map(pair: str) -> pd.DataFrame:
    """`id, slot, urow` -- which unique feature row each of the 10 slots uses."""
    p = paths.sampled_manifest(pair)
    if not p.exists():
        raise SampledError(f"{pair}: no {p}; run stage 25 first")
    return pd.read_parquet(p)


def _gather_index(pair: str, ids: np.ndarray) -> np.ndarray:
    """`[n_rows, n_slots]` of unique-row indices, aligned to the greedy roster.

    Built by reindexing on the roster's own id order rather than by sorting, so
    a pair whose manifest was written in a different order than `rows.parquet`
    still lines up. A mismatch raises instead of producing a shifted matrix --
    the failure mode that silently pairs one example's features with another's
    label.
    """
    slots = slot_map(pair)
    wide = (slots.pivot(index="id", columns="slot", values="urow")
                 .reindex(pd.Index(np.asarray(ids, dtype=object), name="id")))
    if wide.isna().to_numpy().any():
        missing = int(wide.isna().any(axis=1).sum())
        raise SampledError(
            f"{pair}: {missing} roster ids have no complete slot map; the "
            f"manifest and rows.parquet disagree about which examples exist")
    return wide.to_numpy(dtype=np.int64)


def _unique_matrix(pair: str, family: str, segment: str,
                   top_k: int | None) -> np.ndarray:
    """The unique-row feature matrix for one family, from the sampled store.

    Reuses `store.build`'s derivation so a sampled feature is built by exactly
    the same code path as its greedy counterpart -- the two must occupy the same
    space to be concatenated, and a second implementation is how that quietly
    stops being true.
    """
    return store_build.feature_matrix(
        _SampledEntryView(pair), family, segment, top_k=top_k)


class _SampledEntryView:
    """An `Entry`-shaped view whose arrays come from `sampled/` not `raw/`.

    Deliberately a thin shim rather than a second Entry subclass: `store.build`
    needs only `hs`/`diag`/`peaks`/`pair`, and narrowing the surface to those
    four makes it obvious that nothing else about the greedy entry is being
    reinterpreted as sampled.
    """

    def __init__(self, pair: str):
        self.pair = pair
        self._greedy = read.load(pair)
        self.n = _n_unique(pair)

    @property
    def segments(self):
        return self._greedy.segments

    def peaks(self, segment: str = "all") -> dict:
        # Bucket definitions are the GREEDY pass's, by design -- re-deriving them
        # from sampled rows would be circular and would move the feature space.
        return self._greedy.peaks(segment)

    def hs(self, scheme: str, segment: str = "all") -> np.ndarray:
        return self._load("hs", segment, scheme)

    def diag(self, key: str, segment: str = "all") -> np.ndarray:
        return self._load("diag", segment, key)

    def _load(self, kind: str, segment: str, key: str) -> np.ndarray:
        path = paths.sampled_dir(self.pair) / kind / f"{segment}.npz"
        if not path.exists():
            raise SampledError(f"{self.pair}: missing {path}")
        with np.load(path) as z:
            if key not in z:
                raise SampledError(f"{self.pair}/{segment}: {path} has no {key}")
            arr = z[key]
        return np.asarray(arr, dtype=np.float32) if kind == "hs" else arr


@functools.lru_cache(maxsize=8)
def _n_unique(pair: str) -> int:
    p = paths.sampled_unique(pair)
    if not p.exists():
        raise SampledError(f"{pair}: no {p}; run stage 25 first")
    return len(pd.read_parquet(p, columns=["urow"]))


def cloud_block(U: np.ndarray, idx: np.ndarray,
                greedy: np.ndarray) -> np.ndarray:
    """The 10 geometry scalars per row, from the slots' Gram matrix.

    Ported from `25_sampled_router_ovr.aggregate(variant="cloud")` and kept
    numerically identical to it -- `test_cloud_matches_reference_implementation`
    pins that against a literal transcription of the original.

    Everything derives from `G = X @ X.T`, so this is one batched matmul per
    chunk rather than an `n x n x d` broadcast. The broadcast form carries
    Python-level overhead at ~n^2*d and dominated the original run.

    `U` is the unique-row matrix, `idx` the `[n_rows, n_slots]` slot map, and
    `greedy` the per-row greedy matrix (the last two features are displacement
    from it). Computed in float64: the centred Gram is a difference of
    similar-magnitude terms, and in float32 its small eigenvalues lose most of
    their significant digits before `log1p` ever sees them.
    """
    n_rows, n_slots = idx.shape
    if n_slots < 2:
        raise SampledError(
            f"cloud is undefined at n_slots={n_slots}: pairwise distances, "
            f"cosines and the centred Gram all need at least 2 slots")
    n_feat = U.shape[1]
    out = np.empty((n_rows, CLOUD_DIM), dtype=np.float32)
    iu = np.triu_indices(n_slots, 1)
    denom = max(n_slots - 1, 1)
    per_row = max(n_slots * n_feat * 8, 1)
    chunk = int(max(1, min(n_rows, _CLOUD_CHUNK_BYTES // per_row)))

    for s in range(0, n_rows, chunk):
        e = min(s + chunk, n_rows)
        X = U[idx[s:e]].astype(np.float64)             # [c, n_slots, n_feat]
        G = X @ np.swapaxes(X, 1, 2)                   # [c, n_slots, n_slots]
        dg2 = np.diagonal(G, axis1=1, axis2=2)         # [c, n_slots]

        d2 = np.maximum(dg2[:, :, None] + dg2[:, None, :] - 2.0 * G, 0.0)
        dist = np.sqrt(d2[:, iu[0], iu[1]])
        nrm = np.sqrt(np.maximum(dg2, 0.0)) + 1e-12
        cos = (G / (nrm[:, :, None] * nrm[:, None, :]))[:, iu[0], iu[1]]

        mu = X.mean(axis=1)                            # [c, n_feat]
        Xmu = np.einsum("csd,cd->cs", X, mu)           # [c, n_slots]
        mumu = np.einsum("cd,cd->c", mu, mu)
        Gc = (G - Xmu[:, None, :] - Xmu[:, :, None]
              + mumu[:, None, None])
        ev = np.linalg.eigvalsh((Gc + np.swapaxes(Gc, 1, 2)) / 2.0 / denom)
        ev = ev[:, ::-1]                               # descending
        if ev.shape[1] < CLOUD_EIG:
            # n_slots < CLOUD_EIG: zero-pad, as the reference does. Unreachable
            # at the pipeline's n_slots=10, but the padding is what makes the
            # block a fixed CLOUD_DIM wide for any n.
            ev = np.pad(ev, ((0, 0), (0, CLOUD_EIG - ev.shape[1])))
        ev = np.log1p(np.maximum(ev[:, :CLOUD_EIG], 0.0))

        g = greedy[s:e].astype(np.float64)
        Xg = np.einsum("csd,cd->cs", X, g)
        gg = np.einsum("cd,cd->c", g, g)
        dgv = np.sqrt(np.maximum(dg2 - 2.0 * Xg + gg[:, None], 0.0))

        out[s:e] = np.column_stack([
            dist.mean(1), dist.std(1), cos.mean(1), cos.std(1),
            ev, ev.sum(1), dgv.mean(1), dgv.std(1),
        ]).astype(np.float32)
    return out


def aggregate(pair: str, family: str, segment: str = "all", *,
              scheme: str = DEFAULT_SCHEME,
              top_k: int | None = None,
              n_slots: int | None = None) -> np.ndarray:
    """One row per greedy-roster example, in the requested scheme.

    `n_slots` truncates to the FIRST n sampled generations, which is what the
    cost sweep varies -- each extra generation is a real inference cost, so
    `n` is the price of the band label. Slot order is the manifest's, so "first
    n" is well defined and reproducible rather than an arbitrary subset.

    Note the feature WIDTH is constant in `n` for every scheme (`cloud` is 10
    numbers at any n, `mean_std` is 2d), so the sweep varies how well the input
    is estimated, never the model's capacity.
    """
    if scheme not in SCHEMES:
        raise SampledError(f"unknown scheme {scheme!r}; known: {', '.join(SCHEMES)}")
    entry = read.load(pair)
    U = _unique_matrix(pair, family, segment, top_k)
    idx = _gather_index(pair, entry.ids)           # [n_rows, n_slots]
    if n_slots is not None:
        if not 1 <= n_slots <= idx.shape[1]:
            raise SampledError(
                f"{pair}: n_slots={n_slots} outside 1..{idx.shape[1]}")
        idx = idx[:, :n_slots]

    if scheme == "cloud":
        # Needs the slots jointly, not their moments, so it cannot share the
        # streaming accumulation below; it chunks over rows instead.
        greedy = store_build.feature_matrix(entry, family, segment, top_k=top_k)
        _check_greedy(pair, greedy, idx.shape[0])
        return cloud_block(U, idx, greedy)

    # [n_rows, n_slots, n_feat] would be 10x the feature matrix; accumulate the
    # two moments slot by slot instead so peak memory stays at one slot's width.
    n_rows, n_slots = idx.shape
    s1 = np.zeros((n_rows, U.shape[1]), dtype=np.float64)
    s2 = np.zeros_like(s1)
    for k in range(n_slots):
        v = U[idx[:, k]].astype(np.float64)
        s1 += v
        s2 += v * v
    mean = (s1 / n_slots).astype(np.float32)
    if scheme == "mean":
        return mean
    var = np.maximum(s2 / n_slots - (s1 / n_slots) ** 2, 0.0)
    std = np.sqrt(var).astype(np.float32)
    if scheme == "mean_std":
        return np.concatenate([mean, std], axis=1)

    greedy = store_build.feature_matrix(entry, family, segment, top_k=top_k)
    _check_greedy(pair, greedy, n_rows)
    return np.concatenate([greedy, mean, std], axis=1)


def _check_greedy(pair: str, greedy: np.ndarray, n_rows: int) -> None:
    if len(greedy) != n_rows:
        raise SampledError(
            f"{pair}: greedy matrix has {len(greedy)} rows against {n_rows} "
            f"roster ids")


def blocks_by_scheme(pair: str, *, families: tuple[str, ...],
                     segments: tuple[str, ...],
                     schemes: tuple[str, ...] = (DEFAULT_SCHEME,),
                     ) -> dict[str, dict[tuple[str, str], np.ndarray]]:
    """`{scheme: {(family, segment): matrix}}` -- what `run_pair` consumes.

    The per-scheme aggregations are independent, so this is a plain loop; the
    sharing that matters (the greedy-side probe fits) happens downstream in
    `routed_grid._fit_train_side`, not here.
    """
    unknown = [s for s in schemes if s not in SCHEMES]
    if unknown:
        raise SampledError(
            f"unknown scheme(s) {unknown}; known: {', '.join(SCHEMES)}")
    return {s: blocks(pair, families=families, segments=segments, scheme=s)
            for s in schemes}


def blocks(pair: str, *, families: tuple[str, ...],
           segments: tuple[str, ...],
           scheme: str = DEFAULT_SCHEME) -> dict[tuple[str, str], np.ndarray]:
    """`{(family, segment): matrix}` for everything the routed grid will ask for.

    Families the sampled store cannot serve are omitted rather than raising: a
    pair whose sampled extraction covered only some families is a legitimate
    partial state, and the routed grid already reports `_cm` scorers it had to
    skip.
    """
    from csx_probe.store.derive import select as select_mod
    from csx_probe.arms import build as arms_build

    entry = read.load(pair)
    arms = arms_build.build_all(entry)
    out: dict[tuple[str, str], np.ndarray] = {}
    for family in families:
        for segment in segments:
            top_k = None
            if family not in config.HS_FAMILIES:
                top_k = select_mod.select(entry, family, segment,
                                          arms["dse_natural"].train).top_k
            try:
                out[(family, segment)] = aggregate(
                    pair, family, segment, scheme=scheme, top_k=top_k)
            except (SampledError, FileNotFoundError, KeyError):
                continue
    return out
