"""`lapeigvals`, `attn_eigvals`, `attnlogdet` — derived from the stored diagonals.

This is the non-destructive half of L0. Extraction kept the top-50 per head
because `TOP_K_GRID` maxes at 50, so every width the study ever fits is a slice
of what is on disk and **no GPU is needed to change `top_k`**.

Two invariants make that slice legal, both inherited from the writer
(`csx_extract/phase2_attention._reduce`):

**The stored arrays are already sorted descending.** The writer does
`np.sort(...)[..., ::-1][:k]`, which is exactly the published builders'
`sort(descending=True).values[:, :, :top_k]`. So top-`k` here is `[..., :k]` --
not a re-sort, which would be wasted work, and not a `topk` over an unsorted
axis, which would silently reorder within the head.

**Flattening is `[L, H, k] -> L-major`.** `reshape(n, -1)` on `[n, L, H, k]`
reproduces `Tensor.flatten()` on `[L, H, k]` element for element. Getting this
wrong would permute feature columns consistently, which trains and scores without
complaint and quietly destroys any cross-pair comparison of coefficients.

`attnlogdet` is the exception that justifies its own stored key: it is
`mean(log(clamp_min(diag, 1e-12)))` over **all** positions, so it is not a
function of the top-k arrays at any width. It is read, never recomputed.
"""

from __future__ import annotations

import numpy as np

from csx_probe import config
from csx_probe.store.read import Entry, StoreError

# family -> the L0 key it slices. attnlogdet is not a top-k family and is handled
# separately below; it appears here so `DIAG_KEY` covers every spectral family
# and a missing entry is a KeyError rather than a silent skip.
DIAG_KEY = {
    "lapeigvals": "lap_topk",
    "attn_eigvals": "attn_topk",
    "attnlogdet": "attn_logdet",
}

# Only these two have a width to select. attnlogdet has none (it is a mean, not a
# top-k), and `sink` is fixed at SINK_K by the extractor -- see derive/sink.py.
TOPK_FAMILIES = ("lapeigvals", "attn_eigvals")


def top_k_grid(entry: Entry, segment: str = "all") -> list[int]:
    """The widths that are actually available for this entry.

    Clipped to what was stored rather than assumed: a `--limit` extraction or a
    short segment can leave fewer than 50 columns, and CV-selecting a width the
    array cannot serve would fail deep inside a fold instead of here.
    """
    stored = int(entry.diag("attn_topk", segment).shape[-1])
    grid = [k for k in config.TOP_K_GRID if k <= stored]
    if not grid:
        raise StoreError(
            f"{entry.pair}/{segment}: stored top-k width is {stored}, narrower "
            f"than every value in TOP_K_GRID {list(config.TOP_K_GRID)}")
    return grid


def build(entry: Entry, family: str, segment: str = "all",
          top_k: int | None = None) -> np.ndarray:
    """`[n, dim]` float32 for one spectral family.

    `dim` is `L*H*top_k` for the two top-k families and `L*H` for `attnlogdet`.
    """
    if family not in DIAG_KEY:
        raise ValueError(
            f"{family!r} is not a diagonal-derived spectral family; "
            f"known: {', '.join(DIAG_KEY)}")

    if family == "attnlogdet":
        # [n, L, H] -> [n, L*H]. Already fp32 on disk: it is a log-mean, and the
        # values run to about -10, where fp16 would start costing real precision.
        arr = entry.diag("attn_logdet", segment)
        return np.ascontiguousarray(arr.reshape(arr.shape[0], -1),
                                    dtype=np.float32)

    arr = entry.diag(DIAG_KEY[family], segment)          # [n, L, H, K] fp16
    k = _resolve_k(entry, arr, family, segment, top_k)
    sliced = arr[..., :k]
    # Upcast AFTER slicing: at k=5 that is a tenth of the fp32 traffic of
    # upcasting the full stored width first.
    return np.ascontiguousarray(sliced.reshape(sliced.shape[0], -1),
                                dtype=np.float32)


def _resolve_k(entry: Entry, arr: np.ndarray, family: str, segment: str,
               top_k: int | None) -> int:
    stored = int(arr.shape[-1])
    if top_k is None:
        raise ValueError(
            f"{entry.pair}/{family}: top_k is required for this family; it is "
            f"CV-selected per (pair, family) -- see derive/select.py")
    k = int(top_k)
    if k <= 0:
        raise ValueError(f"{entry.pair}/{family}: top_k={k} must be positive")
    if k > stored:
        raise StoreError(
            f"{entry.pair}/{family}/{segment}: top_k={k} but only {stored} "
            f"columns were stored; re-extract at a wider EXTRACT_TOP_K to fit "
            f"this width")
    return k


def dim(entry: Entry, family: str, segment: str = "all",
        top_k: int | None = None) -> int:
    """The width `build` would return, without materialising it."""
    if family == "attnlogdet":
        a = entry.diag("attn_logdet", segment)
        return int(np.prod(a.shape[1:]))
    a = entry.diag(DIAG_KEY[family], segment)
    k = _resolve_k(entry, a, family, segment, top_k)
    return int(a.shape[1] * a.shape[2] * k)


def provenance(entry: Entry, family: str, segment: str = "all",
               top_k: int | None = None) -> dict:
    return {
        "source": f"l0:diag/{DIAG_KEY[family]}",
        "segment": segment,
        "top_k": None if family == "attnlogdet" else int(top_k),
        "n_layers": entry.meta.layers,
        "n_q_heads": entry.meta.raw.get("model", {}).get("n_q_heads"),
    }
