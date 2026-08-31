"""The `sink` family, and the value-norm fusion that is *not* it.

`sink` is the **baseline**: the sink score `D = attn_diag + lap_diag`, top-`SINK_K`
per head, with no value-norm term. That distinction is the reason this module is
separate from `spectral.py` rather than one more entry in its table, and it is
load-bearing for the write-up: the `x ||V||` fusion is this project's own
contribution, so folding it into the family called `sink` would quietly credit the
baseline with the contribution's effect.

Both arrays are stored, gathered at **one** index set. The writer ranks by `D` and
takes `||V||` at those same ranks (`np.take_along_axis` with a shared `idx`), so
`sink_topk[i]` and `sink_vnorm_topk[i]` describe the same position. Sorting them
apart would pair the i-th largest sink score with the i-th largest value norm,
which are different tokens -- a bug that produces a plausible matrix.

`SINK_K` is fixed at extraction (10) and is **not** a CV-selected width. The
published builder used `k=10` and stored nothing wider, so unlike `lapeigvals` and
`attn_eigvals` there is no grid to search here.
"""

from __future__ import annotations

import numpy as np

from csx_probe.store.read import Entry

# The baseline family and the fusion variant. Only `sink` is one of the seven.
BASELINE_KEY = "sink_topk"
FUSED_KEY = "sink_vnorm_topk"


def build(entry: Entry, segment: str = "all", *, fused: bool = False
          ) -> np.ndarray:
    """`[n, L*H*SINK_K]` float32.

    `fused=False` is the `sink` family proper. `fused=True` is the
    `sink x ||V||` variant, which is a separate arm and never silently
    substituted for the baseline.
    """
    key = FUSED_KEY if fused else BASELINE_KEY
    arr = entry.diag(key, segment)
    return np.ascontiguousarray(arr.reshape(arr.shape[0], -1), dtype=np.float32)


def sink_k(entry: Entry, segment: str = "all") -> int:
    return int(entry.diag(BASELINE_KEY, segment).shape[-1])


def dim(entry: Entry, segment: str = "all") -> int:
    a = entry.diag(BASELINE_KEY, segment)
    return int(np.prod(a.shape[1:]))


def provenance(entry: Entry, segment: str = "all", *, fused: bool = False
               ) -> dict:
    return {
        "source": f"l0:diag/{FUSED_KEY if fused else BASELINE_KEY}",
        "segment": segment,
        "sink_k": sink_k(entry, segment),
        "fused_value_norm": bool(fused),
        "n_layers": entry.meta.layers,
    }
