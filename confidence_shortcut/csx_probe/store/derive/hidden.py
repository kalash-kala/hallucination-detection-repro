"""The three `hs_*` families.

Nothing is computed here. The expensive, destructive reduction — per-layer AUROC
to find the peak layers, then `concat[mean_z(mid), mean_z(late), s_ext]` — already
happened inside extraction, because holding raw `[N, L, D]` was ~10 GB per pair
and the raw shards were deleted afterwards (store contract, L0).

So this module is a typed read with the provenance attached. It exists as its own
module anyway: `derive/` is where a consumer looks for "how does family X become a
matrix", and an `hs_wide` that silently had no entry there would read as an
oversight rather than as a deliberate asymmetry.

The consequence of that asymmetry is worth stating where someone will see it:
**changing the hs pooling scheme means re-extracting**, on a GPU. The spectral
side can be re-derived from the store at any width.
"""

from __future__ import annotations

import numpy as np

from csx_probe import config
from csx_probe.store.read import Entry

# `top_k` and `pca_dim` are not knobs for these families: the vector is already
# pooled, and stage_a_common.make_transform sends kind='hs' to StandardScaler
# regardless of pca_dim. Recorded as None in L1 meta so the difference from the
# spectral families is explicit rather than implied by absence.
TOP_K = None
PCA_DIM = None


def build(entry: Entry, family: str, segment: str = "all") -> np.ndarray:
    """`[n, 2*D+1]` float32 for one hs family."""
    if family not in config.HS_FAMILIES:
        raise ValueError(
            f"{family!r} is not an hs family; known: {config.HS_FAMILIES}")
    return entry.hs(family, segment)


def provenance(entry: Entry, family: str, segment: str = "all") -> dict:
    """What this matrix was built from, for L1 `meta.json`.

    The peak layers are the load-bearing part: two pairs' `hs_wide` vectors are
    the same width and utterly incomparable if they pooled different layers, and
    that fact is not recoverable from the matrix.
    """
    return {
        "source": "l0:hs",
        "scheme": family,
        "segment": segment,
        "peaks": entry.peaks(segment),
        "n_layers": entry.meta.layers,
    }
