"""Choosing `(top_k, pca_dim)` for a spectral family.

Regularisation here is `(dimension, C)` **jointly** — the same family is fit at
8,193 dims under `native` and at 100 under `pca100`, and a `C` that is right for
one is wrong for the other. So the basis is selected first, by CV on the train
rows, and `C` is selected against the basis that won (`experiments/c_selection`).

Reproduced from `stage_a_common._spectral_select` / `_spectral_pipe`, including
the detail that is easy to normalise away by accident: **the selection pipeline
fits `LogisticRegression` at its default `C=1.0`**, not at the family's selected
`C`. Selecting the basis at the same `C` the final probe uses would be defensible
— but it is not what produced the published numbers, and this module's job is to
reproduce them.

`pin_from_bundle` exists because that argmax is not perfectly stable across
sklearn builds. When two `(k, pca)` cells are within noise of each other, a
different LAPACK or a different randomized-SVD draw can tip the winner, and every
downstream number moves with it. For the 8 `qa8` parity pairs the published choice
is read back from the bundle cache instead of recomputed — the search is a means
to a frozen constant there, not a live decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from csx_probe import config
from csx_probe.store import read
from csx_probe.store.derive import spectral

N_SPLITS = 5


@dataclass(frozen=True)
class Selection:
    """The chosen basis for one (pair, family, segment), with its CV curve."""
    family: str
    segment: str
    top_k: int | None
    pca_dim: int | None
    cv_auroc: float
    curve: dict[tuple[int | None, int | None], float]
    source: str          # 'cv' | 'bundle' | 'fixed'

    def as_meta(self) -> dict:
        return {"top_k": self.top_k, "pca_dim": self.pca_dim,
                "cv_auroc": self.cv_auroc, "selection_source": self.source}


def _pipe(pca_dim: int | None, n_feat: int, n_samp: int) -> Pipeline:
    """`stage_a_common._spectral_pipe`, verbatim.

    The LR is deliberately at default C -- see the module docstring.
    """
    lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                            random_state=config.SEED)
    if pca_dim is None:
        return Pipeline([("sc", StandardScaler()), ("lr", lr)])
    dim = min(pca_dim, n_feat, n_samp - 1)
    return Pipeline([("pca", PCA(n_components=dim, svd_solver="randomized",
                                 random_state=config.SEED)), ("lr", lr)])


def y_incorrect(categories) -> np.ndarray:
    """The correctness target: 1 == incorrect, matching the global orientation."""
    return np.isin(np.asarray(categories), list(config.I_CATS)).astype(int)


def select(entry: read.Entry, family: str, segment: str, train_rows: np.ndarray,
           *, fixed: tuple[int | None, int | None] | None = None) -> Selection:
    """CV-select the basis on **train rows only**.

    `train_rows` is positional into the entry's arrays. Passing test rows here
    would leak the evaluation set into a hyperparameter choice, which is exactly
    the kind of error that inflates every downstream AUROC by a believable amount.
    """
    if fixed is not None:
        k, p = fixed
        return Selection(family, segment, k, p, float("nan"), {}, "fixed")

    idx = np.asarray(train_rows, dtype=int)
    y = y_incorrect(entry.categories[idx])
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"{entry.pair}/{family}/{segment}: train rows are single-class; "
            f"cannot CV-select a basis")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=config.SEED)
    curve: dict[tuple[int | None, int | None], float] = {}
    best = (-np.inf, None)

    for k in _widths(entry, family, segment):
        X = _matrix(entry, family, segment, k)[idx]
        for pca_dim in config.PCA_GRID:
            cv = float(np.mean(cross_val_score(
                _pipe(pca_dim, X.shape[1], X.shape[0]), X, y, cv=skf,
                scoring="roc_auc", n_jobs=1)))
            curve[(k, pca_dim)] = cv
            # Strict >: ties keep the FIRST cell, and the grids are ascending, so
            # a tie resolves to the narrower basis. Deterministic, and it favours
            # the cheaper model, which is the right default when the CV cannot
            # tell them apart.
            if cv > best[0]:
                best = (cv, (k, pca_dim))
        del X

    cv, (k, pca_dim) = best[0], best[1]
    return Selection(family, segment, k, pca_dim, float(cv), curve, "cv")


def _widths(entry: read.Entry, family: str, segment: str) -> list[int | None]:
    """The top-k values to search, or `[None]` for families that have no width."""
    if family in spectral.TOPK_FAMILIES:
        return list(spectral.top_k_grid(entry, segment))
    return [None]


def _matrix(entry: read.Entry, family: str, segment: str,
            top_k: int | None) -> np.ndarray:
    from csx_probe.store.derive import sink as sink_mod
    if family == "sink":
        return sink_mod.build(entry, segment)
    return spectral.build(entry, family, segment, top_k=top_k)


def curve_rows(sel: Selection, entry: read.Entry) -> list[dict]:
    """The full CV curve as atomic rows, for the `c_selection`-style audit trail.

    Stored rather than discarded for the same reason the `C` curve is: re-deciding
    a basis later should not need refits, and a selection that was nearly a tie is
    a materially different claim from one that won by a mile.
    """
    return [
        {"pair": entry.pair, "family": sel.family, "segment": sel.segment,
         "top_k": k, "pca_dim": p, "cv_auroc": v,
         "selected": (k == sel.top_k and p == sel.pca_dim)}
        for (k, p), v in sorted(sel.curve.items(),
                                key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0))
    ]
