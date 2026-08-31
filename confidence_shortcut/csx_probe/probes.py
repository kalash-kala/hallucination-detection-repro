"""The transform and the two probe heads.

Reproduces `stage_a_common.make_transform` / `hs_lr` / `spectral_lr` /
`score_pos`, with the hyperparameters coming from `frozen_constants.yaml` rather
than from environment variables.

Three details are carried verbatim because changing any of them moves the
published numbers:

**The transform is fit on TRAIN rows only, always.** Both heads then share it.
Fitting the scaler or PCA on train+test is the classic leak that raises every
AUROC by a believable amount, and a shared transform is also what makes the two
heads' coefficient vectors live in one comparable space -- which the alpha
rotation depends on.

**`hs` and `spectral` genuinely differ** (`max_iter` 1000 vs 2000, and only
`spectral` sets `random_state`). Reproduced as-is, not harmonised.

**`kind == 'hs'` ignores `pca_dim` entirely.** The hs vector is already a pooled
reduction, and the published path sends it to `StandardScaler` regardless. That
is why `pca_dim` is recorded as `None` for those families rather than as a value
that was silently not applied.

The two heads are the whole design:

    g    -> y = incorrect            the correctness axis, what we want
    sep  -> y = low-confidence (L)   the confidence axis, the shortcut

Both are fit on the same rows in the same space, so any difference between them
is about the target and not about preprocessing.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from csx_probe import config

HEADS: tuple[str, ...] = config.HEADS


def make_transform(X_train: np.ndarray, kind: str, pca_dim: int | None):
    """Fit the shared coordinate transform on train features only."""
    if kind == "hs" or pca_dim is None:
        return StandardScaler().fit(X_train)
    dim = min(int(pca_dim), X_train.shape[1], X_train.shape[0] - 1)
    return PCA(n_components=dim, svd_solver="randomized",
               random_state=config.SEED).fit(X_train)


def make_lr(family: str, c: float) -> LogisticRegression:
    return LogisticRegression(**config.lr_kwargs(family, c))


def target(categories, head: str) -> np.ndarray:
    """The label vector for one head, in the global orientation.

    `g` is 1 for incorrect; `sep` is 1 for the LOW-confidence band. Both are
    oriented so that a larger score means more incorrect / less confident, which
    is what lets the two be compared without a sign convention per head.
    """
    cats = np.asarray(categories)
    if head == "g":
        return np.isin(cats, config.I_CATS).astype(int)
    if head == "sep":
        return np.isin(cats, config.L_CATS).astype(int)
    raise ValueError(f"{head!r} is not a fitted head; "
                     f"'entropy_only' is unfit and read straight off the rows")


def score_pos(clf, Z) -> np.ndarray:
    """`predict_proba[:, 1]`.

    Rank-identical to `decision_function` for LR -- the sigmoid is strictly
    increasing -- so every AUROC and every percentile bootstrap is unchanged,
    while staying valid for a classifier that has no `decision_function`.
    """
    return clf.predict_proba(Z)[:, 1].astype(float)


def fit_heads(X_train: np.ndarray, categories_train, *, family: str, c: float,
              pca_dim: int | None) -> tuple[object, dict[str, LogisticRegression]]:
    """Fit the transform once and both heads inside it."""
    kind = config.kind_of(family)
    tf = make_transform(X_train, kind, pca_dim)
    Z = tf.transform(X_train)
    heads = {h: make_lr(family, c).fit(Z, target(categories_train, h))
             for h in ("g", "sep")}
    return tf, heads


def axes(tf, heads: dict, X_test: np.ndarray, entropy_test) -> dict[str, np.ndarray]:
    """The three scored axes on one test set.

    `entropy_only` is the unfit control: the raw entropy the run recorded, with
    no probe involved. It is the thing every fitted head has to beat to have
    shown anything.
    """
    Z = tf.transform(X_test)
    return {"g": score_pos(heads["g"], Z),
            "sep": score_pos(heads["sep"], Z),
            "entropy_only": np.asarray(entropy_test, dtype=float)}


def coef(head: LogisticRegression) -> np.ndarray:
    return np.asarray(head.coef_, dtype=float).ravel().copy()
