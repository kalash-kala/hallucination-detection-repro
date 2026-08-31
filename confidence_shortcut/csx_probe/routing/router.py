"""Predicting which band a row belongs to, without seeing the answer.

Three routers, and the difference between them is the whole deployability
argument:

    greedy    band predicted from the SINGLE greedy generation. 1 generation
              total -- no extra sampling cost at all.
    sampled   band predicted from 10 sampled generations. 10 generations, but it
              replaces the NLI/clustering stage those methods usually need.
    oracle    the true band, handed over. NOT DEPLOYABLE -- you would need the
              answer to know the band.

**`oracle` is computed and never tabled.** It exists because the affine-invariance
gate is undefined without it: only under oracle routing is each atomic cell
scored by exactly one expert, which is the premise that makes `z` and
`platt_prior` provably identical within band. Reporting it as a method would be
reporting a scorer that cannot exist at inference time, so `config.PRESENTED_ROUTERS`
excludes it and the renderer refuses it.

**M14 — the greedy router IS the `sep` probe.** In legacy `54` the greedy router
is `make_lr(kind).fit(Ztr, y_hi)`; our `sep` head (`probes.target(..., 'sep')`) is
`1[cat in L_CATS]`. Those are the same binary target with the labels swapped, so
`p(HI) = 1 - p(sep)` and the fitted decision boundary is identical. The
1-generation tier therefore needs no new router machinery, and its band AUROC is
already sitting in `per_pair_long` under `head='sep'`.

That equivalence is worth more than the code it saves. The probe a naive design
reaches for *as an error scorer* -- `sep`, the thing that scores ~0.000 on
`IHvCL` because it is reading confidence rather than correctness -- turns out,
used correctly, to be only a router. Same fit, same features, same rows; the
difference is entirely in what it is asked to do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csx_probe import config, probes


class RouterError(Exception):
    """A router cannot be fit or applied. The message names why."""


HI: str = config.BAND_ORDER[0]
LO: str = config.BAND_ORDER[1]


@dataclass(frozen=True)
class Router:
    """A fitted band predictor. `kind` is one of config.ROUTERS."""

    kind: str
    transform: object | None = None
    clf: object | None = None
    keep: np.ndarray | None = None      # sampled: the surviving feature columns
    n_train: int = 0

    def p_hi(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RouterError(
                f"{self.kind}: has no classifier; `oracle` reads the true band "
                f"and must be called through `assign` with categories")
        Xn = X if self.keep is None else X[:, self.keep]
        p = self.clf.predict_proba(self.transform.transform(Xn))
        return np.asarray(p[:, list(self.clf.classes_).index(1)], dtype=float)

    def assign(self, X: np.ndarray | None = None,
               categories=None) -> np.ndarray:
        """Per-row band labels.

        `oracle` ignores `X` and reads the true band off the categories; the
        fitted routers ignore `categories` and read the features. Passing the
        wrong one is an error rather than a silent default -- an oracle that
        quietly fell back to a fitted router, or vice versa, would change what
        the whole experiment measures while leaving the output shape identical.
        """
        if self.kind == "oracle":
            if categories is None:
                raise RouterError("oracle routing needs the true categories")
            return config.band_of(categories)
        if X is None:
            raise RouterError(f"{self.kind} routing needs the feature matrix")
        return np.where(self.p_hi(X) >= config.ROUTE_THRESHOLD, HI, LO)


def y_hi(categories) -> np.ndarray:
    """1 for the HIGH-confidence band.

    Note this is `1 - probes.target(categories, 'sep')`: `sep` marks the LOW
    band. Same fit, opposite labels -- see the module docstring.
    """
    return (config.band_of(categories) == HI).astype(int)


def fit_oracle() -> Router:
    return Router(kind="oracle")


def fit_greedy(X_train: np.ndarray, categories_train, *, family: str, c: float,
               pca_dim: int | None) -> Router:
    """The 1-generation router: a probe on the greedy pass's own features.

    Mechanically identical to fitting the `sep` head with the labels swapped.
    """
    y = y_hi(categories_train)
    if len(np.unique(y)) < 2:
        raise RouterError("greedy router: train rows are all one band")
    tf = probes.make_transform(X_train, config.kind_of(family), pca_dim)
    clf = probes.make_lr(family, c).fit(tf.transform(X_train), y)
    return Router(kind="greedy", transform=tf, clf=clf,
                  n_train=int(len(y)))


def fit_sampled(R_train: np.ndarray, categories_train, *, family: str, c: float,
                pca_dim: int | None) -> Router:
    """The 10-generation router: a probe on aggregated sampled-generation features.

    `R_train` is the sampled feature block (mean/std over the 10 generations,
    built by the sampled aggregator), NOT the greedy features. The constant-column
    filter is fit on train only, like everything else: a column that is constant
    on train but varies on test carries no fitted information, and keeping it
    would let the transform's scaling differ between the two.
    """
    y = y_hi(categories_train)
    if len(np.unique(y)) < 2:
        raise RouterError("sampled router: train rows are all one band")
    keep = np.asarray(R_train.std(axis=0) > 1e-12)
    if not keep.any():
        raise RouterError(
            "sampled router: every feature column is constant on train")
    tf = probes.make_transform(R_train[:, keep], config.kind_of(family), pca_dim)
    clf = probes.make_lr(family, c).fit(tf.transform(R_train[:, keep]), y)
    return Router(kind="sampled", transform=tf, clf=clf, keep=keep,
                  n_train=int(len(y)))


def band_auroc(assigned: np.ndarray, categories) -> float:
    """How well the router recovered the true band -- the deployability number.

    Reported beside every routed result: a routed scorer's advantage is only
    meaningful next to how often its router was right, and a near-perfect router
    is the signature of `oracle` having leaked in somewhere.
    """
    from csx_probe.metrics import safe_auc
    truth = y_hi(categories)
    pred = (np.asarray(assigned) == HI).astype(int)
    return safe_auc(truth, pred)


def error_rate(assigned: np.ndarray, categories) -> float:
    return float(np.mean(np.asarray(assigned) != config.band_of(categories)))
