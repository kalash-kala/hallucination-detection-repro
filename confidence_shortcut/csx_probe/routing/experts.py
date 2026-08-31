"""The two band experts.

One probe per confidence band, each fit only on that band's rows. Reproduces
`54_routed_vs_generalist_fixedC.fit_expert` / `expert_scores`.

**The transform is refit inside each band, not shared.** A band subset has its
own covariance -- that is the entire premise of specialising -- so reusing the
natural arm's transform would push the experts' inputs back into a shared space
and leak band structure into features that are supposed to be band-local. The
cost is that the two experts' raw logits are no longer commensurable, which is
what `pooling.py` exists to fix, and paying that cost explicitly is better than
hiding it in a shared scaler.

**Every test row is scored by BOTH experts.** With a *predicted* band we cannot
restrict scoring to the rows a band actually owns -- the router may be wrong, and
a row it misroutes still has to receive a score. So both experts score everything
and the router picks per row; `same_band` selection applies to TRAINING only.

**Two training modes, and the difference is the leakage trap.**

    mode="same_band"   train rows chosen by the row's TRUE band
    mode="hier"        train rows chosen by the router's OUT-OF-FOLD prediction

`hier` exists because an expert deployed behind a real router is asked to score
the population the *router* sends it, not the population the true band defines.
Training it on true bands means every misrouted row at test time is out of
distribution. But the out-of-fold discipline is load-bearing: a router fit on all
of train and then asked about those same rows predicts at *training* accuracy, so
its predicted bands nearly equal the true bands and `hier` silently collapses
into `same_band`. The collapse leaves no trace in the output -- the numbers just
quietly become the other design's. `tests/test_routing.py` asserts the
disagreement rate tracks the router's held-out error rate rather than merely
being non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

from csx_probe import config, probes


class ExpertError(Exception):
    """A band expert cannot be fit. The message names the shortfall."""


@dataclass(frozen=True)
class Expert:
    """One band's fitted pipeline, plus the z statistics of its own train logits.

    `mu`/`sd` are taken over the SAME rows the expert was fit on, which is what
    makes `z` pooling label-blind: it reads the shape of the logit distribution
    the expert produced on its own training population, never the labels.
    """

    band: str
    transform: object
    clf: object
    mu: float
    sd: float
    n_train: int
    neg_mode: str

    def logit(self, X: np.ndarray) -> np.ndarray:
        """The raw decision function -- the quantity every pooler is affine in."""
        return np.asarray(
            self.clf.decision_function(self.transform.transform(X)), dtype=float)

    def proba(self, X: np.ndarray) -> np.ndarray:
        """`predict_proba` for the positive (incorrect) class.

        Indexed through `classes_` rather than assuming column 1: an expert fit
        on a band where one class is absent would silently return the wrong
        column, and that failure looks like a merely weak probe.
        """
        p = self.clf.predict_proba(self.transform.transform(X))
        return np.asarray(p[:, list(self.clf.classes_).index(1)], dtype=float)


def _zstats(logits: np.ndarray) -> tuple[float, float]:
    mu, sd = float(np.mean(logits)), float(np.std(logits))
    return mu, (sd if sd > 1e-12 else 1.0)


def band_train_mask(categories, band: str, *, neg_mode: str = "same_band"
                    ) -> np.ndarray:
    """Which train rows this band's expert is fit on.

    `same_band` pairs the band's incorrect rows against its OWN correct rows;
    `pooled` pairs them against both bands' correct rows (the M2 variant, kept
    because the expert-swap ablation needs it).
    """
    inc, cor = config.BANDS[band]
    neg = (cor,) if neg_mode == "same_band" else config.C_POOL
    return np.isin(np.asarray(categories), (inc,) + tuple(neg))


def fit_one(X_train: np.ndarray, categories_train, band: str, *, family: str,
            c: float, pca_dim: int | None, neg_mode: str = "same_band",
            select: np.ndarray | None = None) -> Expert | None:
    """One band expert, or None if a class is too thin to fit honestly.

    `select` overrides the true-band row selection with an arbitrary boolean
    mask -- that is how `mode="hier"` hands in the router's out-of-fold
    assignment. The target is still the TRUE label: routing decides who is
    trained on which rows, never what the right answer is.

    Returning None rather than fitting a degenerate probe is deliberate. With
    `class_weight='balanced'`, a band holding three incorrect rows will fit
    happily and return confident-looking scores; that is an undefined
    measurement wearing the costume of a weak one.
    """
    cats = np.asarray(categories_train)
    inc, _cor = config.BANDS[band]
    m = band_train_mask(cats, band, neg_mode=neg_mode) if select is None \
        else np.asarray(select, dtype=bool)
    if m.sum() == 0:
        return None
    y = (cats[m] == inc).astype(int)
    if min(int(y.sum()), int((y == 0).sum())) < config.MIN_PER_CLASS:
        return None

    tf = probes.make_transform(X_train[m], config.kind_of(family), pca_dim)
    Z = tf.transform(X_train[m])
    clf = probes.make_lr(family, c).fit(Z, y)
    mu, sd = _zstats(clf.decision_function(Z))
    return Expert(band=band, transform=tf, clf=clf, mu=mu, sd=sd,
                  n_train=int(m.sum()), neg_mode=neg_mode)


def fit_experts(X_train: np.ndarray, categories_train, *, family: str, c: float,
                pca_dim: int | None, mode: str = "same_band",
                neg_mode: str = "same_band",
                oof_bands: np.ndarray | None = None
                ) -> dict[str, Expert | None]:
    """`{band: Expert or None}` for every band.

    `mode="hier"` requires `oof_bands` -- the router's out-of-fold band call for
    each train row, from `oof_router_bands`. Refusing to invent them here is the
    point: a caller that forgets gets an error rather than a silent fallback to
    true bands, which is exactly the collapse this design is guarding against.
    """
    if mode not in ("same_band", "hier"):
        raise ExpertError(f"unknown expert mode {mode!r}")
    if mode == "hier" and oof_bands is None:
        raise ExpertError(
            "mode='hier' needs out-of-fold router bands; pass `oof_bands` from "
            "oof_router_bands(). Falling back to true bands would collapse hier "
            "into same_band with no trace in the output.")

    out: dict[str, Expert | None] = {}
    for band in config.BAND_ORDER:
        select = None
        if mode == "hier":
            inc, cor = config.BANDS[band]
            pool = (cor,) if neg_mode == "same_band" else config.C_POOL
            # Routed to this band AND admissible as one of its two classes.
            select = (np.asarray(oof_bands) == band) & np.isin(
                np.asarray(categories_train), (inc,) + tuple(pool))
        out[band] = fit_one(X_train, categories_train, band, family=family, c=c,
                            pca_dim=pca_dim, neg_mode=neg_mode, select=select)
    return out


def oof_router_bands(X_train: np.ndarray, categories_train, *, family: str,
                     c: float, pca_dim: int | None,
                     n_folds: int | None = None,
                     seed: int | None = None) -> np.ndarray:
    """Out-of-fold band predictions for every train row.

    **The whole pipeline is refit inside each fold** -- variance filter and
    transform included, not just the classifier. A transform fit on all of train
    and reused per fold would carry every row's own contribution into the fold
    that is supposed to be holding it out, which is the same leak in a quieter
    place.

    Folds are stratified on the band itself, so each fold sees both bands even
    when the split is lopsided.
    """
    n_folds = n_folds or config.HIER_FOLDS
    seed = config.SEED if seed is None else seed
    y_hi = (config.band_of(categories_train) == config.BAND_ORDER[0]).astype(int)
    out = np.empty(len(y_hi), dtype=object)

    if len(np.unique(y_hi)) < 2:
        out[:] = config.BAND_ORDER[0] if y_hi.all() else config.BAND_ORDER[1]
        return out

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X_train, y_hi):
        tf = probes.make_transform(X_train[tr], config.kind_of(family), pca_dim)
        clf = probes.make_lr(family, c).fit(tf.transform(X_train[tr]), y_hi[tr])
        p_hi = clf.predict_proba(tf.transform(X_train[te]))[
            :, list(clf.classes_).index(1)]
        out[te] = np.where(p_hi >= config.ROUTE_THRESHOLD,
                           config.BAND_ORDER[0], config.BAND_ORDER[1])
    return out


def generalist(X_train: np.ndarray, categories_train, *, family: str, c: float,
               pca_dim: int | None):
    """The single-probe baseline the routed design is argued against.

    Identical fitting to one expert, but on ALL train rows with the pooled
    negatives -- so any difference between it and the routed scorer is about
    band-locality and nothing else.
    """
    cats = np.asarray(categories_train)
    y = np.isin(cats, config.I_CATS).astype(int)
    tf = probes.make_transform(X_train, config.kind_of(family), pca_dim)
    Z = tf.transform(X_train)
    clf = probes.make_lr(family, c).fit(Z, y)
    mu, sd = _zstats(clf.decision_function(Z))
    return Expert(band="ALL", transform=tf, clf=clf, mu=mu, sd=sd,
                  n_train=int(len(cats)), neg_mode="pooled")
