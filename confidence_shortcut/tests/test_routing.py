"""Experts, routers and pooling — and the two traps that fail silently.

**The `hier` collapse.** If the out-of-fold router labelling is fit on all of
train and then asked about those same rows, it predicts at training accuracy, its
bands nearly equal the true bands, and `hier` quietly becomes `same_band`. The
numbers stay plausible; only the design changes. So the test asserts the
disagreement rate *tracks the router's held-out error rate*, not merely that it
is non-zero.

**The affine identity.** Under oracle routing each atomic within-band cell is
scored by exactly one expert, and AUROC inside one band is invariant to a
positive affine map, so `z` and `platt_prior` must agree bit-for-bit there. Under
a real router the same cell is scored by a *mixture* of two affine maps, which is
not affine, so the identity must BREAK — and if it does not, routing is not
actually being applied.
"""

from __future__ import annotations

import numpy as np
import pytest

from csx_probe import config, metrics
from csx_probe.routing import experts, pooling, router

FAM, C = "hs_wide", 1e-4


def synth(n=1200, d=25, *, seed=0, band_signal=1.5):
    """Rows where correctness and band are both learnable but distinct.

    `band_signal` controls how separable the bands are, which is what lets one
    test dial the router from near-perfect down to genuinely error-prone.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    hi = rng.random(n) < 0.55
    inc = (X[:, 0] + np.where(hi, 0.8, -0.8) + rng.normal(scale=0.7, size=n)) > 0.4
    cats = np.where(hi, np.where(inc, "IH", "CH"), np.where(inc, "IL", "CL"))
    X[:, 3] += np.where(hi, band_signal, -band_signal)
    return X, cats


def _fit_all(X, cats, tr):
    exps = experts.fit_experts(X[tr], cats[tr], family=FAM, c=C, pca_dim=None)
    tl, tlab = {}, {}
    for b, e in exps.items():
        m = experts.band_train_mask(cats[tr], b)
        tl[b] = e.logit(X[tr][m])
        tlab[b] = (cats[tr][m] == config.BANDS[b][0]).astype(int)
    return exps, tl, tlab


def _cell_auc(scores, cats, pos, neg):
    m = np.isin(cats, [pos, neg])
    return metrics.safe_auc((cats[m] == pos).astype(int), scores[m])


# ── experts ──────────────────────────────────────────────────────────────────

def test_expert_transform_is_refit_per_band():
    """Each band's expert gets its OWN transform, not a shared one."""
    X, cats = synth()
    tr = np.arange(800)
    exps, _, _ = _fit_all(X, cats, tr)
    hi, lo = exps["HI"], exps["LO"]
    assert hi.transform is not lo.transform
    assert not np.allclose(hi.transform.mean_, lo.transform.mean_), (
        "band subsets have different covariance; identical means would mean the "
        "transform was fit on the shared population")


def test_thin_band_returns_none_rather_than_a_degenerate_fit():
    """A band with too few of either class is undefined, not merely weak."""
    X, cats = synth(n=400)
    cats = cats.copy()
    cats[np.isin(cats, ["IH"])] = "CH"          # wipe out HI's positive class
    e = experts.fit_one(X, cats, "HI", family=FAM, c=C, pca_dim=None)
    assert e is None


def test_hier_requires_oof_bands():
    """Forgetting the OOF labels must raise, never silently use true bands."""
    X, cats = synth()
    with pytest.raises(experts.ExpertError, match="oof_bands"):
        experts.fit_experts(X, cats, family=FAM, c=C, pca_dim=None, mode="hier")


def test_hier_oof_bands_differ_from_true_bands():
    """The collapse guard: OOF disagreement must track held-out router error.

    A router refit inside each fold is being asked about rows it did not see, so
    its error rate on them is a genuine held-out rate. If the OOF labelling were
    (wrongly) produced by a router fit on all of train, disagreement would fall
    far below that rate -- toward zero -- and `hier` would be `same_band` under
    another name.
    """
    X, cats = synth(band_signal=0.6)            # deliberately fallible router
    tr = np.arange(900)
    oof = experts.oof_router_bands(X[tr], cats[tr], family=FAM, c=C,
                                   pca_dim=None)
    disagree = float(np.mean(oof != config.band_of(cats[tr])))

    # Independent estimate of the same quantity: a held-out split, one fit.
    cut = 600
    r = router.fit_greedy(X[tr][:cut], cats[tr][:cut], family=FAM, c=C,
                          pca_dim=None)
    held_out = router.error_rate(r.assign(X[tr][cut:]), cats[tr][cut:])

    assert disagree > 0.02, "OOF bands collapsed onto the true bands"
    assert disagree == pytest.approx(held_out, abs=0.10), (
        f"OOF disagreement {disagree:.3f} should track the router's held-out "
        f"error {held_out:.3f}; a large gap means the folds are leaking")


def test_hier_experts_train_on_router_assigned_rows():
    X, cats = synth(band_signal=0.6)
    tr = np.arange(900)
    oof = experts.oof_router_bands(X[tr], cats[tr], family=FAM, c=C, pca_dim=None)
    hier = experts.fit_experts(X[tr], cats[tr], family=FAM, c=C, pca_dim=None,
                               mode="hier", oof_bands=oof)
    same = experts.fit_experts(X[tr], cats[tr], family=FAM, c=C, pca_dim=None)
    assert hier["HI"].n_train != same["HI"].n_train, (
        "hier experts must see the population the ROUTER sends them")


# ── routers ──────────────────────────────────────────────────────────────────

def test_oracle_needs_categories_and_fitted_routers_need_features():
    X, cats = synth()
    o = router.fit_oracle()
    with pytest.raises(router.RouterError):
        o.assign(X)                                  # no categories
    np.testing.assert_array_equal(o.assign(categories=cats),
                                  config.band_of(cats))
    g = router.fit_greedy(X, cats, family=FAM, c=C, pca_dim=None)
    with pytest.raises(router.RouterError):
        g.assign(None)


def test_greedy_router_is_the_sep_probe_with_labels_swapped():
    """M14: `y_hi == 1 - sep`. Same target, opposite labels."""
    from csx_probe import probes
    _X, cats = synth()
    np.testing.assert_array_equal(router.y_hi(cats),
                                  1 - probes.target(cats, "sep"))


def test_oracle_router_is_never_presented():
    assert "oracle" in config.ROUTERS
    assert "oracle" not in config.PRESENTED_ROUTERS


# ── pooling ──────────────────────────────────────────────────────────────────

def test_platt_needs_labels_but_z_does_not():
    X, cats = synth()
    tr = np.arange(800)
    _e, tl, _tlab = _fit_all(X, cats, tr)
    pooling.fit("z", tl)                                    # label-blind, fine
    with pytest.raises(pooling.PoolingError, match="train labels"):
        pooling.fit("platt_prior", tl)


def test_platt_prior_subtracts_the_band_base_rate():
    """`platt_prior` is exactly `platt` shifted by −logit π_b."""
    X, cats = synth()
    tr = np.arange(800)
    _e, tl, tlab = _fit_all(X, cats, tr)
    p = pooling.fit("platt", tl, tlab)
    q = pooling.fit("platt_prior", tl, tlab)
    for b in p.maps:
        assert q.maps[b].a == pytest.approx(p.maps[b].a)
        assert q.maps[b].b == pytest.approx(p.maps[b].b - p.maps[b].logit_pi)


def test_ece_falls_after_platt_calibration():
    X, cats = synth()
    tr = np.arange(800)
    _e, tl, tlab = _fit_all(X, cats, tr)
    q = pooling.fit("platt_prior", tl, tlab)
    for b, m in q.maps.items():
        assert m.ece_pooled < m.ece_raw, f"band {b}: calibration did not improve"


def test_every_expert_must_score_every_test_row():
    X, cats = synth()
    tr, te = np.arange(800), np.arange(800, len(cats))
    exps, tl, _ = _fit_all(X, cats, tr)
    pz = pooling.fit("z", tl)
    bad = {b: e.logit(X[te])[:-5] for b, e in exps.items()}
    with pytest.raises(pooling.PoolingError, match="every expert"):
        pooling.pooled_scores(pz, bad, config.band_of(cats[te]))


# ── the affine identity: the gate that proves routing is applied ─────────────

@pytest.mark.parametrize("cell", [("IH", "CH"), ("IL", "CL")])
def test_affine_identity_holds_exactly_under_oracle(cell):
    """One expert per cell => a positive affine map => identical AUROC."""
    X, cats = synth()
    tr, te = np.arange(800), np.arange(800, len(cats))
    exps, tl, tlab = _fit_all(X, cats, tr)
    lo = {b: e.logit(X[te]) for b, e in exps.items()}
    asg = router.fit_oracle().assign(categories=cats[te])

    sz = pooling.pooled_scores(pooling.fit("z", tl), lo, asg)
    sp = pooling.pooled_scores(pooling.fit("platt_prior", tl, tlab), lo, asg)
    pos, neg = cell
    assert abs(_cell_auc(sz, cats[te], pos, neg)
               - _cell_auc(sp, cats[te], pos, neg)) <= config.AFFINE_TOL


def test_affine_identity_breaks_under_a_real_router():
    """A mixture of two affine maps is not affine — and if it were, the router
    would not be doing anything."""
    X, cats = synth(band_signal=0.6)
    tr, te = np.arange(800), np.arange(800, len(cats))
    exps, tl, tlab = _fit_all(X, cats, tr)
    lo = {b: e.logit(X[te]) for b, e in exps.items()}
    asg = router.fit_greedy(X[tr], cats[tr], family=FAM, c=C,
                            pca_dim=None).assign(X[te])

    sz = pooling.pooled_scores(pooling.fit("z", tl), lo, asg)
    sp = pooling.pooled_scores(pooling.fit("platt_prior", tl, tlab), lo, asg)
    d = max(abs(_cell_auc(sz, cats[te], p, n) - _cell_auc(sp, cats[te], p, n))
            for p, n in (("IH", "CH"), ("IL", "CL")))
    assert d > config.AFFINE_TOL, (
        "the within-band identity survived a fallible router, which means the "
        "routing step is not actually being applied")


def test_proba_is_not_in_the_affine_family():
    """The negative control: `proba` re-imports the band offset."""
    X, cats = synth()
    tr, te = np.arange(800), np.arange(800, len(cats))
    exps, tl, _ = _fit_all(X, cats, tr)
    lo = {b: e.logit(X[te]) for b, e in exps.items()}
    asg = router.fit_oracle().assign(categories=cats[te])
    sz = pooling.pooled_scores(pooling.fit("z", tl), lo, asg)
    sp = pooling.pooled_scores(pooling.fit("proba", {}), lo, asg)
    # Monotone within band, so within-band cells still agree ...
    assert abs(_cell_auc(sz, cats[te], "IH", "CH")
               - _cell_auc(sp, cats[te], "IH", "CH")) <= config.AFFINE_TOL
    # ... but the cross-band cell does not, because the offset came back.
    assert abs(_cell_auc(sz, cats[te], "IH", "CL")
               - _cell_auc(sp, cats[te], "IH", "CL")) > config.AFFINE_TOL
