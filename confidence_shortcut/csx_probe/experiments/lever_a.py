"""Lever A's four gates — what makes prior-corrected pooling a proof.

Lever A replaces z-pooling with per-band Platt scaling minus the band's own prior
log-odds. The claim is not "a number went up": it is that the two designs are the
*same 2-parameter affine family*, so any difference between them is attributable
to how the parameters were chosen and to nothing else. Four gates make that
checkable. All four must pass; `run_pair` returns a non-zero exit code if any
fails.

**Gate 1 — drift.** `platt` must reproduce the base grid wherever the two designs
coincide. A difference here means something moved that was not supposed to move,
and every downstream comparison is then between two things that differ in more
ways than the one under study.

**Gate 2 — the affine identity, in both directions.** Under **oracle** routing
each atomic within-band cell is scored by exactly one expert, and AUROC inside a
single band is invariant to a positive affine map, so `IHvCH` and `ILvCL` must be
**bit-identical** between `z` and `platt_prior` (published: 0.00e+00, 8/8). Under
a **real** router the same cell is scored by a *mixture* of two affine maps,
which is not affine, so the identity **must break**. Asserting only the first
half would pass trivially for a pipeline that never applied routing at all —
which is exactly the failure this gate exists to catch.

**Gate 3 — calibration.** Per-band ECE must fall (~10× in the published run).
This is the mechanism claim: the prior correction works because the calibrated
scores are actually calibrated.

**Gate 4 — train-only provenance.** Both Platt parameters AND the prior must come
from train rows. Fitting either on test would make the calibration
self-fulfilling and the prior a leak; the gate checks the recorded row counts
against the train arm's own size rather than trusting the call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from csx_probe import config, metrics, probes, results
from csx_probe.arms import build as arms_build
from csx_probe.routing import experts as experts_mod, pooling, router as router_mod
from csx_probe.store import build as store_build, read
from csx_probe.store.derive import select as select_mod

TABLE = "lever_a"

WITHIN_BAND: tuple[str, ...] = ("IHvCH", "ILvCL")
GATES: tuple[str, ...] = ("drift", "affine_identity", "ece_falls",
                          "train_only_provenance")


@dataclass
class GateResult:
    gate: str
    passed: bool
    detail: str
    value: float = float("nan")
    tol: float = float("nan")
    rows: list[dict] = field(default_factory=list)


def _cell_auc(scores: np.ndarray, cats: np.ndarray, contrast: str) -> float:
    pos, neg = config.CONTRASTS[contrast]
    m = np.isin(cats, list(pos) + list(neg))
    return metrics.safe_auc(np.isin(cats[m], list(pos)).astype(int), scores[m])


def check_pair(pair: str, cfg: config.RunConfig, *, family: str = "hs_wide",
               segment: str = "all", train_arm: str = "dse_natural",
               test_arm: str = "dse_natural",
               verbose: bool = True) -> list[GateResult]:
    """Run all four gates for one pair and return their results."""
    entry = read.load(pair)
    if family not in entry.available_families():
        raise ValueError(f"{pair}: family {family!r} is not available")
    arms = arms_build.build_all(entry)
    tr, te = arms[train_arm].train, arms[test_arm].test
    ctr, cte = entry.categories[tr], entry.categories[te]

    top_k, pca_dim = ((None, None) if family in config.HS_FAMILIES else
                      (lambda s: (s.top_k, s.pca_dim))(
                          select_mod.select(entry, family, segment,
                                            arms["dse_natural"].train)))
    X = store_build.feature_matrix(entry, family, segment, top_k=top_k)
    c = cfg.c(family)

    exps = experts_mod.fit_experts(X[tr], ctr, family=family, c=c,
                                   pca_dim=pca_dim)
    if any(e is None for e in exps.values()):
        raise ValueError(f"{pair}/{family}: a band expert could not be fit")

    tl, tlab = {}, {}
    for b, e in exps.items():
        m = experts_mod.band_train_mask(ctr, b)
        tl[b] = e.logit(X[tr][m])
        tlab[b] = (ctr[m] == config.BANDS[b][0]).astype(int)

    p_z = pooling.fit("z", tl)
    p_platt = pooling.fit("platt", tl, tlab)
    p_prior = pooling.fit("platt_prior", tl, tlab)
    logits = {b: e.logit(X[te]) for b, e in exps.items()}

    routers = {"oracle": router_mod.fit_oracle()}
    try:
        routers["greedy"] = router_mod.fit_greedy(X[tr], ctr, family=family,
                                                  c=c, pca_dim=pca_dim)
    except router_mod.RouterError:
        pass

    # Each router's per-row band call on the test rows. Built here because the
    # fitted routers need the test features and `oracle` needs the categories --
    # the gate itself takes the resolved assignments, not the routers.
    assignments = {
        name: (r.assign(categories=cte) if name == "oracle" else r.assign(X[te]))
        for name, r in routers.items()
    }

    out = [
        _gate_drift(entry, X, tr, te, ctr, cte, family, segment, c, pca_dim,
                    p_platt, logits, routers, train_arm=train_arm,
                    test_arm=test_arm),
        affine_gate(cte, logits, assignments, p_z, p_prior),
        _gate_ece(p_prior),
        _gate_provenance(p_prior, exps, n_train=len(tr)),
    ]
    if verbose:
        for g in out:
            print(f"  [{pair}/{family}] {g.gate:22s} "
                  f"{'PASS' if g.passed else 'FAIL'}  {g.detail}", flush=True)
    return out


def _gate_drift(entry, X, tr, te, ctr, cte, family, segment, c, pca_dim,
                p_platt, logits, routers, *, train_arm: str = "dse_natural",
                test_arm: str = "dse_natural") -> GateResult:
    """The base grid must be reproduced where the two designs coincide.

    Pooling acts only on the ROUTED scorers, so the generalist -- which does not
    go through a pooler at all -- must land exactly where the transfer grid put
    it. If it moves, something outside the pooling step changed and every
    comparison below is between two things that differ in more ways than the one
    under study.

    Compared against the stored `per_pair_long` row rather than against a second
    in-process computation: recomputing the same call twice and finding it equal
    proves only that the function is deterministic, which was never in doubt.
    When no baseline is on disk the gate reports itself INAPPLICABLE instead of
    passing -- a gate that cannot run has not been satisfied.
    """
    tf, heads = probes.fit_heads(X[tr], ctr, family=family, c=c,
                                 pca_dim=pca_dim)
    ours = _cell_auc(probes.score_pos(heads["g"], tf.transform(X[te])),
                     cte, "IvC")

    base = results.read_unit("per_pair_long", entry.pair)
    if base is None or not len(base):
        return GateResult("drift", False,
                          "INAPPLICABLE: no stored per_pair_long baseline for "
                          "this pair; run the transfer grid first", ours,
                          config.AFFINE_TOL)
    m = base[(base["family"] == family) & (base["segment"] == segment)
             & (base["train_arm"] == train_arm)
             & (base["test_arm"] == test_arm)
             & (base["head"] == "g") & (base["contrast"] == "IvC")]
    if not len(m):
        return GateResult("drift", False,
                          f"INAPPLICABLE: baseline has no g/IvC row for "
                          f"{family}/{segment}/{train_arm}->{test_arm}", ours,
                          config.AFFINE_TOL)
    ref = float(m["AUROC"].iloc[0])
    d = abs(ours - ref)
    return GateResult("drift", d <= config.AFFINE_TOL,
                      f"generalist IvC {ours:.6f} vs stored {ref:.6f} "
                      f"(delta {d:.2e})", d, config.AFFINE_TOL)


def affine_gate(cte, logits, assignments: dict[str, np.ndarray],
                p_z: pooling.Pooler, p_prior: pooling.Pooler) -> GateResult:
    """Gate 2, both directions.

    `assignments[router_name]` is that router's per-row band call on the test
    rows. `oracle` must give a bit-identical within-band AUROC between `z` and
    `platt_prior`; every other router must give a different one.
    """
    detail, ok = [], True
    worst_oracle, best_real = 0.0, 0.0
    for rname, asg in assignments.items():
        s_z = pooling.pooled_scores(p_z, logits, asg)
        s_p = pooling.pooled_scores(p_prior, logits, asg)
        d = max(abs(_cell_auc(s_z, cte, k) - _cell_auc(s_p, cte, k))
                for k in WITHIN_BAND)
        if rname == "oracle":
            worst_oracle = max(worst_oracle, d)
            if d > config.AFFINE_TOL:
                ok = False
                detail.append(f"oracle differs by {d:.2e} (must be 0)")
        else:
            best_real = max(best_real, d)
            if d <= config.AFFINE_TOL:
                ok = False
                detail.append(
                    f"{rname} is identical within band ({d:.2e}), so routing is "
                    f"not being applied")
    if ok:
        detail.append(f"oracle {worst_oracle:.2e} == 0, real router "
                      f"{best_real:.2e} > 0")
    return GateResult("affine_identity", ok, "; ".join(detail), worst_oracle,
                      config.AFFINE_TOL)


def _gate_ece(p_prior: pooling.Pooler) -> GateResult:
    worst, rows = 1.0, []
    for b, m in p_prior.maps.items():
        ratio = (m.ece_raw / m.ece_pooled) if m.ece_pooled > 0 else np.inf
        rows.append({"band": b, "ece_raw": m.ece_raw,
                     "ece_pooled": m.ece_pooled, "ratio": ratio})
        worst = min(worst, ratio)
    ok = all(r["ece_pooled"] < r["ece_raw"] for r in rows) and bool(rows)
    return GateResult("ece_falls", ok,
                      "; ".join(f"{r['band']}: {r['ece_raw']:.4f} -> "
                                f"{r['ece_pooled']:.4f} ({r['ratio']:.1f}x)"
                                for r in rows),
                      worst, 1.0, rows)


def _gate_provenance(p_prior: pooling.Pooler, exps: dict,
                     n_train: int) -> GateResult:
    """Both Platt parameters and the prior must come from train rows only."""
    bad = []
    for b, m in p_prior.maps.items():
        if m.n_train <= 0 or m.n_train > n_train:
            bad.append(f"{b}: pooled on {m.n_train} rows against a train arm of "
                       f"{n_train}")
        if not np.isfinite(m.logit_pi):
            bad.append(f"{b}: prior log-odds is not finite")
        if m.a <= 0:
            bad.append(f"{b}: Platt slope {m.a:.3f} <= 0, so the expert's logit "
                       f"is inverted relative to the label")
    return GateResult("train_only_provenance", not bad,
                      "; ".join(bad) if bad else
                      f"all bands fit on <= {n_train} train rows, slopes > 0")


def to_frame(pair: str, results_list: list[GateResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "pair": pair, "gate": g.gate, "passed": bool(g.passed),
        "value": float(g.value), "tol": float(g.tol), "detail": g.detail,
    } for g in results_list])


def write(pair: str, results_list: list[GateResult]) -> pd.DataFrame:
    df = to_frame(pair, results_list)
    results.write_unit(TABLE, pair, df)
    return df
