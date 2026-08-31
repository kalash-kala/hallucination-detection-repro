"""Does entropy-matching rotate the correctness probe off the confidence axis?

The transfer grid shows that `g` survives matching while `sep` collapses. That is
consistent with two very different stories: `g` genuinely occupies a **different
direction** in feature space when trained under matching, or it is the *same*
direction measured under less favourable conditions. That is a geometric question,
and this is the experiment that asks it.

    theta(alpha) = angle( w_sep(natural) , w_g(alpha rung) )

with the hypothesis `theta(a=1) > theta(a=0)`, monotone in between.

Three things are held fixed, and each one is a way the result could otherwise be
manufactured:

**One basis.** The transform is fit on natural-train and frozen for every rung.
Weight vectors from different arms otherwise live in different coordinates, and
their cosine would mix rotation with rescaling.

**One reference.** `w_sep` is always the natural-trained one. A per-arm reference
could grow `theta` by moving *itself*, which would prove nothing about `g`.

**One metric.** `Sigma` is the covariance of the natural TEST rows, for every
comparison.

`Sigma` is never materialised: at 39,200 features it would be a 12 GB dense matrix,
and `u'Sigma v` is exactly `(Zc u).(Zc v)` with `Zc` the centred natural-test
matrix. The identity is what makes the whole experiment tractable.

Cosines are **signed, never abs**. Both heads are oriented larger-is-worse, so a
positive cosine is the expected sign and a negative one is a real finding rather
than a convention to normalise away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

from csx_probe import config, probes, results
from csx_probe.arms import alpha as alpha_arms, build as arms_build
from csx_probe.metrics import safe_auc
from csx_probe.store import build as store_build, read
from csx_probe.store.derive import select as select_mod

TABLE = "rotation_long"
VERDICT_TABLE = "verdict"

N_BOOT: int = config.frozen()["alpha"]["n_boot"]
N_BOOT_REF: int = config.frozen()["alpha"]["n_boot_ref"]
RIDGE_ALPHA: float = config.frozen()["alpha"]["ridge_alpha"]
PASS_PCT: float = config.frozen()["alpha"]["null_percentile"]
METRICS = (("theta_sigma_sep", "Sigma"), ("theta_euclid_sep", "Euclid"))


def cos_pair(u: np.ndarray, v: np.ndarray, Zc: np.ndarray) -> tuple[float, float]:
    """`(cos_sigma, cos_euclid)`, signed.

    `cos_sigma` is the correlation between the two probes' scores -- a
    behavioural quantity. `cos_euclid` weights every coordinate equally -- a
    literal one. They are co-primary because they can disagree, and a claim that
    holds under only one of them is a weaker claim that should read as such.
    """
    a, b = Zc @ u, Zc @ v
    den_s = np.sqrt(float(a @ a) * float(b @ b))
    den_e = np.sqrt(float(u @ u) * float(v @ v))
    return (float(a @ b) / den_s if den_s > 0 else np.nan,
            float(u @ v) / den_e if den_e > 0 else np.nan)


def theta(c: float) -> float:
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _angles(w: np.ndarray, refs: dict, Zc: np.ndarray) -> dict:
    out = {}
    for name, r in refs.items():
        cs, ce = cos_pair(r, w, Zc)
        out[f"cos_sigma_{name}"] = cs
        out[f"cos_euclid_{name}"] = ce
        out[f"theta_sigma_{name}"] = theta(cs)
        out[f"theta_euclid_{name}"] = theta(ce)
    return out


def _fit_one(Z: np.ndarray, cat: np.ndarray, family: str, c: float,
             target: str, idx: np.ndarray) -> np.ndarray:
    """One logistic fit on a row subset of the frozen basis.

    Module-level and free of closures so `loky` can pickle it. `Z` arrives as a
    joblib memmap, so the workers share one copy of the basis rather than each
    paying for it -- at `lapeigvals` width that is the difference between 650 MB
    and 650 MB x n_jobs.
    """
    cats = config.L_CATS if target == "sep" else config.I_CATS
    y = np.isin(cat[idx], cats).astype(int)
    return probes.coef(probes.make_lr(family, c).fit(Z[idx], y))


def run_pair(pair: str, cfg: config.RunConfig, *, segment: str = "all",
             families: tuple[str, ...] | None = None,
             verbose: bool = True, n_jobs: int = -1) -> pd.DataFrame:
    entry = read.load(pair)
    families = families or cfg.families
    arms = arms_build.build_all(entry)
    rows: list[dict] = []
    for family in families:
        if family not in entry.available_families():
            continue
        rows += _unit(entry, arms, family, segment, cfg, verbose, n_jobs)
    df = pd.DataFrame(rows)
    if len(df):
        results.write_unit(TABLE, pair, df)
    return df


def _unit(entry: read.Entry, arms: dict, family: str, segment: str,
          cfg: config.RunConfig, verbose: bool, n_jobs: int = -1) -> list[dict]:
    nat, m2 = arms["dse_natural"], arms["dse_matched2"]
    top_k, pca_dim = (None, None)
    if family not in config.HS_FAMILIES:
        sel = select_mod.select(entry, family, segment, nat.train)
        top_k, pca_dim = sel.top_k, sel.pca_dim

    X = store_build.feature_matrix(entry, family, segment, top_k=top_k)
    kind = config.kind_of(family)
    c = cfg.c(family)

    # the frozen basis: natural TRAIN
    tf = probes.make_transform(X[nat.train], kind, pca_dim)
    Z_tr = tf.transform(X[nat.train]).astype(np.float64)
    # the frozen metric: natural TEST
    Z_te = tf.transform(X[nat.test]).astype(np.float64)
    del X
    Zc = Z_te - Z_te.mean(0, keepdims=True)
    y_te = np.isin(entry.categories[nat.test], config.I_CATS).astype(int)

    cat_tr = entry.categories[nat.train]
    ent_tr = entry.entropy[nat.train]
    # position within Z_tr, so a rung's global row ids can index the frozen basis
    pos = {int(r): k for k, r in enumerate(nat.train)}

    w_sep = probes.coef(probes.make_lr(family, c).fit(
        Z_tr, np.isin(cat_tr, config.L_CATS).astype(int)))
    # The entropy reference carries no LR estimation noise, so it cross-checks
    # whether instability in the reference itself is driving any movement.
    w_ent = np.asarray(Ridge(alpha=RIDGE_ALPHA).fit(Z_tr, ent_tr).coef_,
                       dtype=float).ravel()
    refs = {"sep": w_sep, "entropy": w_ent}

    out: list[dict] = []

    def emit(kindtag: str, a: float, draw: int, w: np.ndarray, n: int) -> None:
        out.append({
            "pair": entry.pair, "model": entry.model, "dataset": entry.dataset,
            "family": family, "segment": segment, "C": float(c),
            "c_mode": cfg.c_mode, "kind": kindtag, "alpha": a, "draw": draw,
            "n_train": int(n), "auroc_nat_IvC": safe_auc(y_te, Zc @ w),
            "norm_w": float(np.linalg.norm(w)), **_angles(w, refs, Zc),
        })

    ladder, counts = alpha_arms.build_ladder(entry, nat, m2, "train")
    alpha_arms.assert_nested(ladder)

    # ── plan ────────────────────────────────────────────────────────────────
    # Every fit is independent, but the bootstrap row sets are NOT: they come off
    # one seeded stream per rung. So the index arrays are drawn here, strictly in
    # order, and only the fits are handed out. Drawing them inside the workers
    # would consume the stream in completion order and silently change every
    # bootstrap CI in the programme.
    jobs: list[tuple[str, str, float, int, np.ndarray]] = []  # target,kind,a,draw,idx
    n = len(Z_tr)
    for a in alpha_arms.ALPHAS:
        idx = np.array([pos[int(r)] for r in ladder[a]], dtype=int)
        if len(np.unique(np.isin(cat_tr[idx], config.I_CATS))) < 2:
            continue
        jobs.append(("g", "alpha", a, -1, idx))

        rng = np.random.default_rng(config.SEED + int(a * 1000))
        for b in range(N_BOOT):
            jobs.append(("g", "boot", a, b,
                         idx[rng.integers(0, len(idx), len(idx))]))

        for p in range(alpha_arms.N_PLACEBO):
            prow = alpha_arms.build_placebo(entry, nat, counts[a], p, "train")
            jobs.append(("g", "placebo", a, p,
                         np.array([pos[int(r)] for r in prow], dtype=int)))

    # reference stability: bootstrap w_sep against itself
    rng = np.random.default_rng(config.SEED + 7)
    for b in range(N_BOOT_REF):
        jobs.append(("sep", "refboot", np.nan, b, rng.integers(0, n, n)))

    # ── execute ─────────────────────────────────────────────────────────────
    # `inner_max_num_threads=1` is the whole point: MKL otherwise grabs every
    # core for each individual fit, which on a 2100 x 7169 matrix scales badly
    # and leaves the workers fighting each other for the same cores.
    if verbose:
        print(f"  [{entry.pair}/{family}] {len(jobs)} fits, n_jobs={n_jobs}",
              flush=True)
    ws = Parallel(n_jobs=n_jobs, backend="loky", inner_max_num_threads=1,
                  max_nbytes="1M", verbose=0)(
        delayed(_fit_one)(Z_tr, cat_tr, family, c, tgt, ix)
        for tgt, _, _, _, ix in jobs)

    # ── collect, in the planned order ───────────────────────────────────────
    for (tgt, kindtag, a, draw, ix), w in zip(jobs, ws):
        if kindtag == "refboot":
            cs, ce = cos_pair(w_sep, w, Zc)
            out.append({
                "pair": entry.pair, "model": entry.model,
                "dataset": entry.dataset, "family": family, "segment": segment,
                "C": float(c), "c_mode": cfg.c_mode, "kind": "refboot",
                "alpha": np.nan, "draw": draw, "n_train": n,
                "auroc_nat_IvC": np.nan, "norm_w": float(np.linalg.norm(w)),
                "cos_sigma_sep": cs, "cos_euclid_sep": ce,
                "theta_sigma_sep": theta(cs), "theta_euclid_sep": theta(ce),
            })
        else:
            emit(kindtag, a, draw, w, len(ix))
    return out


def verdict(df: pd.DataFrame) -> pd.DataFrame:
    """Per `(pair, family, metric)`: the observed rotation against its own null.

    The null is formed as all 400 outer differences
    `placebo(a=1)_i - placebo(a=0)_j`, which is the correct general form: the
    placebo draws at the two rungs are independent, so pairing them by index
    would impose a correspondence that does not exist.

    **On this cohort that outer product is inert, and the docstring used to
    overclaim it.** `build_placebo` draws `min(k, len(avail))` rows without
    replacement, and at `a=0` the target count `k` already equals the full
    natural per-cell count -- so every draw returns the entire pool. All 20
    `placebo(a=0)` draws are therefore identical to each other AND to the real
    `a=0` arm (verified: `ptp(p0) == 0` and `theta_real(0) - p0 == 0` exactly, on
    all 63 cells; pinned by `test_placebo_at_alpha0_is_degenerate`).

    With `p0` constant the null is a location shift of `p1`, and the test
    collapses algebraically to

        delta > null_p95   <=>   theta(1) > p95(placebo(a=1))

    Outer, paired and direct forms agree to 1.42e-13 degrees across 126 cells.
    The outer form is kept because it stays correct if a future cohort ever makes
    `a=0` non-degenerate (any pair whose matched2 arm exceeds the natural count
    in some cell), but it must not be described as load-bearing here.

    Note the estimator this leaves: p95 from 20 draws interpolates to index
    `0.95 * 19 = 18.05`, i.e. 95% of the 2nd-largest draw plus 5% of the max.
    Only one headline claim is marginal under that resolution (`attnlogdet`
    under Sigma, 7/9 PASS but holding in 72.7% of placebo resamples).

    The direction is deliberate and is **not** interchangeable with a lower
    tail. The null is "matching adds nothing beyond shrinkage"; under it
    `theta(1)` is just another placebo draw, so rejection requires the UPPER
    tail. A `p05` test is the rejection region for the opposite alternative
    ("matching rotates *less* than shrinkage") and would certify the cells that
    currently, correctly, fail.

    The CI on `delta`, by contrast, IS paired by draw index: there the two
    bootstrap refits share a resampled row set, so `b1 - b0` is a genuinely
    paired difference and pairing removes variance that is common to both.
    """
    out = []
    for (fam, pair, seg), d in df.groupby(["family", "pair", "segment"]):
        for col, label in METRICS:
            obs = {}
            for a in alpha_arms.ALPHAS:
                s = d[(d["kind"] == "alpha") & (d["alpha"] == a)][col]
                obs[a] = float(s.iloc[0]) if len(s) else np.nan
            delta = obs[1.0] - obs[0.0]
            p0 = d[(d["kind"] == "placebo") & (d["alpha"] == 0.0)][col].values
            p1 = d[(d["kind"] == "placebo") & (d["alpha"] == 1.0)][col].values
            if not len(p0) or not len(p1):
                continue
            null = np.subtract.outer(p1, p0).ravel()
            thr = float(np.percentile(null, PASS_PCT))
            b0 = d[(d["kind"] == "boot") & (d["alpha"] == 0.0)][col].values
            b1 = d[(d["kind"] == "boot") & (d["alpha"] == 1.0)][col].values
            bd = (b1 - b0) if len(b0) == len(b1) and len(b0) else np.array([np.nan])
            out.append({
                "pair": pair, "family": fam, "segment": seg, "metric": label,
                "C": float(d["C"].iloc[0]), "c_mode": d["c_mode"].iloc[0],
                "theta_a0": obs[0.0], "theta_a1": obs[1.0], "delta": delta,
                "delta_lo": float(np.nanpercentile(bd, 2.5)),
                "delta_hi": float(np.nanpercentile(bd, 97.5)),
                "null_med": float(np.median(null)), "null_p95": thr,
                "passes": bool(delta > thr),
                **{f"theta_a{int(a * 100):03d}": obs[a]
                   for a in alpha_arms.ALPHAS},
            })
    return pd.DataFrame(out)


def write_verdict(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    v = verdict(df)
    if len(v):
        results.write_unit(VERDICT_TABLE, pair, v)
    return v
