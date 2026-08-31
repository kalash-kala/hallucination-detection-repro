"""M23 -- is the band router metric-agnostic, or did it just learn DSE?

**The reviewer's question.** The sampled router recovers the confidence band from
the internal states of 10 generations at high AUROC. But the band is *defined* by
thresholding discrete semantic entropy, so the whole result could be an artefact
of that one statistic. A deployment is free to decide "low confidence" with
eccentricity, LUQ, degree, or anything else.

So: train the router against the band of metric A, score it against the band of
metric B, for all 64 ordered pairs, independently within each arm. The diagonal
reproduces the existing routed result. The **off-diagonal** is the answer -- a
router that had only learned "DSE" would fall apart there.

    router AUROC(A -> B) = AUROC( band_B(test), router_A.p_hi(test features) )

**Nothing about the router changes.** Same sampled features, same aggregation,
same LR, same frozen transform. The single hook is the *label*. `band_thresholds`
(M21) supplies each metric's own `best_split` threshold; this module recomputes
the `category` field from it and changes nothing else.

**What makes the grid legal.** Re-splitting per metric would give each metric its
own train/test partition, so `natural_A.train` would intersect `natural_B.test`
and every off-diagonal cell would leak. Instead the DSE partition is reused
**verbatim**: `dse_natural` is the frozen arm object itself, not a rebuild, and
`balanced2`/`matched2` are re-quota'd per metric but only ever *subset* it. Then
`natural_A.train == natural_B.train` as row sets for all A, B, and every
cross-metric cell is leak-free by construction. `assert_leak_free` checks it per
pair rather than trusting the argument.

**The honest denominator comes first.** All 8 metrics are computed from the same
10 generations, so their bands agree substantially and some transfer is free.
`agreement()` reports raw band agreement and Cohen's kappa for every metric pair
so a reader can see how much of the off-diagonal is redundancy rather than the
router generalising. The QA run had to concede that agreement genuinely predicts
the drop (Spearman +0.62 to +0.70) and that the claim is therefore *redundancy
modulates the cost, it does not create the effect*. We concede the same or we
report a different finding -- we do not quietly drop the concession.

**`lexical_sim` is the check that matters**: the only metric built without NLI.
If transfer survives to it, the mechanism is not an entailment artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from csx_probe import config, metrics as metrics_mod, results
from csx_probe.arms import build as arms_build, common as arms_common
from csx_probe.experiments import band_thresholds
from csx_probe.routing import router as router_mod
from csx_probe.store import read, sampled as sampled_mod

TABLE = "metric_router_long"
AGREEMENT_TABLE = "band_agreement"

METRICS: tuple[str, ...] = band_thresholds.METRICS
ARMS: tuple[str, ...] = ("dse_natural", "dse_balanced2", "dse_matched2")

# `cloud` and `mean_std` are the two published router inputs. `cloud` is carried
# first because it is the interesting one: 10 numbers against ~2d.
SCHEMES: tuple[str, ...] = ("cloud", "mean_std")

# `matched2` matches on exact strata, which is meaningless for a continuous
# score -- every row would be its own stratum and the arm would collapse. Six of
# the eight metrics are continuous, so those are binned to global quantiles
# first. 40 is the published bin count.
N_QUANTILE_BINS: int = 40
# At or below this many distinct values a metric is treated as naturally
# discrete and matched on its RAW value, as `dse` is.
DISCRETE_MAX_DISTINCT: int = 200


class MetricRouterError(Exception):
    """The metric-router grid cannot be built. The message names why."""


def _stratum_scalar(x: np.ndarray, metric: str) -> np.ndarray:
    """The value `matched2` will key strata on, for one metric.

    Discrete metrics keep their raw value (that is what `dse` does, and it is
    what makes the arm exactly reproduce the published one). Continuous metrics
    are mapped to global quantile bins -- global, not per-band, so that the same
    scalar means the same thing in both bands and a stratum is comparable across
    them.
    """
    x = np.asarray(x, dtype=float)
    if len(np.unique(x)) <= DISCRETE_MAX_DISTINCT:
        return x
    # `rank` then bin, so ties land together and the bins are equal-count rather
    # than equal-width -- the six continuous metrics are all heavily skewed.
    r = pd.Series(x).rank(method="dense")
    q = np.ceil(r / len(np.unique(r)) * N_QUANTILE_BINS)
    return q.to_numpy(dtype=float)


def relabel(entry, metric: str, *, n_cuts: int = band_thresholds.bs.N_CUTS
            ) -> tuple[np.ndarray, np.ndarray]:
    """`(categories, stratum_scalar)` for one metric, aligned to `entry.rows`.

    Only the BAND half of the category is recomputed. Correctness is a property
    of the answer, not of the uncertainty metric, so it is read off the existing
    DSE category (`I*`/`C*`) rather than re-derived -- which also guarantees the
    two labellings differ in exactly one coordinate.
    """
    df = band_thresholds.load_metrics(entry.pair)
    if metric not in df.columns:
        raise MetricRouterError(
            f"{entry.pair}: metric {metric!r} absent; run M20 then M21")
    order = pd.Index(df["id"].to_numpy(dtype=object)).get_indexer(
        pd.Index(np.asarray(entry.ids, dtype=object)))
    if (order < 0).any():
        raise MetricRouterError(
            f"{entry.pair}: {int((order < 0).sum())} roster ids missing from "
            f"the metric table; M20/M21 and rows.parquet disagree")

    x = df[metric].to_numpy(dtype=float)[order]
    band = band_thresholds.bs.band_of(x, band_thresholds.bs.best_split(x, n_cuts))
    correct = np.array([c[0] for c in np.asarray(entry.categories)])  # 'I' / 'C'
    cats = np.char.add(correct.astype(str),
                       np.where(band == "HI", "H", "L"))
    return cats, _stratum_scalar(x, metric)


class _RelabelledEntry:
    """An `Entry`-shaped view carrying one metric's bands instead of DSE's.

    A thin shim rather than a copy: the arm builders read only `pair`, `rows`,
    `categories` and `entropy`, and narrowing the surface to those makes it
    obvious that nothing else about the pair is being reinterpreted. `entropy`
    carries the metric's stratum scalar because that is the field `matched2`
    keys strata on -- the plan's "strata are that metric's own values".
    """

    def __init__(self, entry, metric: str):
        self._e = entry
        self.metric = metric
        self.categories, self.entropy = relabel(entry, metric)

    def __getattr__(self, name):
        return getattr(self._e, name)


def arms_for(entry, metric: str, frozen_nat) -> dict:
    """The three arms under `metric`'s bands, all inside the frozen partition.

    `dse_natural` is the frozen arm OBJECT, returned unchanged -- not rebuilt
    from the relabelled categories. Rebuilding it would stratify on the new
    bands and produce a different partition, which is precisely the leak this
    experiment must not have.
    """
    view = _RelabelledEntry(entry, metric)
    rng = arms_common.pair_rng(f"{entry.pair}|{metric}")
    out = {"dse_natural": frozen_nat}
    for arm in ARMS:
        if arm == "dse_natural":
            continue
        fn = arms_build.BUILDERS.get(arm)
        if fn is None:
            raise MetricRouterError(f"unknown arm {arm!r}")
        try:
            out[arm] = fn(view, frozen_nat, rng)
        except Exception as e:                       # noqa: BLE001
            # A metric whose bands are too skewed to quota is a real, reportable
            # state, not a crash: skip the arm and let the grid record its
            # absence rather than failing the whole pair.
            out[arm] = None
            del e
    return out


def assert_leak_free(by_metric: dict[str, dict]) -> None:
    """No train row of any metric/arm may appear in any test row of any other.

    Checked across all metric pairs and arms, because the containment argument
    is what the whole grid rests on. Cheap next to the fits it protects.
    """
    trains, tests = {}, {}
    for m, arms in by_metric.items():
        for a, arm in arms.items():
            if arm is None:
                continue
            trains[(m, a)] = set(arm.train.tolist())
            tests[(m, a)] = set(arm.test.tolist())
    for ka, tr in trains.items():
        for kb, te in tests.items():
            bad = tr & te
            if bad:
                raise MetricRouterError(
                    f"leak: {ka} train intersects {kb} test in {len(bad)} rows")


def run_pair(pair: str, cfg, *, family: str = "hs_wide", segment: str = "all",
             schemes: tuple[str, ...] = SCHEMES,
             n_samples: tuple[int, ...] = (10,),
             metrics: tuple[str, ...] = METRICS,
             arms: tuple[str, ...] = ARMS) -> pd.DataFrame:
    """The full 8x8 metric grid for one pair. Atomic: no cohort, no medians.

    Cost note: the router is fit ONCE per (arm, scheme, n, train-metric) and
    then scored against all 8 test-metric label sets, so a grid costs 8 fits
    rather than 64.
    """
    if not sampled_mod.is_ready(pair):
        raise MetricRouterError(
            f"{pair}: sampled extraction has not landed; M23 needs it")
    entry = read.load(pair)
    frozen = arms_build.build_all(entry)["dse_natural"]
    by_metric = {m: arms_for(entry, m, frozen) for m in metrics}
    assert_leak_free(by_metric)

    c = cfg.c(family)
    pca_dim = None if family in config.HS_FAMILIES else config.PCA_DIM
    rows: list[dict] = []

    for scheme in schemes:
        for n in n_samples:
            R = sampled_mod.aggregate(pair, family, segment,
                                      scheme=scheme, n_slots=n)
            for arm in arms:
                for a in metrics:                     # the TRAIN metric
                    arm_a = by_metric[a].get(arm)
                    if arm_a is None:
                        continue
                    ctr = _cats_of(by_metric, a, entry)[arm_a.train]
                    if _min_class(ctr) < config.MIN_PER_CLASS:
                        continue
                    try:
                        # `cloud` is 10 columns; PCA to 100 would be a no-op at
                        # best and an error at worst, so it is fit raw.
                        rt = router_mod.fit_sampled(
                            R[arm_a.train], ctr, family=family, c=c,
                            pca_dim=None if scheme == "cloud" else pca_dim)
                    except router_mod.RouterError:
                        continue

                    for b in metrics:                 # the TEST metric
                        arm_b = by_metric[b].get(arm)
                        if arm_b is None:
                            continue
                        cte = _cats_of(by_metric, b, entry)[arm_b.test]
                        if _min_class(cte) < config.MIN_PER_CLASS:
                            continue
                        y_te = router_mod.y_hi(cte)
                        p = rt.p_hi(R[arm_b.test])
                        rows.append({
                            "pair": pair, "model": entry.model,
                            "dataset": entry.dataset,
                            "modality": entry.modality,
                            "family": family, "segment": segment,
                            "C": float(c), "c_mode": cfg.c_mode,
                            "arm": arm, "scheme": scheme, "n_samples": int(n),
                            "train_metric": a, "test_metric": b,
                            "diagonal": a == b,
                            "n_train": int(len(arm_a.train)),
                            "n_test": int(len(arm_b.test)),
                            "frac_hi_train": float(
                                router_mod.y_hi(ctr).mean()),
                            "frac_hi_test": float(y_te.mean()),
                            "router_auroc": metrics_mod.safe_auc(y_te, p),
                        })
    return pd.DataFrame(rows)


_CATS_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _cats_of(by_metric: dict, metric: str, entry) -> np.ndarray:
    """Row-aligned categories under `metric`, memoised per (pair, metric)."""
    key = (entry.pair, metric)
    if key not in _CATS_CACHE:
        _CATS_CACHE[key] = relabel(entry, metric)[0]
    return _CATS_CACHE[key]


def _min_class(cats) -> int:
    y = router_mod.y_hi(cats)
    return int(min(int((y == 1).sum()), int((y == 0).sum())))


def agreement(pair: str) -> pd.DataFrame:
    """Raw band agreement and Cohen's kappa for every ordered metric pair.

    The honest denominator, reported BEFORE any transfer number: it states how
    much of the off-diagonal is free because the metrics are redundant. This is
    shown, not argued away.

    Computed on the frozen natural TEST rows -- the same rows the grid's
    off-diagonal cells are scored on, so the two tables describe one population.
    """
    entry = read.load(pair)
    te = arms_build.build_all(entry)["dse_natural"].test
    band = {m: config.band_of(relabel(entry, m)[0][te]) for m in METRICS}
    rows = []
    for a in METRICS:
        for b in METRICS:
            x, y = band[a], band[b]
            k = (float(cohen_kappa_score(x, y))
                 if len(set(x)) > 1 and len(set(y)) > 1 else np.nan)
            rows.append({"pair": pair, "metric_a": a, "metric_b": b,
                         "n": int(len(x)),
                         "agreement": float((x == y).mean()), "kappa": k})
    return pd.DataFrame(rows)


def write(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    results.write_unit(TABLE, pair, df)
    return df


def write_agreement(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    results.write_unit(AGREEMENT_TABLE, pair, df)
    return df
