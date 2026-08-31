"""Selecting `C`, per `(family, pair)`, and storing the whole CV curve.

The published protocol resolved `C` **per family**, medianing the per-unit best
over all 32 units (4 arms x 8 pairs). That is reproducible but not extensible:
a new pair shifts that median, so every existing pair would have to be refit every
time a model or dataset is added.

We resolve per `(family, pair)` instead, medianing over only that pair's own 4
arm-units. The CV protocol is otherwise identical -- 5-fold
`StratifiedKFold(shuffle, 42)` strictly inside that unit's TRAIN split, transform
refit per fold, argmax mean-fold AUROC -- so nothing about the selection's
legitimacy changes; only the set it is medianed over shrinks from 32 units to 4.

Measured against the published per-family table this agrees on **50 of 56
`(family, pair)` cells**, with `hs_wide` and `attn_eigvals` reproducing 8/8
exactly. `qa8` stays `pinned` regardless, so the parity gate is untouched.

**The full curve is stored, not just the argmax.** Re-resolving `C` under a
different policy then needs no CV refits at all -- it re-reads these columns --
and a selection that was nearly a tie is a materially different claim from one
that won by a mile. Only the grids whose `C` actually moved get refit.

Two details that matter:

**The transform is refit inside each fold.** Fitting it once outside the CV would
leak the held-out fold into the scaling and inflate every curve uniformly, which
is invisible precisely because it moves all of them the same way.

**Test splits are never loaded.** Selection reads the train rows of each arm and
nothing else.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from csx_probe import config, probes, results
from csx_probe.arms import build as arms_build
from csx_probe.store import build as store_build, read
from csx_probe.store.derive import select as select_mod

TABLE = "c_selection"
N_SPLITS = 5


def _pipe(family: str, c: float, pca_dim: int | None, n_feat: int, n_samp: int
          ) -> Pipeline:
    """Transform + LR as ONE pipeline, so the transform refits per fold."""
    kind = config.kind_of(family)
    steps = []
    if kind == "hs" or pca_dim is None:
        from sklearn.preprocessing import StandardScaler
        steps.append(("tf", StandardScaler()))
    else:
        from sklearn.decomposition import PCA
        dim = min(int(pca_dim), n_feat, n_samp - 1)
        steps.append(("tf", PCA(n_components=dim, svd_solver="randomized",
                                random_state=config.SEED)))
    steps.append(("lr", probes.make_lr(family, c)))
    return Pipeline(steps)


def _cv_one(X: np.ndarray, tr: np.ndarray, y: np.ndarray, family: str,
            c: float, pca_dim: int | None) -> float:
    """Mean 5-fold CV AUROC for one `(arm, C)`.

    Module-level and closure-free so `loky` can pickle it. `X` is the whole
    feature matrix rather than the arm's slice: passed once per Parallel call it
    is memmapped once, whereas handing each task its own `X[tr]` copy would dump
    a fresh temp file for all 36 of them.
    """
    Xtr = X[tr]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=config.SEED)
    return float(np.mean(cross_val_score(
        _pipe(family, c, pca_dim, Xtr.shape[1], Xtr.shape[0]),
        Xtr, y, cv=skf, scoring="roc_auc", n_jobs=1)))


def run_pair(pair: str, *, families: tuple[str, ...] | None = None,
             segments: tuple[str, ...] | None = None,
             n_jobs: int = -1, verbose: bool = True) -> pd.DataFrame:
    """The CV curve for every `(family, segment, arm)` unit of one pair."""
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/dev/shm")
    entry = read.load(pair)
    families = families or config.FAMILIES
    segments = segments or entry.segments
    arms = arms_build.build_all(entry)
    grid = config.c_grid()

    rows: list[dict] = []
    for family in families:
        if family not in entry.available_families():
            continue
        for segment in segments:
            top_k, pca_dim = _basis(entry, arms, family, segment)
            X = store_build.feature_matrix(entry, family, segment, top_k=top_k)
            # Every (arm, C) is an independent CV run, so they are dispatched
            # together rather than looped. `inner_max_num_threads=1` matters as
            # much as the worker count: MKL otherwise takes every core for each
            # individual fit and the workers contend for the same cores.
            units = []
            for arm_name, arm in arms.items():
                y = select_mod.y_incorrect(entry.categories[arm.train])
                if len(np.unique(y)) < 2:
                    continue
                units.append((arm_name, arm.train, y))
            if verbose:
                print(f"  [{pair}/{family}/{segment}] {len(units)}x{len(grid)} "
                      f"CV runs, n_jobs={n_jobs}", flush=True)
            flat = Parallel(n_jobs=n_jobs, backend="loky",
                            inner_max_num_threads=1, max_nbytes="1M")(
                delayed(_cv_one)(X, tr, y, family, c, pca_dim)
                for _, tr, y in units for c in grid)

            for u, (arm_name, tr, y) in enumerate(units):
                curve = dict(zip(grid, flat[u * len(grid):(u + 1) * len(grid)]))
                Xtr_shape = (len(tr), X.shape[1])
                best_c = max(curve, key=lambda k: (curve[k], -k))
                rows.append({
                    "space": "pca100" if pca_dim else "native",
                    "pair": pair, "model": entry.model, "dataset": entry.dataset,
                    "family": family, "segment": segment, "train_arm": arm_name,
                    "kind": config.kind_of(family), "pca_dim": pca_dim,
                    "top_k": top_k, "n_train": int(len(tr)),
                    "n_feat": int(Xtr_shape[1]),
                    "best_C": float(best_c),
                    "best_cv_auroc": float(curve[best_c]),
                    **{f"cv_auroc_C={c:g}": v for c, v in curve.items()},
                })
                if verbose:
                    print(f"  [{pair}/{family}/{segment}/{arm_name}] "
                          f"best C={best_c:g} cv={curve[best_c]:.4f}", flush=True)
            del X
    df = pd.DataFrame(rows)
    if len(df):
        results.write_unit(TABLE, pair, df)
    return df


def _basis(entry: read.Entry, arms: dict, family: str, segment: str):
    if family in config.HS_FAMILIES:
        return None, None
    sel = select_mod.select(entry, family, segment, arms["dse_natural"].train)
    return sel.top_k, sel.pca_dim


def per_unit_best(pair: str, segment: str = "all") -> dict[str, list[float]]:
    """`family -> [best C per arm-unit]`, read back from the stored curve.

    This is what `config.resolve_c` medians in `per_pair` mode. Reading it from
    disk rather than recomputing is the point: a policy change costs a parquet
    read, not a CV sweep.
    """
    df = results.read_unit(TABLE, pair)
    if df is None:
        raise FileNotFoundError(
            f"{pair}: no c_selection checkpoint; run cli_probe/03_select_c.py "
            f"for this pair before resolving C in per_pair mode")
    df = df[df["segment"] == segment]
    return {f: g["best_C"].astype(float).tolist()
            for f, g in df.groupby("family")}
