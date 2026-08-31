"""The confound grids: is the shortcut result about entropy, or about n?

`transfer_grid` shows `sep` collapsing on the matched arms while `g` holds up.
The obvious reviewer objection is that the matched arms are also a third the
size of `natural` and differently composed, so the collapse might be nothing
more than a smaller, rebalanced training set. These two grids answer that by
running the *same* grid against controls that differ from `natural` in only one
of those ways at a time.

    ns_A   natural's own skew, A's row count      -> isolates SIZE
    pl_A   A's per-cell counts, uniform draws     -> isolates COMPOSITION
    A      all of the above plus entropy-matching -> the treatment

If `sep` still scores ~0.76 on `ns_matched2`, sample size is not the
explanation. If it also holds on `pl_matched2`, composition is not either, and
what is left is the entropy-matching itself.

**The two grids have deliberately different shapes**, because the two controls
answer different questions:

`pl_*` is a TRAIN-ONLY arm. It is evaluated against the four real test arms, so
its row sits directly alongside the real arm's row in the published grid and the
two are read off the same test population. Giving it its own test split would
compare two probes on two different populations and confound the very thing it
exists to isolate.

`ns_*` is a train AND test arm, evaluated on three test columns: its own test
split (the like-for-like small-population question), `natural.test` (the same
probe on the full population), and the real target's test split (the direct
control-vs-treatment comparison). Those three are the whole point -- a single
diagonal cell would not say whether the difference is in the probe or in the
population it was scored on.

Emits atomic rows only, on the same schema as `per_pair_long` plus `draw` and
`target`, so component 3 can median over draws without a bespoke reader.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from csx_probe import config, metrics, probes, results
from csx_probe.arms import build as arms_build, confound, gates
from csx_probe.store import build as store_build, read
from csx_probe.store.derive import select as select_mod

PLACEBO_TABLE = "placebo_long"
SIZEONLY_TABLE = "sizeonly_long"

# 100, not the main grid's 1000: these rows carry point estimates and the null
# comes from the spread across the 20 draws, so a per-row CI would be 10x the
# cost for a quantity nothing downstream reads.
N_BOOT: int = frozen_n if (frozen_n := config.frozen()["bootstrap"].get(
    "confound_grid_n")) else 100


def _prov(entry: read.Entry) -> dict:
    """The provenance fields, pulled once so workers need no `Entry`."""
    return {"pair": entry.pair, "model": entry.model,
            "dataset": entry.dataset, "modality": entry.modality,
            "prompt_template": entry.prompt_template}


def _score(X: np.ndarray, cats: np.ndarray, ent: np.ndarray, tr: np.ndarray,
           tests: dict[str, np.ndarray], family: str, c: float,
           pca_dim: int | None, n_boot: int, prov: dict, family_seg: tuple,
           train_arm: str, target: str, draw: int) -> list[dict]:
    """Fit both heads on one train arm and score every test column.

    Module-level and closure-free so `loky` can pickle it; `X` arrives as a
    joblib memmap so the workers share one copy of the feature matrix.
    """
    family_name, segment = family_seg
    ctr = cats[tr]
    if any(int((ctr == k).sum()) < config.MIN_PER_CLASS for k in config.CATS):
        return []
    tf, heads = probes.fit_heads(X[tr], ctr, family=family, c=c,
                                 pca_dim=pca_dim)
    out: list[dict] = []
    for test_arm, te in tests.items():
        if not len(te):
            continue
        axes = probes.axes(tf, heads, X[te], ent[te])
        cte = cats[te]
        for head in config.HEADS:
            got: dict[str, float] = {}
            for contrast in config.CONTRAST_ORDER:
                r = metrics.cell(axes[head], cte, contrast, n_boot=n_boot)
                got[contrast] = r["AUROC"]
                out.append({**prov, "family": family_name, "segment": segment,
                            "C": float(c), "target": target, "draw": draw,
                            "train_arm": train_arm, "test_arm": test_arm,
                            "head": head, "contrast": contrast, **r})
            d = metrics.derived(got)
            d["shortcut_IHvCL"] = got.get(config.SHORTCUT, np.nan)
            for name, val in d.items():
                out.append({**prov, "family": family_name, "segment": segment,
                            "C": float(c), "target": target, "draw": draw,
                            "train_arm": train_arm, "test_arm": test_arm,
                            "head": head, "contrast": name, "AUROC": val,
                            "AUROC_lo": np.nan, "AUROC_hi": np.nan,
                            "n": int(len(cte)), "n_pos": -1})
    return out


def run_pair(pair: str, cfg: config.RunConfig, *, kind: str = "placebo",
             segments: tuple[str, ...] | None = None,
             families: tuple[str, ...] | None = None,
             draws: int | None = None, n_boot: int | None = None,
             n_jobs: int = -1, verbose: bool = True) -> pd.DataFrame:
    """One confound grid for one pair. `kind` is `placebo` or `sizeonly`."""
    if kind not in ("placebo", "sizeonly"):
        raise ValueError(f"kind must be 'placebo' or 'sizeonly', got {kind!r}")
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/dev/shm")

    entry = read.load(pair)
    families = families or cfg.families
    segments = segments or entry.segments
    draws = draws if draws is not None else confound.N_DRAWS
    n_boot = n_boot if n_boot is not None else N_BOOT

    real = arms_build.build_all(entry)
    gates.assert_all(entry, real, strict=False)
    ctrl = (confound.placebo_arms(entry, real, draws=draws) if kind == "placebo"
            else confound.sizeonly_arms(entry, real, draws=draws))
    # The controls are drawn from natural's pools, so leak-freedom follows from
    # the family invariant -- but it is asserted rather than argued.
    leaks = gates.check_no_leakage({**real, **ctrl})
    if leaks:
        raise gates.GateError(f"{pair}/{kind}: {leaks[0]}")

    prov = _prov(entry)
    cats, ent = entry.categories, entry.entropy
    rows: list[dict] = []
    for family in families:
        if family not in entry.available_families():
            continue
        for segment in segments:
            rows += _family_segment(entry, real, ctrl, family, segment, cfg,
                                    kind, n_boot, n_jobs, prov, cats, ent,
                                    verbose)
    df = pd.DataFrame(rows)
    if len(df):
        results.write_unit(
            PLACEBO_TABLE if kind == "placebo" else SIZEONLY_TABLE, pair, df)
    return df


def _tests_for(kind: str, name: str, arm, real: dict) -> dict[str, np.ndarray]:
    """The test columns this control is scored on -- see the module docstring."""
    if kind == "placebo":
        return {a: real[a].test for a in real}
    tgt = _target_of(name)
    return {name: arm.test,
            "dse_natural": real["dse_natural"].test,
            tgt: real[tgt].test}


def _parse(name: str) -> tuple[str, int]:
    """`ns_dse_matched2_d07` -> `('dse_matched2', 7)`.

    `rsplit`, not `split`: the arm names contain `_d` inside `_dse_` as well as
    in the draw suffix, so a left split silently returns `se_matched2`.
    """
    stem, draw = name.rsplit("_d", 1)
    return stem.split("_", 1)[1], int(draw)


def _target_of(name: str) -> str:
    return _parse(name)[0]


def _family_segment(entry: read.Entry, real: dict, ctrl: dict, family: str,
                    segment: str, cfg: config.RunConfig, kind: str,
                    n_boot: int, n_jobs: int, prov: dict, cats: np.ndarray,
                    ent: np.ndarray, verbose: bool) -> list[dict]:
    top_k, pca_dim = (None, None)
    if family not in config.HS_FAMILIES:
        sel = select_mod.select(entry, family, segment,
                                real["dse_natural"].train)
        top_k, pca_dim = sel.top_k, sel.pca_dim
    X = store_build.feature_matrix(entry, family, segment, top_k=top_k)
    c = cfg.c(family)

    jobs = []
    for name, arm in sorted(ctrl.items()):
        jobs.append((name, arm, _tests_for(kind, name, arm, real)))
    if verbose:
        print(f"  [{entry.pair}/{family}/{segment}/{kind}] {len(jobs)} control "
              f"arms, n_jobs={n_jobs}", flush=True)

    out = Parallel(n_jobs=n_jobs, backend="loky", inner_max_num_threads=1,
                   max_nbytes="1M")(
        delayed(_score)(X, cats, ent, arm.train, tests, family, c, pca_dim,
                        n_boot, prov, (family, segment), name,
                        *_parse(name))
        for name, arm, tests in jobs)
    del X
    return [r for chunk in out for r in chunk]
