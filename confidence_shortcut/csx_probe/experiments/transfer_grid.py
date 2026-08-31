"""The transfer grid: train on arm *i*, test on arm *j*, for every ordered pair.

This is the experiment the write-up's headline rests on. The question it answers
is not "can a probe predict correctness" -- the natural arm answers that, and it
answers it partly by reading confidence, which is the problem. It is:

    when the confidence channel is closed by construction, how much of the
    apparent correctness signal survives?

The off-diagonal cells are where that gets decided. A probe trained on
`dse_natural` and tested on `dse_matched2` is being asked to work where entropy is
*exactly* uninformative about correctness within band -- so whatever AUROC it
retains there cannot be coming from the shortcut.

Three heads are scored on every test arm:

    g             fitted on correctness    -- what we want to measure
    sep           fitted on the band       -- the shortcut, made explicit
    entropy_only  unfit, the raw entropy   -- the floor everything must beat

Fitting `sep` at all is the point. It makes the confidence axis a measured
quantity rather than a confound argued about in prose: on `IHvCL` it scores near
0.000, which is the shortcut being caught in the act, and any codebase that
flipped that sign to keep it above 0.5 could not report the finding.

**One transform, both heads, fit on the train arm only.** The two heads then
differ solely in their target, so the comparison between them is about what was
predicted and not about how the features were scaled.

Emits atomic rows only -- one per `(pair, family, segment, train_arm, test_arm,
head, contrast)`. No medians, no cohorts; component 3 does that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from csx_probe import config, metrics, probes, results
from csx_probe.arms import build as arms_build, gates
from csx_probe.store import build as store_build, read
from csx_probe.store.derive import select as select_mod

TABLE = "per_pair_long"


def run_pair(pair: str, cfg: config.RunConfig, *,
             segments: tuple[str, ...] | None = None,
             families: tuple[str, ...] | None = None,
             basis: dict | None = None,
             n_boot: int | None = None,
             verbose: bool = True) -> pd.DataFrame:
    """Every grid cell for one pair. The unit of resumption is the pair."""
    entry = read.load(pair)
    families = families or cfg.families
    segments = segments or entry.segments
    n_boot = n_boot if n_boot is not None else cfg.n_boot

    arms = arms_build.build_all(entry)
    # Gate before fitting, not after: a leaking arm family makes every number
    # below meaningless, and finding that out after the grid has run costs the
    # whole grid.
    gates.assert_all(entry, arms, strict=False)

    rows: list[dict] = []
    for family in families:
        if family not in entry.available_families():
            if verbose:
                print(f"  [{pair}/{family}] skipped: needs a phase that is "
                      f"not done", flush=True)
            continue
        for segment in segments:
            rows += _family_segment(entry, arms, family, segment, cfg,
                                    basis, n_boot, verbose)
    df = pd.DataFrame(rows)
    if len(df):
        results.write_unit(TABLE, pair, df)
    return df


def _family_segment(entry: read.Entry, arms: dict, family: str, segment: str,
                    cfg: config.RunConfig, basis: dict | None, n_boot: int,
                    verbose: bool) -> list[dict]:
    top_k, pca_dim = _basis_for(entry, arms, family, segment, basis)
    X = store_build.feature_matrix(entry, family, segment, top_k=top_k)
    c = cfg.c(family)
    out: list[dict] = []

    for train_arm, arm in arms.items():
        tr = arm.train
        ctr = entry.categories[tr]
        # A head fitted where a cell is nearly empty is not a weak measurement,
        # it is an undefined one; the class_weight='balanced' reweighting would
        # happily fit it and return a confident-looking number.
        if any(int((ctr == k).sum()) < config.MIN_PER_CLASS for k in config.CATS):
            if verbose:
                print(f"  [{entry.pair}/{family}/{segment}/{train_arm}] skipped: "
                      f"a train cell is below min_per_class", flush=True)
            continue

        tf, heads = probes.fit_heads(X[tr], ctr, family=family, c=c,
                                     pca_dim=pca_dim)

        for test_arm, other in arms.items():
            te = other.test
            axes = probes.axes(tf, heads, X[te], entry.entropy[te])
            cte = entry.categories[te]
            for head in config.HEADS:
                got: dict[str, float] = {}
                for contrast in config.CONTRAST_ORDER:
                    r = metrics.cell(axes[head], cte, contrast, n_boot=n_boot)
                    got[contrast] = r["AUROC"]
                    out.append(_row(entry, family, segment, c, cfg, train_arm,
                                    test_arm, head, contrast, **r))
                d = metrics.derived(got)
                d["shortcut_IHvCL"] = got.get(config.SHORTCUT, np.nan)
                for name, val in d.items():
                    out.append(_row(
                        entry, family, segment, c, cfg, train_arm, test_arm,
                        head, name, AUROC=val, AUROC_lo=np.nan,
                        AUROC_hi=np.nan, n=int(len(cte)), n_pos=-1))
        del tf, heads
    del X
    return out


def _basis_for(entry: read.Entry, arms: dict, family: str, segment: str,
               basis: dict | None) -> tuple[int | None, int | None]:
    """`(top_k, pca_dim)` for this family.

    Selected on the NATURAL train rows, once per (pair, family, segment), and
    reused for every arm in the grid. Re-selecting per train-arm would make the
    off-diagonal cells compare two probes that differ in basis as well as in
    training distribution -- and the basis difference would be invisible in the
    output.
    """
    if family in config.HS_FAMILIES:
        return None, None
    if basis and (family, segment) in basis:
        return basis[(family, segment)]
    sel = select_mod.select(entry, family, segment, arms["dse_natural"].train)
    return sel.top_k, sel.pca_dim


def _row(entry: read.Entry, family: str, segment: str, c: float,
         cfg: config.RunConfig, train_arm: str, test_arm: str, head: str,
         contrast: str, **rest) -> dict:
    return {
        "pair": entry.pair, "model": entry.model, "dataset": entry.dataset,
        "modality": entry.modality, "family": family, "segment": segment,
        "C": float(c), "c_mode": cfg.c_mode,
        "prompt_template": entry.prompt_template,
        "train_arm": train_arm, "test_arm": test_arm,
        "diagonal": train_arm == test_arm,
        "head": head, "contrast": contrast, **rest,
    }


def arm_stats(pair: str) -> pd.DataFrame:
    """Composition of every arm split, for the `arm_stats` table."""
    entry = read.load(pair)
    arms = arms_build.build_all(entry)
    rows = []
    for name, arm in arms.items():
        for split in ("train", "test"):
            sel = arm.rows(split)
            rows.append({
                "pair": pair, "arm": name, "split": split,
                **metrics.arm_stats(entry.categories[sel], entry.entropy[sel]),
                "note": arm.note, "ratio": arm.ratio,
            })
    df = pd.DataFrame(rows)
    results.write_unit("arm_stats", pair, df)
    return df
