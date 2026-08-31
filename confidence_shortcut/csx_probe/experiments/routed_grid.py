"""The routed grid: band-routed specialists against a single generalist.

Answers the question the transfer grid leaves open. The transfer grid shows how
much correctness signal survives when the confidence channel is closed; this one
asks whether a *routed* architecture — two band-local experts plus a predicted
band — retains more of it than one generalist probe, and at what deployment cost.

Emits `routed_long`, schema-compatible with `per_pair_long` plus two columns:

    router   oracle | sampled | greedy | '-'   how the band was decided
    scorer   which design produced the score

so `csx_report` reuses its loaders unchanged and the two tables concatenate.

**The scorers, and what each one isolates.**

| scorer | routed | information | isolates |
|---|---|---|---|
| `entropy_only` | no | 1 gen | the floor: raw entropy, unfit |
| `sep` | no | 1 gen | the confidence axis, made explicit |
| `generalist` | no | 1 gen | one probe, no routing |
| `generalist_cm` | no | 10 gens | the generalist at *equal information* |
| `spec1_z` | yes | 1 gen | routing, z-pooled |
| `spec1_platt` | yes | 1 gen | routing, Lever A pooling |
| `spec1_z_cm` | yes | 10 gens | routing at equal information |
| `spec1_z_hier` | yes | 1 gen | experts trained on router-assigned rows |

`generalist_cm` is not optional. Without it, `generalist` vs `spec1_z` compares a
1-generation scorer against a design that also gets 10 generations for its
router — so a win could just mean "10 generations beat 1". The `_cm` pair is the
comparison to read for *does routing beat a generalist at equal information*.

**Every expert scores every test row, then the router picks.** With a predicted
band we cannot restrict scoring to the rows a band actually owns: the router may
be wrong, and a misrouted row still needs a number. Restricting instead would
quietly evaluate each expert only where it was already going to do well.

**Plan-then-execute, so the seeded bootstrap stream stays serial.** Work units
are enumerated first and dispatched as one `Parallel` call; bootstrap indices come
off `default_rng(seed)` inside each unit, never in worker-completion order, so the
CIs reproduce regardless of how loky schedules the workers.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from csx_probe import config, metrics, probes, results
from csx_probe.arms import build as arms_build, gates
from csx_probe.routing import experts as experts_mod, pooling, router as router_mod
from csx_probe.store import build as store_build, read
from csx_probe.store.derive import select as select_mod

TABLE = "routed_long"

UNROUTED: tuple[str, ...] = ("entropy_only", "sep", "generalist", "generalist_cm")
ROUTED: tuple[str, ...] = ("spec1_z", "spec1_platt", "spec1_z_cm", "spec1_z_hier")

# `sampled_scheme` for cells that never read the sampled block. Those cells are
# identical under every scheme, so they are emitted once under this tag rather
# than copied per scheme -- a report grouping by scheme must therefore treat
# NO_SCHEME rows as shared, not as a fourth variant.
NO_SCHEME: str = "-"

# Which pooling method each routed scorer uses, and whether it wants the
# 10-generation (cost-matched) feature block or the greedy one.
SCORER_SPEC: dict[str, dict] = {
    "spec1_z":      {"pool": "z",           "cm": False, "mode": "same_band"},
    "spec1_platt":  {"pool": "platt_prior", "cm": False, "mode": "same_band"},
    "spec1_z_cm":   {"pool": "z",           "cm": True,  "mode": "same_band"},
    "spec1_z_hier": {"pool": "z",           "cm": False, "mode": "hier"},
}


class RoutedError(Exception):
    """The routed grid cannot run for this unit. The message names why."""


def run_pair(pair: str, cfg: config.RunConfig, *,
             segments: tuple[str, ...] | None = None,
             families: tuple[str, ...] | None = None,
             scorers: tuple[str, ...] | None = None,
             sampled: dict[str, np.ndarray] | None = None,
             n_boot: int | None = None, n_jobs: int = 1,
             verbose: bool = True) -> pd.DataFrame:
    """Every routed cell for one pair. The unit of resumption is the pair.

    `sampled` is `{scheme: {(family, segment): matrix}}` -- the aggregated
    10-generation features, row-aligned to the entry, one block per aggregation
    scheme. Absent, the `_cm` scorers and the `sampled` router are skipped with
    a note rather than silently substituting greedy features -- a cost-matched
    comparison fed single-pass features is not cost-matched, and would be
    indistinguishable in the output from one that was.

    Every emitted row carries `sampled_scheme`. Cells that do not read the
    sampled block are emitted ONCE under `NO_SCHEME`, so requesting a second
    scheme adds only the cells that actually differ under it.
    """
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/dev/shm")
    entry = read.load(pair)
    families = families or cfg.families
    segments = segments or entry.segments
    scorers = scorers or (UNROUTED + ROUTED)
    n_boot = n_boot if n_boot is not None else cfg.n_boot

    arms = arms_build.build_all(entry)
    gates.assert_all(entry, arms, strict=False)

    plan = [(f, s) for f in families if f in entry.available_families()
            for s in segments]
    if verbose:
        print(f"[{pair}] routed grid: {len(plan)} (family, segment) units, "
              f"n_jobs={n_jobs}", flush=True)

    if n_jobs == 1:
        chunks = [_family_segment(entry, arms, f, s, cfg, scorers, sampled,
                                  n_boot, verbose) for f, s in plan]
    else:
        chunks = Parallel(n_jobs=n_jobs, backend="loky",
                          inner_max_num_threads=1, max_nbytes="1M")(
            delayed(_family_segment)(entry, arms, f, s, cfg, scorers, sampled,
                                     n_boot, False) for f, s in plan)

    rows = [r for c in chunks for r in c]
    df = pd.DataFrame(rows)
    if len(df):
        results.write_unit(TABLE, pair, df)
    return df


def _family_segment(entry: read.Entry, arms: dict, family: str, segment: str,
                    cfg: config.RunConfig, scorers: tuple[str, ...],
                    sampled: dict | None, n_boot: int,
                    verbose: bool) -> list[dict]:
    top_k, pca_dim = _basis_for(entry, arms, family, segment)
    X = store_build.feature_matrix(entry, family, segment, top_k=top_k)
    Rs = {sch: blk[(family, segment)]
          for sch, blk in (sampled or {}).items()
          if (family, segment) in blk}
    for sch, R in Rs.items():
        if len(R) != len(X):
            raise RoutedError(
                f"{entry.pair}/{family}/{segment}: sampled block for scheme "
                f"{sch!r} has {len(R)} rows against {len(X)} greedy rows; they "
                f"must be row-aligned")
    c = cfg.c(family)
    out: list[dict] = []

    for train_arm, arm in arms.items():
        tr = arm.train
        ctr = entry.categories[tr]
        if any(int((ctr == k).sum()) < config.MIN_PER_CLASS
               for k in config.CATS):
            if verbose:
                print(f"  [{entry.pair}/{family}/{segment}/{train_arm}] "
                      f"skipped: a train cell is below min_per_class",
                      flush=True)
            continue
        fitted = _fit_train_side(X, Rs, ctr, tr, family, c, pca_dim, scorers)

        for test_arm, other in arms.items():
            te = other.test
            cte = entry.categories[te]
            scored = _score_test_side(fitted, X, Rs, te, cte, entry, scorers)
            for (scorer, router_name, scheme), s in scored.items():
                out += _emit(entry, family, segment, c, cfg, train_arm,
                             test_arm, scorer, router_name, scheme, s, cte,
                             n_boot)
        del fitted
    del X
    return out


def _fit_train_side(X, Rs, ctr, tr, family, c, pca_dim, scorers) -> dict:
    """Everything fit on the train arm: experts, poolers, routers, baselines.

    `Rs` is `{scheme: sampled_block}`. The greedy-side artifacts (the generalist
    heads, the `greedy` router, and the non-`_cm` experts) do NOT depend on the
    scheme, so they are fit ONCE and shared across every scheme -- that is the
    whole reason the scheme loop lives here rather than around `run_pair`. On
    `hs_wide` those are the expensive fits (~14k features); the `cloud` blocks
    are 10 features and nearly free by comparison, so adding a scheme costs far
    less than the grid it appears to duplicate.

    Scheme-dependent artifacts are keyed by scheme; scheme-free ones by `None`.
    """
    f: dict = {"family": family, "c": c, "pca_dim": pca_dim}
    Rs = Rs or {}

    if "generalist" in scorers or "sep" in scorers:
        tf, heads = probes.fit_heads(X[tr], ctr, family=family, c=c,
                                     pca_dim=pca_dim)
        f["generalist"] = (tf, heads)
    f["generalist_cm"] = {}
    if "generalist_cm" in scorers:
        for sch, R in Rs.items():
            f["generalist_cm"][sch] = probes.fit_heads(
                R[tr], ctr, family=family, c=c, pca_dim=pca_dim)

    # Routers, keyed `(name, scheme)`. `oracle` needs no fitting; `greedy` is
    # the sep probe's twin; `sampled` is fit once per scheme.
    f["routers"] = {("oracle", None): router_mod.fit_oracle()}
    try:
        f["routers"][("greedy", None)] = router_mod.fit_greedy(
            X[tr], ctr, family=family, c=c, pca_dim=pca_dim)
    except router_mod.RouterError:
        pass
    for sch, R in Rs.items():
        try:
            f["routers"][("sampled", sch)] = router_mod.fit_sampled(
                R[tr], ctr, family=family, c=c, pca_dim=pca_dim)
        except router_mod.RouterError:
            pass

    # Band experts, one set per (feature block, training mode).
    f["experts"] = {}
    f["poolers"] = {}
    for scorer in [s for s in scorers if s in ROUTED]:
        spec = SCORER_SPEC[scorer]
        blocks = list(Rs.items()) if spec["cm"] else [(None, X)]
        for sch, M in blocks:
            if M is None:
                continue
            key = (sch, spec["mode"])
            if key not in f["experts"]:
                oof = None
                if spec["mode"] == "hier":
                    oof = experts_mod.oof_router_bands(
                        M[tr], ctr, family=family, c=c, pca_dim=pca_dim)
                f["experts"][key] = experts_mod.fit_experts(
                    M[tr], ctr, family=family, c=c, pca_dim=pca_dim,
                    mode=spec["mode"], oof_bands=oof)
            exps = f["experts"][key]
            if any(e is None for e in exps.values()):
                continue
            # Pooling parameters come from each expert's OWN training rows --
            # the same rows it learned on, so the prior is that population's.
            tl, tlab = {}, {}
            for b, e in exps.items():
                m = experts_mod.band_train_mask(ctr, b)
                tl[b] = e.logit(M[tr][m])
                tlab[b] = (ctr[m] == config.BANDS[b][0]).astype(int)
            f["poolers"][(scorer, sch)] = pooling.fit(spec["pool"], tl, tlab)
    return f


def _score_test_side(f, X, Rs, te, cte, entry, scorers) -> dict:
    """`{(scorer, router, scheme): scores}` for one test arm.

    `scheme` is `NO_SCHEME` for any cell that does not touch the sampled block,
    so the greedy-only baselines appear exactly once no matter how many schemes
    were requested rather than being duplicated per scheme.

    A cell can depend on the scheme through its expert block (`_cm` scorers),
    through its router (`@sampled`), or both. When both, the two are pinned to
    the SAME scheme: a `cloud` expert routed by a `greedy_mean_std` router is a
    chimera that corresponds to no real deployment, and reporting it would put
    cells in the table that nothing could actually be run as.
    """
    out: dict[tuple[str, str, str], np.ndarray] = {}
    Rs = Rs or {}

    if "entropy_only" in scorers:
        out[("entropy_only", "-", NO_SCHEME)] = np.asarray(
            entry.entropy[te], dtype=float)
    if "generalist" in f:
        tf, heads = f["generalist"]
        ax = probes.axes(tf, heads, X[te], entry.entropy[te])
        if "generalist" in scorers:
            out[("generalist", "-", NO_SCHEME)] = ax["g"]
        if "sep" in scorers:
            out[("sep", "-", NO_SCHEME)] = ax["sep"]
    for sch, (tf, heads) in f.get("generalist_cm", {}).items():
        out[("generalist_cm", "-", sch)] = probes.score_pos(
            heads["g"], tf.transform(Rs[sch][te]))

    for (scorer, sch), pooler in f["poolers"].items():
        spec = SCORER_SPEC[scorer]
        M = Rs[sch] if spec["cm"] else X
        exps = f["experts"][(sch, spec["mode"])]
        logits = {b: e.logit(M[te]) for b, e in exps.items()}
        for (rname, rsch), r in f["routers"].items():
            if spec["cm"] and rname == "sampled" and rsch != sch:
                continue                      # no cross-scheme chimeras
            asg = (r.assign(categories=cte) if rname == "oracle"
                   else r.assign((Rs[rsch] if rname == "sampled" else X)[te]))
            tag = sch if spec["cm"] else (rsch if rname == "sampled"
                                          else NO_SCHEME)
            out[(scorer, rname, tag)] = pooling.pooled_scores(
                pooler, logits, asg)
            out.setdefault(("__router__", rname, rsch or NO_SCHEME), asg)
    return out


def _emit(entry, family, segment, c, cfg, train_arm, test_arm, scorer,
          router_name, scheme, s, cte, n_boot) -> list[dict]:
    """One scorer's cells, plus the router's own band AUROC as a diagnostic."""
    if scorer == "__router__":
        return [_row(entry, family, segment, c, cfg, train_arm, test_arm,
                     "router_band", router_name, scheme, "band_AUROC",
                     AUROC=router_mod.band_auroc(s, cte),
                     AUROC_lo=np.nan, AUROC_hi=np.nan, n=int(len(cte)),
                     n_pos=-1)]
    rows, got = [], {}
    for contrast in config.CONTRAST_ORDER:
        r = metrics.cell(s, cte, contrast, n_boot=n_boot)
        got[contrast] = r["AUROC"]
        rows.append(_row(entry, family, segment, c, cfg, train_arm, test_arm,
                         scorer, router_name, scheme, contrast, **r))
    d = metrics.derived(got)
    d["shortcut_IHvCL"] = got.get(config.SHORTCUT, np.nan)
    for name, val in d.items():
        rows.append(_row(entry, family, segment, c, cfg, train_arm, test_arm,
                         scorer, router_name, scheme, name, AUROC=val,
                         AUROC_lo=np.nan, AUROC_hi=np.nan,
                         n=int(len(cte)), n_pos=-1))
    return rows


def _row(entry, family, segment, c, cfg, train_arm, test_arm, scorer,
         router_name, scheme, contrast, **rest) -> dict:
    return {
        "pair": entry.pair, "model": entry.model, "dataset": entry.dataset,
        "modality": entry.modality, "family": family, "segment": segment,
        "C": float(c), "c_mode": cfg.c_mode,
        "prompt_template": entry.prompt_template,
        "train_arm": train_arm, "test_arm": test_arm,
        "diagonal": train_arm == test_arm,
        "scorer": scorer, "router": router_name,
        # `-` means the cell never touched the sampled block, so it is shared
        # across schemes rather than missing from them.
        "sampled_scheme": scheme,
        # `head` keeps routed_long readable by per_pair_long's loaders.
        "head": scorer, "contrast": contrast, **rest,
    }


def _basis_for(entry, arms, family, segment) -> tuple[int | None, int | None]:
    if family in config.HS_FAMILIES:
        return None, None
    sel = select_mod.select(entry, family, segment, arms["dse_natural"].train)
    return sel.top_k, sel.pca_dim
