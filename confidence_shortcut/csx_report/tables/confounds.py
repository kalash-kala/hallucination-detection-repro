"""The confound ladder: how much of the matched-arm effect is size, or composition?

Four training regimes, all scored on the SAME test population (`natural.test`),
so the rungs differ only in what the probe was trained on:

    natural      the full skewed pool                      -- the leaky baseline
    ns_A         natural's skew, A's row count             -- SIZE alone
    pl_A         A's per-cell counts, uniform inside cells -- COMPOSITION alone
    A            all of that plus entropy-matching         -- the treatment

Reading `natural -> ns_A` gives the cost of shrinking; `ns_A -> pl_A` the cost of
rebalancing; `pl_A -> A` what is left for the entropy-matching itself. That last
step is the number the write-up's claim actually rests on, and before these grids
existed it was not separable from the other two.

**Why `natural.test` and not each arm's own test split.** A ladder whose rungs
were each scored on their own population would confound a change in the probe
with a change in what it was scored against -- exactly the ambiguity the controls
exist to remove. One test population, four probes.

**Draws are medianed within a pair before pairs are medianed.** The 20 draws are
replicates of one control, not 20 independent pairs; pooling them into the
across-pair median would weight a pair by its draw count and shrink the CI to
something the design does not support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from csx_report.aggregate import Grouped, _warn_mixed, hier_median_ci

TEST_ARM = "dse_natural"
RUNGS = ("natural", "size_only", "placebo", "matched")


def _per_pair_control(df: pd.DataFrame, *, head: str, contrast: str,
                      segment: str) -> pd.DataFrame:
    """One value per `(pair, family, target)`: the median over the 20 draws."""
    if not len(df):
        return pd.DataFrame(columns=["pair", "family", "target", "AUROC"])
    d = df[(df["head"] == head) & (df["contrast"] == contrast)
           & (df["test_arm"] == TEST_ARM) & (df["segment"] == segment)]
    if not len(d):
        return pd.DataFrame(columns=["pair", "family", "target", "AUROC"])
    return (d.groupby(["pair", "family", "target"], as_index=False)["AUROC"]
             .median())


def _per_pair_real(per_pair: pd.DataFrame, *, train_arm: str, head: str,
                   contrast: str, segment: str) -> pd.DataFrame:
    d = per_pair[(per_pair["head"] == head)
                 & (per_pair["contrast"] == contrast)
                 & (per_pair["test_arm"] == TEST_ARM)
                 & (per_pair["segment"] == segment)
                 & (per_pair["train_arm"] == train_arm)]
    return d[["pair", "family", "AUROC"]].copy()


def ladder(per_pair: pd.DataFrame, placebo: pd.DataFrame,
           sizeonly: pd.DataFrame, *, head: str = "sep",
           contrast: str = "IvC", segment: str = "all",
           cohort: str | None = None) -> Grouped:
    """The four-rung ladder per `(family, target)`, medianed over pairs.

    Defaults to `head='sep'`: the confidence probe is the one whose collapse the
    size objection was raised against, so it is the head the controls were built
    to answer for. `head='g'` is the companion question and is rendered too.
    """
    targets = sorted(set(placebo.get("target", pd.Series(dtype=str)).unique())
                     | set(sizeonly.get("target", pd.Series(dtype=str)).unique()))
    pl = _per_pair_control(placebo, head=head, contrast=contrast,
                           segment=segment)
    ns = _per_pair_control(sizeonly, head=head, contrast=contrast,
                           segment=segment)
    nat = _per_pair_real(per_pair, train_arm="dse_natural", head=head,
                         contrast=contrast, segment=segment)

    rows = []
    incomplete: set = set()
    for target in targets:
        real = _per_pair_real(per_pair, train_arm=target, head=head,
                              contrast=contrast, segment=segment)
        fams = sorted(set(nat["family"]) | set(real["family"])
                      | set(pl["family"]) | set(ns["family"]))
        for family in fams:
            frames = {
                "natural": nat[nat["family"] == family],
                "size_only": ns[(ns["family"] == family)
                                & (ns["target"] == target)],
                "placebo": pl[(pl["family"] == family)
                              & (pl["target"] == target)],
                "matched": real[real["family"] == family],
            }
            if not any(len(v) for v in frames.values()):
                continue
            # Every rung must be medianed over the SAME pairs. Otherwise a rung
            # that only some pairs have -- normal mid-run, when the controls lag
            # the transfer grid -- is differenced against a rung over a wider
            # group, and the step reads as an effect that is really a change of
            # population. That inverts the sign in practice, so it is enforced
            # rather than warned about.
            common = set.intersection(*(set(f["pair"]) for f in frames.values()))
            missing = sorted(set(nat["pair"]) - common)
            if not common:
                continue
            vals = {k: f[f["pair"].isin(common)]["AUROC"]
                    for k, f in frames.items()}
            rec: dict = {"family": family, "target": target, "head": head,
                         "contrast": contrast, "segment": segment}
            for rung in RUNGS:
                v = vals[rung].to_numpy(dtype=float)
                if len(v):
                    med, lo, hi = hier_median_ci(v)
                    rec[rung] = med
                    rec[f"{rung}_lo"], rec[f"{rung}_hi"] = lo, hi
                    rec[f"n_{rung}"] = int(len(v))
                else:
                    rec[rung] = np.nan
                    rec[f"{rung}_lo"] = rec[f"{rung}_hi"] = np.nan
                    rec[f"n_{rung}"] = 0
            # The three steps, which is what the ladder is read for.
            rec["d_size"] = rec["size_only"] - rec["natural"]
            rec["d_composition"] = rec["placebo"] - rec["size_only"]
            rec["d_matching"] = rec["matched"] - rec["placebo"]
            rec["n_common"] = len(common)
            rec["excluded"] = ",".join(missing)
            rows.append(rec)
            if missing:
                incomplete.add((family, target, tuple(missing)))

    pairs = sorted(set(per_pair["pair"].unique()))
    warn = _warn_mixed(per_pair)
    if incomplete:
        dropped = sorted({p for _, _, ms in incomplete for p in ms})
        warn.append(
            f"the confound grids do not yet cover every pair, so each ladder is "
            f"restricted to the pairs that have all four rungs; excluded here: "
            f"{', '.join(dropped)}")
    if len(placebo) and len(sizeonly):
        pl_pairs = set(placebo["pair"].unique())
        ns_pairs = set(sizeonly["pair"].unique())
        if pl_pairs != ns_pairs:
            warn.append(
                f"placebo and size-only cover different pairs "
                f"(only-placebo={sorted(pl_pairs - ns_pairs)}, "
                f"only-sizeonly={sorted(ns_pairs - pl_pairs)}); the two ladders "
                f"are not over the same group")
    return Grouped(pd.DataFrame(rows), len(pairs), pairs, warn)


# ── rendering ────────────────────────────────────────────────────────────────

def _f(x, nd: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{nd}f}"


def _signed(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.3f}"


def table_for(g: Grouped, *, family: str) -> list[str]:
    d = g.table[g.table["family"] == family]
    if not len(d):
        return []
    out = [f"### {family}", "",
           "| target | natural | +size | +composition | +matching | "
           "Δsize | Δcomp | Δmatch |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in d.sort_values("target").iterrows():
        out.append(
            f"| {r['target']} | {_f(r['natural'])} | {_f(r['size_only'])} | "
            f"{_f(r['placebo'])} | {_f(r['matched'])} | "
            f"{_signed(r['d_size'])} | {_signed(r['d_composition'])} | "
            f"{_signed(r['d_matching'])} |")
    out.append("")
    return out


def report(by_head: dict[str, Grouped], *, cohort: str | None = None,
           dropped: list[str] | None = None,
           families: list[str] | None = None) -> str:
    """`CONFOUND_REPORT.md` -- the ladder for each head."""
    any_g = next(iter(by_head.values()))
    lines = ["# Confound ladder — is it entropy, or is it n?", ""]
    lines.append(
        f"**Group:** {cohort or 'custom group'} — {any_g.n_pairs} pair(s): "
        f"{', '.join(any_g.pairs)}")
    if dropped:
        lines.append(f"> Dropped (no results): {', '.join(sorted(dropped))}")
    lines.append("")
    for w in dict.fromkeys(w for g in by_head.values() for w in g.warnings):
        lines.append(f"> **Caveat:** {w}.")
        lines.append("")
    lines += [
        f"All four rungs are scored on the same test population "
        f"(`{TEST_ARM}.test`), so they differ only in what the probe was "
        f"trained on. Control draws are medianed within a pair before pairs "
        f"are medianed.", "",
        "`Δsize` is the cost of shrinking alone, `Δcomp` of rebalancing on top "
        "of that, and `Δmatch` what is left for the entropy-matching itself — "
        "the step the claim rests on.", ""]

    for head, g in by_head.items():
        if not len(g.table):
            continue
        lbl = {"sep": "`sep` — the confidence probe",
               "g": "`g` — the correctness probe"}.get(head, f"`{head}`")
        lines += [f"## {lbl}", ""]
        fams = families or sorted(g.table["family"].unique())
        for family in fams:
            lines += table_for(g, family=family)
    return "\n".join(lines) + "\n"
