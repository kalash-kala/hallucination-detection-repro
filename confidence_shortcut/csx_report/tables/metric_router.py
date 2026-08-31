"""The metric-router transfer grid: does a router learn 'the confidence band',
or just 'this metric's particular threshold'?

Each cell is `AUROC(router trained on metric A's band, scored on metric B's
band)`. The diagonal reproduces the ordinary routed result. The off-diagonal
is the transfer question, and `drop = diagonal(B) - offdiag(A->B)` is read
against `band_agreement`, the honest denominator: metrics built from the same
10 generations agree on the band substantially before any router is fit, so
some transfer is free. `lexical_sim` is the one metric with no NLI in it — if
transfer survives there, the effect is not an entailment artefact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt(x, nd: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{nd}f}"


def summary(grid: pd.DataFrame, agreement: pd.DataFrame) -> pd.DataFrame:
    """One row per pair: diagonal / off-diagonal AUROC, mean band agreement,
    and how much of the drop is explained by disagreement (Spearman).

    `drop(A->B) = AUROC(train A, test B) - AUROC(train B, test B)` -- the
    PLAN's sign, offdiag minus diagonal, so drop is usually <= 0 and LESS
    negative (closer to 0) means better transfer. Spearman(agreement, drop)
    is then positive when higher agreement predicts less degradation, which
    is the sign the QA reference (+0.62 to +0.70) is reported in. Getting
    this backwards silently flips the whole "redundancy explains it away"
    reading, so it is spelled out rather than left implicit in a minus sign.
    """
    from scipy.stats import spearmanr

    rows = []
    for pair, g in grid.groupby("pair"):
        # Averaged over scheme/arm: a cell can be scored under more than one
        # sampled aggregation, and the diagonal/drop comparison is about the
        # metric, not the scheme.
        dia = (g[g["diagonal"]].groupby("train_metric")["router_auroc"].mean())
        off = (g[~g["diagonal"]]
              .groupby(["train_metric", "test_metric"], as_index=False)
              ["router_auroc"].mean())
        ag = agreement[agreement["pair"] == pair]
        merged = off.merge(
            ag.rename(columns={"metric_a": "train_metric",
                               "metric_b": "test_metric"}),
            on=["train_metric", "test_metric"], how="left")
        merged["drop"] = (merged["router_auroc"]
                          - merged["test_metric"].map(dia))
        rho = (spearmanr(merged["agreement"], merged["drop"]).statistic
              if merged["agreement"].notna().sum() >= 3 else float("nan"))
        lex = merged[(merged["train_metric"] == "lexical_sim")
                     | (merged["test_metric"] == "lexical_sim")]
        rows.append({
            "pair": pair,
            "n_cells": len(g),
            "diagonal": float(dia.mean()),
            "off_diagonal": float(off["router_auroc"].mean()),
            "mean_agreement_offdiag": float(
                ag[ag["metric_a"] != ag["metric_b"]]["agreement"].mean()),
            "spearman_agreement_vs_drop": float(rho),
            "lexical_sim_mean_drop": float(lex["drop"].mean()) if len(lex) else float("nan"),
        })
    return pd.DataFrame(rows)


def cell_table(grid: pd.DataFrame, *, pair: str) -> list[str]:
    d = grid[grid["pair"] == pair]
    if not len(d):
        return []
    metrics = sorted(set(d["train_metric"]) | set(d["test_metric"]))
    piv = d.pivot_table(index="train_metric", columns="test_metric",
                        values="router_auroc")
    out = [f"### {pair}", "",
           "| train \\ test | " + " | ".join(metrics) + " |",
           "|---" * (len(metrics) + 1) + "|"]
    for m in metrics:
        row = piv.loc[m] if m in piv.index else None
        cells = " | ".join(
            _fmt(row[c]) if row is not None and c in row.index else "--"
            for c in metrics)
        out.append(f"| {m} | {cells} |")
    out.append("")
    return out


def report(grid: pd.DataFrame, agreement: pd.DataFrame, *,
           cohort: str | None = None, dropped: list[str] | None = None,
           per_pair_grids: bool = False) -> str:
    """`METRIC_ROUTER_REPORT.md` analogue."""
    pairs = sorted(grid["pair"].unique())
    lines = ["# Metric-router grid — does the band router generalise across metrics?",
             ""]
    lines.append(f"**Group:** {cohort or 'custom group'} — {len(pairs)} pair(s): "
                 f"{', '.join(pairs)}")
    if dropped:
        lines.append(f"> Dropped (no results): {', '.join(sorted(dropped))}")
    lines.append("")
    lines += [
        "Diagonal reproduces the ordinary routed result for that metric's own "
        "band; off-diagonal trains on one metric's band and scores against "
        "another's — legal only because the DSE train/test partition is reused "
        "verbatim across metrics (`assert_leak_free`), never re-split.", "",
        "`spearman(agreement, drop)` is the concession the QA run had to make: "
        "some of the transfer is redundancy between metrics built on the same "
        "10 generations, not a router property. A LOWER value here is the "
        "better result for the claim.", "",
    ]

    s = summary(grid, agreement)
    lines += ["| pair | cells | diagonal | off-diagonal | mean band agreement "
             "(off-diag) | Spearman(agreement, drop) | lexical_sim mean drop |",
             "|---|---|---|---|---|---|---|"]
    for _, r in s.sort_values("pair").iterrows():
        lines.append(
            f"| {r['pair']} | {int(r['n_cells'])} | {_fmt(r['diagonal'])} | "
            f"{_fmt(r['off_diagonal'])} | {_fmt(r['mean_agreement_offdiag'])} | "
            f"{_fmt(r['spearman_agreement_vs_drop'])} | "
            f"{_fmt(r['lexical_sim_mean_drop'])} |")
    lines.append("")
    lines.append(
        f"**Cohort mean:** diagonal {_fmt(s['diagonal'].mean())}, "
        f"off-diagonal {_fmt(s['off_diagonal'].mean())}, "
        f"Spearman(agreement, drop) {_fmt(s['spearman_agreement_vs_drop'].mean())}, "
        f"`lexical_sim` mean drop {_fmt(s['lexical_sim_mean_drop'].mean())}.")
    lines.append("")

    if per_pair_grids:
        lines += ["## Per-pair 8x8 grids", ""]
        for p in pairs:
            lines += cell_table(grid, pair=p)

    return "\n".join(lines) + "\n"
