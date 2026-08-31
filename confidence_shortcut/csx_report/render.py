"""Rendering the aggregated tables to markdown.

Every number here is pulled from the atomic tables at render time. Nothing is
transcribed, so a report cannot drift from the results it describes.

The caveats travel with the numbers rather than living in a paragraph someone has
to remember to update: a group that mixes `C` or prompt templates says so on the
page, a dropped pair is named, and the pass-rule bar is printed next to the count
that is being compared against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from csx_report.aggregate import Grouped


def _fmt(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def header(title: str, g: Grouped, *, cohort: str | None = None,
           dropped: list[str] | None = None) -> list[str]:
    out = [f"# {title}", ""]
    label = cohort or "custom group"
    out.append(f"**Group:** {label} — {g.n_pairs} pair(s): "
               f"{', '.join(g.pairs)}")
    if dropped:
        # A shrinking denominator changes every median, so it is stated, never
        # silently absorbed.
        out.append("")
        out.append(f"> **{len(dropped)} pair(s) named but absent from the "
                   f"results and dropped:** {', '.join(dropped)}. "
                   f"Every median below is over the {g.n_pairs} that remain.")
    for w in g.warnings:
        out.append("")
        out.append(f"> **Caveat:** {w}.")
    out.append("")
    return out


def transfer_matrix(g: Grouped, *, family: str, segment: str = "all",
                    value: str = "AUROC") -> list[str]:
    """The train-arm x test-arm grid for one family, as a markdown table."""
    d = g.table[(g.table["family"] == family) &
                (g.table["segment"] == segment)]
    if not len(d):
        return [f"_no rows for {family}/{segment}_", ""]
    piv = d.pivot(index="train_arm", columns="test_arm",
                  values=f"{value}_median")
    cols = list(piv.columns)
    out = [f"### {family} ({segment})", "",
           "| train \\ test | " + " | ".join(c.replace("dse_", "") for c in cols)
           + " |",
           "|---" * (len(cols) + 1) + "|"]
    for arm, row in piv.iterrows():
        cells = " | ".join(_fmt(row[c]) for c in cols)
        out.append(f"| {arm.replace('dse_', '')} | {cells} |")
    out.append("")
    return out


def verdict_table(g: Grouped) -> list[str]:
    """The pass-rule rollup, with the bar printed beside every count."""
    if not len(g.table):
        return ["_no verdict rows_", ""]
    out = ["| family | metric | median Δθ | passes | bar | meets bar |",
           "|---|---|---|---|---|---|"]
    for _, r in g.table.sort_values(["family", "metric"]).iterrows():
        out.append(
            f"| {r['family']} | {r['metric']} | {_fmt(r['delta_median'], 2)}° "
            f"| {int(r['n_pass'])}/{int(r['n_pairs'])} | >= {int(r['threshold'])} "
            f"| {'yes' if r['meets_bar'] else 'no'} |")
    out.append("")
    out.append(f"_Bar is 6/8 for `qa8` verbatim, `ceil(0.75 n)` otherwise; the "
               f"two coincide at n=8._")
    out.append("")
    return out


def shortcut_report(g: Grouped, *, cohort: str | None = None,
                    dropped: list[str] | None = None,
                    families: list[str] | None = None,
                    segment: str = "all") -> str:
    lines = header("Confidence shortcut — transfer grid", g,
                   cohort=cohort, dropped=dropped)
    lines += [
        "Median AUROC across pairs, head `g` (correctness), contrast `IvC`.",
        "",
        "Rows are the arm a probe was **trained** on; columns the arm it was ",
        "**tested** on. The off-diagonal cells are the question: a probe trained ",
        "on `natural` and tested on `matched2` is being asked to work where ",
        "entropy is exactly uninformative about correctness within band, so ",
        "whatever it retains there is not coming from the shortcut.",
        "",
    ]
    fams = families or sorted(g.table["family"].unique())
    for f in fams:
        lines += transfer_matrix(g, family=f, segment=segment)
    return "\n".join(lines)


def rotation_report(g: Grouped, *, cohort: str | None = None,
                    dropped: list[str] | None = None) -> str:
    lines = header("Alpha rotation — verdict", g, cohort=cohort,
                   dropped=dropped)
    lines += [
        "Does entropy-matching move the correctness probe to a genuinely ",
        "different direction? `Δθ` is the rotation from the natural-trained ",
        "probe (α=0) to the matched-trained one (α=1), against the confidence ",
        "axis. A pair passes when `Δθ` exceeds the 95th percentile of **its own** ",
        "placebo null — never when it merely exceeds the natural arm, since a ",
        "smaller training set rotates away from any fixed reference for free.",
        "",
    ]
    lines += verdict_table(g)
    return "\n".join(lines)
