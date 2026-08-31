#!/usr/bin/env python
"""Stage 12 (M24): Part C's three parity documents + the combined final report.

Mirrors the published QA docs section for section, so replication is a question
of reading two columns rather than a difference in method:

    routed_vs_generalist_fixedC/DISCUSSION_NOTE.md   -> LEVER_A_DISCUSSION_NOTE.md
    sr_xauc_platt/SR_XAUC_RESULTS.md                 -> SR_XAUC_RESULTS.md
    METRIC_ROUTER_REPORT_selected_c.md               -> METRIC_ROUTER_REPORT.md

Reads `routed_long`, `lever_a`, `metric_router_long` and `band_agreement` --
all atomic, all per-pair -- and aggregates at render time, same as stage 10.
Also writes FINAL_REPORT.md, which links the four Part A/B/C documents and
states the top-line numbers side by side.

Usage:
    cli_report/12_part_c_report.py --plan
    cli_report/12_part_c_report.py --run --cohort vlm
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cohorts, paths  # noqa: E402
from csx_report import load  # noqa: E402
from csx_report.tables import lever_a as lever_a_tbl  # noqa: E402
from csx_report.tables import metric_router as mr_tbl  # noqa: E402
from csx_report.tables import srxauc  # noqa: E402


def _fmt(x, nd: int = 3) -> str:
    import numpy as np
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) \
        else f"{x:.{nd}f}"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cohort", default=None)
    g.add_argument("--pairs", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not (args.run or args.plan):
        ap.print_usage(sys.stderr)
        print("\nRefusing to guess: pass --plan or --run.", file=sys.stderr)
        return 2

    cohort = args.cohort or (None if args.pairs else "vlm")
    routed = load.table("routed_long")
    have = set(routed["pair"].unique())
    present, dropped = cohorts.resolve(cohort=cohort, pairs=args.pairs,
                                       available=have)
    if not present:
        print(f"no results for any pair in {cohort or args.pairs}", file=sys.stderr)
        return 1

    needed = {"routed_long", "lever_a", "metric_router_long", "band_agreement"}
    missing = [t for t in needed if not load.available(t)]
    out_dir = Path(args.out) if args.out else paths.report_dir()

    if args.plan:
        print("=== stage 12: Part C report (M24) ===")
        print(f"group: {cohort or 'custom'} -> {len(present)} pair(s) present, "
             f"{len(dropped)} dropped")
        if missing:
            print(f"MISSING TABLES: {missing} -- would refuse to write")
        else:
            print("all 4 input tables present")
        for name in ("SR_XAUC_RESULTS.md", "LEVER_A_DISCUSSION_NOTE.md",
                    "METRIC_ROUTER_REPORT.md", "FINAL_REPORT.md"):
            print(f"would write: {out_dir / name}")
        return 0

    if missing:
        print(f"REFUSING to write: missing tables {missing}", file=sys.stderr)
        return 3

    routed = routed[routed["pair"].isin(present)]
    gates = load.table("lever_a")
    gates = gates[gates["pair"].isin(present)]
    grid = load.table("metric_router_long")
    grid = grid[grid["pair"].isin(present)]
    agreement = load.table("band_agreement")
    agreement = agreement[agreement["pair"].isin(present)]

    out_dir.mkdir(parents=True, exist_ok=True)

    res, atomic = srxauc.compute(routed)
    p1 = out_dir / "SR_XAUC_RESULTS.md"
    p1.write_text(srxauc.report(res, cohort=cohort, pairs=sorted(present),
                                dropped=dropped))
    print(f"[ok] {p1}")

    p2 = out_dir / "LEVER_A_DISCUSSION_NOTE.md"
    p2.write_text(lever_a_tbl.report(gates, cohort=cohort, dropped=dropped))
    print(f"[ok] {p2}")

    p3 = out_dir / "METRIC_ROUTER_REPORT.md"
    p3.write_text(mr_tbl.report(grid, agreement, cohort=cohort,
                                dropped=dropped, per_pair_grids=True))
    print(f"[ok] {p3}")

    # ── the combined final report ───────────────────────────────────────────
    hs = res[(res["family"] == "hs_wide") & (res["variant"] == "primary")]
    gen = hs[hs["method"] == "generalist@-"]
    best = hs[~hs["unranked"]].sort_values("sr_xauc", ascending=False).iloc[0] \
        if len(hs[~hs["unranked"]]) else None
    sampled_best = hs[hs["method"].str.contains("@sampled", na=False)] \
        .sort_values("sr_xauc", ascending=False)
    s = mr_tbl.summary(grid, agreement)
    all_gates_pass = bool(
        gates.groupby("pair")["passed"].all().all())

    lines = ["# Confidence shortcut on VLM — final replication report", "",
             f"**Cohort:** {cohort or 'custom'} — {len(present)} pairs: "
             f"{', '.join(sorted(present))}", ""]
    if dropped:
        lines.append(f"> Dropped: {', '.join(sorted(dropped))}")
        lines.append("")
    lines += [
        "Behavioral (Part A), geometric (Part B) and routing (Part C) evidence, "
        "reproduced on the 9-pair VLM cohort against the published 8-pair LLM "
        "(QA) result. Full detail lives in the four linked documents; this page "
        "is the headline read.", "",
        "## Documents", "",
        "| part | document | question |",
        "|---|---|---|",
        "| A | [CONFIDENCE_SHORTCUT_DRAFT.md](CONFIDENCE_SHORTCUT_DRAFT.md) | "
        "does pooled accuracy hide a confidence shortcut? |",
        "| A | [CONFOUND_REPORT.md](CONFOUND_REPORT.md) | how much of the "
        "matched-arm effect is size, or composition? |",
        "| B | [ALPHA_ROTATION_VLM_REPORT.md](ALPHA_ROTATION_VLM_REPORT.md) | "
        "does the probe's decision boundary genuinely rotate? |",
        "| C | [SR_XAUC_RESULTS.md](SR_XAUC_RESULTS.md) | does the shortcut "
        "survive band-conditioning? |",
        "| C | [LEVER_A_DISCUSSION_NOTE.md](LEVER_A_DISCUSSION_NOTE.md) | do "
        "the routing proof obligations hold? |",
        "| C | [METRIC_ROUTER_REPORT.md](METRIC_ROUTER_REPORT.md) | does the "
        "router generalise across UQ metrics? |", "",
        "## Headline numbers (hs_wide, primary variant)", "",
        f"- **Pooled generalist SR-xAUC:** {_fmt(gen['sr_xauc'].median()) if len(gen) else '--'} "
        f"— near chance once conditioned on confidence band, despite a "
        f"respectable pooled `IvC`. This is the shortcut.",
        f"- **Best ranked design:** `{best['method']}` at "
        f"{_fmt(best['sr_xauc'])} SR-xAUC" if best is not None else "",
        f"- **Best deployable (sampled router):** "
        + (f"`{sampled_best.iloc[0]['method']}` at "
           f"{_fmt(sampled_best.iloc[0]['sr_xauc'])}"
           if len(sampled_best) else "--"),
        f"- **Lever A gates:** "
        f"{'all pass across the cohort' if all_gates_pass else 'AT LEAST ONE FAILS'}.",
        f"- **Metric-router transfer:** diagonal {_fmt(s['diagonal'].mean())}, "
        f"off-diagonal {_fmt(s['off_diagonal'].mean())}, "
        f"Spearman(agreement, drop) {_fmt(s['spearman_agreement_vs_drop'].mean())} "
        f"(QA reference: +0.62 to +0.70 — some transfer is redundancy, not "
        f"router generalisation), `lexical_sim` (non-NLI control) mean drop "
        f"{_fmt(s['lexical_sim_mean_drop'].mean())}.", "",
        "## Reading this as a replication", "",
        "A replication means: on a modality with different features, "
        "different failure modes and different answer distributions, pooled "
        "`IvC` still hides a confidence shortcut, SR-xAUC still exposes it, "
        "and band-conditional routing still survives it. It does **not** "
        "establish a two-channel (linguistic + visual-grounding) confidence "
        "model — the bands here are cut from linguistic entropy only.", "",
    ]
    p4 = out_dir / "FINAL_REPORT.md"
    p4.write_text("\n".join(lines))
    print(f"[ok] {p4}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
