#!/usr/bin/env python
"""Stage 10: aggregate the atomic tables into a report, for any grouping.

The cohort is a runtime argument, so re-grouping costs seconds and refits
nothing: `--cohort vlm`, `--cohort qa8`, or an arbitrary `--pairs` list, plus
`--split-by` for free cross-tabs.

Usage:
    cli_report/10_report.py --plan
    cli_report/10_report.py --run --cohort vlm
    cli_report/10_report.py --run --pairs qwen25vl_advqa,gemma3_12b_advqa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cohorts, paths  # noqa: E402
from csx_report import aggregate, completeness, draft, load, render  # noqa: E402
from csx_report.tables import confounds as confound_tbl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--cohort", default=None)
    g.add_argument("--pairs", default=None)
    ap.add_argument("--segment", default="all")
    ap.add_argument("--families", default=None)
    ap.add_argument("--out", default=None, help="directory (default: <store>/reports)")
    ap.add_argument("--split-by", default=None,
                    help="also emit one report per value of this column "
                         "(model, dataset, modality). Free cross-tabs: the "
                         "atomic tables are already per-pair, so this refits "
                         "nothing.")
    ap.add_argument("--headline-family", default="hs_wide",
                    help="the family the per-cell and transfer sections detail")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the report even though pairs or (family, "
                         "segment) cells are missing, stamping the shortfall "
                         "into the document. Without this a partial group is "
                         "refused, because a partial report gets read as the "
                         "result.")
    args = ap.parse_args()

    if not (args.run or args.plan):
        ap.print_usage(sys.stderr)
        print("\nRefusing to guess: pass --plan or --run.", file=sys.stderr)
        return 2

    cohort = args.cohort or (None if args.pairs else "vlm")
    per_pair = load.table("per_pair_long")
    have = set(per_pair["pair"].unique())
    present, dropped = cohorts.resolve(cohort=cohort, pairs=args.pairs,
                                       available=have)
    if not present:
        print(f"no results for any pair in {cohort or args.pairs}", file=sys.stderr)
        return 1

    cov = completeness.check(per_pair, present, dropped)

    out_dir = Path(args.out) if args.out else paths.report_dir()
    if args.plan:
        print(f"=== stage 10: report ===")
        print(f"group: {cohort or 'custom'} -> {len(present)} pair(s) present, "
              f"{len(dropped)} dropped")
        if cov.complete:
            print(f"coverage: complete ({len(cov.reference)} "
                  f"(family, segment) cells per pair)")
        else:
            print("coverage: INCOMPLETE")
            for r in cov.reasons():
                print(f"  - {r}")
            print("  would refuse to write (pass --allow-partial to override)"
                  if not args.allow_partial else
                  "  would write anyway, stamped PARTIAL (--allow-partial)")
        print(f"would write: {out_dir/'CONFIDENCE_SHORTCUT_REPORT.md'}")
        if load.available("verdict"):
            print(f"             {out_dir/'ALPHA_ROTATION_REPORT.md'}")
        return 0

    if not cov.complete and not args.allow_partial:
        print(f"=== stage 10: report ===", file=sys.stderr)
        print(f"REFUSING to write a report for {cohort or 'custom'}: the group "
              f"is incomplete.\n", file=sys.stderr)
        for r in cov.reasons():
            print(f"  - {r}", file=sys.stderr)
        print("\nFinish the missing units and re-run, or pass --allow-partial "
              "to write it anyway with the shortfall stamped into the "
              "document.", file=sys.stderr)
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    fams = args.families.split(",") if args.families else None

    def _sub(name, keep):
        """One results table restricted to `keep`, or empty if never written."""
        if not load.available(name):
            return pd.DataFrame()
        t = load.table(name)
        return t[t["pair"].isin(keep)]

    def emit(keep: list[str], *, label: str | None, suffix: str = "") -> None:
        sub = per_pair[per_pair["pair"].isin(keep)]
        if not len(sub):
            return
        # Recomputed per group: a split-by subset can be complete even when the
        # parent is, or the reverse, and the banner has to describe the document
        # it is actually attached to.
        sub_cov = completeness.check(per_pair, keep,
                                     dropped if keep == present else [])
        head = completeness.banner(sub_cov)

        def _write(path: Path, text: str) -> None:
            path.write_text(head + "\n" + text if head else text)

        grp = aggregate.shortcut_table(sub, cohort=cohort)
        p = out_dir / f"CONFIDENCE_SHORTCUT_REPORT{suffix}.md"
        _write(p, render.shortcut_report(
            grp, cohort=label, dropped=dropped, families=fams,
            segment=args.segment))
        print(f"[ok] {p} ({grp.n_pairs} pairs)"
              + ("  [PARTIAL]" if head else ""))
        for w in grp.warnings:
            print(f"  caveat: {w}")

        pl, ns = _sub("placebo_long", keep), _sub("sizeonly_long", keep)
        if len(pl) or len(ns):
            by_head = {h: confound_tbl.ladder(sub, pl, ns, head=h,
                                              segment=args.segment,
                                              cohort=label)
                       for h in ("sep", "g")}
            if any(len(g.table) for g in by_head.values()):
                p3 = out_dir / f"CONFOUND_REPORT{suffix}.md"
                _write(p3, confound_tbl.report(
                    by_head, cohort=label, dropped=dropped, families=fams))
                print(f"[ok] {p3}")

        v = _sub("verdict", keep)
        if len(v):
            vg = aggregate.pass_counts(v, cohort=label)
            p2 = out_dir / f"ALPHA_ROTATION_REPORT{suffix}.md"
            _write(p2, render.rotation_report(vg, cohort=label,
                                             dropped=dropped))
            print(f"[ok] {p2}")

        # The assembled Part A + Part B document, in the published draft's own
        # order, so the two can be read side by side.
        p4 = out_dir / f"CONFIDENCE_SHORTCUT_DRAFT{suffix}.md"
        _write(p4, draft.render(
            per_pair=sub, arm_stats=_sub("arm_stats", keep),
            rotation=_sub("rotation_long", keep), verdict=v,
            placebo=pl, sizeonly=ns, cohort=label, dropped=dropped,
            families=fams, segment=args.segment,
            headline_family=args.headline_family))
        print(f"[ok] {p4}")

    emit(present, label=cohort)

    if args.split_by:
        col = args.split_by
        if col not in per_pair.columns:
            print(f"cannot split by {col!r}: not a column in per_pair_long "
                  f"({sorted(per_pair.columns)})", file=sys.stderr)
            return 1
        sub = per_pair[per_pair["pair"].isin(present)]
        for val, g in sub.groupby(col, dropna=False):
            keep = sorted(g["pair"].unique())
            slug = str(val).replace("/", "-")
            print(f"\n--- {col}={val} ({len(keep)} pairs) ---")
            emit(keep, label=f"{cohort or 'custom'} / {col}={val}",
                 suffix=f"__{col}_{slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
