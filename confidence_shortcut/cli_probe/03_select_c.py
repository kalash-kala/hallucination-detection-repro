#!/usr/bin/env python
"""Stage 03: CV-select `C` per (family, pair), storing the full curve.

Required before stage 04 for any pair in `per_pair` mode -- which is every pair
except the 8 `qa8` ones, whose `C` is pinned to the published table.

The curve is stored, not just the argmax, so switching a pair between policies
later costs a parquet read rather than a CV sweep.

Usage:
    cli_probe/03_select_c.py --plan
    cli_probe/03_select_c.py --run --pairs qwen25vl_advqa
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import config, results  # noqa: E402
from csx_probe.experiments import c_selection  # noqa: E402
from csx_probe.store import read  # noqa: E402


def units(pairs, families):
    out = []
    for key in pairs:
        ok, why = read.is_usable(key)
        mode = config.c_policy().get("assignment", {}).get(
            key, config.c_policy()["default"])
        unmet = () if ok else (why,)
        note = f"mode={mode}"
        if mode == "pinned":
            note += " (pinned: stage 04 reads the frozen table, this is audit only)"
        out.append(cli.Unit(
            key=key,
            outputs=(paths.results_units(c_selection.TABLE, key),),
            unmet=unmet, note=note,
            cost=f"{len(families)} fam x 4 arms x {len(config.c_grid())} C x 5 folds",
        ))
    return out


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0], inner_jobs=True)
    ap.add_argument("--families", default=None,
                    help="comma-separated (default: all 7)")
    ap.add_argument("--segments", default="all",
                    help="comma-separated (default: 'all' only -- the image/text "
                         "segments multiply the cost by 3)")
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)

    families = (tuple(args.families.split(",")) if args.families
                else config.FAMILIES)
    segments = tuple(args.segments.split(","))
    pairs = [p.key for p in registry.resolve(
        args.pairs, include_pending=args.include_pending)]
    us = units(pairs, families)

    if mode == "plan":
        cli.print_plan("stage 03: select C", us, force=args.force)
        return 0

    # Both levels of parallelism come out of ONE core budget: `do` opens
    # its own pool per unit, so the outer width has to be resolved first.
    pair_jobs = cli.resolve_pair_jobs(
        args.pair_jobs, sum(1 for u in us if not u.blocked))
    n_jobs = cli.resolve_inner_jobs(args.n_jobs, pair_jobs)

    def do(u: cli.Unit) -> str:
        df = c_selection.run_pair(u.key, families=families, segments=segments,
                                  n_jobs=n_jobs, verbose=False)
        if not len(df):
            return "no units (families unavailable)"
        best = df.groupby("family")["best_C"].median()
        return ", ".join(f"{f}={c:g}" for f, c in best.items())

    rc = cli.run_units("stage 03: select C", us, do, force=args.force,
                        pair_jobs=args.pair_jobs, inner_fanout=n_jobs)
    try:
        results.consolidate(c_selection.TABLE)
    except FileNotFoundError:
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
