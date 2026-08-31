#!/usr/bin/env python
"""Stage 06: the placebo and size-only confound grids.

Separates *entropy-matching* from its two side effects. Each control differs
from `natural` in exactly one way -- `ns_*` in size alone, `pl_*` in composition
alone -- so the difference between a control and its real arm is attributable to
the matching itself rather than to the training set having shrunk or rebalanced.

Cost is dominated by refits: per (pair, family, kind) it is
`3 targets x n_draws` logistic fits plus scoring, at the full feature width.
Both grids are run unless `--kind` names one.

Usage:
    cli_probe/06_confounds.py --plan
    cli_probe/06_confounds.py --run --pairs qwen25vl_advqa --families hs_wide
    cli_probe/06_confounds.py --run --kind sizeonly --n-jobs 16
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import config, results  # noqa: E402
from csx_probe.arms import confound  # noqa: E402
from csx_probe.experiments import c_selection, confounds  # noqa: E402
from csx_probe.store import read  # noqa: E402

KINDS = ("placebo", "sizeonly")
TABLE_OF = {"placebo": confounds.PLACEBO_TABLE,
            "sizeonly": confounds.SIZEONLY_TABLE}


def units(pairs, families, kinds, draws):
    out = []
    for key in pairs:
        ok, why = read.is_usable(key)
        unmet = [] if ok else [why]
        pol = config.c_policy()
        mode = pol.get("assignment", {}).get(key, pol["default"])
        if ok and mode == "per_pair" and not results.has_unit(
                c_selection.TABLE, key):
            unmet.append("needs stage 03 (per_pair C is unresolved)")
        out.append(cli.Unit(
            key=key,
            outputs=tuple(paths.results_units(TABLE_OF[k], key) for k in kinds),
            unmet=tuple(unmet),
            cost=f"{len(families)} fam x {len(kinds)} kind x 3 targets x "
                 f"{draws} draws fits",
        ))
    return out


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0], inner_jobs=True)
    ap.add_argument("--families", default=None)
    ap.add_argument("--segments", default="all")
    ap.add_argument("--kind", default=None, choices=KINDS,
                    help="default: both")
    ap.add_argument("--draws", type=int, default=confound.N_DRAWS)
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)

    families = (tuple(args.families.split(",")) if args.families
                else config.FAMILIES)
    segments = tuple(args.segments.split(","))
    kinds = (args.kind,) if args.kind else KINDS
    pairs = [p.key for p in registry.resolve(
        args.pairs, include_pending=args.include_pending)]
    us = units(pairs, families, kinds, args.draws)

    if mode == "plan":
        cli.print_plan("stage 06: confound grids", us, force=args.force)
        return 0

    # Both levels of parallelism come out of ONE core budget: `do` opens
    # its own pool per unit, so the outer width has to be resolved first.
    pair_jobs = cli.resolve_pair_jobs(
        args.pair_jobs, sum(1 for u in us if not u.blocked))
    n_jobs = cli.resolve_inner_jobs(args.n_jobs, pair_jobs)

    def do(u: cli.Unit) -> str:
        pol = config.c_policy()
        m = pol.get("assignment", {}).get(u.key, pol["default"])
        per_unit = c_selection.per_unit_best(u.key) if m == "per_pair" else None
        cfg = config.RunConfig.for_pair(u.key, per_unit_best=per_unit,
                                        families=families)
        got = []
        for kind in kinds:
            df = confounds.run_pair(u.key, cfg, kind=kind, families=families,
                                    segments=segments, draws=args.draws,
                                    n_jobs=n_jobs, verbose=False)
            got.append(f"{kind} {len(df)}")
        return ", ".join(got) + " rows"

    rc = cli.run_units("stage 06: confound grids", us, do, force=args.force,
                        pair_jobs=args.pair_jobs, inner_fanout=n_jobs)
    for kind in kinds:
        try:
            results.consolidate(TABLE_OF[kind])
        except FileNotFoundError:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
