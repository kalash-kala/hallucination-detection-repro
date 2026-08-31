#!/usr/bin/env python
"""Stage 05: the alpha rotation, and its per-pair verdict.

Asks whether entropy-matching moves the correctness probe to a genuinely
different direction, or merely measures the same direction more noisily. The
placebo null is what separates those two, so it is not optional.

Cost is dominated by refits: per (pair, family) it is
`5 rungs x (1 + n_boot + n_placebo)` logistic fits, at the full feature width.
With the published `n_boot=100` that is ~605 fits per family. `--n-boot` is
offered for iteration; the default is the published value.

Usage:
    cli_probe/05_alpha_rotation.py --plan
    cli_probe/05_alpha_rotation.py --run --pairs qwen25vl_advqa --families hs_wide
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import config, results  # noqa: E402
from csx_probe.experiments import alpha_rotation as ar, c_selection  # noqa: E402
from csx_probe.store import read  # noqa: E402


def units(pairs, families):
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
            outputs=(paths.results_units(ar.TABLE, key),),
            unmet=tuple(unmet),
            cost=f"{len(families)} fam x 5 rungs x (1+{ar.N_BOOT}+"
                 f"{config.frozen()['alpha']['n_placebo']}) fits",
        ))
    return out


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0], inner_jobs=True)
    ap.add_argument("--families", default=None)
    ap.add_argument("--segment", default="all")
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)

    families = (tuple(args.families.split(",")) if args.families
                else config.FAMILIES)
    pairs = [p.key for p in registry.resolve(
        args.pairs, include_pending=args.include_pending)]
    us = units(pairs, families)

    if mode == "plan":
        cli.print_plan("stage 05: alpha rotation", us, force=args.force)
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
        df = ar.run_pair(u.key, cfg, segment=args.segment, families=families,
                         verbose=False, n_jobs=n_jobs)
        v = ar.write_verdict(u.key, df)
        n_pass = int(v["passes"].sum()) if len(v) else 0
        return f"{len(df)} rows, {n_pass}/{len(v)} verdict cells pass"

    rc = cli.run_units("stage 05: alpha rotation", us, do, force=args.force,
                        pair_jobs=args.pair_jobs, inner_fanout=n_jobs)
    for table in (ar.TABLE, ar.VERDICT_TABLE):
        try:
            results.consolidate(table)
        except FileNotFoundError:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
