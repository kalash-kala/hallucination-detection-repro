#!/usr/bin/env python
"""Stage 04: the transfer grid — train on arm i, test on arm j.

The headline experiment. Emits atomic per-pair rows only; medians and cohorts are
component 3's job.

Cost is the thing to watch. Per (pair, family, segment) it is 4 fits plus
4x4x3x9 scored cells, each with a 1000-draw bootstrap. `--n-boot` trades CI
precision for wall time when iterating; the published value is 1000 and the
default here is the published value.

Usage:
    cli_probe/04_transfer_grid.py --plan
    cli_probe/04_transfer_grid.py --run --pairs qwen25vl_advqa --segments all
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import config, results  # noqa: E402
from csx_probe.experiments import c_selection, transfer_grid  # noqa: E402
from csx_probe.store import read  # noqa: E402


def _cfg(pair: str, families: tuple[str, ...], n_boot: int) -> config.RunConfig:
    """Resolve C for this pair, reading the stored CV curve if the policy needs it."""
    pol = config.c_policy()
    mode = pol.get("assignment", {}).get(pair, pol["default"])
    per_unit = None
    if mode == "per_pair":
        per_unit = c_selection.per_unit_best(pair)
    return config.RunConfig.for_pair(pair, per_unit_best=per_unit,
                                     families=families, n_boot=n_boot)


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
            outputs=(paths.results_units(transfer_grid.TABLE, key),),
            unmet=tuple(unmet), note=f"C mode={mode}",
            cost=f"{len(families)} fam x 4 train-arms x 4 test-arms",
        ))
    return out


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0])
    ap.add_argument("--families", default=None)
    ap.add_argument("--segments", default="all",
                    help="default: all only -- the 8-QA parity target never "
                         "had image/text, pass e.g. --segments all,image,text "
                         "to opt into the VLM-only segment extension")
    ap.add_argument("--n-boot", type=int,
                    default=config.frozen()["bootstrap"]["row_level_n"])
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)

    families = (tuple(args.families.split(",")) if args.families
                else config.FAMILIES)
    segments = tuple(args.segments.split(",")) if args.segments else None
    pairs = [p.key for p in registry.resolve(
        args.pairs, include_pending=args.include_pending)]
    us = units(pairs, families)

    if mode == "plan":
        cli.print_plan("stage 04: transfer grid", us, force=args.force)
        return 0

    def do(u: cli.Unit) -> str:
        cfg = _cfg(u.key, families, args.n_boot)
        df = transfer_grid.run_pair(u.key, cfg, segments=segments,
                                    families=families, verbose=False)
        return f"{len(df)} rows"

    rc = cli.run_units("stage 04: transfer grid", us, do, force=args.force,
                        pair_jobs=args.pair_jobs)
    try:
        df = results.consolidate(transfer_grid.TABLE)
        print(f"{transfer_grid.TABLE}: {len(df)} rows over "
              f"{df['pair'].nunique()} pairs")
    except FileNotFoundError:
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
