#!/usr/bin/env python
"""Stage 08: Lever A's four gates. CPU only. Non-zero exit on any failure.

These are what make prior-corrected pooling a proof rather than a number that
moved:

    drift                  the base grid is reproduced where the designs coincide
    affine_identity        z == platt_prior within band under ORACLE routing,
                           and NOT under a real router (both halves required)
    ece_falls              per-band calibration error drops
    train_only_provenance  both Platt parameters and the prior come from train

Gate 2's second half is the one that catches a silently broken pipeline: if the
within-band identity survives a fallible router, the routing step is not being
applied at all, and every routed number in the report is really an unrouted one.

Usage:
    cli_probe/08_lever_a.py --plan
    cli_probe/08_lever_a.py --run --pairs qwen25vl_advqa --family hs_wide
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import config, results  # noqa: E402
from csx_probe.experiments import c_selection, lever_a  # noqa: E402
from csx_probe.store import read  # noqa: E402


def _cfg(pair: str, families: tuple[str, ...]) -> config.RunConfig:
    pol = config.c_policy()
    mode = pol.get("assignment", {}).get(pair, pol["default"])
    per_unit = c_selection.per_unit_best(pair) if mode == "per_pair" else None
    return config.RunConfig.for_pair(pair, per_unit_best=per_unit,
                                     families=families)


def units(pairs, family: str):
    out = []
    for key in pairs:
        ok, why = read.is_usable(key)
        unmet = [] if ok else [why]
        if ok and not results.has_unit("per_pair_long", key):
            unmet.append("needs stage 04 (the drift gate compares against the "
                         "stored transfer-grid baseline)")
        out.append(cli.Unit(
            key=key,
            outputs=(paths.results_units(lever_a.TABLE, key),),
            unmet=tuple(unmet), cost=f"4 gates on {family}",
        ))
    return out


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0])
    ap.add_argument("--family", default="hs_wide")
    ap.add_argument("--segment", default="all")
    ap.add_argument("--train-arm", default="dse_natural")
    ap.add_argument("--test-arm", default="dse_natural")
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)

    pairs = [p.key for p in registry.resolve(
        args.pairs, include_pending=args.include_pending)]
    us = units(pairs, args.family)

    if mode == "plan":
        cli.print_plan("stage 08: lever A gates", us, force=args.force)
        return 0

    failures: list[str] = []

    def do(u: cli.Unit) -> str:
        cfg = _cfg(u.key, (args.family,))
        gs = lever_a.check_pair(u.key, cfg, family=args.family,
                                segment=args.segment, train_arm=args.train_arm,
                                test_arm=args.test_arm, verbose=True)
        lever_a.write(u.key, gs)
        bad = [g.gate for g in gs if not g.passed]
        if bad:
            failures.append(f"{u.key}: {', '.join(bad)}")
        return (f"{len(gs) - len(bad)}/{len(gs)} gates pass"
                + (f" -- FAILED: {', '.join(bad)}" if bad else ""))

    rc = cli.run_units("stage 08: lever A gates", us, do, force=args.force,
                        pair_jobs=args.pair_jobs)
    if failures:
        print("\nGATE FAILURES (Lever A is not established for these pairs):")
        for f in failures:
            print(f"  {f}")
        return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
