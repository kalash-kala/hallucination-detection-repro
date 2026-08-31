#!/usr/bin/env python
"""Stage 01: build the four arms for each pair, and check every gate.

CPU-only and cheap, but it is the gate the whole grid rests on: if containment or
the entropy-dead property fails, every downstream number is meaningless while
still looking entirely plausible. Running it as its own stage means that failure
surfaces in seconds rather than after a multi-hour fit.

Usage:
    cli_probe/01_build_arms.py --plan
    cli_probe/01_build_arms.py --run --pairs qwen25vl_advqa
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import results  # noqa: E402
from csx_probe.arms import build as arms_build, gates  # noqa: E402
from csx_probe.experiments import transfer_grid  # noqa: E402
from csx_probe.store import read  # noqa: E402


def units(pairs, force: bool):
    out = []
    for key in pairs:
        ok, why = read.is_usable(key)
        out.append(cli.Unit(
            key=key,
            outputs=(paths.results_units("arm_stats", key),),
            unmet=() if ok else (why,),
            cost="4 arms + gates",
        ))
    return out


def do(u: cli.Unit) -> str:
    entry = read.load(u.key)
    arms = arms_build.build_all(entry)
    problems = gates.check_all(entry, arms, strict=False)
    if problems:
        raise gates.GateError("; ".join(problems[:3]))
    transfer_grid.arm_stats(u.key)
    bits = [f"{n.replace('dse_', '')}={a.n('train')}/{a.n('test')}"
            for n, a in arms.items()]
    notes = [a.note for a in arms.values() if a.note]
    return " ".join(bits) + (f"  [{'; '.join(notes)}]" if notes else "")


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0])
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)
    pairs = [p.key for p in registry.resolve(
        args.pairs, include_pending=args.include_pending)]
    us = units(pairs, args.force)
    if mode == "plan":
        cli.print_plan("stage 01: arms", us, force=args.force)
        return 0
    rc = cli.run_units("stage 01: arms", us, do, force=args.force,
                        pair_jobs=args.pair_jobs)
    try:
        df = results.consolidate("arm_stats")
        print(f"arm_stats: {len(df)} rows over {df['pair'].nunique()} pairs")
    except FileNotFoundError:
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
