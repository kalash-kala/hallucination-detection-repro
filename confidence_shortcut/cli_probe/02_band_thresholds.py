#!/usr/bin/env python
"""Stage 02: `best_split` band thresholds per (metric, pair). CPU, seconds.

M21. One 1-D 2-means threshold for each of the 8 uncertainty metrics, so M23's
metric-router grid can relabel `category` per metric without ever re-splitting
the train/test partition.

The `dse` row is a gate, not a result: its threshold is already stored as
`DSE_threshold`, so recovering it exactly proves the binarisation still matches
the one that produced every band label downstream. `--strict` (the default)
exits non-zero if any pair fails that check.

Usage:
    cli_probe/02_band_thresholds.py --plan
    cli_probe/02_band_thresholds.py --run --pairs qwen25vl_advqa
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import results  # noqa: E402
from csx_probe.experiments import band_thresholds as bt  # noqa: E402


def units(pairs):
    out = []
    for key in pairs:
        unmet = []
        # M20's output is the hard dependency: 7 of the 8 metrics live there.
        if not paths.uq_metrics(key).exists():
            unmet.append("needs M20 (cli_extract/28_uq_metrics.py): no "
                         "uq_metrics.parquet")
        out.append(cli.Unit(
            key=key,
            outputs=(paths.results_units(bt.TABLE, key),),
            unmet=tuple(unmet),
            cost=f"{len(bt.METRICS)} metrics x {bt.bs.N_CUTS} cuts",
        ))
    return out


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0])
    ap.add_argument("--n-cuts", type=int, default=None,
                    help="scan resolution; the stored thresholds are only "
                         "reproducible at the default 100")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="report a failed dse gate instead of exiting non-zero")
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)

    n_cuts = args.n_cuts or bt.bs.N_CUTS
    pairs = [p.key for p in registry.resolve(
        args.pairs, include_pending=args.include_pending)]
    us = units(pairs)

    if mode == "plan":
        cli.print_plan("stage 02: band thresholds", us, force=args.force)
        return 0

    failures: list[str] = []

    def do(u: cli.Unit) -> str:
        df = bt.run_pair(u.key, n_cuts=n_cuts)
        bt.write(u.key, df)
        gate = df.loc[df["metric"] == bt.DSE_METRIC]
        ok = bool(gate["reproduces_tau"].iloc[0])
        if not ok:
            failures.append(
                f"{u.key}: scan gave {gate['threshold'].iloc[0]:.8f}, "
                f"stored tau is {gate['stored_tau'].iloc[0]:.8f}")
        # Surface the negated metrics: their thresholds are the ones the
        # published grid would have silently gotten wrong.
        neg = df[df["negated"]]
        bad_legacy = int((~neg["legacy_grid_valid"]).sum())
        return (f"dse gate {'PASS' if ok else 'FAIL'}, "
                f"{len(df)} metrics, {bad_legacy} needed the min-start fix")

    rc = cli.run_units("stage 02: band thresholds", us, do, force=args.force,
                       pair_jobs=args.pair_jobs)

    if failures:
        print("\nDSE GATE FAILURES (the binarisation no longer reproduces the "
              "stored thresholds):")
        for f in failures:
            print(f"  {f}")
        if args.strict:
            return 1
    if rc == 0 and not failures:
        try:
            df = results.consolidate(bt.TABLE)
            print(f"{bt.TABLE}: {len(df)} rows over {df['pair'].nunique()} pairs")
        except FileNotFoundError:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
