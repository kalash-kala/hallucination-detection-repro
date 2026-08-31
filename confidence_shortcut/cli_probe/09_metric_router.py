#!/usr/bin/env python
"""Stage 09: the metric-router grid -- all 64 ordered metric pairs. CPU.

M23. Trains the sampled band router against metric A's band and scores it
against metric B's, for every ordered `(A, B)`, inside each arm. The diagonal
reproduces stage 07's routed result; the off-diagonal answers whether the router
learned "the confidence band" or merely "discrete semantic entropy".

Two tables are written per pair, and the SECOND is the one to read first:

    metric_router_long   the 8x8 transfer grid
    band_agreement       raw band agreement + Cohen's kappa per metric pair

All 8 metrics come from the same 10 generations, so their bands are correlated
and some transfer is free. `band_agreement` states how much, so the off-diagonal
is read against an honest denominator rather than in a vacuum.

The train/test partition is NEVER re-split: `dse_natural` is reused verbatim and
the other arms only subset it, which is what keeps every cross-metric cell
leak-free. `metric_router.assert_leak_free` verifies that per pair instead of
trusting the argument.

Usage:
    cli_probe/09_metric_router.py --plan
    cli_probe/09_metric_router.py --run --pairs qwen25vl_advqa
    cli_probe/09_metric_router.py --run --schemes cloud --n-samples 1,3,10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import config, results  # noqa: E402
from csx_probe.experiments import (c_selection, metric_router as mr,  # noqa: E402
                                   band_thresholds as bt)


def _cfg(pair: str, families: tuple[str, ...]):
    pol = config.c_policy()
    mode = pol.get("assignment", {}).get(pair, pol["default"])
    per_unit = c_selection.per_unit_best(pair) if mode == "per_pair" else None
    return config.RunConfig.for_pair(pair, per_unit_best=per_unit,
                                     families=families, n_boot=0)


def _sampled_ready(pair: str) -> bool:
    meta = paths.sampled_meta(pair)
    if not meta.exists():
        return False
    try:
        return bool(json.loads(meta.read_text())
                    .get("extraction", {}).get("done"))
    except Exception:
        return False


def units(pairs, schemes, n_samples, family):
    out = []
    for key in pairs:
        unmet = []
        if not _sampled_ready(key):
            unmet.append("needs M19 (sampled extraction) for this pair")
        if not results.has_unit(bt.TABLE, key):
            unmet.append("needs stage 02 (band thresholds)")
        pol = config.c_policy()
        mode = pol.get("assignment", {}).get(key, pol["default"])
        if mode == "per_pair" and not results.has_unit(c_selection.TABLE, key):
            unmet.append("needs stage 03 (per_pair C is unresolved)")
        n_fits = len(mr.ARMS) * len(schemes) * len(n_samples) * len(mr.METRICS)
        out.append(cli.Unit(
            key=key,
            outputs=(paths.results_units(mr.TABLE, key),
                     paths.results_units(mr.AGREEMENT_TABLE, key)),
            unmet=tuple(unmet),
            cost=(f"{family}: {len(mr.METRICS)}x{len(mr.METRICS)} grid x "
                  f"{len(mr.ARMS)} arms x {len(schemes)} schemes x "
                  f"{len(n_samples)} n -- {n_fits} router fits"),
        ))
    return out


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0])
    ap.add_argument("--family", default="hs_wide",
                    help="feature family the router reads (default hs_wide)")
    ap.add_argument("--segment", default="all")
    ap.add_argument("--schemes", default=",".join(mr.SCHEMES),
                    help="sampled aggregation schemes (cloud,mean_std)")
    ap.add_argument("--n-samples", default="10",
                    help="comma-separated sweep over the number of sampled "
                         "generations, e.g. 1,2,3,5,10. Each extra generation "
                         "is a real inference cost, so this is the price of "
                         "the band label. `cloud` is undefined at n=1.")
    ap.add_argument("--metrics", default=None,
                    help="restrict the grid to these metrics")
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)

    schemes = tuple(s for s in args.schemes.split(",") if s)
    n_samples = tuple(int(s) for s in args.n_samples.split(",") if s)
    metrics = (tuple(args.metrics.split(",")) if args.metrics else mr.METRICS)
    pairs = [p.key for p in registry.resolve(
        args.pairs, include_pending=args.include_pending)]
    us = units(pairs, schemes, n_samples, args.family)

    if mode == "plan":
        cli.print_plan("stage 09: metric-router grid", us, force=args.force)
        return 0

    def do(u: cli.Unit) -> str:
        cfg = _cfg(u.key, (args.family,))
        # The denominator first, deliberately: it is cheap, it never fails, and
        # it is what the transfer numbers have to be read against.
        ag = mr.agreement(u.key)
        mr.write_agreement(u.key, ag)

        df = mr.run_pair(u.key, cfg, family=args.family, segment=args.segment,
                         schemes=schemes, n_samples=n_samples, metrics=metrics)
        mr.write(u.key, df)

        off = df[~df["diagonal"]]["router_auroc"]
        dia = df[df["diagonal"]]["router_auroc"]
        mean_ag = float(ag[ag["metric_a"] != ag["metric_b"]]["agreement"].mean())
        return (f"{len(df)} rows; diagonal {dia.mean():.3f}, "
                f"off-diagonal {off.mean():.3f} "
                f"(mean off-diagonal band agreement {mean_ag:.3f})")

    rc = cli.run_units("stage 09: metric-router grid", us, do,
                       force=args.force, pair_jobs=args.pair_jobs)
    for table in (mr.TABLE, mr.AGREEMENT_TABLE):
        try:
            df = results.consolidate(table)
            print(f"{table}: {len(df)} rows over {df['pair'].nunique()} pairs")
        except FileNotFoundError:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
