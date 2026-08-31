#!/usr/bin/env python
"""Stage 07: the routed grid — band experts vs a single generalist.

CPU only. Emits `routed_long`, schema-compatible with `per_pair_long` plus
`router`/`scorer`, so component 3 concatenates the two without a special case.

The 10-generation scorers (`generalist_cm`, `spec1_z_cm`) and the `sampled`
router need the aggregated sampled feature block. Without it they are SKIPPED and
the skip is reported -- never silently backed off to greedy features, which would
turn a cost-matched comparison into an unmatched one that looks identical in the
output.

Usage:
    cli_probe/07_routed_grid.py --plan
    cli_probe/07_routed_grid.py --run --pairs qwen25vl_advqa --n-jobs 8
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import config, results  # noqa: E402
from csx_probe.experiments import c_selection, routed_grid  # noqa: E402
from csx_probe.store import read  # noqa: E402


def _cfg(pair: str, families: tuple[str, ...], n_boot: int) -> config.RunConfig:
    pol = config.c_policy()
    mode = pol.get("assignment", {}).get(pair, pol["default"])
    per_unit = c_selection.per_unit_best(pair) if mode == "per_pair" else None
    return config.RunConfig.for_pair(pair, per_unit_best=per_unit,
                                     families=families, n_boot=n_boot)


def sampled_schemes_default() -> str:
    """Both published router variants by default.

    `greedy_mean_std` is the incumbent; `cloud` is the 10-scalar geometry block
    whose competitiveness against a ~14k-wide `mean_std` was one of the QA run's
    sharper findings and is an open question on VLM. Running them together is
    what makes them comparable -- they then share a train/test partition, a `C`,
    and a frozen basis, so a difference between them is the aggregation and
    nothing else.
    """
    from csx_probe.store import sampled as _s
    return f"{_s.DEFAULT_SCHEME},cloud"


def _sampled_ready(pair: str) -> bool:
    """Whether the 10-generation tier's extraction has landed for this pair."""
    meta = paths.sampled_meta(pair)
    if not meta.exists():
        return False
    import json
    try:
        return bool(json.loads(meta.read_text())
                    .get("extraction", {}).get("done"))
    except Exception:
        return False


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
        cm = ("with 10-gen scorers" if _sampled_ready(key) else
              "greedy-only (no sampled store; _cm scorers and the sampled "
              "router will be skipped)")
        out.append(cli.Unit(
            key=key,
            outputs=(paths.results_units(routed_grid.TABLE, key),),
            unmet=tuple(unmet), note=f"C mode={mode}; {cm}",
            cost=f"{len(families)} fam x 4x4 arms x 8 scorers x 3 routers",
        ))
    return out


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0], inner_jobs=True)
    ap.add_argument("--families", default=None)
    ap.add_argument("--segments", default="all",
                    help="default: all only -- the 8-QA parity target never "
                         "had image/text, pass e.g. --segments all,image,text "
                         "to opt into the VLM-only segment extension")
    ap.add_argument("--scorers", default=None,
                    help=f"default: {','.join(routed_grid.UNROUTED + routed_grid.ROUTED)}")
    ap.add_argument("--schemes", default=sampled_schemes_default(),
                    help="sampled aggregation schemes, comma-separated "
                         "(greedy_mean_std,mean,mean_std,cloud). Cells that do "
                         "not read the sampled block are emitted once, so a "
                         "second scheme costs only the cells that differ.")
    ap.add_argument("--n-boot", type=int,
                    default=config.frozen()["bootstrap"]["row_level_n"])
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)

    families = (tuple(args.families.split(",")) if args.families
                else config.FAMILIES)
    segments = tuple(args.segments.split(",")) if args.segments else None
    scorers = tuple(args.scorers.split(",")) if args.scorers else None
    pairs = [p.key for p in registry.resolve(
        args.pairs, include_pending=args.include_pending)]
    us = units(pairs, families)

    if mode == "plan":
        cli.print_plan("stage 07: routed grid", us, force=args.force)
        return 0

    # Both levels of parallelism come out of ONE core budget: `do` opens
    # its own pool per unit, so the outer width has to be resolved first.
    pair_jobs = cli.resolve_pair_jobs(
        args.pair_jobs, sum(1 for u in us if not u.blocked))
    n_jobs = cli.resolve_inner_jobs(args.n_jobs, pair_jobs)

    def do(u: cli.Unit) -> str:
        cfg = _cfg(u.key, families, args.n_boot)
        sampled = None
        if _sampled_ready(u.key):
            from csx_probe.store import sampled as sampled_mod
            sampled = sampled_mod.blocks_by_scheme(
                u.key, families=families,
                segments=segments or cfg.segments,
                schemes=tuple(args.schemes.split(",")) if args.schemes else ())
        df = routed_grid.run_pair(u.key, cfg, segments=segments,
                                  families=families, scorers=scorers,
                                  sampled=sampled, n_jobs=n_jobs,
                                  verbose=False)
        return f"{len(df)} rows"

    rc = cli.run_units("stage 07: routed grid", us, do, force=args.force,
                        pair_jobs=args.pair_jobs, inner_fanout=n_jobs)
    try:
        df = results.consolidate(routed_grid.TABLE)
        print(f"{routed_grid.TABLE}: {len(df)} rows over "
              f"{df['pair'].nunique()} pairs")
    except FileNotFoundError:
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
