#!/usr/bin/env python
"""Stage 28 -- the 7 text-only NLI uncertainty metrics (Part C, M20).

GPU, but small: DeBERTa-v2-xlarge-mnli, not the VLM. No image is loaded and the
vision tower never runs -- each row costs ~90 batched entailment forward passes
over `(question, one of the 10 sampled answers)` string pairs. Independent of
stage 26 (M19); needs only the greedy roster (stage 20) and the source run
CSV's `n_generations` column, so it can run on either GPU whenever there is
headroom, including concurrently with a stage-26 run on the other pair.

Output merges into `csx_probe`'s row table (`uq_rows.parquet`) as 7 new
columns via `paths.uq_metrics(pair)` -- see `csx_probe/uq.py`.

Usage:
    python cli_extract/28_uq_metrics.py --pairs qwen25vl_advqa --plan
    python cli_extract/28_uq_metrics.py --pairs qwen25vl_advqa --run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_extract import uq_metrics  # noqa: E402


def _unmet(pair_key: str) -> list[str]:
    unmet = []
    if not paths.raw_rows(pair_key).exists():
        unmet.append(f"greedy pipeline not run (missing {paths.raw_rows(pair_key)}); "
                     f"run stages 20-22 first")
    p = registry.get(pair_key)
    if not p.csv_path.exists():
        unmet.append(f"source CSV not found at {p.csv_path}")
    return unmet


def main(argv=None) -> int:
    ap = cli.base_parser(__doc__, pair_jobs_default="1")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N roster rows (smoke test)")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="independent DeBERTa instances on this one GPU -- "
                         "the model is ~3.6G against 80G VRAM, so this is a "
                         "real lever here (see csx_extract/uq_metrics.py); "
                         "cross-GPU fan-out is still one pinned process per "
                         "GPU at the shell level")
    args = ap.parse_args(argv)
    mode = cli.require_mode(ap, args)

    pairs = registry.resolve(args.pairs, include_pending=args.include_pending)
    units = []
    for p in pairs:
        units.append(cli.Unit(
            key=p.key,
            outputs=[paths.uq_metrics(p.key)],
            inputs=[paths.raw_rows(p.key)],
            cost=f"{len(uq_metrics.METHODS)} methods, ~90 entailment passes/row",
            unmet=_unmet(p.key),
        ))

    if mode == "plan":
        cli.print_plan("stage 28: uq metrics", units, force=args.force)
        return 0

    def do(u: cli.Unit) -> str:
        info = uq_metrics.extract(u.key, limit=args.limit,
                                  n_workers=args.n_workers)
        return (f"{info['n_scored_this_run']}/{info['n_rows']} rows scored "
                f"in {info['minutes']} min ({info['n_workers']} workers)")

    return cli.run_units("stage 28: uq metrics", units, do, force=args.force,
                         pair_jobs=args.pair_jobs)


if __name__ == "__main__":
    raise SystemExit(main())
