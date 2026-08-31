#!/usr/bin/env python
"""Stage 25 -- build the sampled-generation manifest for each pair. CPU only.

Dedups the 10 sampled generations per row down to the unique (id, text) forward
passes actually needed (measured 35-80% dedup across the 9 VLM pairs), and
writes the slot map that lets aggregation (mean/std/cloud) gather features back
into original sample order. Validates the CSV/roster join before any GPU time is
spent on the actual extraction (stage 26).

Usage:
    python cli_extract/25_sampled_manifest.py --pairs qwen25vl_advqa --plan
    python cli_extract/25_sampled_manifest.py --modality vlm --run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_extract import sampled_manifest as sm  # noqa: E402


def main(argv=None) -> int:
    ap = cli.base_parser(__doc__, pair_jobs_default="1")
    ap.add_argument("--modality", choices=("text", "vlm"), default=None)
    args = ap.parse_args(argv)
    mode = cli.require_mode(ap, args)

    pairs = registry.resolve(args.pairs, include_pending=args.include_pending,
                             modality=args.modality)
    units = [
        cli.Unit(
            key=p.key,
            outputs=[paths.sampled_manifest(p.key), paths.sampled_unique(p.key)],
            inputs=[paths.raw_rows(p.key), p.csv_path],
            cost=f"{p.dataset.key}",
        )
        for p in pairs
    ]

    if mode == "plan":
        cli.print_plan("stage 25: sampled manifest", units, force=args.force)
        return 0

    def do(u: cli.Unit) -> str:
        built = sm.build(u.key)
        sm.write(u.key, built)
        return (f"{built.n_rows} rows, {built.n_slots} slots -> "
                f"{built.n_unique} unique ({built.dedup_frac:.1%} dedup)")

    return cli.run_units("stage 25: sampled manifest", units, do, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
