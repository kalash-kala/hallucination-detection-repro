#!/usr/bin/env python
"""Stage 21 -- phase 1: hidden states + s_ext (sdpa). GPU.

Linear in sequence length and comparatively cheap. Completing this stage alone
makes hs_wide / hs_narrow / hs_peak_only available to csx_probe, so probing can
start before the expensive eager pass is scheduled.

Usage:
    python cli_extract/21_hidden.py --pairs qwen25vl_advqa --plan
    python cli_extract/21_hidden.py --pairs qwen25vl_advqa --run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_extract import phase1_hidden, writer  # noqa: E402


def main(argv=None) -> int:
    ap = cli.base_parser(__doc__, pair_jobs_default="1")
    ap.add_argument("--modality", choices=("text", "vlm"), default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="extract only the first N rows (smoke test)")
    args = ap.parse_args(argv)
    mode = cli.require_mode(ap, args)

    pairs = registry.resolve(args.pairs, include_pending=args.include_pending,
                             modality=args.modality)
    units = []
    for p in pairs:
        d = paths.raw_dir(p.key)
        units.append(cli.Unit(
            key=p.key,
            # `_shards` is the phase's real output; peaks/hs come from stage 22.
            outputs=[d / "_shards" / p.segments[0] / "layer1.npy"],
            inputs=[paths.raw_rows(p.key)],
            cost=f"{p.model.layers}L x {len(p.segments)} seg, sdpa",
        ))

    if mode == "plan":
        cli.print_plan("stage 21: phase 1 (hidden, sdpa)", units, force=args.force)
        return 0

    def do(u: cli.Unit) -> str:
        info = phase1_hidden.run(u.key, limit=args.limit)
        # phase1 is NOT marked done here. The contract requires hs/<seg>.npz
        # whenever phase1 is done, and those are written by stage 22. Flipping
        # the flag now would advertise an entry csx_probe cannot actually read.
        writer.update_meta(u.key, n_rows=info["n_rows"], phase1_pass=info)
        return f"{info['n_rows']} rows in {info['minutes']} min (run stage 22 next)"

    return cli.run_units("stage 21: phase 1", units, do, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
