#!/usr/bin/env python
"""Stage 22 -- peak layers, pooled hs_* features, then delete the raw shards. CPU.

DESTRUCTIVE by design, following full_natural/02_reduce_hidden.py: the raw
per-layer hidden states are ~10 GB per pair at 62 layers and are not worth
keeping once pooled. Changing the pooling scheme later means re-running stage 21.
`--keep-raw` skips the delete for debugging.

This stage marks phase 1 done, because it is the point at which the entry's
hs/<segment>.npz files exist and csx_probe can actually read the pair.

Usage:
    python cli_extract/22_reduce.py --pairs qwen25vl_advqa --run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_extract import reduce_hidden, writer  # noqa: E402


def main(argv=None) -> int:
    ap = cli.base_parser(__doc__, pair_jobs_default="1")
    ap.add_argument("--modality", choices=("text", "vlm"), default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="reduce even though phase 1 ran under --limit; the "
                         "resulting entry has a silently small n")
    ap.add_argument("--keep-raw", action="store_true",
                    help="do not delete the per-layer shards")
    args = ap.parse_args(argv)
    mode = cli.require_mode(ap, args)

    pairs = registry.resolve(args.pairs, include_pending=args.include_pending,
                             modality=args.modality)
    units = []
    for p in pairs:
        d = paths.raw_dir(p.key)
        units.append(cli.Unit(
            key=p.key,
            outputs=[d / "hs" / f"{s}.npz" for s in p.segments]
                    + [d / "hs" / "peaks.json"],
            inputs=[d / "_shards" / p.segments[0] / "layer1.npy"],
            cost=f"{len(p.segments)} segment(s), CPU",
            note="deletes raw shards" if not args.keep_raw else "keeps raw",
        ))

    if mode == "plan":
        cli.print_plan("stage 22: reduce hidden", units, force=args.force)
        return 0

    def do(u: cli.Unit) -> str:
        info = reduce_hidden.run(u.key, keep_raw=args.keep_raw,
                                 allow_partial=args.allow_partial)
        writer.finish_phase(u.key, "phase1", **{
            k: v for k, v in info.items() if k != "peaks"})
        freed = info["raw_bytes_freed"] / 1e9
        return f"pooled; {freed:.1f} GB freed" if freed else "pooled"

    return cli.run_units("stage 22: reduce hidden", units, do, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
