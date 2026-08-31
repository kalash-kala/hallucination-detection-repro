#!/usr/bin/env python
"""Stage 20 -- build rows.parquet + meta.json for each pair. CPU only, no model.

This is the cheap gate before any GPU time is spent. It joins the L2 labels with
the generations folder, applies the declared image-path rewrite, verifies every
image file exists, subsamples to the dataset cap, and writes an entry with both
phases marked not-done.

If a roster entry is wrong -- the wrong generations folder, a rewrite that does
not apply, a partial transfer -- it fails here, in seconds.

Usage:
    python cli_extract/20_rows.py --pairs qwen25vl_advqa --plan
    python cli_extract/20_rows.py --modality vlm --run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_extract import config, rows as rows_mod, writer  # noqa: E402


def main(argv=None) -> int:
    ap = cli.base_parser(__doc__, pair_jobs_default="1")
    ap.add_argument("--modality", choices=("text", "vlm"), default=None)
    ap.add_argument("--n-target", type=int, default=None,
                    help="override the dataset's subsample cap")
    args = ap.parse_args(argv)
    mode = cli.require_mode(ap, args)

    pairs = registry.resolve(args.pairs, include_pending=args.include_pending,
                             modality=args.modality)
    units = [
        cli.Unit(
            key=p.key,
            outputs=[paths.raw_rows(p.key), paths.raw_meta(p.key)],
            inputs=[paths.uq_table()] + (
                [p.generations_path] if p.needs_generations() else []),
            cost=f"{p.dataset.key}, cap={p.dataset.n_target or 'all'}",
            note=("rewrite " + p.path_rewrite[0] if p.path_rewrite else ""),
        )
        for p in pairs
    ]

    if mode == "plan":
        cli.print_plan("stage 20: rows", units, force=args.force)
        return 0

    def do(u: cli.Unit) -> str:
        p = registry.get(u.key)
        built = rows_mod.build(u.key, n_target=args.n_target)
        writer.init_entry(
            pair=u.key,
            model={"key": p.model.key, "hf_id": p.model.hf_id,
                   "layers": p.model.layers,
                   "image_token_ids": list(p.model.image_token_ids)},
            dataset=p.dataset.key,
            modality=p.model.modality,
            prompt_template=p.prompt_template,
            segments=list(p.segments),
            rows=built.rows,
            n_pool=built.n_pool,
            subsample=built.subsample_meta,
            top_k=config.EXTRACT_TOP_K,
            sink_k=config.SINK_K,
            extractor_version=config.EXTRACTOR_VERSION,
        )
        sm = built.subsample_meta
        return (f"{built.n} rows"
                + (f" (subsampled from {sm['n_pool']})" if sm["applied"] else ""))

    return cli.run_units("stage 20: rows", units, do, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
