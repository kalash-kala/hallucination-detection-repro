#!/usr/bin/env python
"""Stage 26 -- extract internal states for the 10-sample router/SR-xAUC tier.

GPU, expensive: one recomputed-stats sdpa pass over greedy-train rows, plus one
combined eager pass (hidden + attention) per UNIQUE (id, text) in the manifest
stage 25 built. Order by cost: advqa before okvqa before vqav2, Pixtral last
within each (its median prompt is 1,143 tokens against gemma-3's ~293) -- same
ordering discipline as stage 23.

Independent of stage 22 (reduce)/stage 23 (attention) -- it reads the greedy
pair's `hs/peaks.json` for bucket definitions (must exist) and its `rows.parquet`
for context (image_path/question/prompt_token_ids), but recomputes its own
z-score statistics rather than reading anything phase 1 wrote.

The sampled store is KEPT after this runs, not purged -- see
docs/PART_C_ROUTING_PLAN.md. Estimated store size: ~35-45 GB across all 9 VLM
pairs, against 337 G free on /data.

Usage:
    python cli_extract/26_extract_sampled.py --pairs qwen25vl_advqa --plan
    python cli_extract/26_extract_sampled.py --pairs qwen25vl_advqa --run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_extract import sampled_extract, sampled_writer  # noqa: E402

PROMPT_TOKENS = {"qwen25vl": 391, "gemma3_12b": 293, "pixtral12b": 1143}


def _unmet(pair_key: str) -> list[str]:
    peaks_path = paths.raw_dir(pair_key) / "hs" / "peaks.json"
    manifest_path = paths.sampled_manifest(pair_key)
    unmet = []
    if not peaks_path.exists():
        unmet.append(f"greedy pipeline not reduced (missing {peaks_path}); "
                     f"run stages 20-22 first")
    if not manifest_path.exists():
        unmet.append("sampled manifest not built; run stage 25 first")
    return unmet


def _n_unique(pair_key: str) -> int | None:
    p = paths.sampled_unique(pair_key)
    if not p.exists():
        return None
    import pandas as pd
    return len(pd.read_parquet(p, columns=["urow"]))


def main(argv=None) -> int:
    ap = cli.base_parser(__doc__, pair_jobs_default="1")
    ap.add_argument("--modality", choices=("text", "vlm"), default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="extract only the first N unique rows (smoke test)")
    args = ap.parse_args(argv)
    mode = cli.require_mode(ap, args)

    pairs = registry.resolve(args.pairs, include_pending=args.include_pending,
                             modality=args.modality)
    units = []
    for p in pairs:
        d = paths.sampled_dir(p.key)
        tok = PROMPT_TOKENS.get(p.model.key)
        nu = _n_unique(p.key)
        units.append(cli.Unit(
            key=p.key,
            outputs=[d / "hs" / f"{s}.npz" for s in p.segments]
                    + [d / "diag" / f"{s}.npz" for s in p.segments],
            inputs=[paths.sampled_manifest(p.key)],
            cost=(f"eager, ~{tok} tok/row, {nu} unique rows" if nu is not None
                  else "eager"),
            unmet=_unmet(p.key),
        ))

    if mode == "plan":
        cli.print_plan("stage 26: extract sampled", units, force=args.force)
        return 0

    def do(u: cli.Unit) -> str:
        meta = {"pair": u.key, "extraction": {"done": False}}
        paths.sampled_meta(u.key).parent.mkdir(parents=True, exist_ok=True)
        paths.sampled_meta(u.key).write_text(json.dumps(meta, indent=2))
        info = sampled_extract.extract(u.key, limit=args.limit)
        sampled_writer.finish_extraction(u.key, **info)
        return f"{info['n_unique']} unique rows in {info['minutes']} min"

    return cli.run_units("stage 26: extract sampled", units, do,
                         force=args.force, pair_jobs=args.pair_jobs)


if __name__ == "__main__":
    raise SystemExit(main())
