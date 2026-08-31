#!/usr/bin/env python
"""Stage 23 -- phase 2: attention/Laplacian/sink diagonals (eager). GPU, expensive.

Cost is quadratic in sequence length, so order matters: advqa (3,000 rows) before
okvqa (14,055) before vqav2 (15,000 after subsampling), and Pixtral last within
each -- its median prompt is 1,143 tokens against gemma-3's ~293.

Unlocks lapeigvals, attn_eigvals, attnlogdet and sink.

Independent of stage 22 (reduce), but NOT of stage 21: phase 2 reads each row's
span from rows.parquet rather than recomputing it, which is what guarantees the
two passes describe the same sequence. So stage 21 must have run first, and a
pair whose spans are still unfilled is reported as blocked rather than attempted.

Usage:
    python cli_extract/23_attention.py --pairs qwen25vl_advqa --plan
    python cli_extract/23_attention.py --pairs qwen25vl_advqa --run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_extract import phase2_attention, writer  # noqa: E402

# Measured medians / maxima of prompt_token_ids across the generations folders.
# Used only to make --plan honest about relative cost.
PROMPT_TOKENS = {"qwen25vl": 391, "gemma3_12b": 293, "pixtral12b": 1143}


def _unmet(pair_key: str) -> list[str]:
    """Phase-2 preconditions that file existence cannot express.

    rows.parquet exists from stage 20, so an existence check calls this ready.
    What phase 2 actually needs is the SPANS inside it, which only stage 21
    fills; without this, --plan promises work that dies on the first row.
    """
    try:
        rows = writer.read_rows(pair_key)
    except Exception:
        return []          # the missing-input check already covers this
    if "seq_len" not in rows.columns or (rows["seq_len"] <= 0).all():
        return ["phase 1 has not run (rows.parquet has no spans); run stage 21"]
    n_missing = int((rows["seq_len"] <= 0).sum())
    if n_missing:
        return [f"phase 1 is incomplete ({n_missing} of {len(rows)} rows have "
                f"no span); re-run stage 21"]
    return []


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
        tok = PROMPT_TOKENS.get(p.model.key)
        units.append(cli.Unit(
            key=p.key,
            outputs=[d / "diag" / f"{s}.npz" for s in p.segments],
            inputs=[paths.raw_rows(p.key)],
            cost=(f"eager, ~{tok} tok/row" if tok else "eager"),
            unmet=_unmet(p.key),
        ))

    if mode == "plan":
        cli.print_plan("stage 23: phase 2 (attention, eager)", units,
                       force=args.force)
        return 0

    def do(u: cli.Unit) -> str:
        info = phase2_attention.run(u.key, limit=args.limit)
        writer.finish_phase(u.key, "phase2", **info)
        return f"{info['n_rows']} rows in {info['minutes']} min"

    return cli.run_units("stage 23: phase 2", units, do, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
