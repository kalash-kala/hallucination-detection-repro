#!/usr/bin/env python
"""STAGE 00 — ingest every run CSV into the L2 row table.

The cheapest layer in the store and the one with the widest reach: it needs no
GPU and no internal states, so it covers every registered pair immediately. Arm
construction, the entropy_only verification and the alpha-ladder leak table all
run off this alone -- which is how VLM pairs produce results before their
extraction is anywhere near done.

Enforces the invariants the rest of the study rests on (orientation H = LOW
entropy, correctness from LLM_verdict rather than `accuracy`, one tau per pair);
see csx_probe/uq.py for why each is checked rather than assumed.

Outputs:
    <store>/uq/uq_rows.parquet                  one row per example, all pairs
    <store>/uq/generations/<pair>.parquet       optional, --with-generations
    <store>/uq/summary.csv                      per-pair cell counts + tau

Usage:
    PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
    $PY cli_probe/00_ingest_uq.py --plan
    $PY cli_probe/00_ingest_uq.py --run
    $PY cli_probe/00_ingest_uq.py --run --pairs qwen25vl_vqav2 --with-generations
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csx_common import cli, paths, registry  # noqa: E402
from csx_probe import uq  # noqa: E402


def main() -> int:
    ap = cli.base_parser(__doc__.split("\n\n")[0])
    ap.add_argument("--with-generations", action="store_true",
                    help="also write the 10 sampled answer strings per row "
                         "(bulky; needed only by the 10-sample extraction)")
    ap.add_argument("--lenient", action="store_true",
                    help="report and skip pairs that fail a gate instead of aborting")
    args = ap.parse_args()
    mode = cli.require_mode(ap, args)

    pairs = registry.resolve(args.pairs, include_pending=args.include_pending)
    out = paths.uq_table()

    units = [
        cli.Unit(
            key=p.key,
            inputs=[p.csv_path],
            outputs=[],  # the table is written once, for all pairs together
            note=f"{p.model.key}/{p.dataset.key} ({p.model.modality})",
        )
        for p in pairs
    ]

    if mode == "plan":
        n = cli.print_plan(f"L2 ingest -> {out}", units, force=True)
        if out.exists():
            print(f"note: {out} already exists and would be overwritten")
        return 0 if n else 1

    rows, gens, problems = uq.build(
        [p.key for p in pairs],
        with_generations=args.with_generations,
        strict=not args.lenient,
    )
    for prob in problems:
        print(f"[FAIL] {prob['pair']}: {prob['error']}", file=sys.stderr)

    written = uq.write(rows, gens)
    summary = uq.summarise(rows)
    spath = out.parent / "summary.csv"
    summary.to_csv(spath, index=False)
    written.append(spath)

    print(summary.to_string(index=False))
    print(f"\n{len(rows):,} rows across {rows['pair'].nunique()} pairs")
    for w in written:
        print(f"  wrote {w}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
