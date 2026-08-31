"""`csx-store verify` — validate one L0 entry, standalone.

This is the handoff between component 1 and component 2. It is deliberately
runnable by whoever ran the extraction, on the machine that ran it, with no
experiment code involved and no torch import: a GPU job finishes, this says
usable or names the reason, and the pair is available to `csx_probe` immediately.
There is no "finish everything, then start experiments" gate.

Every check reports rather than raising on the first failure, so one run tells
you everything that is wrong with an entry instead of one thing at a time.

Usage:
    csx-store verify --pair qwen25vl_vqav2
    csx-store verify --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from csx_common import paths, registry  # noqa: E402
from csx_common.store_schema import (  # noqa: E402
    DIAG_KEYS, DIAG_SINK_KEYS, DIAG_TOPK_KEYS, HS_SCHEMES, ROW_COLUMNS,
    ContractError, EntryMeta, Problem,
)

from . import config  # noqa: E402


def verify_entry(pair: str, *, check_l2: bool = True) -> list[Problem]:
    """Run every check against one entry. Empty list == usable."""
    probs: list[Problem] = []
    d = paths.raw_dir(pair)
    if not d.is_dir():
        return [Problem("exists", f"no entry at {d}")]

    # 1. meta
    try:
        meta = EntryMeta.load(paths.raw_meta(pair))
    except ContractError as exc:
        return [Problem("meta", str(exc))]
    if meta.pair != pair:
        probs.append(Problem("meta", f"meta.pair={meta.pair!r} but entry is {pair!r}"))

    try:
        reg = registry.get(pair)
    except KeyError as exc:
        probs.append(Problem("registry", str(exc)))
        reg = None
    if reg is not None:
        if tuple(meta.segments) != reg.segments:
            probs.append(Problem(
                "segments",
                f"meta declares {meta.segments} but a {reg.model.modality} pair "
                f"must have {reg.segments}"))
        if meta.raw["prompt_template"] != reg.prompt_template:
            probs.append(Problem(
                "meta", f"prompt_template {meta.raw['prompt_template']!r} != "
                        f"registry {reg.prompt_template!r}"))

    # 2. rows
    rpath = paths.raw_rows(pair)
    if not rpath.exists():
        probs.append(Problem("rows", f"missing {rpath}"))
        return probs
    rows = pd.read_parquet(rpath)
    n = meta.n_rows
    missing_cols = [c for c in ROW_COLUMNS if c not in rows.columns]
    if missing_cols:
        probs.append(Problem("rows", f"missing columns {missing_cols}"))
    if len(rows) != n:
        probs.append(Problem("rows", f"{len(rows)} rows but meta says n_rows={n}"))
    if "row" in rows and not np.array_equal(
            np.sort(rows["row"].to_numpy()), np.arange(len(rows))):
        probs.append(Problem("rows", "`row` is not exactly 0..n-1"))
    if "id" in rows and rows["id"].duplicated().any():
        probs.append(Problem("rows", "duplicate ids"))

    # 3. spans
    probs += _check_spans(rows, meta)

    # 4-6. per-segment artifacts for whichever phases claim to be done
    for seg in meta.segments:
        if meta.phase_done("phase1"):
            probs += _check_hs(pair, seg, n)
        if meta.phase_done("phase2"):
            probs += _check_diag(pair, seg, n, meta)

    if not meta.phase_done("phase1") and not meta.phase_done("phase2"):
        probs.append(Problem("phase", "neither phase is marked done; nothing usable yet"))

    # 7. L2 cross-check
    if check_l2 and "id" in rows:
        probs += _check_against_l2(pair, rows)

    return probs


def _check_spans(rows: pd.DataFrame, meta: EntryMeta) -> list[Problem]:
    """Spans decide which tokens every feature is pooled over, so a silent error
    here would corrupt every family at once while looking perfectly plausible."""
    out: list[Problem] = []
    need = {"answer_start", "answer_end", "seq_len", "image_start", "image_end"}
    if not need <= set(rows.columns):
        return out
    a0, a1 = rows["answer_start"].to_numpy(), rows["answer_end"].to_numpy()
    sl = rows["seq_len"].to_numpy()
    bad = int(((a0 < 0) | (a0 >= a1) | (a1 > sl)).sum())
    if bad:
        out.append(Problem("spans", f"{bad} rows violate 0 <= answer_start < "
                                    f"answer_end <= seq_len"))
    i0, i1 = rows["image_start"].to_numpy(), rows["image_end"].to_numpy()
    if meta.modality == "vlm":
        bad = int(((i0 < 0) | (i0 >= i1) | (i1 > sl)).sum())
        if bad:
            out.append(Problem("spans", f"{bad} VLM rows have an invalid image span"))
        if int((i1 > a0).sum()):
            out.append(Problem(
                "spans",
                f"{int((i1 > a0).sum())} rows have the image span overlapping the "
                f"answer span; image tokens must precede the answer"))
    else:
        if not ((i0 == -1) & (i1 == -1)).all():
            out.append(Problem("spans", "text pair must use image_start/end = -1"))
    return out


def _check_hs(pair: str, seg: str, n: int) -> list[Problem]:
    p = paths.raw_dir(pair) / "hs" / f"{seg}.npz"
    if not p.exists():
        return [Problem("hs", f"phase1 done but missing {p}")]
    out: list[Problem] = []
    with np.load(p) as z:
        for scheme in HS_SCHEMES:
            if scheme not in z:
                out.append(Problem("hs", f"{seg}: missing scheme {scheme}"))
                continue
            arr = z[scheme]
            if arr.ndim != 2:
                out.append(Problem("hs", f"{seg}/{scheme}: expected 2-D, got {arr.shape}"))
            elif arr.shape[0] != n:
                out.append(Problem("hs", f"{seg}/{scheme}: {arr.shape[0]} rows != {n}"))
    peaks = paths.raw_dir(pair) / "hs" / "peaks.json"
    if not peaks.exists():
        out.append(Problem("hs", f"missing {peaks}; the pooled features are "
                                 f"uninterpretable without their peak layers"))
    return out


def _check_diag(pair: str, seg: str, n: int, meta: EntryMeta) -> list[Problem]:
    p = paths.raw_dir(pair) / "diag" / f"{seg}.npz"
    if not p.exists():
        return [Problem("diag", f"phase2 done but missing {p}")]
    out: list[Problem] = []
    with np.load(p) as z:
        for key in DIAG_KEYS:
            if key not in z:
                # attn_logdet is the dangerous one: attnlogdet is a mean over ALL
                # positions, so it cannot be rebuilt from the top-k arrays, and
                # its absence would make that family silently wrong.
                why = (" -- attnlogdet cannot be derived from the top-k arrays"
                       if key == "attn_logdet" else "")
                out.append(Problem("diag", f"{seg}: missing {key}{why}"))
        shapes = {k: z[k].shape for k in DIAG_KEYS if k in z}
        for k, s in shapes.items():
            if s[0] != n:
                out.append(Problem("diag", f"{seg}/{k}: {s[0]} rows != {n}"))
        lh = {k: s[1:3] for k, s in shapes.items() if len(s) >= 3}
        if len(set(map(tuple, lh.values()))) > 1:
            out.append(Problem("diag", f"{seg}: inconsistent [L,H] across keys: {lh}"))
        elif lh and meta.layers is not None:
            L = next(iter(lh.values()))[0]
            if int(L) != meta.layers:
                out.append(Problem(
                    "diag", f"{seg}: diagonals have L={L} but meta says "
                            f"model.layers={meta.layers}"))
        topk = {k: shapes[k][-1] for k in DIAG_TOPK_KEYS if k in shapes}
        for keys, want in ((DIAG_SINK_KEYS, config.SINK_K),
                           (tuple(k for k in DIAG_TOPK_KEYS
                                  if k not in DIAG_SINK_KEYS),
                            config.EXTRACT_TOP_K)):
            got = {k: topk[k] for k in keys if k in topk}
            bad = {k: v for k, v in got.items() if v != want}
            if bad:
                out.append(Problem(
                    "diag", f"{seg}: top-k width != {want}: {bad}"))
        # meta records the attention/Laplacian width; the sink arrays have their
        # own constant and are checked above.
        meta_k = meta.raw.get("top_k")
        if meta_k is not None and int(meta_k) != config.EXTRACT_TOP_K:
            out.append(Problem(
                "diag", f"{seg}: meta.top_k={meta_k} != "
                        f"EXTRACT_TOP_K={config.EXTRACT_TOP_K}"))
    return out


def _check_against_l2(pair: str, rows: pd.DataFrame) -> list[Problem]:
    """A pair must not be extracted against rows the study does not know about."""
    try:
        t = paths.uq_table()
        if not t.exists():
            return []
        l2 = pd.read_parquet(t, columns=["pair", "id"])
    except Exception as exc:  # noqa: BLE001
        return [Problem("l2", f"could not read L2: {exc}")]
    known = set(l2.loc[l2["pair"] == pair, "id"])
    if not known:
        return [Problem("l2", f"L2 has no rows for {pair}")]
    unknown = set(rows["id"].astype(str)) - known
    if unknown:
        ex = sorted(unknown)[:3]
        return [Problem("l2", f"{len(unknown)} extracted ids are absent from L2, "
                              f"e.g. {ex}")]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="csx-store", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="validate L0 entries")
    g = v.add_mutually_exclusive_group(required=True)
    g.add_argument("--pair")
    g.add_argument("--all", action="store_true")
    v.add_argument("--json", action="store_true", help="machine-readable output")
    v.add_argument("--no-l2", action="store_true",
                   help="skip the cross-check against the L2 row table")
    args = ap.parse_args(argv)

    pairs = ([p.key for p in registry.resolve()] if args.all else [args.pair])
    report, failed = {}, 0
    for pair in pairs:
        probs = verify_entry(pair, check_l2=not args.no_l2)
        report[pair] = [str(p) for p in probs]
        if probs:
            failed += 1

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for pair, probs in report.items():
            if not probs:
                print(f"OK    {pair}")
            else:
                print(f"FAIL  {pair}")
                for p in probs:
                    print(f"        {p}")
        print(f"\n{len(pairs) - failed}/{len(pairs)} entries usable")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
