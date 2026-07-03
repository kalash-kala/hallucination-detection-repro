"""
Package self-check: compare freshly produced Part 9 / Part 11 CSVs against the
original study's CSVs (value-level, tolerant to float formatting).

Usage:
  python scripts/validate_against_original.py \
      --original-dir /path/to/original/per_category_analysis \
      [--check part9] [--check part11]

Exits 0 if every shared (key, column) matches within --atol (default 5e-4,
i.e. AUROCs equal to ~3–4 decimals), 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "per_category_analysis"

CHECKS = {
    "part9": ("per_category_pairwise_auroc.csv", ["pair", "method"]),
    "part11": ("per_category_band_specialist_auroc.csv", ["pair", "method", "band"]),
}


def compare(name: str, ours: Path, orig: Path, keys: list[str], atol: float) -> bool:
    if not ours.exists():
        print(f"[{name}] MISSING ours: {ours}")
        return False
    if not orig.exists():
        print(f"[{name}] MISSING original: {orig}")
        return False
    a = pd.read_csv(ours)
    b = pd.read_csv(orig)
    keys = [k for k in keys if k in a.columns and k in b.columns]
    a = a.set_index(keys).sort_index()
    b = b.set_index(keys).sort_index()
    shared_rows = a.index.intersection(b.index)
    shared_cols = [c for c in a.columns if c in b.columns]
    if len(shared_rows) == 0:
        print(f"[{name}] FAIL: no shared rows")
        return False
    n_bad = 0
    for c in shared_cols:
        av = pd.to_numeric(a.loc[shared_rows, c], errors="coerce")
        bv = pd.to_numeric(b.loc[shared_rows, c], errors="coerce")
        if av.isna().all():   # non-numeric column: compare as strings
            mism = (a.loc[shared_rows, c].astype(str) != b.loc[shared_rows, c].astype(str)).sum()
        else:
            diff = (av - bv).abs()
            mism = int(((diff > atol) & ~(av.isna() & bv.isna())).sum())
        if mism:
            n_bad += mism
            print(f"[{name}] column {c!r}: {mism} mismatching rows")
    print(f"[{name}] rows compared: {len(shared_rows)} "
          f"(ours {len(a)}, original {len(b)}), columns: {len(shared_cols)}, "
          f"mismatches: {n_bad}")
    return n_bad == 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original-dir", required=True, type=Path)
    p.add_argument("--check", action="append", choices=list(CHECKS),
                   help="default: all")
    p.add_argument("--atol", type=float, default=5e-4)
    args = p.parse_args()

    ok = True
    for name in (args.check or list(CHECKS)):
        fn, keys = CHECKS[name]
        ok &= compare(name, OUT_DIR / fn, args.original_dir / fn, keys, args.atol)
    print("VALIDATION:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()