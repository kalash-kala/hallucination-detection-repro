#!/usr/bin/env python3
"""
STAGE 00c — build frozen 70/30 train/test splits for a new (model, dataset) pair.

This step did not exist anywhere in the repro package before: splits for the
original 12 pairs are shipped, frozen data (recovery-gaps-data/data/
ranking_experiment_{model}_{dataset}/splits/) and are never regenerated — the
original repo built them once via ranking/run_experiment.py's step 1 (which
also does the full forward-pass extraction, so it isn't ported here).

This script does ONLY that step-1 slice for a NEW pair, using the ranking/
data_loader.py that already ships in this package:
  1. load_samples()  -- requires `candidate_list` (see generate_distractors_hf.py)
  2. split_samples()  -- 70/30, stratified by correct/incorrect, at the question level
  3. save_split()      -- writes splits/{train,test}.{jsonl,ids.json}

Once this has run for a pair, scripts/ranking/extract_cache.py and friends can
target it exactly like the original 12 pairs (after their own MODEL_MAP/PAIRS
lists are extended -- a separate step, see REPRODUCTION_GUIDE.md).

Usage:
    python make_splits.py \\
        --input_csv ../../data/uncertainty_runs/sampled_1400_..._with_distractors.csv \\
        --model qwen3_14b --dataset sciq
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RANKING_DIR = REPO_ROOT / "scripts" / "ranking"
sys.path.insert(0, str(RANKING_DIR))

from config import ExperimentConfig, ensure_dirs  # noqa: E402
from data_loader import load_samples, split_samples, save_split  # noqa: E402

DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"


def build_splits(input_csv: Path, model: str, dataset: str, test_fraction: float, seed: int) -> None:
    if not input_csv.exists():
        raise SystemExit(f"ERROR: input CSV not found: {input_csv}")
    if "candidate_list" not in Path(input_csv).read_text().splitlines()[0]:
        raise SystemExit(
            f"ERROR: {input_csv.name} has no 'candidate_list' column -- "
            "run generate_distractors_hf.py on it first."
        )

    output_dir = DATA_DIR / f"ranking_experiment_{model}_{dataset}"
    cfg = ExperimentConfig(
        input_csv=input_csv,
        output_dir=output_dir,
        test_fraction=test_fraction,
        seed=seed,
    )
    ensure_dirs(cfg)

    print(f"[{model}/{dataset}] loading samples from {input_csv}")
    samples = load_samples(cfg)
    print(f"[{model}/{dataset}] loaded {len(samples)} samples")

    train, test = split_samples(samples, cfg)
    save_split(train, test, cfg)

    n_correct = sum(1 for s in samples if s.open_text_label)
    n_incorrect = len(samples) - n_correct
    print(
        f"[{model}/{dataset}] train={len(train)} test={len(test)} "
        f"(correct={n_correct} incorrect={n_incorrect}) -> {output_dir / 'splits'}"
    )

    if samples and samples[0].category is not None:
        from collections import Counter
        cats = Counter(s.category for s in samples)
        print(f"[{model}/{dataset}] category breakdown: {dict(cats)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True, type=Path,
                         help="*_with_distractors.csv produced by generate_distractors_hf.py")
    parser.add_argument("--model", required=True, help="Model tag, e.g. qwen3_14b, gemma3_27b")
    parser.add_argument("--dataset", required=True, choices=["sciq", "triviaqa", "math"])
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_splits(args.input_csv, args.model, args.dataset, args.test_fraction, args.seed)


if __name__ == "__main__":
    main()