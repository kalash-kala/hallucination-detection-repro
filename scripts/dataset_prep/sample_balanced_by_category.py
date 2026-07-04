#!/usr/bin/env python3
"""
STAGE 00a — balanced category sampling for onboarding a new (model, dataset) pair.

Ported from recovery-gaps-experiment/scripts/sample_balanced_by_category.py.
Samples n rows from a concentration-output CSV (data/uncertainty_runs/) with
balanced strata across the `category` column (correct_high/correct_low/
incorrect_high/incorrect_low).

Precedent from the original gemma-3-12b-it onboarding (run_gemma_pipeline.sh):
  - sciq / triviaqa: --n_sample 1400 (balanced across the 4 categories)
  - math: NOT balanced-sampled — the full concentration_output CSV is used
    as-is (row count is already small), just copied with a sampled_all_ prefix.
    See copy_math_as_sampled_all() below.

Usage:
    python sample_balanced_by_category.py --n_sample 1400 --seed 42
"""

import argparse
import ast
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
UNCERTAINTY_RUNS_DIR = REPO_ROOT / "data" / "uncertainty_runs"

# ============================================================================
# Balanced-sample these (sciq / triviaqa) — new pairs being onboarded.
# ============================================================================
FILES_TO_PROCESS = [
    UNCERTAINTY_RUNS_DIR / "uncertainty_run_qwen3_14b_sciq_combined_full_llm_verdict_concentration_output.csv",
    UNCERTAINTY_RUNS_DIR / "uncertainty_run_gemma3_27b_sciq_combined_full_llm_verdict_concentration_output.csv",
    # Add once the judge job finishes (batch_size=8 run, see logs/llm_judge_verdict_qwen3_14b_triviaqa_bs8_*.log):
    # UNCERTAINTY_RUNS_DIR / "uncertainty_run_qwen3_14b_triviaqa_combined_50K_llm_verdict_concentration_output.csv",
    # Add once gemma3_27b triviaqa generation + judging exists:
    # UNCERTAINTY_RUNS_DIR / "uncertainty_run_gemma3_27b_triviaqa_combined_50K_llm_verdict_concentration_output.csv",
]

# ============================================================================
# Copy-as-is (math) — small enough that no balanced sampling is applied.
# ============================================================================
MATH_FILES_TO_COPY = [
    UNCERTAINTY_RUNS_DIR / "uncertainty_run_qwen3_14b_answerable_math_combined_llm_verdict_concentration_output.csv",
    # Add once gemma3_27b answerable_math generation + judging exists:
    # UNCERTAINTY_RUNS_DIR / "uncertainty_run_gemma3_27b_answerable_math_combined_llm_verdict_concentration_output.csv",
]
# ============================================================================


def extract_first(val: str) -> str:
    """Convert a Python list literal string like \"['gin']\" to its first element."""
    try:
        result = ast.literal_eval(str(val))
        if isinstance(result, list) and result:
            return str(result[0]).strip()
    except Exception:
        pass
    return str(val).strip("[]'\"")


def process_file(input_path: str, output_path: str, n: int, seed: int) -> None:
    """Process a single CSV file with category-based balanced sampling."""

    df = pd.read_csv(input_path)

    print(f"\n{'='*70}")
    print(f"Processing: {Path(input_path).name}")
    print(f"{'='*70}")

    # Check for required columns
    if "category" not in df.columns:
        sys.exit("ERROR: CSV does not have 'category' column.")

    if "LLM_verdict" not in df.columns:
        if "accuracy" not in df.columns:
            sys.exit("ERROR: CSV has neither 'LLM_verdict' nor 'accuracy' column.")
        print("NOTE: 'LLM_verdict' not found; deriving from 'accuracy' column.")
        df["LLM_verdict"] = df["accuracy"].map(lambda v: float(v) > 0)

    # Normalise LLM_verdict to bool
    df["LLM_verdict"] = df["LLM_verdict"].map(
        lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true"
    )

    # Normalise ground_truth to plain string
    if "ground_truth" in df.columns:
        df["ground_truth"] = df["ground_truth"].map(extract_first)

    # Remove duplicate questions
    if "question" in df.columns:
        df = df.drop_duplicates(subset=['question'], keep='first').reset_index(drop=True)
        print(f"After removing duplicates: {len(df)} unique questions")

    # Drop rows where low_t_generation is NaN
    if "low_t_generation" in df.columns:
        df = df.dropna(subset=['low_t_generation']).reset_index(drop=True)
        print(f"After removing NaN in low_t_generation: {len(df)} rows")

    # Get unique categories
    unique_categories = df["category"].unique()
    num_categories = len(unique_categories)
    rows_per_category = n // num_categories

    print(f"\nFound {num_categories} categories:")
    for cat in sorted(unique_categories):
        count = len(df[df["category"] == cat])
        print(f"  {cat}: {count} rows")

    # Split by category
    sampled_dfs = []
    for category in unique_categories:
        category_df = df[df["category"] == category]

        if len(category_df) < rows_per_category:
            print(f"WARNING: category '{category}' has only {len(category_df)} rows, "
                  f"need {rows_per_category}. Using all available.")
            sampled_dfs.append(category_df)
        else:
            sampled_dfs.append(
                category_df.sample(n=rows_per_category, random_state=seed)
            )

    # Combine and shuffle
    sampled = pd.concat(sampled_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)

    sampled.to_csv(output_path, index=False)

    print(f"\nSampled {len(sampled)} rows → {output_path}")
    print("Sampling breakdown:")
    for category in sorted(unique_categories):
        count = len(sampled[sampled["category"] == category])
        print(f"  {category}: {count}")


def copy_math_as_sampled_all(input_path: Path) -> None:
    """Math pairs skip balanced sampling: just copy with a sampled_all_ prefix."""
    output_path = input_path.parent / f"sampled_all_{input_path.name}"
    if output_path.exists():
        print(f"  [skip] {output_path.name} already exists")
        return
    shutil.copy(input_path, output_path)
    print(f"  created {output_path} (copy of full math dataset)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balanced stratified sampler by category for semantic-uncertainty CSVs."
    )
    parser.add_argument("--n_sample", required=True, type=int, help="Total number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    if not FILES_TO_PROCESS and not MATH_FILES_TO_COPY:
        print("ERROR: No files specified. Add CSV paths to FILES_TO_PROCESS / MATH_FILES_TO_COPY.")
        sys.exit(1)

    for input_file in FILES_TO_PROCESS:
        input_file = Path(input_file)
        if not input_file.exists():
            print(f"ERROR: File not found: {input_file}")
            continue

        output_file = input_file.parent / f"sampled_{args.n_sample}_{input_file.name}"
        process_file(str(input_file), str(output_file), args.n_sample, args.seed)

    for math_file in MATH_FILES_TO_COPY:
        math_file = Path(math_file)
        if not math_file.exists():
            print(f"ERROR: File not found: {math_file}")
            continue
        copy_math_as_sampled_all(math_file)


if __name__ == "__main__":
    main()