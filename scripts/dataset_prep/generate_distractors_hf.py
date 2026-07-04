#!/usr/bin/env python3
"""
STAGE 00b — distractor generation for onboarding a new (model, dataset) pair.

Ported from recovery-gaps-experiment/scripts/generate_distractors_hf.py.
Generates 3 distractors per row using HuggingFace Transformers with
device_map="auto" (Llama-3.3-70B-Instruct judge, same model as
llm_judge_verdict_hf.py), for automatic multi-GPU distribution.

For each row:
  - LLM_verdict True  -> 3 plausible but wrong distractors (none equal to ground truth)
  - LLM_verdict False -> distractor[0] is the model's greedy answer (low_t_generation),
                         distractors[1-2] are additional plausible wrong answers

Output CSV preserves all input columns and appends a `candidate_list` column
containing the 3 distractors as a JSON list string. This candidate_list column
is required by ranking/data_loader.py (config.candidate_column) -- the 70/30
train/test split cannot be built without it.

Usage:
    python generate_distractors_hf.py --cuda_device 0,1 --batch_size 8

    nohup command:
    nohup python3 generate_distractors_hf.py --cuda_device 0,1 --batch_size 8 \\
        > ../../logs/distractors_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
UNCERTAINTY_RUNS_DIR = REPO_ROOT / "data" / "uncertainty_runs"

DEFAULT_JUDGE_MODEL = (
    "/data/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/"
    "snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
)

# ============================================================================
# MANUAL FILE LIST - sampled_* / sampled_all_* outputs of
# sample_balanced_by_category.py for the new (model, dataset) pairs.
# ============================================================================
FILES_TO_PROCESS = [
    UNCERTAINTY_RUNS_DIR / "sampled_1400_uncertainty_run_qwen3_14b_sciq_combined_full_llm_verdict_concentration_output.csv",
    UNCERTAINTY_RUNS_DIR / "sampled_1400_uncertainty_run_gemma3_27b_sciq_combined_full_llm_verdict_concentration_output.csv",
    UNCERTAINTY_RUNS_DIR / "sampled_all_uncertainty_run_qwen3_14b_answerable_math_combined_llm_verdict_concentration_output.csv",
    # Add once available (see sample_balanced_by_category.py FILES_TO_PROCESS):
    # UNCERTAINTY_RUNS_DIR / "sampled_1400_uncertainty_run_qwen3_14b_triviaqa_combined_50K_llm_verdict_concentration_output.csv",
    # UNCERTAINTY_RUNS_DIR / "sampled_1400_uncertainty_run_gemma3_27b_triviaqa_combined_50K_llm_verdict_concentration_output.csv",
    # UNCERTAINTY_RUNS_DIR / "sampled_all_uncertainty_run_gemma3_27b_answerable_math_combined_llm_verdict_concentration_output.csv",
]
# ============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

DISTRACTOR_RULES_CORRECT = """You are a distractor generator for multiple-choice questions.

You will be given a question and its correct answer. Your task is to generate exactly 3 distractor answers — plausible but WRONG options that could genuinely confuse someone who doesn't know the answer well.

=== RULES ===
- All 3 distractors must be WRONG (not equal to the correct answer, even as a paraphrase or variant).
- Distractors must be relevant and topically plausible — a reader should be tempted to pick them.
- Distractors must be meaningfully distinct from each other (no paraphrases, no plural variants).
- For NUMERIC answers: use genuinely different numbers (e.g. if correct is 4, use 3, 5, 7 — not 4.0).
- Keep each distractor concise (same style/length as the correct answer).
- Do NOT include explanations, labels, or any text outside the JSON.

=== OUTPUT FORMAT ===
Return a single JSON object with exactly one key: "distractors" — a list of exactly 3 strings.
Start your response with { and end with }. No markdown. No code fences.

Example (for illustration only):
{"distractors": ["vodka", "rum", "tequila"]}
"""

DISTRACTOR_RULES_INCORRECT = """You are a distractor generator for multiple-choice questions.

You will be given a question, its correct answer, and a wrong answer produced by a language model. Your task is to generate exactly 3 distractor answers. The first distractor MUST be the model's wrong answer; generate 2 additional plausible but WRONG distractors.

=== RULES ===
- distractors[0] MUST be the exact model wrong answer provided — do not alter it.
- distractors[1] and distractors[2] must be WRONG (not equal to the correct answer or to distractors[0]).
- All 3 distractors must be relevant and topically plausible.
- Distractors must be meaningfully distinct from each other (no paraphrases, no plural variants).
- For NUMERIC answers: use genuinely different numbers.
- Keep each distractor concise (same style/length as the correct answer).
- Do NOT include explanations, labels, or any text outside the JSON.

=== OUTPUT FORMAT ===
Return a single JSON object with exactly one key: "distractors" — a list of exactly 3 strings.
distractors[0] must be the model's wrong answer. Start with { and end with }. No markdown. No code fences.

Example (for illustration only):
{"distractors": ["bourbon", "vodka", "rum"]}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

def build_user_prompt(question: str, ground_truth: str, llm_verdict: bool, greedy: str) -> str:
    if llm_verdict:
        return (
            f"{DISTRACTOR_RULES_CORRECT}\n\n"
            f"=== INPUT ===\n"
            f"Question      : {question}\n"
            f"Correct answer: {ground_truth}\n\n"
            f"=== YOUR JSON RESPONSE ==="
        )
    else:
        return (
            f"{DISTRACTOR_RULES_INCORRECT}\n\n"
            f"=== INPUT ===\n"
            f"Question          : {question}\n"
            f"Correct answer    : {ground_truth}\n"
            f"Model wrong answer: {greedy}\n\n"
            f"=== YOUR JSON RESPONSE ==="
        )


def build_chat_prompt(tokenizer, question: str, ground_truth: str, llm_verdict: bool, greedy: str) -> str:
    messages = [{"role": "user", "content": build_user_prompt(question, ground_truth, llm_verdict, greedy)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ──────────────────────────────────────────────────────────────────────────────
# JSON extraction and validation
# ──────────────────────────────────────────────────────────────────────────────

def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json(text: str) -> str:
    text = strip_code_fences(text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError(f"No JSON object found in response: {text[:200]!r}")


def validate_distractors(data: dict, llm_verdict: bool, greedy: str, ground_truth: str) -> None:
    if "distractors" not in data:
        raise ValueError(f"Missing 'distractors' key. Got: {list(data.keys())}")
    d = data["distractors"]
    if not isinstance(d, list) or len(d) != 3:
        raise ValueError(f"'distractors' must be a list of 3 strings. Got: {d!r}")
    for item in d:
        if not str(item).strip():
            raise ValueError(f"Distractor must not be empty. Got: {item!r}")
    if not llm_verdict:
        if str(d[0]).strip().lower() != greedy.strip().lower():
            raise ValueError(
                f"distractors[0] must equal the model wrong answer '{greedy}'. Got: '{d[0]}'"
            )
    for item in d:
        if str(item).strip().lower() == ground_truth.strip().lower():
            raise ValueError(f"Distractor '{item}' matches the ground truth '{ground_truth}'.")


# ──────────────────────────────────────────────────────────────────────────────
# Resume support
# ──────────────────────────────────────────────────────────────────────────────

def load_existing_output(output_path: str) -> pd.DataFrame:
    if os.path.exists(output_path):
        try:
            return pd.read_csv(output_path)
        except Exception:
            print(f"WARNING: Could not read existing output at {output_path}. Starting fresh.")
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(
    prompts: list[str],
    tokenizer,
    model,
    max_new_tokens: int,
    first_device,
) -> list[str]:
    """Run a single batch through the model and return decoded new tokens per prompt."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    input_ids      = inputs["input_ids"].to(first_device)
    attention_mask = inputs["attention_mask"].to(first_device)
    input_len      = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for out in output_ids:
        new_tokens = out[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(text)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

def process_csv(
    input_path: str,
    output_path: str,
    model_path: str,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    max_retries: int = 3,
    model=None,
    tokenizer=None,
) -> None:
    if tokenizer is None:
        print(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if model is None:
        print(f"Loading model from    : {model_path}")
        print("  Using device_map='auto' — distributes layers across available GPUs")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        print("Model loaded.\n")

    first_device = next(model.parameters()).device

    df = pd.read_csv(input_path)

    if "LLM_verdict" not in df.columns:
        if "accuracy" not in df.columns:
            raise SystemExit("ERROR: CSV has neither 'LLM_verdict' nor 'accuracy' column.")
        print("NOTE: 'LLM_verdict' not found; deriving from 'accuracy' column.")
        df["LLM_verdict"] = df["accuracy"].map(lambda v: float(v) > 0)

    df["LLM_verdict"] = df["LLM_verdict"].map(
        lambda v: v if isinstance(v, bool) else str(v).strip().lower() == "true"
    )

    existing = load_existing_output(output_path)
    processed_questions: set = set(existing["question"].tolist()) if not existing.empty else set()
    results: list[dict] = existing.to_dict("records") if not existing.empty else []
    print(f"Resuming from {len(results)} already-processed entries.\n")

    # Filter to unprocessed rows only
    pending = df[~df["question"].isin(processed_questions)].copy()
    total   = len(df)
    pending_total = len(pending)
    print(f"Rows to process: {pending_total} / {total}\n")

    new_count = 0
    errors    = 0

    pending_list = pending.to_dict("records")

    for batch_start in range(0, len(pending_list), batch_size):
        batch_rows = pending_list[batch_start: batch_start + batch_size]
        batch_end  = min(batch_start + batch_size, pending_total)
        print(f"[{batch_start + 1}–{batch_end} / {pending_total}]")

        # ── Build prompts ────────────────────────────────────────────────────
        prompts = []
        for row in batch_rows:
            prompts.append(build_chat_prompt(
                tokenizer,
                question=str(row["question"]).strip(),
                ground_truth=str(row["ground_truth"]).strip(),
                llm_verdict=bool(row["LLM_verdict"]),
                greedy=str(row["low_t_generation"]).strip(),
            ))

        # ── Attempt batch; fall back to single-row retries on failure ────────
        raw_outputs = run_batch(prompts, tokenizer, model, max_new_tokens, first_device)

        for row, raw in zip(batch_rows, raw_outputs):
            question     = str(row["question"]).strip()
            ground_truth = str(row["ground_truth"]).strip()
            llm_verdict  = bool(row["LLM_verdict"])
            greedy       = str(row["low_t_generation"]).strip()

            distractors = None
            last_error  = None

            # Try to parse output from the batch run first, then retry individually
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt == 1:
                        text = raw
                    else:
                        # Re-run this single row with temperature > 0
                        temperature = 0.3 * (attempt - 1)
                        single_prompt = build_chat_prompt(
                            tokenizer, question, ground_truth, llm_verdict, greedy
                        )
                        inputs = tokenizer(
                            [single_prompt], return_tensors="pt",
                            padding=True, truncation=True, max_length=2048,
                        )
                        input_ids      = inputs["input_ids"].to(first_device)
                        attention_mask = inputs["attention_mask"].to(first_device)
                        input_len      = input_ids.shape[1]
                        with torch.no_grad():
                            out_ids = model.generate(
                                input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        text = tokenizer.decode(
                            out_ids[0][input_len:], skip_special_tokens=True
                        ).strip()

                    print(f"  {'OK' if attempt == 1 else f'Retry {attempt}'} | raw: {text[:150]}")
                    json_str    = extract_json(text)
                    parsed      = json.loads(json_str)
                    validate_distractors(parsed, llm_verdict, greedy, ground_truth)
                    distractors = parsed["distractors"]
                    break

                except Exception as e:
                    last_error = e
                    print(f"  Attempt {attempt}/{max_retries} failed: {e}")

            if distractors is None:
                print(f"  SKIPPING '{question[:60]}' after {max_retries} attempts. Last error: {last_error}")
                errors += 1
                continue

            entry = dict(row)
            entry["candidate_list"] = json.dumps(distractors, ensure_ascii=False)
            results.append(entry)
            processed_questions.add(question)
            new_count += 1

        # Persist after every batch
        pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\nFinished. New: {new_count}  |  Skipped (already done): {total - pending_total}  |  Errors: {errors}")
    print(f"Output: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3 distractors per row using HuggingFace Transformers (device_map=auto)."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
        help="HuggingFace model path (default: local Llama-3.3-70B-Instruct snapshot)",
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default=None,
        help="CUDA device index or indices (e.g. '0' or '0,1'). Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--batch_size",     type=int,   default=8,   help="Inference batch size (default: 8)")
    parser.add_argument("--max_new_tokens", type=int,   default=256, help="Max tokens to generate per row (default: 256)")
    parser.add_argument("--max_retries",    type=int,   default=3,   help="Retry attempts per row on failure (default: 3)")
    args = parser.parse_args()

    if not FILES_TO_PROCESS:
        print("ERROR: No files specified in FILES_TO_PROCESS list.")
        print("Please add CSV file paths to the FILES_TO_PROCESS list in this script.")
        return

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    # Load model once and reuse across all FILES_TO_PROCESS to avoid OOM on reload.
    print(f"Loading tokenizer from: {args.model}")
    shared_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    shared_tokenizer.padding_side = "left"
    if shared_tokenizer.pad_token is None:
        shared_tokenizer.pad_token = shared_tokenizer.eos_token

    print(f"Loading model from    : {args.model}")
    print("  Using device_map='auto' — distributes layers across available GPUs")
    shared_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    shared_model.eval()
    print("Model loaded.\n")

    for input_file in FILES_TO_PROCESS:
        input_file = Path(input_file)
        if not input_file.exists():
            print(f"ERROR: File not found: {input_file}")
            continue

        # Generate output filename: input.csv → input_with_distractors.csv
        output_file = input_file.parent / f"{input_file.stem}_with_distractors.csv"

        print(f"\n{'='*70}")
        print(f"Processing: {input_file.name}")
        print(f"Output: {output_file.name}")
        print(f"{'='*70}\n")

        process_csv(
            input_path=str(input_file),
            output_path=str(output_file),
            model_path=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_retries=args.max_retries,
            model=shared_model,
            tokenizer=shared_tokenizer,
        )


if __name__ == "__main__":
    main()