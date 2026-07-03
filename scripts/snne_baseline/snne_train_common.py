"""Shared config for regenerating SNNE generations + scores on the TRAIN split.

Plan B for the SNNE-combo experiment: the cached SNNE per-example scores only
cover the TEST split (results/snne_baseline/scores/<pair>.csv). To fold SNNE into
the train->test 2-axis combos (entropy/lap/hidden), we need SNNE scores on the
TRAIN split too. This module centralises the per-pair generation recipe so the
vLLM (llama/mistral/qwen) and HF (gemma) generators stay byte-consistent.

Everything writes to NEW namespaces so existing artifacts are untouched:
    results/snne_baseline/generations_train/<pair>.jsonl
    results/snne_baseline/scores_train/<pair>.csv

Generation recipe (verified against the source SU runs + regenerate_gaps.py):
  - 5-shot completion prefix ("Answer ... as briefly as possible." + 5 Q/A),
    resolved PER PAIR from the same run that produced that pair's TEST split.
  - n=10 samples, temperature=1.0, max_new_tokens=15, stop at newline, seed=10.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"
SU_DIR = Path("/data/kalashkala/semantic_uncertainty_data/uncertainty")

GEN_TRAIN_DIR = REPO_ROOT / "results" / "snne_baseline" / "generations_train"
SCORE_TRAIN_DIR = REPO_ROOT / "results" / "snne_baseline" / "scores_train"

N_SAMPLES = 10
TEMPERATURE = 1.0
MAX_TOKENS = 15          # SU runs used model_max_new_tokens=15 for every dataset
SEED = 10

MODEL_HF = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "gemma":   "google/gemma-3-12b-it",
}

DATASETS = ("sciq", "triviaqa", "math")
ALL_PAIRS = [f"{m}_{d}" for m in MODEL_HF for d in DATASETS]

# Run that produced each pair's TEST split -> use its 5-shot prefix for TRAIN too.
# 8 pairs come from their own SU run; the 4 gap pairs (no same-pair run) reuse the
# dataset-level prefix exactly as regenerate_gaps.py did for their test split.
READY_RUNS = {
    "llama_sciq":     "sciq__meta-llama__Llama-3.1-8B-Instruct__seed10__pid3815758__20260428_195641",
    "mistral_sciq":   "sciq__mistralai__Mistral-7B-Instruct-v0.3__seed10__pid3091374__20260513_143011",
    "llama_math":     "answerable_math__meta-llama__Llama-3.1-8B-Instruct__seed10__pid2950350__20260513_124800",
    "mistral_math":   "answerable_math__mistralai__Mistral-7B-Instruct-v0.3__seed10__pid3089414__20260513_142744",
    "qwen_math":      "answerable_math__Qwen__Qwen2.5-7B-Instruct__seed10__pid2950888__20260513_124831",
    "gemma_sciq":     "sciq__google__gemma-3-12b-it__seed10__pid1154278__20260609_192835",
    "gemma_triviaqa": "trivia_qa_nocontext__google__gemma-3-12b-it__seed10__pid1543419__20260610_104719",
    "gemma_math":     "answerable_math__google__gemma-3-12b-it__seed10__pid1175779__20260609_194248",
}
# Dataset-level fallback prefix (llama seed10 run), matching regenerate_gaps.py.
PREFIX_RUN = {
    "sciq":     "sciq__meta-llama__Llama-3.1-8B-Instruct__seed10__pid3815758__20260428_195641",
    "triviaqa": "trivia_qa__meta-llama__Llama-3.1-8B-Instruct__seed10__pid3241938__20260411_090924",
    "math":     "answerable_math__meta-llama__Llama-3.1-8B-Instruct__seed10__pid2950350__20260513_124800",
}


def pair_model_dataset(pair: str) -> tuple[str, str]:
    model, dataset = pair.split("_", 1)
    return model, dataset


def few_shot_prefix(pair: str) -> str:
    """5-shot completion prefix for this pair (own run if available, else dataset)."""
    _, dataset = pair_model_dataset(pair)
    run = READY_RUNS.get(pair) or PREFIX_RUN[dataset]
    d = pickle.load(open(SU_DIR / run / "experiment_details.pkl", "rb"))
    p = d["prompt"]
    assert p.count("Question:") == 5 and p.endswith("\n\n"), \
        f"unexpected prompt format for {pair} ({run})"
    return p


def load_train(pair: str):
    """[(id, question, label_bool)] from OUR train split."""
    rows = []
    path = DATA_DIR / f"ranking_experiment_{pair}" / "splits" / "train.jsonl"
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["id"], d["question"].strip(), bool(d["open_text_label"])))
    return rows


def build_prompts(pair: str):
    """(rows, prompts) for one pair; prompt = prefix + 'Question: q\\nAnswer:'."""
    prefix = few_shot_prefix(pair)
    rows = load_train(pair)
    prompts = [prefix + f"Question: {q}\nAnswer:" for _, q, _ in rows]
    return rows, prompts