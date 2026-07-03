"""
LapEigvals baseline — STEP 1: capture attention and reduce to diagonals.

The recovery-gaps caches store hidden states only (no attention), so the
LapEigvals family cannot be computed from existing data. This script re-runs one
TEACHER-FORCED forward pass per example over (question + greedy answer) with
`output_attentions=True` and `attn_implementation="eager"` (eager is required —
SDPA/flash do not return attention weights), then immediately reduces each
example's attention to the cheap [#layers, #heads, #seq] diagonals that the
LapEigvals features need, discarding the full seq×seq tensors.

Faithful-adaptation note: the upstream paper captures attention during greedy
*generation* (rows = generated tokens). A teacher-forced pass over the cached
greedy string yields the identical answer-position attention deterministically
and avoids any answer drift vs the labels we already scored. We compute the
diagonals over the full causal matrix (all positions), exactly matching the
upstream feature builders, which top-k / log-mean over every sequence position.

Runs both the TRAIN and TEST splits (the probe is supervised, trained on train).
Label convention matches Classifier A: y = 1 for a hallucination (wrong answer),
i.e. y = int(not open_text_label).

Outputs (one file per pair+split):
  results/lapeigvals_baseline/diags/{model}_{dataset}_{split}.pt
    {"attn_diags": [Tensor[L,H,S] fp16], "lap_diags": [...],
     "labels": LongTensor[N], "ids": [str], "categories": [str|None]}

Usage:
  PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
  CUDA_VISIBLE_DEVICES=0 $PY extract_attention.py --model llama --dataset sciq
  CUDA_VISIBLE_DEVICES=0 $PY extract_attention.py --all
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# reuse the ranking pipeline's split loader / prompt tokenisation
RANKING_DIR = Path(__file__).resolve().parents[1] / "ranking"
sys.path.insert(0, str(RANKING_DIR))
from config import ExperimentConfig          # noqa: E402
from data_loader import load_split, Sample   # noqa: E402
from model_utils import _answer_token_span   # noqa: E402

from lapeigvals_features import attention_diagonal, laplacian_diagonal_from_attn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"
OUT_DIR = REPO_ROOT / "results" / "lapeigvals_baseline" / "diags"

MODEL_MAP = {
    "llama":   ("meta-llama/Llama-3.1-8B-Instruct", 32),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", 32),
    "qwen":    ("Qwen/Qwen2.5-7B-Instruct", 28),
    "gemma":   ("google/gemma-3-12b-it", 48),
}
PAIRS = [(m, d) for m in ("llama", "mistral", "qwen", "gemma") for d in ("sciq", "triviaqa", "math")]
SPLITS = ("train", "test")


def make_cfg(model: str, dataset: str, gpu_id: int) -> ExperimentConfig:
    model_name, _ = MODEL_MAP[model]
    return ExperimentConfig(
        output_dir=DATA_DIR / f"ranking_experiment_{model}_{dataset}",
        model_name=model_name,
        gpu_id=gpu_id,
        dtype="bfloat16",
        alpha=1.0,
        layer_1idx=1,
    )


def load_model_eager(model_name: str, gpu_id: int):
    """Like ranking.model_utils.load_model but forces eager attention (so that
    output_attentions returns the weights) and does NOT use the cached global."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


@torch.no_grad()
def diagonals_one(s: Sample, model, tokenizer, device):
    """One teacher-forced pass -> (attn_diag, lap_diag) fp16 on CPU, or None."""
    greedy = (s.greedy_prediction or "").strip()
    if greedy == "":
        return None
    full_ids, _answer_ids, _prefix_len = _answer_token_span(tokenizer, s.question, greedy)
    input_ids = torch.tensor([full_ids], device=device)

    outputs = model(input_ids=input_ids, output_attentions=True, use_cache=False)
    # tuple over layers of [1, H, S, S] -> list of [H, S, S] in fp32 for stable math
    item_attn = [a[0].float() for a in outputs.attentions]

    attn_diag = attention_diagonal(item_attn)                              # [L, H, S]
    lap_diag = laplacian_diagonal_from_attn(item_attn, vertical_edges=False)  # [L, H, S]

    out = (attn_diag.half().cpu(), lap_diag.half().cpu())
    del outputs, item_attn, attn_diag, lap_diag, input_ids
    return out


def process_pair(model: str, dataset: str, gpu_id: int, overwrite: bool) -> None:
    cfg = make_cfg(model, dataset, gpu_id)
    targets = {
        split: OUT_DIR / f"{model}_{dataset}_{split}.pt" for split in SPLITS
    }
    if all(p.exists() for p in targets.values()) and not overwrite:
        print(f"=== {model}/{dataset} === all splits cached — skipping (use --overwrite)", flush=True)
        return

    model_name = MODEL_MAP[model][0]
    print(f"\n=== {model}/{dataset} === loading {model_name} (eager attn)", flush=True)
    t0 = time.time()
    model_obj, tokenizer, device = load_model_eager(model_name, gpu_id)
    print(f"  model loaded in {time.time()-t0:.1f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        out_path = targets[split]
        if out_path.exists() and not overwrite:
            print(f"  [{split}] cached — skipping", flush=True)
            continue
        samples = load_split(cfg, split)
        attn_diags, lap_diags, labels, ids, cats = [], [], [], [], []
        n_skip = 0
        for s in tqdm(samples, desc=f"{model}/{dataset}/{split}"):
            res = diagonals_one(s, model_obj, tokenizer, device)
            if res is None:
                n_skip += 1
                continue
            a_diag, l_diag = res
            attn_diags.append(a_diag)
            lap_diags.append(l_diag)
            labels.append(int(not s.open_text_label))  # 1 = hallucination (wrong)
            ids.append(s.id)
            cats.append(s.category)
        torch.save(
            {
                "attn_diags": attn_diags,
                "lap_diags": lap_diags,
                "labels": torch.tensor(labels, dtype=torch.long),
                "ids": ids,
                "categories": cats,
                "model": model,
                "dataset": dataset,
                "split": split,
            },
            out_path,
        )
        pos = int(sum(labels))
        print(f"  [{split}] wrote {len(ids)} examples "
              f"(pos/wrong={pos}, skipped empty greedy={n_skip}) -> {out_path}", flush=True)

    del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_MAP))
    p.add_argument("--dataset", choices=["sciq", "triviaqa", "math"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.all:
        pairs = PAIRS
    elif args.model and args.dataset:
        pairs = [(args.model, args.dataset)]
    else:
        raise SystemExit("specify --all or both --model and --dataset")
    for model, dataset in pairs:
        process_pair(model, dataset, args.gpu_id, args.overwrite)


if __name__ == "__main__":
    main()