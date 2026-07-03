"""
Compute greedy + distractor sidecars for the TRAIN split.

The existing greedy_sidecar.jsonl and distractor_sidecar.jsonl only cover the 420
test examples. This script produces the equivalent files for the 980 train examples,
enabling Classifier A to be trained on the full train split.

Instead of generating new distractors (which requires expensive LLM generation),
we score the existing candidate_list distractors from train.jsonl — these are the
same distractors used for probe training and are available without extra inference.

Outputs (per pair):
  cache/greedy_sidecar_train.jsonl   — same format as greedy_sidecar.jsonl
  cache/distractor_sidecar_train.jsonl — same format as distractor_sidecar.jsonl

Usage:
  PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
  CUDA_VISIBLE_DEVICES=0 $PY compute_train_sidecars.py --model llama --dataset sciq --gpu_id 0
  CUDA_VISIBLE_DEVICES=0,1 $PY compute_train_sidecars.py --all --gpu_id 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"

sys.path.insert(0, str(Path(__file__).parent))
from config import ExperimentConfig
from data_loader import load_split
from model_utils import load_model, _answer_token_span
import model_utils as _model_utils

MODEL_MAP = {
    "llama":   ("meta-llama/Llama-3.1-8B-Instruct", 32),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", 32),
    "qwen":    ("Qwen/Qwen2.5-7B-Instruct", 28),
    "gemma":   ("google/gemma-3-12b-it", 48),
}
PAIRS = [(m, d) for m in ("llama", "mistral", "qwen", "gemma") for d in ("sciq", "triviaqa", "math")]


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


def load_all_probes(cfg: ExperimentConfig, layers: list[int]) -> dict[int, tuple]:
    probe_dir = cfg.paths()["probe"]
    out: dict[int, tuple] = {}
    for L in layers:
        clf_p = probe_dir / f"probe_layer{L}.joblib"
        scl_p = probe_dir / f"scaler_layer{L}.joblib"
        if not (clf_p.exists() and scl_p.exists()):
            continue
        clf = joblib.load(clf_p)
        scaler = joblib.load(scl_p)
        out[L] = (clf.coef_.ravel().astype(np.float64),
                  float(clf.intercept_[0]),
                  scaler)
    return out


@torch.no_grad()
def score_answer(
    question: str, answer: str, model_obj, tokenizer, device,
    final_norm, lm_head, layers: list[int], probes: dict[int, tuple], alpha: float,
) -> dict | None:
    answer = (answer or "").strip()
    if answer == "":
        return None

    full_ids, answer_ids, prefix_len = _answer_token_span(tokenizer, question, answer)
    input_ids = torch.tensor([full_ids], device=device)
    outputs = model_obj(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states

    pred_positions = list(range(prefix_len - 1, len(full_ids) - 1))
    n_tok = len(answer_ids)
    if n_tok == 0:
        return None
    tgt = torch.tensor(answer_ids, device=device)

    h_stack = torch.stack([hidden_states[L][0, pred_positions, :] for L in layers], dim=0)
    h_normed = final_norm(h_stack)
    logits_ll = lm_head(h_normed)
    log_probs = torch.log_softmax(logits_ll.float(), dim=-1)
    tgt_exp = tgt.unsqueeze(0).unsqueeze(-1).expand(len(layers), -1, 1)
    token_lps = log_probs.gather(2, tgt_exp).squeeze(-1)
    sum_lps = token_lps.sum(dim=1).tolist()
    s_ext_L = {str(L): sum_lps[i] / max(n_tok, 1) ** alpha for i, L in enumerate(layers)}

    s_int: dict[str, float] = {}
    for L in layers:
        if L not in probes:
            continue
        coef, intercept, scaler = probes[L]
        h_last = hidden_states[L][0, -1].float().cpu().numpy().reshape(1, -1)
        z = float((scaler.transform(h_last).ravel() @ coef) + intercept)
        s_int[str(L)] = 1.0 / (1.0 + np.exp(-z))

    return {
        "candidate": answer,
        "n_answer_tokens": n_tok,
        "s_ext_L": s_ext_L,
        "s_int": s_int,
    }


def process_dataset(
    model: str, dataset: str, gpu_id: int, overwrite: bool,
    model_obj, tokenizer, device, final_norm, lm_head,
) -> None:
    cfg = make_cfg(model, dataset, gpu_id)
    n_layers = MODEL_MAP[model][1]
    layers = list(range(1, n_layers + 1))
    cache_dir = cfg.paths()["cache"]

    greedy_out = cache_dir / "greedy_sidecar_train.jsonl"
    dist_out   = cache_dir / "distractor_sidecar_train.jsonl"

    if greedy_out.exists() and dist_out.exists() and not overwrite:
        n_g = sum(1 for _ in greedy_out.open())
        n_d = sum(1 for _ in dist_out.open())
        print(f"\n=== {model}/{dataset} ===  train sidecars exist "
              f"(greedy={n_g}, dist={n_d}) — use --overwrite to recompute; skipping.")
        return

    train = load_split(cfg, "train")
    print(f"\n=== {model}/{dataset} ===  train={len(train)}", flush=True)

    probes = load_all_probes(cfg, layers)
    print(f"  loaded probes for {len(probes)}/{n_layers} layers", flush=True)

    greedy_rows = []
    dist_rows   = []
    n_skip = 0

    for s in tqdm(train, desc=f"{model}/{dataset}"):
        g = score_answer(s.question, s.greedy_prediction, model_obj, tokenizer,
                         device, final_norm, lm_head, layers, probes, cfg.alpha)
        if g is None:
            n_skip += 1
            continue
        greedy_rows.append({
            "id": s.id,
            "candidate": g["candidate"],
            "open_text_label": s.open_text_label,
            "category": s.category,
            "n_answer_tokens": g["n_answer_tokens"],
            "s_ext_L": g["s_ext_L"],
            "s_int": g["s_int"],
        })

        scored_dists = []
        for dist_text in (s.candidate_list or []):
            d = score_answer(s.question, dist_text, model_obj, tokenizer,
                             device, final_norm, lm_head, layers, probes, cfg.alpha)
            if d is not None:
                scored_dists.append(d)
        dist_rows.append({
            "id": s.id,
            "category": s.category,
            "open_text_label": s.open_text_label,
            "n_distractors": len(scored_dists),
            "distractors": scored_dists,
        })

    cache_dir.mkdir(parents=True, exist_ok=True)
    with greedy_out.open("w") as f:
        for r in greedy_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  [written] greedy_sidecar_train: {len(greedy_rows)} rows → {greedy_out}", flush=True)
    with dist_out.open("w") as f:
        for r in dist_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  [written] distractor_sidecar_train: {len(dist_rows)} rows → {dist_out}", flush=True)
    print(f"  skipped {n_skip} empty greedy", flush=True)


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
        print("Specify --model + --dataset or --all")
        sys.exit(1)

    # Group by model to load each model only once
    from itertools import groupby
    for model_name, group in groupby(pairs, key=lambda x: x[0]):
        datasets = [d for _, d in group]
        print(f"\n>>> Loading {model_name} model...", flush=True)
        # Use first dataset's cfg just to get model path / gpu
        cfg0 = make_cfg(model_name, datasets[0], args.gpu_id)
        model_obj, tokenizer, device = load_model(cfg0)
        final_norm = model_obj.model.norm if hasattr(model_obj.model, "norm") else model_obj.model.language_model.norm
        lm_head = model_obj.lm_head

        for dataset in datasets:
            process_dataset(
                model_name, dataset, args.gpu_id, args.overwrite,
                model_obj, tokenizer, device, final_norm, lm_head,
            )

        # Free GPU memory before loading next model
        del model_obj, final_norm, lm_head
        _model_utils._MODEL = _model_utils._TOKENIZER = _model_utils._DEVICE = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
