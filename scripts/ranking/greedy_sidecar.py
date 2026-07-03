"""
Standalone sidecar scorer for the model's GREEDY generation (low_t_generation).

Motivation
----------
For CORRECT samples (LLM_verdict=True) the greedy is semantically equal to GT and
was therefore NEVER added to candidate_pool, so the main pipeline never scored it
(no hidden state, no s_ext_L, no s_int).  For INCORRECT samples the greedy IS in
the pool and was scored, but it lives mixed into the per-layer cache files.

To get a clean, deployment-honest view of "what the model thinks of its OWN output"
across all four categories, this script computes — for the greedy string of EVERY
test sample — both scores at EVERY layer, in ONE forward pass per sample:

  * s_ext_L : logit-lens length-normalised log-prob (final_norm + lm_head applied
              to hidden_states[L] at the answer prediction positions).  Identical
              math to logitlens_scorer._score_pair.
  * s_int   : per-layer probe probability (StandardScaler -> LogisticRegression)
              applied to hidden_states[L][0, -1] (last answer token).  Identical
              to internal_scorer / probe_dataset conventions.

It writes ONE small sidecar file and NEVER touches the existing cache:

  cache/greedy_sidecar.jsonl
    {"id", "candidate"(=greedy), "open_text_label", "category",
     "n_answer_tokens",
     "s_ext_L": {"1": .., ..., "L": ..},
     "s_int":   {"1": .., ..., "L": ..}}

This computes greedy for ALL test samples (correct + incorrect) so the trajectory
plot reads the greedy curve from a single, method-consistent source for every
category.

Usage
-----
  PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
  CUDA_VISIBLE_DEVICES=0 $PY greedy_sidecar.py --model llama --dataset sciq --gpu_id 0
  CUDA_VISIBLE_DEVICES=0 $PY greedy_sidecar.py --all --gpu_id 0
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

sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig
from data_loader import load_split, Sample
from model_utils import load_model, _answer_token_span

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"

MODEL_MAP = {
    "llama":   ("meta-llama/Llama-3.1-8B-Instruct", 32),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", 32),
    "qwen":    ("Qwen/Qwen2.5-7B-Instruct", 28),
    "gemma":   ("google/gemma-3-12b-it", 48),
}
PAIRS = [(m, d) for m in ("llama", "mistral", "qwen", "gemma") for d in ("sciq", "triviaqa", "math")]

SIDECAR_NAME = "greedy_sidecar.jsonl"


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
    """Return {L: (coef_vector, intercept, scaler)} for layers whose probe exists.

    We extract raw numpy arrays so the probe can be applied without going through
    sklearn's version-sensitive predict_proba code path.
    """
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
def score_greedy_one(
    s: Sample, model, tokenizer, device, final_norm, lm_head,
    layers: list[int], probes: dict[int, tuple], alpha: float,
) -> dict | None:
    """One forward pass over (question, greedy); per-layer s_ext_L and s_int."""
    greedy = (s.greedy_prediction or "").strip()
    if greedy == "":
        return None

    full_ids, answer_ids, prefix_len = _answer_token_span(tokenizer, s.question, greedy)
    input_ids = torch.tensor([full_ids], device=device)
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # tuple len num_blocks+1

    # ---- logit-lens s_ext_L (answer prediction positions) ----
    pred_positions = list(range(prefix_len - 1, len(full_ids) - 1))
    n_tok = len(answer_ids)
    tgt = torch.tensor(answer_ids, device=device)

    h_stack = torch.stack([hidden_states[L][0, pred_positions, :] for L in layers], dim=0)
    h_normed = final_norm(h_stack)
    logits_ll = lm_head(h_normed)
    log_probs = torch.log_softmax(logits_ll.float(), dim=-1)
    tgt_exp = tgt.unsqueeze(0).unsqueeze(-1).expand(len(layers), -1, 1)
    token_lps = log_probs.gather(2, tgt_exp).squeeze(-1)      # [n_layers, n_ans]
    sum_lps = token_lps.sum(dim=1).tolist()                   # [n_layers]
    s_ext_L = {str(L): sum_lps[i] / max(n_tok, 1) ** alpha
               for i, L in enumerate(layers)}

    # ---- internal probe s_int (last answer token) ----
    s_int: dict[str, float] = {}
    for L in layers:
        if L not in probes:
            continue
        coef, intercept, scaler = probes[L]
        h_last = hidden_states[L][0, -1].float().cpu().numpy().reshape(1, -1)
        z = float((scaler.transform(h_last).ravel() @ coef) + intercept)
        s_int[str(L)] = 1.0 / (1.0 + np.exp(-z))

    return {
        "id": s.id,
        "candidate": greedy,
        "open_text_label": s.open_text_label,
        "category": s.category,
        "n_answer_tokens": n_tok,
        "s_ext_L": s_ext_L,
        "s_int": s_int,
    }


def process_pair(model: str, dataset: str, gpu_id: int, overwrite: bool) -> None:
    cfg = make_cfg(model, dataset, gpu_id)
    n_layers = MODEL_MAP[model][1]
    layers = list(range(1, n_layers + 1))
    cache_dir = cfg.paths()["cache"]
    out_path = cache_dir / SIDECAR_NAME

    test = load_split(cfg, "test")
    print(f"\n=== {model}/{dataset} ===  test={len(test)}", flush=True)

    if out_path.exists() and not overwrite:
        n_done = sum(1 for _ in out_path.open())
        print(f"  sidecar exists ({n_done} rows) — use --overwrite to recompute; skipping.", flush=True)
        return

    probes = load_all_probes(cfg, layers)
    print(f"  loaded probes for {len(probes)}/{n_layers} layers", flush=True)

    model_obj, tokenizer, device = load_model(cfg)
    final_norm = model_obj.model.norm if hasattr(model_obj.model, "norm") else model_obj.model.language_model.norm
    lm_head = model_obj.lm_head

    rows = []
    n_skip = 0
    for s in tqdm(test, desc="greedy sidecar"):
        rec = score_greedy_one(s, model_obj, tokenizer, device,
                               final_norm, lm_head, layers, probes, cfg.alpha)
        if rec is None:
            n_skip += 1
            continue
        rows.append(rec)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} rows (skipped {n_skip} empty greedy) → {out_path}", flush=True)


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
