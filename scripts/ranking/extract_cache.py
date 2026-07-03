"""
REPRO STAGE 01a — hidden-state cache extraction (trimmed).

Builds, for one (model, dataset) pair, exactly the cache files that the Part 9 /
Part 11 analyses read — nothing else (no logit-lens sweep, no probe training,
no internal probe scoring, no distractor generation):

  cache/hidden_layer{L}.npz    last-answer-token hidden state per (sample, candidate)
  cache/index_layer{L}.json    "{id}|||{candidate}" -> row index
  cache/scores_layer{L}.jsonl  s_ext (layer-independent log-prob) per candidate

This is the "external sweep" step of the original ranking pipeline
(ranking/run_experiment.py, later re-run standalone by rebuild_clean_cache.py),
with samples loaded from the SHIPPED frozen splits — the splits are data, they
are never regenerated, so train/test membership is identical to the original
experiments by construction.

One teacher-forced forward pass per (sample, candidate) with
output_hidden_states=True captures every layer simultaneously.

Idempotent: external_scorer.score_all_layers() returns immediately if every
layer's cache file already exists (pass --overwrite to force a rebuild).

Usage:
  PY=.../envs/semantic_uncertainty/bin/python
  CUDA_VISIBLE_DEVICES=0 $PY scripts/ranking/extract_cache.py --model llama --dataset sciq
  CUDA_VISIBLE_DEVICES=0 $PY scripts/ranking/extract_cache.py --all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig
from data_loader import load_split
from external_scorer import score_all_layers

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"

MODEL_MAP = {
    "llama":   ("meta-llama/Llama-3.1-8B-Instruct",   32),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3",  32),
    "qwen":    ("Qwen/Qwen2.5-7B-Instruct",            28),
    "gemma":   ("google/gemma-3-12b-it",               48),
}
DATASETS = ("sciq", "triviaqa", "math")


def extract_pair(model: str, dataset: str, gpu_id: int, overwrite: bool) -> None:
    exp_dir = DATA_DIR / f"ranking_experiment_{model}_{dataset}"
    model_name, n_layers = MODEL_MAP[model]
    layers = list(range(1, n_layers + 1))

    cfg = ExperimentConfig(
        output_dir=exp_dir,
        model_name=model_name,
        gpu_id=gpu_id,
        dtype="bfloat16",
        alpha=1.0,
        layer_1idx=1,
    )

    cache_dir = cfg.paths()["cache"]
    done = all((cache_dir / f"hidden_layer{L}.npz").exists()
               and (cache_dir / f"index_layer{L}.json").exists()
               and (cache_dir / f"scores_layer{L}.jsonl").exists()
               for L in layers)
    if done and not overwrite:
        print(f"[skip] {model}/{dataset}: cache complete "
              f"({n_layers} layers) — use --overwrite to rebuild", flush=True)
        return
    if overwrite:
        for L in layers:
            for fn in (f"hidden_layer{L}.npz", f"index_layer{L}.json",
                       f"scores_layer{L}.jsonl"):
                (cache_dir / fn).unlink(missing_ok=True)

    train_samples = load_split(cfg, "train")
    test_samples = load_split(cfg, "test")
    all_samples = train_samples + test_samples
    print(f"=== {model}/{dataset}: external sweep — "
          f"{len(all_samples)} samples × {n_layers} layers ===", flush=True)
    t0 = time.time()
    score_all_layers(cfg, all_samples, layers)
    print(f"=== {model}/{dataset}: done in {(time.time() - t0) / 60:.1f} min ===",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--model", choices=list(MODEL_MAP))
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--all", action="store_true", help="run every (model, dataset) pair")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.all:
        pairs = [(m, d) for m in MODEL_MAP for d in DATASETS]
    else:
        if not (args.model and args.dataset):
            p.error("either --all or both --model and --dataset")
        pairs = [(args.model, args.dataset)]

    for model, dataset in pairs:
        extract_pair(model, dataset, args.gpu_id, args.overwrite)


if __name__ == "__main__":
    main()