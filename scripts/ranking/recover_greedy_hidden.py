"""
Recover MISSING greedy-answer hidden states for the 2-axis / classifier features.

Background
----------
The per-layer hidden cache (cache/hidden_layer{L}.npz + index_layer{L}.json) only
ever stored vectors for entries of each sample's *candidate_pool* (see
external_scorer.score_and_hidden_multilayer, which stores hidden_states[L][0, -1],
the LAST answer-token hidden state). For CORRECT samples the greedy generation was
usually NOT added to the pool (it was semantically equal to GT and dropped), so when
the greedy string does not string-match any cached candidate, the example has NO
cached greedy hidden state and is silently dropped from every hidden-state feature.

That is why hidden-state combos in two_axis_results.csv run on fewer test rows than
lap_only / entropy_only (e.g. qwen_sciq 420 -> 314). This script fills exactly those
gaps so all combos can be evaluated on the SAME full test set.

What it does
------------
For every test sample whose greedy key `{id}|||{greedy}` is absent from the existing
hidden index at the late layers (>=22), it runs ONE forward pass over (question,
greedy) and stores hidden_states[L][0, -1] for every late layer L — byte-identical
math to external_scorer / model_utils.score_and_hidden_multilayer.

Output (NEVER touches the existing cache):
  cache/greedy_late_hidden.npz   -> arrays {"L<late>": [n_missing, hidden]} + "keys"
  cache/greedy_late_hidden_index.json -> {"<id>|||<greedy>": row_idx}
  cache/greedy_late_hidden_meta.json  -> {late_layers, n_recovered, hidden_size}

Usage
-----
  PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
  CUDA_VISIBLE_DEVICES=1 $PY recover_greedy_hidden.py --all --gpu_id 0
  CUDA_VISIBLE_DEVICES=1 $PY recover_greedy_hidden.py --model qwen --dataset sciq --gpu_id 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import model_utils
from config import ExperimentConfig
from data_loader import load_split
from model_utils import load_model, score_and_hidden_multilayer


def reset_model() -> None:
    """Drop the cached model singleton so the NEXT load_model picks up a different
    model_name. Without this, load_model returns the first model loaded and every
    later pair is silently scored with the wrong network."""
    model_utils._MODEL = None
    model_utils._TOKENIZER = None
    model_utils._DEVICE = None
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"

MODEL_MAP = {
    "llama":   ("meta-llama/Llama-3.1-8B-Instruct", 32),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", 32),
    "qwen":    ("Qwen/Qwen2.5-7B-Instruct", 28),
}
PAIRS = [(m, d) for m in ("llama", "mistral", "qwen") for d in ("sciq", "triviaqa", "math")]

LATE_MIN = 22  # matches train_2axis load_hidden (layers >= 22)


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


def existing_late_layers(cache_dir: Path) -> list[int]:
    return sorted(
        int(p.stem.replace("hidden_layer", ""))
        for p in cache_dir.glob("hidden_layer*.npz")
        if int(p.stem.replace("hidden_layer", "")) >= LATE_MIN
    )


def missing_keys(cache_dir: Path, late_layers: list[int], id2greedy: dict[str, str]) -> set[str]:
    """ids (as `{id}|||{greedy}`) whose greedy hidden is absent in ANY late layer index."""
    indices = {L: json.loads((cache_dir / f"index_layer{L}.json").read_text()) for L in late_layers}
    miss = set()
    for qid, greedy in id2greedy.items():
        key = f"{qid}|||{greedy}"
        if not all(key in indices[L] for L in late_layers):
            miss.add(key)
    return miss


def process_pair(model: str, dataset: str, split: str, gpu_id: int, overwrite: bool) -> None:
    cfg = make_cfg(model, dataset, gpu_id)
    cache_dir = cfg.paths()["cache"]
    out_npz = cache_dir / f"greedy_late_hidden_{split}.npz"
    out_idx = cache_dir / f"greedy_late_hidden_{split}_index.json"
    out_meta = cache_dir / f"greedy_late_hidden_{split}_meta.json"

    late_layers = existing_late_layers(cache_dir)
    if not late_layers:
        print(f"=== {model}/{dataset} [{split}] === no late hidden layers found, skipping", flush=True)
        return

    samples = load_split(cfg, split)
    # greedy string per id, mirroring greedy_sidecar (s.greedy_prediction.strip())
    id2greedy = {str(s.id): (s.greedy_prediction or "").strip()
                 for s in samples if (s.greedy_prediction or "").strip()}
    sample_by_id = {str(s.id): s for s in samples}

    miss = missing_keys(cache_dir, late_layers, id2greedy)
    print(f"\n=== {model}/{dataset} [{split}] ===  n={len(samples)}  late_layers={late_layers}  "
          f"missing_greedy_hidden={len(miss)}", flush=True)

    if not miss:
        print("  nothing missing — all greedy hidden states already cached.", flush=True)
        return
    if out_npz.exists() and not overwrite:
        print(f"  {out_npz.name} exists — use --overwrite to recompute; skipping.", flush=True)
        return

    load_model(cfg)  # warm the singleton
    keys: list[str] = []
    rows = {L: [] for L in late_layers}
    for key in tqdm(sorted(miss), desc="recover greedy hidden"):
        qid, greedy = key.split("|||", 1)
        s = sample_by_id[qid]
        _, _, hs = score_and_hidden_multilayer(s.question, greedy, late_layers, cfg)
        keys.append(key)
        for L in late_layers:
            rows[L].append(hs[L].astype(np.float32))

    arrays = {f"L{L}": np.stack(rows[L], axis=0) for L in late_layers}
    arrays["keys"] = np.array(keys)
    np.savez_compressed(out_npz, **arrays)
    out_idx.write_text(json.dumps({k: i for i, k in enumerate(keys)}))
    out_meta.write_text(json.dumps({
        "late_layers": late_layers,
        "n_recovered": len(keys),
        "hidden_size": int(arrays[f"L{late_layers[0]}"].shape[1]),
        "late_min": LATE_MIN,
        "split": split,
    }))
    print(f"  recovered {len(keys)} greedy hidden states -> {out_npz.name}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_MAP))
    p.add_argument("--dataset", choices=["sciq", "triviaqa", "math"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--splits", nargs="+", default=["train", "test"],
                   choices=["train", "test"])
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
    prev_model = None
    for model, dataset in pairs:
        if model != prev_model:
            reset_model()  # force reload of the correct network when model changes
            prev_model = model
        for split in args.splits:
            process_pair(model, dataset, split, args.gpu_id, args.overwrite)


if __name__ == "__main__":
    main()