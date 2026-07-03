"""
Clean rebuild of the cache for a single (model, dataset) pair.

Use case
--------
The llama/triviaqa cache was corrupted by a failed topup merge (layers 1-8 polluted
with extra alias rows, layer 9 hidden npz truncated).  This script rebuilds it from
scratch in four steps:

  1. Safety rename: cache/ -> cache_backup_{timestamp}/   (instant, nothing lost)
  2. External sweep: one forward pass per (sample, candidate) capturing ALL layers
     simultaneously.  Writes hidden_layer{L}.npz + scores_layer{L}.jsonl +
     index_layer{L}.json for every L.  Runs on ALL samples (train + test) to match
     the original run_experiment.py convention.
  3. Logitlens sweep: one forward pass per (sample, candidate) applying
     final_norm + lm_head to each layer's hidden state at answer positions.
     Writes logitlens_layer{L}.jsonl for every L.  Also runs on ALL samples.
  4. Internal scoring: for each layer L loads hidden_layer{L}.npz + the
     EXISTING probe_layer{L}.joblib / scaler_layer{L}.joblib from probe/ and
     writes internal_scores_layer{L}.jsonl.  Runs on test samples only.
     The probe is NOT retrained — the restored _DAMAGED_probe was trained on
     the original clean data and is still valid.

Staging strategy
----------------
All three GPU stages write to  cache_rebuild_{timestamp}/  (staging).
Only if ALL expected files are present at the end does the staging dir get
renamed to  cache/.  If the script dies mid-run, the backup and staging
dirs are left untouched for inspection.

Usage
-----
  PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
  CUDA_VISIBLE_DEVICES=1 nohup $PY rebuild_clean_cache.py \\
      --model llama --dataset triviaqa --gpu_id 1 \\
      > /tmp/rebuild_llama_triviaqa_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  echo "PID=$!"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig
from data_loader import load_split, Sample
from model_utils import load_model, _answer_token_span
from external_scorer import score_all_layers as ext_score_all_layers
from logitlens_scorer import score_all_layers as ll_score_all_layers

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "recovery-gaps-data" / "data"

MODEL_MAP = {
    "llama":   ("meta-llama/Llama-3.1-8B-Instruct",   32),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3",  32),
    "qwen":    ("Qwen/Qwen2.5-7B-Instruct",            28),
}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def make_cfg(model: str, dataset: str, gpu_id: int, output_dir: Path) -> ExperimentConfig:
    """ExperimentConfig with output_dir set to the given directory.

    input_csv is left at its default — it is only needed by load_samples() which
    we never call here (we load from saved splits instead).
    """
    model_name, _ = MODEL_MAP[model]
    return ExperimentConfig(
        output_dir=output_dir,
        model_name=model_name,
        gpu_id=gpu_id,
        dtype="bfloat16",
        alpha=1.0,
        layer_1idx=1,
    )


# ---------------------------------------------------------------------------
# Step 4: apply existing probes to newly computed hidden states
# ---------------------------------------------------------------------------

def run_internal_scoring(
    test_samples: list[Sample],
    layers: list[int],
    cache_dir: Path,
    probe_dir: Path,
) -> None:
    log(f"  [internal] applying probes for {len(layers)} layers ...")
    missing_probes = []

    for L in tqdm(layers, desc="internal scoring"):
        clf_path = probe_dir / f"probe_layer{L}.joblib"
        scl_path = probe_dir / f"scaler_layer{L}.joblib"
        if not (clf_path.exists() and scl_path.exists()):
            missing_probes.append(L)
            continue

        hidden_path = cache_dir / f"hidden_layer{L}.npz"
        index_path  = cache_dir / f"index_layer{L}.json"
        if not (hidden_path.exists() and index_path.exists()):
            log(f"  [internal] layer {L}: hidden/index missing — skipping")
            continue

        clf    = joblib.load(clf_path)
        scaler = joblib.load(scl_path)
        coef   = clf.coef_.ravel().astype(np.float64)
        intercept = float(clf.intercept_[0])

        full_hidden = np.load(hidden_path)["hidden"]
        index       = json.loads(index_path.read_text())

        rows = []
        for s in test_samples:
            for cand in s.candidate_pool:
                key = f"{s.id}|||{cand}"
                if key not in index:
                    continue
                h = full_hidden[index[key]].reshape(1, -1).astype(np.float64)
                z = float((scaler.transform(h).ravel() @ coef) + intercept)
                s_int = 1.0 / (1.0 + np.exp(-z))
                rows.append({"id": s.id, "candidate": cand, "s_int": float(s_int)})

        out_path = cache_dir / f"internal_scores_layer{L}.jsonl"
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if missing_probes:
        log(f"  [internal] WARNING: probes missing for layers {missing_probes}")
    log(f"  [internal] done")


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

def verify_staging(staging_dir: Path, layers: list[int]) -> list[str]:
    """Return list of missing filenames (empty = all good)."""
    patterns = [
        "hidden_layer{L}.npz",
        "index_layer{L}.json",
        "scores_layer{L}.jsonl",
        "logitlens_layer{L}.jsonl",
        "internal_scores_layer{L}.jsonl",
    ]
    missing = []
    for L in layers:
        for pattern in patterns:
            fn = pattern.format(L=L)
            if not (staging_dir / fn).exists():
                missing.append(fn)
    return missing


def print_summary(cache_dir: Path, n_layers: int) -> None:
    log("--- Cache summary ---")
    for label, glob in [
        ("hidden npz",      "hidden_layer*.npz"),
        ("index json",      "index_layer*.json"),
        ("scores jsonl",    "scores_layer*.jsonl"),
        ("logitlens jsonl", "logitlens_layer*.jsonl"),
        ("internal jsonl",  "internal_scores_layer*.jsonl"),
    ]:
        n = len(list(cache_dir.glob(glob)))
        ok = "✓" if n == n_layers else f"✗ (got {n}, want {n_layers})"
        log(f"  {label:22s}: {n:3d}  {ok}")

    for name, path in [("scores_layer1", cache_dir / "scores_layer1.jsonl"),
                        ("internal_layer1", cache_dir / "internal_scores_layer1.jsonl")]:
        if path.exists():
            n = sum(1 for _ in path.open())
            log(f"  {name} rows        : {n}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def rebuild(model: str, dataset: str, gpu_id: int) -> None:
    exp_dir  = DATA_DIR / f"ranking_experiment_{model}_{dataset}"
    n_layers = MODEL_MAP[model][1]
    layers   = list(range(1, n_layers + 1))
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")

    real_cache  = exp_dir / "cache"
    backup_dir  = exp_dir / f"cache_backup_{ts}"
    staging_dir = exp_dir / f"cache_rebuild_{ts}"
    probe_dir   = exp_dir / "probe"

    log(f"=== Rebuild: {model}/{dataset} | layers=1..{n_layers} | gpu={gpu_id} ===")
    log(f"  exp_dir   : {exp_dir}")
    log(f"  probe_dir : {probe_dir}  ({len(list(probe_dir.glob('*.joblib')))} joblibs found)")

    # ------------------------------------------------------------------
    # Step 1 — Safety backup of existing cache
    # ------------------------------------------------------------------
    if real_cache.exists():
        real_cache.rename(backup_dir)
        log(f"[step 1] Backed up cache/ -> {backup_dir.name}")
    else:
        log(f"[step 1] No existing cache/ to back up — starting fresh")

    staging_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load samples from existing saved splits (avoids needing input_csv)
    # ------------------------------------------------------------------
    split_cfg = make_cfg(model, dataset, gpu_id, output_dir=exp_dir)
    train_samples = load_split(split_cfg, "train")
    test_samples  = load_split(split_cfg, "test")
    all_samples   = train_samples + test_samples
    log(f"[data] train={len(train_samples)}  test={len(test_samples)}  total={len(all_samples)}")

    # Build a cfg whose cache path points to staging_dir.
    # ExperimentConfig.paths()["cache"] = output_dir / "cache"
    # So we need output_dir / "cache" == staging_dir.
    # staging_dir = exp_dir / "cache_rebuild_{ts}", not "cache".
    # Solution: make a temporary parent dir and symlink its "cache" child to staging_dir.
    tmp_parent = exp_dir / f"_tmp_rebuild_parent_{ts}"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    cache_link = tmp_parent / "cache"
    cache_link.symlink_to(staging_dir.resolve())
    sweep_cfg = make_cfg(model, dataset, gpu_id, output_dir=tmp_parent)

    try:
        # ------------------------------------------------------------------
        # Step 2 — External sweep: hidden states + s_ext
        # ------------------------------------------------------------------
        log(f"\n[step 2] External sweep (hidden states + s_ext) — {len(all_samples)} samples × {n_layers} layers")
        t0 = time.time()
        ext_score_all_layers(sweep_cfg, all_samples, layers)
        log(f"[step 2] Done in {(time.time()-t0)/60:.1f} min")

        # ------------------------------------------------------------------
        # Step 3 — Logitlens sweep: s_ext_L per layer
        # ------------------------------------------------------------------
        log(f"\n[step 3] Logitlens sweep (s_ext_L) — {len(all_samples)} samples × {n_layers} layers")
        t0 = time.time()
        ll_score_all_layers(sweep_cfg, all_samples, layers)
        log(f"[step 3] Done in {(time.time()-t0)/60:.1f} min")

    finally:
        # Always clean up the symlink parent, even if steps 2/3 fail
        try:
            cache_link.unlink()
            tmp_parent.rmdir()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 4 — Internal scoring (no GPU needed, reads staging_dir)
    # ------------------------------------------------------------------
    log(f"\n[step 4] Internal scoring (s_int) — {len(test_samples)} test samples × {n_layers} layers")
    t0 = time.time()
    run_internal_scoring(
        test_samples=test_samples,
        layers=layers,
        cache_dir=staging_dir,
        probe_dir=probe_dir,
    )
    log(f"[step 4] Done in {(time.time()-t0)/60:.1f} min")

    # ------------------------------------------------------------------
    # Verify and promote
    # ------------------------------------------------------------------
    log(f"\n[verify] Checking staging dir completeness ...")
    missing = verify_staging(staging_dir, layers)

    if missing:
        log(f"[verify] WARNING — {len(missing)} files missing:")
        for fn in missing[:20]:
            log(f"         {fn}")
        log(f"[verify] Staging preserved at {staging_dir.name}")
        log(f"[verify] Backup preserved at  {backup_dir.name}")
        log(f"[verify] Fix missing files then rename staging -> cache/ manually")
    else:
        total = len(layers) * 5
        log(f"[verify] All {total} expected files present — promoting staging -> cache/")
        staging_dir.rename(real_cache)
        log(f"[promote] Done.  Backup remains at {backup_dir.name}")

    print_summary(real_cache if not missing else staging_dir, n_layers)
    log(f"\n=== Rebuild finished: {model}/{dataset} ===")


def parse_args():
    p = argparse.ArgumentParser(description="Clean rebuild of one (model, dataset) cache.")
    p.add_argument("--model",   choices=list(MODEL_MAP), required=True)
    p.add_argument("--dataset", choices=["sciq", "triviaqa", "math"], required=True)
    p.add_argument("--gpu_id",  type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rebuild(args.model, args.dataset, args.gpu_id)