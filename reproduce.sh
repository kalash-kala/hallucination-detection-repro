#!/usr/bin/env bash
# ============================================================================
# reproduce.sh — end-to-end reproduction of the Part 9 / Part 11 results
#
#   ./reproduce.sh extract    GPU stage: build every derived artifact (run once;
#                             every step is idempotent — reruns skip finished work)
#   ./reproduce.sh snne       GPU stage: SNNE scoring (separate conda env)
#   ./reproduce.sh prep       CPU stage: baseline CV configs + hs_lr classifiers
#   ./reproduce.sh analyze    CPU stage: Part 9 + Part 11 tables
#   ./reproduce.sh all        everything in order
#
# Prerequisites: the two conda envs in environment/*.yml, and (for `extract`)
# one GPU with ~32 GB VRAM (gemma-3-12b bf16 weights ~24 GB; the smaller models
# fit on 24 GB) plus HF access to the four models. Alternatively, restore the
# precomputed artifacts tarball (see REPRODUCTION_GUIDE.md) and skip straight
# to `prep`.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${REPRO_PY:-$HOME/miniconda3/envs/semantic_uncertainty/bin/python}"
PY_SNNE="${REPRO_PY_SNNE:-$HOME/miniconda3/envs/snne/bin/python}"
GPU="${REPRO_GPU:-0}"
MODELS=(llama mistral qwen gemma)
DATASETS=(sciq triviaqa math)
mkdir -p logs

log() { echo "[$(date +%H:%M:%S)] $*"; }

stage_extract() {
  # 01a — hidden-state cache: one teacher-forced pass per (sample, candidate),
  #        all layers at once -> hidden_layer{L}.npz / index / scores_layer{L}.jsonl
  log "STAGE 01a — hidden-state caches (12 pairs)"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/ranking/extract_cache.py --all --gpu_id 0

  # 01b — greedy sidecars: id -> greedy string (+ per-layer s_ext_L); test & train
  log "STAGE 01b — greedy sidecars"
  for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/ranking/greedy_sidecar.py         --model "$m" --dataset "$d" --gpu_id 0
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/ranking/compute_train_sidecars.py --model "$m" --dataset "$d" --gpu_id 0
  done; done

  # 01c — recover greedy hidden states missing from the candidate-pool cache
  #        (correct samples' greedy strings) -> greedy_late_hidden_{split}.npz
  log "STAGE 01c — recover greedy hidden states"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/ranking/recover_greedy_hidden.py --all --gpu_id 0

  # 02 — attention diagonals for the LapEigvals family (eager attention)
  log "STAGE 02 — attention diagonals (LapEigvals family)"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/lapeigvals_baseline/extract_attention.py --all --gpu_id 0

  # 03 — per-head value norms for the sinkhole (sink×‖V‖) features
  log "STAGE 03 — value norms (sinkhole)"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/sinkhole/extract_value_norms.py --all --gpu_id 0
}

stage_snne() {
  # 04 — SNNE per-example scores (DeBERTa entailment; SEPARATE conda env).
  #      Generations ship with the package (results/snne_baseline/generations*).
  log "STAGE 04 — SNNE scores (test + train)"
  CUDA_VISIBLE_DEVICES=$GPU $PY_SNNE scripts/snne_baseline/dump_snne_scores.py
  CUDA_VISIBLE_DEVICES=$GPU $PY_SNNE scripts/snne_baseline/dump_train_scores.py
  log "STAGE 04b — SNNE per-category table + best-method selection"
  $PY_SNNE scripts/snne_baseline/snne_per_category.py
  $PY_SNNE scripts/snne_baseline/compare.py
}

stage_prep() {
  # 05 — LapEigvals-family CV (top_k / PCA per pair) -> all_pairs_metrics.csv
  log "STAGE 05 — LapEigvals baseline CV configs"
  $PY scripts/lapeigvals_baseline/train_lapeigvals.py

  # 06 — hs_lr: peak layers + bucketed classifiers (layer_stats.pkl per variant)
  log "STAGE 06 — hs_lr peak layers + classifiers"
  $PY scripts/classifier/compute_peak_layers.py
  $PY scripts/classifier/train_classifier.py

  # 07 — per-example scores for the spectral baselines (+SNNE column)
  log "STAGE 07 — per_category_analysis (per_example_scores.csv)"
  $PY scripts/per_category_analysis.py --snne
}

stage_analyze() {
  # 08 — Part 9: band-stratified pairwise/pooled correctness AUROC, all methods
  log "STAGE 08 — Part 9 (per_category_pairwise_auroc.csv)"
  $PY scripts/per_category_pairwise_auroc.py

  # 09 — Part 11: band-specialist LRs (dedicated HI / LO classifiers)
  log "STAGE 09 — Part 11 (per_category_band_specialist_auroc.csv)"
  $PY scripts/per_category_band_specialist_auroc.py

  log "DONE — outputs in results/per_category_analysis/"
}

case "${1:-all}" in
  extract) stage_extract ;;
  snne)    stage_snne ;;
  prep)    stage_prep ;;
  analyze) stage_analyze ;;
  all)     stage_extract; stage_snne; stage_prep; stage_analyze ;;
  *) echo "usage: $0 {extract|snne|prep|analyze|all}"; exit 1 ;;
esac