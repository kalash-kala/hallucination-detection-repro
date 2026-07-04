#!/bin/bash
# =============================================================================
# prep_new_pair.sh — STAGE 00: onboard a new (model, dataset) pair into the
# repro package by building the frozen 70/30 splits it needs before Tier 1
# GPU extraction (extract_cache.py etc.) can run against it.
#
#   00a  sample_balanced_by_category.py   sciq/triviaqa -> balanced n=1400
#                                          math          -> copied as sampled_all_
#   00b  generate_distractors_hf.py       adds candidate_list (Llama-3.3-70B, HF)
#   00c  make_splits.py                   70/30 split -> recovery-gaps-data/data/
#                                          ranking_experiment_{model}_{dataset}/splits/
#
# FILES_TO_PROCESS in 00a/00b are edited by hand per onboarding batch (same
# convention as llm_judge_verdict_hf.py / split_by_concentration.py upstream).
# This script assumes those lists already point at the pair(s) you want.
#
# Usage:
#   cd scripts/dataset_prep
#   ./prep_new_pair.sh <model_tag> <dataset> <n_sample>
#   e.g. ./prep_new_pair.sh qwen3_14b sciq 1400
# =============================================================================
set -euo pipefail

MODEL="${1:?usage: prep_new_pair.sh <model_tag> <dataset> <n_sample>}"
DATASET="${2:?usage: prep_new_pair.sh <model_tag> <dataset> <n_sample>}"
N_SAMPLE="${3:-1400}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/uncertainty_runs"
PY="${REPRO_PY:-$HOME/miniconda3/envs/semantic_uncertainty/bin/python3}"

log() { echo -e "\n========== [$(date '+%F %T')] $* =========="; }

log "STAGE 00a — balanced sampling / math copy (edit FILES_TO_PROCESS by hand first)"
$PY "$SCRIPT_DIR/sample_balanced_by_category.py" --n_sample "$N_SAMPLE"

log "STAGE 00b — distractor generation (edit FILES_TO_PROCESS by hand first)"
$PY "$SCRIPT_DIR/generate_distractors_hf.py" --cuda_device 0,1 --batch_size 8

# Resolve the *_with_distractors.csv this (model, dataset) pair produced.
if [ "$DATASET" = "math" ]; then
  STEM="sampled_all_uncertainty_run_${MODEL}_answerable_math"
else
  STEM="sampled_${N_SAMPLE}_uncertainty_run_${MODEL}_${DATASET}"
fi
WITH_DISTRACTORS=$(find "$DATA_DIR" -maxdepth 1 -name "${STEM}*_with_distractors.csv" | head -1)
[ -n "$WITH_DISTRACTORS" ] || { echo "ERROR: no *_with_distractors.csv matching ${STEM}* in $DATA_DIR"; exit 1; }

log "STAGE 00c — 70/30 split -> recovery-gaps-data/data/ranking_experiment_${MODEL}_${DATASET}"
$PY "$SCRIPT_DIR/make_splits.py" \
    --input_csv "$WITH_DISTRACTORS" \
    --model "$MODEL" \
    --dataset "$DATASET"

log "DONE — ${MODEL}/${DATASET} splits ready. Next: register ${MODEL} in the Tier 1/2 \
MODEL_MAP/PAIRS lists (extract_cache.py, greedy_sidecar.py, extract_attention.py, \
extract_value_norms.py, train_lapeigvals.py, build_features.py, run_baselines.py, \
per_category_analysis.py) before running reproduce.sh extract."