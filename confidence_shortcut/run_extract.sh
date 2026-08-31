#!/usr/bin/env bash
# Component 1: GPU extraction for one pair. One pair at a time by design --
# each stage is resumable and phase 2 is the expensive one.
#
#   CUDA_VISIBLE_DEVICES=0 ./run_extract.sh --pairs qwen25vl_advqa
#   ./run_extract.sh --pairs qwen25vl_advqa --plan
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/root/miniconda3/envs/semantic_uncertainty/bin/python}
MODE=--run
ARGS=()
for a in "$@"; do
  [[ $a == --plan ]] && MODE=--plan || ARGS+=("$a")
done
for s in 20_rows 21_hidden 22_reduce 23_attention 24_verify; do
  echo; echo "### $s"
  "$PY" cli_extract/$s.py $MODE "${ARGS[@]}"
done
