#!/usr/bin/env bash
# Component 3: render. Seconds, and refits nothing -- the atomic tables are
# already per-pair, so any grouping is a runtime argument.
#
#   ./run_report.sh --cohort vlm
#   ./run_report.sh --cohort vlm --split-by dataset
#   ./run_report.sh --pairs qwen25vl_advqa,gemma3_12b_advqa
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/root/miniconda3/envs/semantic_uncertainty/bin/python}
MODE=--run
ARGS=()
for a in "$@"; do
  [[ $a == --plan ]] && MODE=--plan || ARGS+=("$a")
done
"$PY" cli_report/10_report.py $MODE "${ARGS[@]}"
