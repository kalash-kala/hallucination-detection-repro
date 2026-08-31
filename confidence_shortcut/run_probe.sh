#!/usr/bin/env bash
# Component 2: arms, C, and every experiment, for one cohort or pair list.
#
# Every stage checkpoints per pair, so a kill resumes from the last completed
# unit. NJOBS is the real core count -- each worker is pinned to one BLAS
# thread, so it is not a multiplier on it.
#
#   ./run_probe.sh --cohort vlm
#   ./run_probe.sh --pairs qwen25vl_advqa --families hs_wide
#   NJOBS=8 ./run_probe.sh --cohort vlm --plan
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/root/miniconda3/envs/semantic_uncertainty/bin/python}
NJOBS=${NJOBS:-$(nproc)}
MODE=--run
export JOBLIB_TEMP_FOLDER=${JOBLIB_TEMP_FOLDER:-/dev/shm}

ARGS=()
for a in "$@"; do
  [[ $a == --plan ]] && MODE=--plan || ARGS+=("$a")
done

run () { echo; echo "### $1"; shift; "$PY" "$@" $MODE "${ARGS[@]}"; }

run "01 arms"          cli_probe/01_build_arms.py
run "03 select C"      cli_probe/03_select_c.py     --n-jobs "$NJOBS"
run "04 transfer grid" cli_probe/04_transfer_grid.py
run "05 alpha rotation" cli_probe/05_alpha_rotation.py --n-jobs "$NJOBS"
run "06 confounds"     cli_probe/06_confounds.py    --n-jobs "$NJOBS"
echo; echo "### done -- render with ./run_report.sh"
