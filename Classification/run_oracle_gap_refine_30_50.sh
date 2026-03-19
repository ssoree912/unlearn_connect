#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-oracle_gap_refine_30_50_v1}"
export RATIOS_CSV="${RATIOS_CSV:-30,40,50}"
export SELECTOR_MODE="${SELECTOR_MODE:-retrain_oracle}"
export TUNING_SKIP_MIA="${TUNING_SKIP_MIA:-0}"
export SELECTOR_SCORE_COLS="${SELECTOR_SCORE_COLS:-ua,acc_retain,acc_test,mia}"
export SELECTOR_SCORE_WEIGHTS="${SELECTOR_SCORE_WEIGHTS:-ua=1.0,acc_retain=1.0,acc_test=1.0,mia=1.0}"
export SALUN_GRID_PRESET="${SALUN_GRID_PRESET:-oracle_gap_refine_30_50}"
export SALUN_DISABLE_EXPLICIT_CANDIDATES="${SALUN_DISABLE_EXPLICIT_CANDIDATES:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

bash "${SCRIPT_DIR}/run_nested_ratio_sweep.sh"
