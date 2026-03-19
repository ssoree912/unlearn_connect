#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export RATIO="${RATIO:-20}"
export RUN_NAME="${RUN_NAME:-gap20_clean_20260318}"
export KEEP_GRID_CSV="${KEEP_GRID_CSV:-0.60,0.65,0.70,0.75}"
export LR_GRID_CSV="${LR_GRID_CSV:-0.01,0.011,0.012,0.013,0.015,0.017}"
export UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
export UNLEARN_DECAY="${UNLEARN_DECAY:-91,136}"

bash "${SCRIPT_DIR}/run_gap20_clean.sh"
