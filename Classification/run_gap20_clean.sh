#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1

RATIO="${RATIO:-20}"
if [[ "${RATIO}" != "20" ]]; then
  echo "This script is currently configured for the 20% clean gap rerun only." >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-gap20_clean_20260318}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-experiments/${RUN_NAME}}"
RATIO_ROOT="${EXPERIMENT_ROOT}/ratio${RATIO}"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/run.log"

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
GPU="${GPU:-0}"
FORGET_SEED="${FORGET_SEED:-1}"
TUNE_SEED="${TUNE_SEED:-7}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-182}"
RETRAIN_LR="${RETRAIN_LR:-0.1}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
UNLEARN_DECAY="${UNLEARN_DECAY:-91,136}"
BASE_CKPT="${BASE_CKPT:-runs/baseline/0checkpoint.pth.tar}"
FORGET_IDX="${FORGET_IDX:-runs/20/forget_indices.npy}"
KEEP_GRID_CSV="${KEEP_GRID_CSV:-0.40,0.45,0.50,0.55,0.60}"
LR_GRID_CSV="${LR_GRID_CSV:-0.0005,0.001,0.002,0.005,0.01,0.02,0.05}"

RETRAIN_DIR="${RATIO_ROOT}/retrain"
MASK_DIR="${RATIO_ROOT}/mask"
TUNE_ROOT="${RATIO_ROOT}/tuning"
BEST_ENV="${RATIO_ROOT}/best_salun.env"
BEST_CSV="${RATIO_ROOT}/best_salun_leaderboard.csv"
SUMMARY_TXT="${RATIO_ROOT}/summary.txt"

mkdir -p "${LOG_DIR}" "${RATIO_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_if_missing() {
  local marker="$1"
  shift
  if [[ -e "${marker}" ]]; then
    echo "[$(timestamp)] [skip] ${marker} exists"
    return 0
  fi
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file is missing: ${path}" >&2
    exit 1
  fi
}

run_retrain() {
  mkdir -p "${RETRAIN_DIR}"
  python -u main_forget.py     --arch "${ARCH}"     --dataset "${DATASET}"     --gpu "${GPU}"     --save_dir "${RETRAIN_DIR}"     --model_path "${BASE_CKPT}"     --unlearn retrain     --unlearn_epochs "${RETRAIN_EPOCHS}"     --unlearn_lr "${RETRAIN_LR}"     --forget_seed "${FORGET_SEED}"     --forget_index_path "${FORGET_IDX}"

  python -u evaluate_checkpoints.py     --arch "${ARCH}"     --dataset "${DATASET}"     --gpu "${GPU}"     --run_dir "${RETRAIN_DIR}"     --unlearn retrain     --forget_seed "${FORGET_SEED}"     --forget_index_path "${FORGET_IDX}"     --include_final_checkpoint     --label "retrain_${RATIO}"
}

run_mask_generation() {
  local mask_keep_grid_csv="${1:-${KEEP_GRID_CSV}}"
  mkdir -p "${MASK_DIR}"
  python -u generate_mask.py     --arch "${ARCH}"     --dataset "${DATASET}"     --gpu "${GPU}"     --save_dir "${MASK_DIR}"     --model_path "${BASE_CKPT}"     --forget_seed "${FORGET_SEED}"     --forget_index_path "${FORGET_IDX}"     --unlearn_seed "${FORGET_SEED}"     --unlearn_epochs 1     --mask_keep_ratios "${mask_keep_grid_csv}"
}

run_trial() {
  local keep="$1"
  local lr="$2"
  local trial_dir="${TUNE_ROOT}/keep_${keep}/lr_${lr}/epochs_${UNLEARN_EPOCHS}"

  mkdir -p "${trial_dir}"
  python -u main_random.py     --arch "${ARCH}"     --dataset "${DATASET}"     --gpu "${GPU}"     --unlearn RL     --unlearn_epochs "${UNLEARN_EPOCHS}"     --unlearn_lr "${lr}"     --decreasing_lr "${UNLEARN_DECAY}"     --model_path "${BASE_CKPT}"     --save_dir "${trial_dir}"     --mask_path "${MASK_DIR}/with_${keep}.pt"     --forget_seed "${FORGET_SEED}"     --forget_index_path "${FORGET_IDX}"     --unlearn_seed "${TUNE_SEED}"

  python -u evaluate_checkpoints.py     --arch "${ARCH}"     --dataset "${DATASET}"     --gpu "${GPU}"     --run_dir "${trial_dir}"     --unlearn RL     --forget_seed "${FORGET_SEED}"     --forget_index_path "${FORGET_IDX}"     --include_final_checkpoint     --label "r${RATIO}_k${keep}_lr${lr}"
}

echo "[$(timestamp)] Starting clean 20% gap rerun"
echo "[$(timestamp)] Experiment root: ${EXPERIMENT_ROOT}"
echo "[$(timestamp)] Base checkpoint: ${BASE_CKPT}"
echo "[$(timestamp)] Forget index path: ${FORGET_IDX}"
echo "[$(timestamp)] GPU: ${GPU}"
echo "[$(timestamp)] Keep grid: ${KEEP_GRID_CSV}"
echo "[$(timestamp)] LR grid: ${LR_GRID_CSV}"
echo "[$(timestamp)] SalUn decay: ${UNLEARN_DECAY}"

require_file "${BASE_CKPT}"
require_file "${FORGET_IDX}"

run_if_missing "${RETRAIN_DIR}/endpoint_metrics.csv" run_retrain

IFS=',' read -r -a KEEP_GRID <<< "${KEEP_GRID_CSV}"
IFS=',' read -r -a LR_GRID <<< "${LR_GRID_CSV}"

MISSING_KEEP_GRID=()
for keep in "${KEEP_GRID[@]}"; do
  if [[ ! -f "${MASK_DIR}/with_${keep}.pt" ]]; then
    MISSING_KEEP_GRID+=("${keep}")
  fi
done

if (( ${#MISSING_KEEP_GRID[@]} > 0 )); then
  MISSING_KEEP_GRID_CSV="$(IFS=,; echo "${MISSING_KEEP_GRID[*]}")"
  echo "[$(timestamp)] Generating missing masks for keep ratios: ${MISSING_KEEP_GRID_CSV}"
  run_mask_generation "${MISSING_KEEP_GRID_CSV}"
else
  echo "[$(timestamp)] [skip] all requested masks already exist"
fi

for keep in "${KEEP_GRID[@]}"; do
  require_file "${MASK_DIR}/with_${keep}.pt"
  for lr in "${LR_GRID[@]}"; do
    trial_dir="${TUNE_ROOT}/keep_${keep}/lr_${lr}/epochs_${UNLEARN_EPOCHS}"
    run_if_missing "${trial_dir}/endpoint_metrics.csv" run_trial "${keep}" "${lr}"
  done
done

python -u select_best_salun_legacy.py   --tune_root "${TUNE_ROOT}"   --retrain_metrics_path "${RETRAIN_DIR}/endpoint_metrics.csv"   --output_env "${BEST_ENV}"   --output_csv "${BEST_CSV}"   --reference_mode retrain_oracle   --reference_name "retrain_oracle_ratio_${RATIO}"   --score_cols ua,acc_retain,acc_test,mia

{
  echo "Generated at $(timestamp)"
  echo
  echo "[retrain]"
  sed -n '1,2p' "${RETRAIN_DIR}/endpoint_metrics.csv"
  echo
  echo "[top_candidates]"
  sed -n '1,6p' "${BEST_CSV}"
  echo
  echo "[best_env]"
  sed -n '1,20p' "${BEST_ENV}"
} > "${SUMMARY_TXT}"

echo "[$(timestamp)] Completed clean 20% gap rerun"
echo "[$(timestamp)] Best leaderboard: ${BEST_CSV}"
echo "[$(timestamp)] Best config env: ${BEST_ENV}"
echo "[$(timestamp)] Summary: ${SUMMARY_TXT}"
