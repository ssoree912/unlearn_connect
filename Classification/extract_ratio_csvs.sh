#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
if [[ -z "${RUNS_DIR:-}" ]]; then
  if [[ -n "${EXPERIMENT_ROOT}" ]]; then
    RUNS_DIR="${EXPERIMENT_ROOT}/runs"
  elif [[ -n "${EXPERIMENT_NAME}" ]]; then
    RUNS_DIR="experiments/${EXPERIMENT_NAME}/runs"
  else
    RUNS_DIR="runs"
  fi
fi
SUMMARY_DIR="${SUMMARY_DIR:-${RUNS_DIR}/summary}"
FORGET_SEED="${FORGET_SEED:-1}"
RATIOS_CSV="${RATIOS_CSV:-10,20,30,40,50}"
CKPT_EPOCHS="${CKPT_EPOCHS:-0,1,3,5,10}"
RUN_ENDPOINTS="${RUN_ENDPOINTS:-1}"
RUN_INTERPOLATION="${RUN_INTERPOLATION:-1}"
RUN_AGGREGATION="${RUN_AGGREGATION:-1}"

has_epoch_checkpoints() {
  local checkpoint_dir="$1"
  if [[ ! -d "${checkpoint_dir}" ]]; then
    return 1
  fi

  if compgen -G "${checkpoint_dir}/epoch_*.pth.tar" > /dev/null; then
    return 0
  fi

  return 1
}

has_any_checkpoint() {
  local checkpoint_dir="$1"
  local final_checkpoint="$2"

  if has_epoch_checkpoints "${checkpoint_dir}"; then
    return 0
  fi

  if [[ -f "${final_checkpoint}" ]]; then
    return 0
  fi

  return 1
}

evaluate_run() {
  local run_dir="$1"
  local unlearn_name="$2"
  local label="$3"
  local forget_idx="$4"
  local checkpoint_dir="${run_dir}/checkpoints"
  local final_checkpoint="${run_dir}/${unlearn_name}checkpoint.pth.tar"

  if ! has_any_checkpoint "${checkpoint_dir}" "${final_checkpoint}"; then
    echo "Skipping endpoint CSV for ${run_dir}: no checkpoints found"
    return 0
  fi

  python evaluate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_dir "${run_dir}" \
    --unlearn "${unlearn_name}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --include_final_checkpoint \
    --label "${label}"
}

IFS=',' read -ra RATIOS <<< "${RATIOS_CSV}"
for RATIO in "${RATIOS[@]}"; do
  RATIO_DIR="${RUNS_DIR}/${RATIO}"
  FORGET_IDX="${RATIO_DIR}/forget_indices.npy"
  RETRAIN_DIR="${RATIO_DIR}/retrain"
  FT_DIR="${RATIO_DIR}/ft"
  GA_DIR="${RATIO_DIR}/ga"
  SALUN_A_DIR="${RATIO_DIR}/salun_A"
  SALUN_B_DIR="${RATIO_DIR}/salun_B"
  INTERP_DIR="${RATIO_DIR}/interpolation"

  if [[ ! -f "${FORGET_IDX}" ]]; then
    echo "Skipping ratio ${RATIO}: missing forget index file ${FORGET_IDX}"
    continue
  fi

  echo "=== Extracting CSVs for ratio ${RATIO}% ==="

  if [[ "${RUN_ENDPOINTS}" == "1" ]]; then
    evaluate_run "${RETRAIN_DIR}" "retrain" "retrain_${RATIO}" "${FORGET_IDX}"
    evaluate_run "${SALUN_A_DIR}" "RL" "salun_A_${RATIO}" "${FORGET_IDX}"
    evaluate_run "${SALUN_B_DIR}" "RL" "salun_B_${RATIO}" "${FORGET_IDX}"
    evaluate_run "${FT_DIR}" "FT" "ft_${RATIO}" "${FORGET_IDX}"
    evaluate_run "${GA_DIR}" "GA" "ga_${RATIO}" "${FORGET_IDX}"
  fi

  if [[ "${RUN_INTERPOLATION}" == "1" ]]; then
    if has_epoch_checkpoints "${SALUN_A_DIR}/checkpoints" && has_epoch_checkpoints "${SALUN_B_DIR}/checkpoints"; then
      INTERP_ARGS=(
        --arch "${ARCH}"
        --dataset "${DATASET}"
        --run_a_dir "${SALUN_A_DIR}"
        --run_b_dir "${SALUN_B_DIR}"
        --curve_epochs "${CKPT_EPOCHS}"
        --forget_seed "${FORGET_SEED}"
        --forget_index_path "${FORGET_IDX}"
        --output_dir "${INTERP_DIR}"
        --label_a "salun_A_${RATIO}"
        --label_b "salun_B_${RATIO}"
      )

      if [[ -f "${RETRAIN_DIR}/endpoint_metrics.csv" ]]; then
        INTERP_ARGS+=(--retrain_metrics_path "${RETRAIN_DIR}/endpoint_metrics.csv")
      else
        echo "Warning: retrain metrics missing for ratio ${RATIO}; retrain gap summary will be skipped"
      fi

      python interpolate_checkpoints.py "${INTERP_ARGS[@]}"
    else
      echo "Skipping interpolation CSVs for ratio ${RATIO}: missing SalUn A/B epoch checkpoints"
    fi
  fi
done

if [[ "${RUN_AGGREGATION}" == "1" ]]; then
  python aggregate_ratio_summaries.py \
    --runs_dir "${RUNS_DIR}" \
    --output_dir "${SUMMARY_DIR}"
fi
