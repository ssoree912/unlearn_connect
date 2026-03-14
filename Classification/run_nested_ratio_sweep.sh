#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
RUNS_DIR="${RUNS_DIR:-runs}"
SUMMARY_DIR="${SUMMARY_DIR:-${RUNS_DIR}/summary}"
BASE_DIR="${BASE_DIR:-${RUNS_DIR}/baseline}"
BASE_CKPT="${BASE_CKPT:-${BASE_DIR}/0checkpoint.pth.tar}"
FORGET_SEED="${FORGET_SEED:-1}"
RATIOS_CSV="${RATIOS_CSV:-10,20,30,40,50}"
CKPT_EPOCHS="${CKPT_EPOCHS:-0,1,3,5,10}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-182}"
RETRAIN_LR="${RETRAIN_LR:-0.1}"
UNLEARN_SEED_A="${UNLEARN_SEED_A:-11}"
UNLEARN_SEED_B="${UNLEARN_SEED_B:-22}"
TRAIN_BASELINE="${TRAIN_BASELINE:-0}"
RUN_RETRAIN="${RUN_RETRAIN:-1}"
RUN_SALUN="${RUN_SALUN:-1}"
RUN_INTERPOLATION="${RUN_INTERPOLATION:-1}"
RUN_FT="${RUN_FT:-0}"
RUN_GA="${RUN_GA:-0}"
RUN_AGGREGATION="${RUN_AGGREGATION:-1}"

declare -A SALUN_MASK_CENTER=(
  [10]=0.5
  [20]=0.6
  [30]=0.7
  [40]=0.8
  [50]=0.8
)

declare -A SALUN_LR_CENTER=(
  [10]=0.013
  [20]=0.008
  [30]=0.005
  [40]=0.003
  [50]=0.001
)

declare -A FT_LR_CENTER=(
  [10]=0.01
  [20]=0.003
  [30]=0.002
  [40]=0.002
  [50]=0.001
)

declare -A GA_LR_CENTER=(
  [10]=3e-5
  [20]=1e-5
  [30]=3e-6
  [40]=3e-6
  [50]=1e-6
)

declare -A SALUN_EPOCHS=(
  [10]=10
  [20]=10
  [30]=10
  [40]=10
  [50]=10
)

declare -A FT_EPOCHS=(
  [10]=10
  [20]=10
  [30]=10
  [40]=10
  [50]=10
)

declare -A GA_EPOCHS=(
  [10]=5
  [20]=5
  [30]=5
  [40]=5
  [50]=5
)

require_ratio_config() {
  local ratio="$1"
  if [[ -z "${SALUN_MASK_CENTER[$ratio]+x}" || -z "${SALUN_LR_CENTER[$ratio]+x}" ]]; then
    echo "Missing SalUn configuration for ratio ${ratio}" >&2
    exit 1
  fi
  if [[ -z "${FT_LR_CENTER[$ratio]+x}" || -z "${GA_LR_CENTER[$ratio]+x}" ]]; then
    echo "Missing FT/GA configuration for ratio ${ratio}" >&2
    exit 1
  fi
  if [[ -z "${SALUN_EPOCHS[$ratio]+x}" || -z "${FT_EPOCHS[$ratio]+x}" || -z "${GA_EPOCHS[$ratio]+x}" ]]; then
    echo "Missing epoch configuration for ratio ${ratio}" >&2
    exit 1
  fi
}

mkdir -p "${RUNS_DIR}"

if [[ "${TRAIN_BASELINE}" == "1" || ! -f "${BASE_CKPT}" ]]; then
  python main_train.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --lr 0.1 \
    --epochs 182 \
    --save_dir "${BASE_DIR}" \
    --forget_seed "${FORGET_SEED}"
fi

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Baseline checkpoint not found: ${BASE_CKPT}" >&2
  exit 1
fi

python make_nested_forget_indices.py \
  --arch "${ARCH}" \
  --dataset "${DATASET}" \
  --forget_seed "${FORGET_SEED}" \
  --percentages "${RATIOS_CSV}" \
  --output_root "${RUNS_DIR}" \
  --permutation_output_path "${RUNS_DIR}/forget_permutation.npy"

IFS=',' read -ra RATIOS <<< "${RATIOS_CSV}"
for RATIO in "${RATIOS[@]}"; do
  require_ratio_config "${RATIO}"

  RATIO_DIR="${RUNS_DIR}/${RATIO}"
  FORGET_IDX="${RATIO_DIR}/forget_indices.npy"
  MASK_DIR="${RATIO_DIR}/mask"
  RETRAIN_DIR="${RATIO_DIR}/retrain"
  FT_DIR="${RATIO_DIR}/ft"
  GA_DIR="${RATIO_DIR}/ga"
  SALUN_A_DIR="${RATIO_DIR}/salun_A"
  SALUN_B_DIR="${RATIO_DIR}/salun_B"
  INTERP_DIR="${RATIO_DIR}/interpolation"

  MASK_THRESHOLD="${SALUN_MASK_CENTER[$RATIO]}"
  SALUN_LR="${SALUN_LR_CENTER[$RATIO]}"
  FT_LR="${FT_LR_CENTER[$RATIO]}"
  GA_LR="${GA_LR_CENTER[$RATIO]}"
  SALUN_EPOCH="${SALUN_EPOCHS[$RATIO]}"
  FT_EPOCH="${FT_EPOCHS[$RATIO]}"
  GA_EPOCH="${GA_EPOCHS[$RATIO]}"
  MASK_PATH="${MASK_DIR}/with_${MASK_THRESHOLD}.pt"

  mkdir -p "${RATIO_DIR}" "${MASK_DIR}" "${RETRAIN_DIR}" "${FT_DIR}" "${GA_DIR}" "${SALUN_A_DIR}" "${SALUN_B_DIR}" "${INTERP_DIR}"

  echo "=== Ratio ${RATIO}% ==="
  echo "SalUn: mask=${MASK_THRESHOLD}, lr=${SALUN_LR}, epochs=${SALUN_EPOCH}"
  echo "FT: lr=${FT_LR}, epochs=${FT_EPOCH}"
  echo "GA: lr=${GA_LR}, epochs=${GA_EPOCH}"

  if [[ "${RUN_RETRAIN}" == "1" ]]; then
    python main_forget.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --save_dir "${RETRAIN_DIR}" \
      --model_path "${BASE_CKPT}" \
      --unlearn retrain \
      --unlearn_epochs "${RETRAIN_EPOCHS}" \
      --unlearn_lr "${RETRAIN_LR}" \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --unlearn_seed "${FORGET_SEED}"

    python evaluate_checkpoints.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --run_dir "${RETRAIN_DIR}" \
      --unlearn retrain \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --include_final_checkpoint \
      --label "retrain_${RATIO}"
  fi

  if [[ "${RUN_SALUN}" == "1" ]]; then
    python generate_mask.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --save_dir "${MASK_DIR}" \
      --model_path "${BASE_CKPT}" \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --unlearn_seed "${FORGET_SEED}" \
      --unlearn_epochs 1

    python main_random.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --unlearn RL \
      --unlearn_epochs "${SALUN_EPOCH}" \
      --unlearn_lr "${SALUN_LR}" \
      --model_path "${BASE_CKPT}" \
      --save_dir "${SALUN_A_DIR}" \
      --mask_path "${MASK_PATH}" \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --unlearn_seed "${UNLEARN_SEED_A}" \
      --checkpoint_epochs "${CKPT_EPOCHS}"

    python main_random.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --unlearn RL \
      --unlearn_epochs "${SALUN_EPOCH}" \
      --unlearn_lr "${SALUN_LR}" \
      --model_path "${BASE_CKPT}" \
      --save_dir "${SALUN_B_DIR}" \
      --mask_path "${MASK_PATH}" \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --unlearn_seed "${UNLEARN_SEED_B}" \
      --checkpoint_epochs "${CKPT_EPOCHS}"

    python evaluate_checkpoints.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --run_dir "${SALUN_A_DIR}" \
      --unlearn RL \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --include_final_checkpoint \
      --label "salun_A_${RATIO}"

    python evaluate_checkpoints.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --run_dir "${SALUN_B_DIR}" \
      --unlearn RL \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --include_final_checkpoint \
      --label "salun_B_${RATIO}"
  fi

  if [[ "${RUN_INTERPOLATION}" == "1" ]]; then
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
  fi

  if [[ "${RUN_FT}" == "1" ]]; then
    python main_forget.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --save_dir "${FT_DIR}" \
      --model_path "${BASE_CKPT}" \
      --unlearn FT \
      --unlearn_epochs "${FT_EPOCH}" \
      --unlearn_lr "${FT_LR}" \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --unlearn_seed "${FORGET_SEED}"

    python evaluate_checkpoints.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --run_dir "${FT_DIR}" \
      --unlearn FT \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --include_final_checkpoint \
      --label "ft_${RATIO}"
  fi

  if [[ "${RUN_GA}" == "1" ]]; then
    python main_forget.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --save_dir "${GA_DIR}" \
      --model_path "${BASE_CKPT}" \
      --unlearn GA \
      --unlearn_epochs "${GA_EPOCH}" \
      --unlearn_lr "${GA_LR}" \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --unlearn_seed "${FORGET_SEED}"

    python evaluate_checkpoints.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --run_dir "${GA_DIR}" \
      --unlearn GA \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --include_final_checkpoint \
      --label "ga_${RATIO}"
  fi
done

if [[ "${RUN_AGGREGATION}" == "1" ]]; then
  python aggregate_ratio_summaries.py \
    --runs_dir "${RUNS_DIR}" \
    --output_dir "${SUMMARY_DIR}"
fi
