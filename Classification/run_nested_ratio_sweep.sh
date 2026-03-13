#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
RUNS_DIR="${RUNS_DIR:-runs}"
BASE_DIR="${BASE_DIR:-${RUNS_DIR}/baseline}"
BASE_CKPT="${BASE_CKPT:-${BASE_DIR}/0checkpoint.pth.tar}"
FORGET_SEED="${FORGET_SEED:-1}"
RATIOS_CSV="${RATIOS_CSV:-10,20,30,40,50}"
CKPT_EPOCHS="${CKPT_EPOCHS:-0,1,3,5,10}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.5}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-10}"
UNLEARN_LR="${UNLEARN_LR:-0.013}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-182}"
RETRAIN_LR="${RETRAIN_LR:-0.1}"
UNLEARN_SEED_A="${UNLEARN_SEED_A:-11}"
UNLEARN_SEED_B="${UNLEARN_SEED_B:-22}"
TRAIN_BASELINE="${TRAIN_BASELINE:-0}"

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
  RATIO_DIR="${RUNS_DIR}/${RATIO}"
  FORGET_IDX="${RATIO_DIR}/forget_indices.npy"
  MASK_DIR="${RATIO_DIR}/mask"
  MASK_PATH="${MASK_DIR}/with_${MASK_THRESHOLD}.pt"
  RETRAIN_DIR="${RATIO_DIR}/retrain"
  FT_DIR="${RATIO_DIR}/ft"
  GA_DIR="${RATIO_DIR}/ga"
  SALUN_A_DIR="${RATIO_DIR}/salun_A"
  SALUN_B_DIR="${RATIO_DIR}/salun_B"
  INTERP_DIR="${RATIO_DIR}/interpolation"

  mkdir -p "${RATIO_DIR}"

  python generate_mask.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --save_dir "${MASK_DIR}" \
    --model_path "${BASE_CKPT}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${FORGET_IDX}" \
    --unlearn_seed "${FORGET_SEED}" \
    --unlearn_epochs 1

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

  python main_forget.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --save_dir "${FT_DIR}" \
    --model_path "${BASE_CKPT}" \
    --unlearn FT \
    --unlearn_epochs "${UNLEARN_EPOCHS}" \
    --unlearn_lr "${UNLEARN_LR}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${FORGET_IDX}" \
    --unlearn_seed "${FORGET_SEED}"

  python main_forget.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --save_dir "${GA_DIR}" \
    --model_path "${BASE_CKPT}" \
    --unlearn GA \
    --unlearn_epochs "${UNLEARN_EPOCHS}" \
    --unlearn_lr "${UNLEARN_LR}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${FORGET_IDX}" \
    --unlearn_seed "${FORGET_SEED}"

  python main_random.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --unlearn RL \
    --unlearn_epochs "${UNLEARN_EPOCHS}" \
    --unlearn_lr "${UNLEARN_LR}" \
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
    --unlearn_epochs "${UNLEARN_EPOCHS}" \
    --unlearn_lr "${UNLEARN_LR}" \
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
    --run_dir "${RETRAIN_DIR}" \
    --unlearn retrain \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${FORGET_IDX}" \
    --include_final_checkpoint \
    --label "retrain_${RATIO}"

  python evaluate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_dir "${FT_DIR}" \
    --unlearn FT \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${FORGET_IDX}" \
    --include_final_checkpoint \
    --label "ft_${RATIO}"

  python evaluate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_dir "${GA_DIR}" \
    --unlearn GA \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${FORGET_IDX}" \
    --include_final_checkpoint \
    --label "ga_${RATIO}"

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

  python interpolate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_a_dir "${SALUN_A_DIR}" \
    --run_b_dir "${SALUN_B_DIR}" \
    --curve_epochs "${CKPT_EPOCHS}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${FORGET_IDX}" \
    --output_dir "${INTERP_DIR}" \
    --retrain_metrics_path "${RETRAIN_DIR}/endpoint_metrics.csv" \
    --label_a "salun_A_${RATIO}" \
    --label_b "salun_B_${RATIO}"
done
