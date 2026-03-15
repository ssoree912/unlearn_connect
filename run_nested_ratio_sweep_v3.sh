#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
RUNS_DIR="${RUNS_DIR:-runs}"
SUMMARY_DIR="${SUMMARY_DIR:-${RUNS_DIR}/summary}"
TUNE_DIR="${TUNE_DIR:-${RUNS_DIR}/_tuning}"
BASE_DIR="${BASE_DIR:-${RUNS_DIR}/baseline}"
BASE_CKPT="${BASE_CKPT:-${BASE_DIR}/0checkpoint.pth.tar}"
FORGET_SEED="${FORGET_SEED:-1}"
RATIOS_CSV="${RATIOS_CSV:-10,20,30,40,50}"
CKPT_EPOCHS="${CKPT_EPOCHS:-0,1,3,5,10}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-182}"
RETRAIN_LR="${RETRAIN_LR:-0.1}"
UNLEARN_SEED_TUNE="${UNLEARN_SEED_TUNE:-7}"
UNLEARN_SEED_A="${UNLEARN_SEED_A:-11}"
UNLEARN_SEED_B="${UNLEARN_SEED_B:-22}"
TRAIN_BASELINE="${TRAIN_BASELINE:-0}"
RUN_RETRAIN="${RUN_RETRAIN:-1}"
RUN_TUNING="${RUN_TUNING:-1}"
RUN_SALUN="${RUN_SALUN:-1}"
RUN_INTERPOLATION="${RUN_INTERPOLATION:-1}"
RUN_FT="${RUN_FT:-0}"
RUN_GA="${RUN_GA:-0}"
RUN_AGGREGATION="${RUN_AGGREGATION:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# --- Ratio-aware SalUn search space ---
# NOTE:
#   generate_mask.py writes files named with_{x}.pt where x is the fraction of
#   coordinates allowed to update (keep ratio), not the paper's sparsity label.
#   Larger forgetting ratios usually prefer smaller keep ratios.
declare -A SALUN_KEEP_GRID=(
  [10]="0.3 0.5 0.7"
  [20]="0.3 0.4 0.5"
  [30]="0.2 0.3 0.4"
  [40]="0.1 0.2 0.3"
  [50]="0.1 0.2 0.3 0.4"
)

declare -A SALUN_LR_GRID=(
  [10]="0.005 0.013 0.02"
  [20]="0.003 0.008 0.013"
  [30]="0.002 0.005 0.008"
  [40]="0.001 0.003 0.005"
  [50]="0.0005 0.001 0.003"
)

declare -A SALUN_EPOCHS=(
  [10]=10 [20]=10 [30]=10 [40]=10 [50]=10
)

# Optional endpoint-only baselines.
declare -A FT_LR_CENTER=(
  [10]=0.01 [20]=0.005 [30]=0.003 [40]=0.002 [50]=0.001
)

declare -A GA_LR_CENTER=(
  [10]=3e-5 [20]=1e-5 [30]=3e-6 [40]=1e-6 [50]=1e-6
)

declare -A FT_EPOCHS=(
  [10]=10 [20]=10 [30]=10 [40]=10 [50]=10
)

declare -A GA_EPOCHS=(
  [10]=5 [20]=5 [30]=5 [40]=5 [50]=5
)

require_ratio_config() {
  local ratio="$1"
  [[ -n "${SALUN_KEEP_GRID[$ratio]+x}" ]] || { echo "Missing SALUN_KEEP_GRID for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${SALUN_LR_GRID[$ratio]+x}" ]] || { echo "Missing SALUN_LR_GRID for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${SALUN_EPOCHS[$ratio]+x}" ]] || { echo "Missing SALUN_EPOCHS for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${FT_LR_CENTER[$ratio]+x}" ]] || { echo "Missing FT_LR_CENTER for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${GA_LR_CENTER[$ratio]+x}" ]] || { echo "Missing GA_LR_CENTER for ratio ${ratio}" >&2; exit 1; }
}

maybe_run() {
  local marker="$1"
  shift
  if [[ "${SKIP_EXISTING}" == "1" && -e "${marker}" ]]; then
    echo "[skip] ${marker} exists"
    return 0
  fi
  "$@"
}

ensure_retrain_metrics() {
  local retrain_csv="$1"
  if [[ ! -f "${retrain_csv}" ]]; then
    echo "Retrain metrics missing: ${retrain_csv}" >&2
    echo "Run with RUN_RETRAIN=1 first, or place endpoint_metrics.csv in the retrain directory." >&2
    exit 1
  fi
}

mkdir -p "${RUNS_DIR}" "${SUMMARY_DIR}" "${TUNE_DIR}"

if [[ "${TRAIN_BASELINE}" == "1" || ! -f "${BASE_CKPT}" ]]; then
  python main_train.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --lr 0.1 \
    --epochs 182 \
    --save_dir "${BASE_DIR}" \
    --forget_seed "${FORGET_SEED}"
fi

[[ -f "${BASE_CKPT}" ]] || { echo "Baseline checkpoint not found: ${BASE_CKPT}" >&2; exit 1; }

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
  BEST_ENV="${RATIO_DIR}/best_salun.env"
  BEST_CSV="${RATIO_DIR}/best_salun_leaderboard.csv"
  TUNE_RATIO_DIR="${TUNE_DIR}/${RATIO}"

  SALUN_EPOCH="${SALUN_EPOCHS[$RATIO]}"
  FT_EPOCH="${FT_EPOCHS[$RATIO]}"
  GA_EPOCH="${GA_EPOCHS[$RATIO]}"
  FT_LR="${FT_LR_CENTER[$RATIO]}"
  GA_LR="${GA_LR_CENTER[$RATIO]}"

  mkdir -p "${RATIO_DIR}" "${MASK_DIR}" "${RETRAIN_DIR}" "${SALUN_A_DIR}" "${SALUN_B_DIR}" "${INTERP_DIR}" "${TUNE_RATIO_DIR}"

  echo "=== Ratio ${RATIO}% ==="
  echo "SalUn keep grid: ${SALUN_KEEP_GRID[$RATIO]}"
  echo "SalUn lr grid:   ${SALUN_LR_GRID[$RATIO]}"

  # 1) Retrain once per ratio.
  if [[ "${RUN_RETRAIN}" == "1" ]]; then
    maybe_run "${RETRAIN_DIR}/endpoint_metrics.csv" \
      bash -lc "python main_forget.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --save_dir '${RETRAIN_DIR}' \
        --model_path '${BASE_CKPT}' \
        --unlearn retrain \
        --unlearn_epochs '${RETRAIN_EPOCHS}' \
        --unlearn_lr '${RETRAIN_LR}' \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --unlearn_seed '${FORGET_SEED}' && \
      python evaluate_checkpoints.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --run_dir '${RETRAIN_DIR}' \
        --unlearn retrain \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --include_final_checkpoint \
        --label 'retrain_${RATIO}'"
  fi
  ensure_retrain_metrics "${RETRAIN_DIR}/endpoint_metrics.csv"

  # 2) Generate all mask files once per ratio.
  maybe_run "${MASK_DIR}/with_1.0.pt" \
    python generate_mask.py \
      --arch "${ARCH}" \
      --dataset "${DATASET}" \
      --save_dir "${MASK_DIR}" \
      --model_path "${BASE_CKPT}" \
      --forget_seed "${FORGET_SEED}" \
      --forget_index_path "${FORGET_IDX}" \
      --unlearn_seed "${FORGET_SEED}" \
      --unlearn_epochs 1

  # 3) Tune SalUn per ratio with one seed and final-only evaluation.
  if [[ "${RUN_TUNING}" == "1" ]]; then
    for KEEP_RATIO in ${SALUN_KEEP_GRID[$RATIO]}; do
      MASK_PATH="${MASK_DIR}/with_${KEEP_RATIO}.pt"
      [[ -f "${MASK_PATH}" ]] || { echo "Missing mask file: ${MASK_PATH}" >&2; exit 1; }
      for LR in ${SALUN_LR_GRID[$RATIO]}; do
        TRIAL_DIR="${TUNE_RATIO_DIR}/keep_${KEEP_RATIO}/lr_${LR}"
        mkdir -p "${TRIAL_DIR}"
        maybe_run "${TRIAL_DIR}/endpoint_metrics.csv" \
          bash -lc "python main_random.py \
            --arch '${ARCH}' \
            --dataset '${DATASET}' \
            --unlearn RL \
            --unlearn_epochs '${SALUN_EPOCH}' \
            --unlearn_lr '${LR}' \
            --model_path '${BASE_CKPT}' \
            --save_dir '${TRIAL_DIR}' \
            --mask_path '${MASK_PATH}' \
            --forget_seed '${FORGET_SEED}' \
            --forget_index_path '${FORGET_IDX}' \
            --unlearn_seed '${UNLEARN_SEED_TUNE}' && \
          python evaluate_checkpoints.py \
            --arch '${ARCH}' \
            --dataset '${DATASET}' \
            --run_dir '${TRIAL_DIR}' \
            --unlearn RL \
            --forget_seed '${FORGET_SEED}' \
            --forget_index_path '${FORGET_IDX}' \
            --include_final_checkpoint \
            --label 'tune_r${RATIO}_k${KEEP_RATIO}_lr${LR}'"
      done
    done

    python select_best_salun.py \
      --tune_root "${TUNE_RATIO_DIR}" \
      --retrain_metrics_path "${RETRAIN_DIR}/endpoint_metrics.csv" \
      --output_env "${BEST_ENV}" \
      --output_csv "${BEST_CSV}"
  fi

  [[ -f "${BEST_ENV}" ]] || { echo "Missing best config env file: ${BEST_ENV}" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "${BEST_ENV}"
  echo "[best] ratio=${RATIO} keep=${BEST_KEEP_RATIO} lr=${BEST_LR} score=${BEST_SCORE}"

  BEST_MASK_PATH="${MASK_DIR}/with_${BEST_KEEP_RATIO}.pt"
  [[ -f "${BEST_MASK_PATH}" ]] || { echo "Missing best mask file: ${BEST_MASK_PATH}" >&2; exit 1; }

  # 4) Run the actual two-seed LMC experiment with the selected SalUn hyperparameters.
  if [[ "${RUN_SALUN}" == "1" ]]; then
    maybe_run "${SALUN_A_DIR}/endpoint_metrics.csv" \
      bash -lc "python main_random.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --unlearn RL \
        --unlearn_epochs '${SALUN_EPOCH}' \
        --unlearn_lr '${BEST_LR}' \
        --model_path '${BASE_CKPT}' \
        --save_dir '${SALUN_A_DIR}' \
        --mask_path '${BEST_MASK_PATH}' \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --unlearn_seed '${UNLEARN_SEED_A}' \
        --checkpoint_epochs '${CKPT_EPOCHS}' && \
      python evaluate_checkpoints.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --run_dir '${SALUN_A_DIR}' \
        --unlearn RL \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --include_final_checkpoint \
        --label 'salun_A_${RATIO}'"

    maybe_run "${SALUN_B_DIR}/endpoint_metrics.csv" \
      bash -lc "python main_random.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --unlearn RL \
        --unlearn_epochs '${SALUN_EPOCH}' \
        --unlearn_lr '${BEST_LR}' \
        --model_path '${BASE_CKPT}' \
        --save_dir '${SALUN_B_DIR}' \
        --mask_path '${BEST_MASK_PATH}' \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --unlearn_seed '${UNLEARN_SEED_B}' \
        --checkpoint_epochs '${CKPT_EPOCHS}' && \
      python evaluate_checkpoints.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --run_dir '${SALUN_B_DIR}' \
        --unlearn RL \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --include_final_checkpoint \
        --label 'salun_B_${RATIO}'"
  fi

  # 5) Interpolation on the selected A/B runs.
  if [[ "${RUN_INTERPOLATION}" == "1" ]]; then
    maybe_run "${INTERP_DIR}/barrier_summary.csv" \
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
  fi

  # 6) Optional endpoint-only baselines using ratio-specific centers.
  if [[ "${RUN_FT}" == "1" ]]; then
    maybe_run "${FT_DIR}/endpoint_metrics.csv" \
      bash -lc "python main_forget.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --save_dir '${FT_DIR}' \
        --model_path '${BASE_CKPT}' \
        --unlearn FT \
        --unlearn_epochs '${FT_EPOCH}' \
        --unlearn_lr '${FT_LR}' \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --unlearn_seed '${FORGET_SEED}' && \
      python evaluate_checkpoints.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --run_dir '${FT_DIR}' \
        --unlearn FT \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --include_final_checkpoint \
        --label 'ft_${RATIO}'"
  fi

  if [[ "${RUN_GA}" == "1" ]]; then
    maybe_run "${GA_DIR}/endpoint_metrics.csv" \
      bash -lc "python main_forget.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --save_dir '${GA_DIR}' \
        --model_path '${BASE_CKPT}' \
        --unlearn GA \
        --unlearn_epochs '${GA_EPOCH}' \
        --unlearn_lr '${GA_LR}' \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --unlearn_seed '${FORGET_SEED}' && \
      python evaluate_checkpoints.py \
        --arch '${ARCH}' \
        --dataset '${DATASET}' \
        --run_dir '${GA_DIR}' \
        --unlearn GA \
        --forget_seed '${FORGET_SEED}' \
        --forget_index_path '${FORGET_IDX}' \
        --include_final_checkpoint \
        --label 'ga_${RATIO}'"
  fi
done

if [[ "${RUN_AGGREGATION}" == "1" ]]; then
  python aggregate_ratio_summaries.py \
    --runs_dir "${RUNS_DIR}" \
    --output_dir "${SUMMARY_DIR}"
fi
