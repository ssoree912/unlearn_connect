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
TUNE_DIR="${TUNE_DIR:-${RUNS_DIR}/_tuning}"
BASE_DIR="${BASE_DIR:-runs/baseline}"
BASE_CKPT="${BASE_CKPT:-${BASE_DIR}/0checkpoint.pth.tar}"
FORGET_SEED="${FORGET_SEED:-1}"
RATIOS_CSV="${RATIOS_CSV:-10,20,30,40,50}"
CKPT_EPOCHS="${CKPT_EPOCHS:-0,1,3,5,10}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-182}"
RETRAIN_LR="${RETRAIN_LR:-0.1}"
UNLEARN_SEED_TUNE="${UNLEARN_SEED_TUNE:-7}"
UNLEARN_SEED_A="${UNLEARN_SEED_A:-11}"
UNLEARN_SEED_B="${UNLEARN_SEED_B:-22}"
TUNE_CKPT_EPOCHS="${TUNE_CKPT_EPOCHS:-6,8,10,12,15}"
TRAIN_BASELINE="${TRAIN_BASELINE:-0}"
RUN_RETRAIN="${RUN_RETRAIN:-1}"
RUN_TUNING="${RUN_TUNING:-1}"
RUN_SALUN="${RUN_SALUN:-0}"
RUN_INTERPOLATION="${RUN_INTERPOLATION:-0}"
RUN_FT="${RUN_FT:-0}"
RUN_GA="${RUN_GA:-0}"
RUN_AGGREGATION="${RUN_AGGREGATION:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SELECTOR_MODE="${SELECTOR_MODE:-retrain_oracle}"
if [[ -z "${TUNING_SKIP_MIA+x}" ]]; then
  if [[ "${SELECTOR_MODE}" == "retrain_oracle" ]]; then
    TUNING_SKIP_MIA="0"
  else
    TUNING_SKIP_MIA="0"
  fi
fi
if [[ -z "${SELECTOR_SCORE_COLS+x}" ]]; then
  SELECTOR_SCORE_COLS="ua,acc_retain,acc_test,mia"
fi
if [[ -z "${SELECTOR_SCORE_WEIGHTS+x}" ]]; then
  if [[ "${SELECTOR_MODE}" == "retrain_oracle" ]]; then
    SELECTOR_SCORE_WEIGHTS="ua=2.5,mia=1.0,acc_test=0.7,acc_retain=0.7"
  else
    SELECTOR_SCORE_WEIGHTS=""
  fi
fi
EXPERIMENT_CONFIG_PATH="${EXPERIMENT_CONFIG_PATH:-${RUNS_DIR}/experiment_config.env}"

declare -A SALUN_KEEP_GRID=(
  [10]="0.2 0.3 0.4 0.5 0.6 0.7"
  [20]="0.45 0.48 0.50 0.55 0.60"
  [30]="0.33 0.34 0.35 0.36"
  [40]="0.340 0.342 0.345 0.348"
  [50]="0.380 0.382 0.385 0.390 0.395"
)

declare -A SALUN_LR_GRID=(
  [10]="0.005 0.008 0.013 0.02 0.03"
  [20]="0.0175 0.018 0.019"
  [30]="0.0165 0.017 0.0175"
  [40]="0.0170 0.0172"
  [50]="0.0190 0.0192 0.0193 0.0195"
)

declare -A SALUN_EPOCH_GRID=(
  [10]="10"
  [20]="15"
  [30]="10"
  [40]="10"
  [50]="10"
)

declare -A SALUN_CANDIDATE_SPECS=(
  [20]="0.50:0.0175:15 0.50:0.018:15 0.55:0.0175:15 0.55:0.018:15 0.45:0.018:15 0.48:0.018:15 0.60:0.0175:15 0.50:0.019:15"
  [30]="0.33:0.0165:10 0.34:0.0165:10 0.34:0.017:10 0.35:0.0165:10 0.35:0.017:10 0.34:0.0175:10 0.36:0.017:10 0.36:0.0175:10"
  [40]="0.340:0.0172:10 0.342:0.0170:10 0.342:0.0172:10 0.345:0.0170:10 0.345:0.0172:10 0.348:0.0172:10"
  [50]="0.380:0.0192:10 0.382:0.0190:10 0.382:0.0192:10 0.385:0.0190:10 0.385:0.0192:10 0.380:0.0195:10 0.390:0.0190:10 0.395:0.0193:10"
)

declare -A FT_LR_CENTER=(
  [10]=0.01
  [20]=0.005
  [30]=0.003
  [40]=0.002
  [50]=0.001
)

declare -A GA_LR_CENTER=(
  [10]=3e-5
  [20]=1e-5
  [30]=3e-6
  [40]=1e-6
  [50]=1e-6
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

declare -A PAPER_TARGET_UA=(
  [10]=2.85
  [20]=3.73
  [30]=6.22
  [40]=6.86
  [50]=7.75
)

declare -A PAPER_TARGET_ACC_RETAIN=(
  [10]=99.62
  [20]=98.61
  [30]=95.91
  [40]=95.01
  [50]=94.28
)

declare -A PAPER_TARGET_ACC_TEST=(
  [10]=93.93
  [20]=92.75
  [30]=90.72
  [40]=89.76
  [50]=89.29
)

declare -A PAPER_TARGET_MIA=(
  [10]=14.39
  [20]=13.18
  [30]=14.11
  [40]=15.15
  [50]=16.99
)

declare -A SELECTOR_MIN_ACC_RETAIN=(
  [20]=98.3
)

declare -A SELECTOR_MIN_ACC_TEST=(
  [20]=92.8
)

declare -A TUNE_CKPT_EPOCHS_BY_RATIO=(
  [20]="10,12,15"
  [30]="6,8,10"
  [40]="4,5,6,7,8,10"
  [50]="4,5,6,7,8,10"
)

require_ratio_config() {
  local ratio="$1"
  [[ -n "${SALUN_KEEP_GRID[$ratio]+x}" ]] || { echo "Missing SALUN_KEEP_GRID for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${SALUN_LR_GRID[$ratio]+x}" ]] || { echo "Missing SALUN_LR_GRID for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${SALUN_EPOCH_GRID[$ratio]+x}" ]] || { echo "Missing SALUN_EPOCH_GRID for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${FT_LR_CENTER[$ratio]+x}" ]] || { echo "Missing FT_LR_CENTER for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${GA_LR_CENTER[$ratio]+x}" ]] || { echo "Missing GA_LR_CENTER for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${FT_EPOCHS[$ratio]+x}" ]] || { echo "Missing FT_EPOCHS for ratio ${ratio}" >&2; exit 1; }
  [[ -n "${GA_EPOCHS[$ratio]+x}" ]] || { echo "Missing GA_EPOCHS for ratio ${ratio}" >&2; exit 1; }
  if [[ "${SELECTOR_MODE}" == "paper_target" ]]; then
    [[ -n "${PAPER_TARGET_UA[$ratio]+x}" ]] || { echo "Missing PAPER_TARGET_UA for ratio ${ratio}" >&2; exit 1; }
    [[ -n "${PAPER_TARGET_ACC_RETAIN[$ratio]+x}" ]] || { echo "Missing PAPER_TARGET_ACC_RETAIN for ratio ${ratio}" >&2; exit 1; }
    [[ -n "${PAPER_TARGET_ACC_TEST[$ratio]+x}" ]] || { echo "Missing PAPER_TARGET_ACC_TEST for ratio ${ratio}" >&2; exit 1; }
    [[ -n "${PAPER_TARGET_MIA[$ratio]+x}" ]] || { echo "Missing PAPER_TARGET_MIA for ratio ${ratio}" >&2; exit 1; }
  fi
}

ratio_tune_ckpt_spec() {
  local ratio="$1"
  if [[ -n "${TUNE_CKPT_EPOCHS_BY_RATIO[$ratio]+x}" ]]; then
    echo "${TUNE_CKPT_EPOCHS_BY_RATIO[$ratio]}"
  else
    echo "${TUNE_CKPT_EPOCHS}"
  fi
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

decreasing_lr_for_unlearn_epochs() {
  local epochs="$1"
  case "${epochs}" in
    10)
      echo "5,8"
      ;;
    12)
      echo "6,10"
      ;;
    15)
      echo "8,12"
      ;;
    *)
      local first=$(( epochs * 60 / 100 ))
      local second=$(( epochs * 80 / 100 ))
      if (( first < 1 )); then
        first=1
      fi
      if (( second <= first )); then
        second=$(( first + 1 ))
      fi
      if (( second >= epochs )); then
        second=$(( epochs - 1 ))
      fi
      echo "${first},${second}"
      ;;
  esac
}

filter_epoch_spec() {
  local spec="$1"
  local max_epoch="$2"
  local filtered=()
  local token
  IFS=',' read -ra TOKENS <<< "${spec}"
  for token in "${TOKENS[@]}"; do
    token="${token// /}"
    if [[ -z "${token}" ]]; then
      continue
    fi
    if (( token <= max_epoch )); then
      filtered+=("${token}")
    fi
  done

  if [[ "${#filtered[@]}" -eq 0 ]]; then
    return 0
  fi

  local joined=""
  local item
  for item in "${filtered[@]}"; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${item}"
  done
  echo "${joined}"
}

if [[ "${TUNING_SKIP_MIA}" == "1" && "${SELECTOR_SCORE_COLS}" == *"mia"* ]]; then
  echo "SELECTOR_SCORE_COLS including mia requires TUNING_SKIP_MIA=0" >&2
  exit 1
fi

write_experiment_config() {
  mkdir -p "$(dirname "${EXPERIMENT_CONFIG_PATH}")"
  {
    echo "ARCH=${ARCH}"
    echo "DATASET=${DATASET}"
    echo "RUNS_DIR=${RUNS_DIR}"
    echo "SUMMARY_DIR=${SUMMARY_DIR}"
    echo "TUNE_DIR=${TUNE_DIR}"
    echo "BASE_CKPT=${BASE_CKPT}"
    echo "FORGET_SEED=${FORGET_SEED}"
    echo "RATIOS_CSV=${RATIOS_CSV}"
    echo "CKPT_EPOCHS=${CKPT_EPOCHS}"
    echo "RETRAIN_EPOCHS=${RETRAIN_EPOCHS}"
    echo "RETRAIN_LR=${RETRAIN_LR}"
    echo "UNLEARN_SEED_TUNE=${UNLEARN_SEED_TUNE}"
    echo "UNLEARN_SEED_A=${UNLEARN_SEED_A}"
    echo "UNLEARN_SEED_B=${UNLEARN_SEED_B}"
    echo "TUNE_CKPT_EPOCHS=${TUNE_CKPT_EPOCHS}"
    echo "SELECTOR_MODE=${SELECTOR_MODE}"
    echo "TUNING_SKIP_MIA=${TUNING_SKIP_MIA}"
    echo "SELECTOR_SCORE_COLS=${SELECTOR_SCORE_COLS}"
    echo "SELECTOR_SCORE_WEIGHTS=${SELECTOR_SCORE_WEIGHTS}"
    for ratio in 10 20 30 40 50; do
      if [[ -n "${SALUN_KEEP_GRID[$ratio]+x}" ]]; then
        echo "SALUN_KEEP_GRID_${ratio}=${SALUN_KEEP_GRID[$ratio]}"
      fi
      if [[ -n "${SALUN_LR_GRID[$ratio]+x}" ]]; then
        echo "SALUN_LR_GRID_${ratio}=${SALUN_LR_GRID[$ratio]}"
      fi
      if [[ -n "${SALUN_EPOCH_GRID[$ratio]+x}" ]]; then
        echo "SALUN_EPOCH_GRID_${ratio}=${SALUN_EPOCH_GRID[$ratio]}"
      fi
      if [[ -n "${SALUN_CANDIDATE_SPECS[$ratio]+x}" ]]; then
        echo "SALUN_CANDIDATE_SPECS_${ratio}=${SALUN_CANDIDATE_SPECS[$ratio]}"
      fi
      if [[ -n "${TUNE_CKPT_EPOCHS_BY_RATIO[$ratio]+x}" ]]; then
        echo "TUNE_CKPT_EPOCHS_BY_RATIO_${ratio}=${TUNE_CKPT_EPOCHS_BY_RATIO[$ratio]}"
      fi
      if [[ -n "${PAPER_TARGET_UA[$ratio]+x}" ]]; then
        echo "PAPER_TARGET_UA_${ratio}=${PAPER_TARGET_UA[$ratio]}"
        echo "PAPER_TARGET_ACC_RETAIN_${ratio}=${PAPER_TARGET_ACC_RETAIN[$ratio]}"
        echo "PAPER_TARGET_ACC_TEST_${ratio}=${PAPER_TARGET_ACC_TEST[$ratio]}"
        echo "PAPER_TARGET_MIA_${ratio}=${PAPER_TARGET_MIA[$ratio]}"
      fi
      if [[ -n "${SELECTOR_MIN_ACC_RETAIN[$ratio]+x}" ]]; then
        echo "SELECTOR_MIN_ACC_RETAIN_${ratio}=${SELECTOR_MIN_ACC_RETAIN[$ratio]}"
      fi
      if [[ -n "${SELECTOR_MIN_ACC_TEST[$ratio]+x}" ]]; then
        echo "SELECTOR_MIN_ACC_TEST_${ratio}=${SELECTOR_MIN_ACC_TEST[$ratio]}"
      fi
    done
  } > "${EXPERIMENT_CONFIG_PATH}"
}

ensure_retrain_metrics() {
  local retrain_csv="$1"
  if [[ ! -f "${retrain_csv}" ]]; then
    echo "Retrain metrics missing: ${retrain_csv}" >&2
    echo "Run with RUN_RETRAIN=1 first, or place endpoint_metrics.csv in the retrain directory." >&2
    exit 1
  fi
}

run_retrain_and_eval() {
  local ratio="$1"
  local retrain_dir="$2"
  local forget_idx="$3"

  python main_forget.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --save_dir "${retrain_dir}" \
    --model_path "${BASE_CKPT}" \
    --unlearn retrain \
    --unlearn_epochs "${RETRAIN_EPOCHS}" \
    --unlearn_lr "${RETRAIN_LR}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --unlearn_seed "${FORGET_SEED}"

  python evaluate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_dir "${retrain_dir}" \
    --unlearn retrain \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --include_final_checkpoint \
    --label "retrain_${ratio}"
}

generate_masks() {
  local mask_dir="$1"
  local forget_idx="$2"
  local keep_ratios_csv="$3"

  python generate_mask.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --save_dir "${mask_dir}" \
    --model_path "${BASE_CKPT}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --unlearn_seed "${FORGET_SEED}" \
    --unlearn_epochs 1 \
    --mask_keep_ratios "${keep_ratios_csv}"
}

mask_keep_ratios_csv() {
  local ratio="$1"
  local joined=""
  local keep_ratio
  for keep_ratio in ${SALUN_KEEP_GRID[$ratio]}; do
    if [[ -n "${joined}" ]]; then
      joined+=","
    fi
    joined+="${keep_ratio}"
  done
  echo "${joined}"
}

all_required_masks_exist() {
  local mask_dir="$1"
  local ratio="$2"
  local keep_ratio

  for keep_ratio in ${SALUN_KEEP_GRID[$ratio]}; do
    if [[ ! -f "${mask_dir}/with_${keep_ratio}.pt" ]]; then
      return 1
    fi
  done

  return 0
}

run_tuning_trial_and_eval() {
  local ratio="$1"
  local keep_ratio="$2"
  local lr="$3"
  local salun_epoch="$4"
  local trial_dir="$5"
  local mask_path="$6"
  local forget_idx="$7"
  local tuning_mia_args=()
  local tuning_decreasing_lr
  local tuning_checkpoint_epochs

  tuning_decreasing_lr="$(decreasing_lr_for_unlearn_epochs "${salun_epoch}")"
  tuning_checkpoint_epochs="$(filter_epoch_spec "$(ratio_tune_ckpt_spec "${ratio}")" "${salun_epoch}")"

  if [[ "${TUNING_SKIP_MIA}" == "1" ]]; then
    tuning_mia_args+=(--skip_mia)
  fi

  python main_random.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --unlearn RL \
    --unlearn_epochs "${salun_epoch}" \
    --unlearn_lr "${lr}" \
    --model_path "${BASE_CKPT}" \
    --save_dir "${trial_dir}" \
    --mask_path "${mask_path}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --unlearn_seed "${UNLEARN_SEED_TUNE}" \
    --decreasing_lr "${tuning_decreasing_lr}" \
    --checkpoint_epochs "${tuning_checkpoint_epochs}" \
    "${tuning_mia_args[@]}"

  python evaluate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_dir "${trial_dir}" \
    --unlearn RL \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --include_final_checkpoint \
    --label "tune_r${ratio}_k${keep_ratio}_lr${lr}" \
    "${tuning_mia_args[@]}"
}

select_best_config() {
  local ratio="$1"
  local tune_root="$2"
  local retrain_csv="$3"
  local output_env="$4"
  local output_csv="$5"

  local selector_args=(
    --tune_root "${tune_root}"
    --output_env "${output_env}"
    --output_csv "${output_csv}"
    --score_cols "${SELECTOR_SCORE_COLS}"
    --score_weights "${SELECTOR_SCORE_WEIGHTS}"
    --reference_mode "${SELECTOR_MODE}"
  )

  if [[ "${SELECTOR_MODE}" == "paper_target" ]]; then
    selector_args+=(
      --reference_name "paper_salun_ratio_${ratio}"
      --target_ua "${PAPER_TARGET_UA[$ratio]}"
      --target_acc_retain "${PAPER_TARGET_ACC_RETAIN[$ratio]}"
      --target_acc_test "${PAPER_TARGET_ACC_TEST[$ratio]}"
      --target_mia "${PAPER_TARGET_MIA[$ratio]}"
    )
  else
    selector_args+=(
      --reference_name "retrain_oracle_ratio_${ratio}"
      --retrain_metrics_path "${retrain_csv}"
    )
  fi

  if [[ -n "${SELECTOR_MIN_ACC_RETAIN[$ratio]+x}" ]]; then
    selector_args+=(--min_acc_retain "${SELECTOR_MIN_ACC_RETAIN[$ratio]}")
  fi
  if [[ -n "${SELECTOR_MIN_ACC_TEST[$ratio]+x}" ]]; then
    selector_args+=(--min_acc_test "${SELECTOR_MIN_ACC_TEST[$ratio]}")
  fi

  python select_best_salun.py "${selector_args[@]}"
}

collect_tuning_rows() {
  local ratio="$1"
  local tune_root="$2"
  local retrain_csv="$3"
  local output_csv="$4"

  local collector_args=(
    --tune_root "${tune_root}"
    --output_csv "${output_csv}"
    --score_cols "${SELECTOR_SCORE_COLS}"
    --score_weights "${SELECTOR_SCORE_WEIGHTS}"
    --reference_mode "${SELECTOR_MODE}"
  )

  if [[ "${SELECTOR_MODE}" == "paper_target" ]]; then
    collector_args+=(
      --reference_name "paper_salun_ratio_${ratio}"
      --target_ua "${PAPER_TARGET_UA[$ratio]}"
      --target_acc_retain "${PAPER_TARGET_ACC_RETAIN[$ratio]}"
      --target_acc_test "${PAPER_TARGET_ACC_TEST[$ratio]}"
      --target_mia "${PAPER_TARGET_MIA[$ratio]}"
    )
  else
    collector_args+=(
      --reference_name "retrain_oracle_ratio_${ratio}"
      --retrain_metrics_path "${retrain_csv}"
    )
  fi

  if [[ -n "${SELECTOR_MIN_ACC_RETAIN[$ratio]+x}" ]]; then
    collector_args+=(--min_acc_retain "${SELECTOR_MIN_ACC_RETAIN[$ratio]}")
  fi
  if [[ -n "${SELECTOR_MIN_ACC_TEST[$ratio]+x}" ]]; then
    collector_args+=(--min_acc_test "${SELECTOR_MIN_ACC_TEST[$ratio]}")
  fi

  python collect_tuning_trials.py "${collector_args[@]}"
}

run_salun_seed_and_eval() {
  local ratio="$1"
  local run_dir="$2"
  local seed="$3"
  local salun_epoch="$4"
  local salun_lr="$5"
  local mask_path="$6"
  local forget_idx="$7"
  local run_label="$8"
  local run_decreasing_lr
  local run_checkpoint_epochs

  run_decreasing_lr="$(decreasing_lr_for_unlearn_epochs "${salun_epoch}")"
  run_checkpoint_epochs="$(filter_epoch_spec "${CKPT_EPOCHS}" "${salun_epoch}")"

  python main_random.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --unlearn RL \
    --unlearn_epochs "${salun_epoch}" \
    --unlearn_lr "${salun_lr}" \
    --model_path "${BASE_CKPT}" \
    --save_dir "${run_dir}" \
    --mask_path "${mask_path}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --unlearn_seed "${seed}" \
    --decreasing_lr "${run_decreasing_lr}" \
    --checkpoint_epochs "${run_checkpoint_epochs}"

  python evaluate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_dir "${run_dir}" \
    --unlearn RL \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --include_final_checkpoint \
    --label "${run_label}"
}

run_interpolation() {
  local ratio="$1"
  local salun_a_dir="$2"
  local salun_b_dir="$3"
  local interp_dir="$4"
  local retrain_csv="$5"
  local forget_idx="$6"

  python interpolate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_a_dir "${salun_a_dir}" \
    --run_b_dir "${salun_b_dir}" \
    --curve_epochs "${CKPT_EPOCHS}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --output_dir "${interp_dir}" \
    --retrain_metrics_path "${retrain_csv}" \
    --label_a "salun_A_${ratio}" \
    --label_b "salun_B_${ratio}"
}

run_ft_and_eval() {
  local ratio="$1"
  local ft_dir="$2"
  local ft_epoch="$3"
  local ft_lr="$4"
  local forget_idx="$5"

  python main_forget.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --save_dir "${ft_dir}" \
    --model_path "${BASE_CKPT}" \
    --unlearn FT \
    --unlearn_epochs "${ft_epoch}" \
    --unlearn_lr "${ft_lr}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --unlearn_seed "${FORGET_SEED}"

  python evaluate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_dir "${ft_dir}" \
    --unlearn FT \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --include_final_checkpoint \
    --label "ft_${ratio}"
}

run_ga_and_eval() {
  local ratio="$1"
  local ga_dir="$2"
  local ga_epoch="$3"
  local ga_lr="$4"
  local forget_idx="$5"

  python main_forget.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --save_dir "${ga_dir}" \
    --model_path "${BASE_CKPT}" \
    --unlearn GA \
    --unlearn_epochs "${ga_epoch}" \
    --unlearn_lr "${ga_lr}" \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --unlearn_seed "${FORGET_SEED}"

  python evaluate_checkpoints.py \
    --arch "${ARCH}" \
    --dataset "${DATASET}" \
    --run_dir "${ga_dir}" \
    --unlearn GA \
    --forget_seed "${FORGET_SEED}" \
    --forget_index_path "${forget_idx}" \
    --include_final_checkpoint \
    --label "ga_${ratio}"
}

mkdir -p "${RUNS_DIR}" "${SUMMARY_DIR}" "${TUNE_DIR}"
write_experiment_config

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
  ALL_TUNING_CSV="${RATIO_DIR}/all_tuning_trials.csv"
  TUNE_RATIO_DIR="${TUNE_DIR}/${RATIO}"

  FT_EPOCH="${FT_EPOCHS[$RATIO]}"
  GA_EPOCH="${GA_EPOCHS[$RATIO]}"
  FT_LR="${FT_LR_CENTER[$RATIO]}"
  GA_LR="${GA_LR_CENTER[$RATIO]}"

  mkdir -p \
    "${RATIO_DIR}" \
    "${MASK_DIR}" \
    "${RETRAIN_DIR}" \
    "${FT_DIR}" \
    "${GA_DIR}" \
    "${SALUN_A_DIR}" \
    "${SALUN_B_DIR}" \
    "${INTERP_DIR}" \
    "${TUNE_RATIO_DIR}"

  echo "=== Ratio ${RATIO}% ==="
  echo "Selector mode: ${SELECTOR_MODE}"
  echo "Selector score weights: ${SELECTOR_SCORE_WEIGHTS}"
  echo "SalUn keep grid: ${SALUN_KEEP_GRID[$RATIO]}"
  echo "SalUn lr grid:   ${SALUN_LR_GRID[$RATIO]}"
  echo "SalUn epoch grid: ${SALUN_EPOCH_GRID[$RATIO]}"
  if [[ -n "${SALUN_CANDIDATE_SPECS[$RATIO]+x}" ]]; then
    echo "SalUn candidate specs: ${SALUN_CANDIDATE_SPECS[$RATIO]}"
  fi
  if [[ "${SELECTOR_MODE}" == "paper_target" ]]; then
    echo "Paper target: ua=${PAPER_TARGET_UA[$RATIO]}, acc_retain=${PAPER_TARGET_ACC_RETAIN[$RATIO]}, acc_test=${PAPER_TARGET_ACC_TEST[$RATIO]}, mia=${PAPER_TARGET_MIA[$RATIO]}"
  fi

  if [[ "${RUN_RETRAIN}" == "1" ]]; then
    maybe_run "${RETRAIN_DIR}/endpoint_metrics.csv" \
      run_retrain_and_eval "${RATIO}" "${RETRAIN_DIR}" "${FORGET_IDX}"
  fi
  if [[ "${SELECTOR_MODE}" == "retrain_oracle" || "${RUN_INTERPOLATION}" == "1" ]]; then
    ensure_retrain_metrics "${RETRAIN_DIR}/endpoint_metrics.csv"
  fi

  if all_required_masks_exist "${MASK_DIR}" "${RATIO}"; then
    echo "[skip] all required masks already exist under ${MASK_DIR}"
  else
    generate_masks "${MASK_DIR}" "${FORGET_IDX}" "$(mask_keep_ratios_csv "${RATIO}")"
  fi

  if [[ "${RUN_TUNING}" == "1" ]]; then
    if [[ -n "${SALUN_CANDIDATE_SPECS[$RATIO]+x}" ]]; then
      for CANDIDATE_SPEC in ${SALUN_CANDIDATE_SPECS[$RATIO]}; do
        IFS=':' read -r KEEP_RATIO LR SALUN_EPOCH <<< "${CANDIDATE_SPEC}"
        MASK_PATH="${MASK_DIR}/with_${KEEP_RATIO}.pt"
        [[ -f "${MASK_PATH}" ]] || { echo "Missing mask file: ${MASK_PATH}" >&2; exit 1; }
        TRIAL_DIR="${TUNE_RATIO_DIR}/keep_${KEEP_RATIO}/lr_${LR}/epochs_${SALUN_EPOCH}"
        mkdir -p "${TRIAL_DIR}"
        maybe_run "${TRIAL_DIR}/endpoint_metrics.csv" \
          run_tuning_trial_and_eval \
            "${RATIO}" \
            "${KEEP_RATIO}" \
            "${LR}" \
            "${SALUN_EPOCH}" \
            "${TRIAL_DIR}" \
            "${MASK_PATH}" \
            "${FORGET_IDX}"
      done
    else
      for KEEP_RATIO in ${SALUN_KEEP_GRID[$RATIO]}; do
        MASK_PATH="${MASK_DIR}/with_${KEEP_RATIO}.pt"
        [[ -f "${MASK_PATH}" ]] || { echo "Missing mask file: ${MASK_PATH}" >&2; exit 1; }
        for LR in ${SALUN_LR_GRID[$RATIO]}; do
          for SALUN_EPOCH in ${SALUN_EPOCH_GRID[$RATIO]}; do
            TRIAL_DIR="${TUNE_RATIO_DIR}/keep_${KEEP_RATIO}/lr_${LR}/epochs_${SALUN_EPOCH}"
            mkdir -p "${TRIAL_DIR}"
            maybe_run "${TRIAL_DIR}/endpoint_metrics.csv" \
              run_tuning_trial_and_eval \
                "${RATIO}" \
                "${KEEP_RATIO}" \
                "${LR}" \
                "${SALUN_EPOCH}" \
                "${TRIAL_DIR}" \
                "${MASK_PATH}" \
                "${FORGET_IDX}"
          done
        done
      done
    fi

    maybe_run "${BEST_ENV}" \
      select_best_config \
        "${RATIO}" \
        "${TUNE_RATIO_DIR}" \
        "${RETRAIN_DIR}/endpoint_metrics.csv" \
        "${BEST_ENV}" \
        "${BEST_CSV}"

    collect_tuning_rows \
      "${RATIO}" \
      "${TUNE_RATIO_DIR}" \
      "${RETRAIN_DIR}/endpoint_metrics.csv" \
      "${ALL_TUNING_CSV}"
  fi

  [[ -f "${BEST_ENV}" ]] || { echo "Missing best config env file: ${BEST_ENV}" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "${BEST_ENV}"
  echo "[best] ratio=${RATIO} keep=${BEST_KEEP_RATIO} lr=${BEST_LR} epoch=${BEST_EPOCH} score=${BEST_SCORE}"

  BEST_MASK_PATH="${MASK_DIR}/with_${BEST_KEEP_RATIO}.pt"
  [[ -f "${BEST_MASK_PATH}" ]] || { echo "Missing best mask file: ${BEST_MASK_PATH}" >&2; exit 1; }

  if [[ "${RUN_SALUN}" == "1" ]]; then
    maybe_run "${SALUN_A_DIR}/endpoint_metrics.csv" \
      run_salun_seed_and_eval \
        "${RATIO}" \
        "${SALUN_A_DIR}" \
        "${UNLEARN_SEED_A}" \
        "${BEST_EPOCH}" \
        "${BEST_LR}" \
        "${BEST_MASK_PATH}" \
        "${FORGET_IDX}" \
        "salun_A_${RATIO}"

    maybe_run "${SALUN_B_DIR}/endpoint_metrics.csv" \
      run_salun_seed_and_eval \
        "${RATIO}" \
        "${SALUN_B_DIR}" \
        "${UNLEARN_SEED_B}" \
        "${BEST_EPOCH}" \
        "${BEST_LR}" \
        "${BEST_MASK_PATH}" \
        "${FORGET_IDX}" \
        "salun_B_${RATIO}"
  fi

  if [[ "${RUN_INTERPOLATION}" == "1" ]]; then
    maybe_run "${INTERP_DIR}/barrier_summary.csv" \
      run_interpolation \
        "${RATIO}" \
        "${SALUN_A_DIR}" \
        "${SALUN_B_DIR}" \
        "${INTERP_DIR}" \
        "${RETRAIN_DIR}/endpoint_metrics.csv" \
        "${FORGET_IDX}"
  fi

  if [[ "${RUN_FT}" == "1" ]]; then
    maybe_run "${FT_DIR}/endpoint_metrics.csv" \
      run_ft_and_eval "${RATIO}" "${FT_DIR}" "${FT_EPOCH}" "${FT_LR}" "${FORGET_IDX}"
  fi

  if [[ "${RUN_GA}" == "1" ]]; then
    maybe_run "${GA_DIR}/endpoint_metrics.csv" \
      run_ga_and_eval "${RATIO}" "${GA_DIR}" "${GA_EPOCH}" "${GA_LR}" "${FORGET_IDX}"
  fi
done

if [[ "${RUN_AGGREGATION}" == "1" ]]; then
  python aggregate_ratio_summaries.py \
    --runs_dir "${RUNS_DIR}" \
    --output_dir "${SUMMARY_DIR}"
fi
