#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
UNLEARN="${UNLEARN:-RL}"
EXP_ID="${EXP_ID:-conn_mode_v1}"
RATIO="${RATIO:-10}"
RUN_ROOT="${RUN_ROOT:-runs/${RATIO}/step_connectivity}"
BASE_CKPT="${BASE_CKPT:-runs/baseline/0checkpoint.pth.tar}"
FORGET_SEED="${FORGET_SEED:-1}"
FORGET_INDEX_PATH="${FORGET_INDEX_PATH:-runs/${RATIO}/forget_indices.npy}"
UNLEARN_SEED_A="${UNLEARN_SEED_A:-11}"
UNLEARN_SEED_B="${UNLEARN_SEED_B:-22}"
UNLEARN_SEED_C="${UNLEARN_SEED_C:-}"
DECREASING_LR="${DECREASING_LR:-5,8}"
SALUN_LR="${SALUN_LR:-}"
SALUN_EPOCHS="${SALUN_EPOCHS:-10}"
MASK_PATH="${MASK_PATH:-}"
BEST_ENV_PATH="${BEST_ENV_PATH:-}"
KEEP_RATIO="${KEEP_RATIO:-}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-0}"
ALPHA_GRID="${ALPHA_GRID:-0.0:0.1:1.0}"
BETA="${BETA:-0.3}"
DELTA="${DELTA:-0.0}"
REFINE_RADIUS="${REFINE_RADIUS:-0.1}"
REFINE_STEP="${REFINE_STEP:-0.02}"
SIMPLEX_WEIGHT_STEP="${SIMPLEX_WEIGHT_STEP:-0.1}"
CONNECTIVITY_MODE="${CONNECTIVITY_MODE:-linear}"
CONTROL_RUN_DIRS="${CONTROL_RUN_DIRS:-}"
SELECTION_MODE="${SELECTION_MODE:-dr_min_then_val}"
ABS_BALANCE_A="${ABS_BALANCE_A:-1.0}"
ABS_BALANCE_B="${ABS_BALANCE_B:-1.0}"
PAIR_ID="${PAIR_ID:-A${UNLEARN_SEED_A}_B${UNLEARN_SEED_B}}"
NOTES="${NOTES:-}"
ALLOW_SAME_UNLEARN_SEED="${ALLOW_SAME_UNLEARN_SEED:-0}"
GPU="${GPU:-0}"
AUTO_SUMMARY="${AUTO_SUMMARY:-0}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-plot_stepwise_ab_avg_merge_vs_retrain.py}"
SUMMARY_RUNS_ROOT="${SUMMARY_RUNS_ROOT:-}"
SUMMARY_RETRAIN_ROOT="${SUMMARY_RETRAIN_ROOT:-}"
SUMMARY_DIR="${SUMMARY_DIR:-}"
SUMMARY_OUTPUT_PREFIX="${SUMMARY_OUTPUT_PREFIX:-}"
SUMMARY_RATIO_RUN_SPECS="${SUMMARY_RATIO_RUN_SPECS:-}"
SUMMARY_PREPEND_RUNS_ROOT="${SUMMARY_PREPEND_RUNS_ROOT:-}"
SUMMARY_PREPEND_RETRAIN_ROOT="${SUMMARY_PREPEND_RETRAIN_ROOT:-}"
SUMMARY_PREPEND_RATIO_RUN_SPECS="${SUMMARY_PREPEND_RATIO_RUN_SPECS:-}"
SUMMARY_EPOCH_OFFSET="${SUMMARY_EPOCH_OFFSET:-0.0}"
SUMMARY_PREPEND_EPOCH_OFFSET="${SUMMARY_PREPEND_EPOCH_OFFSET:-0.0}"
SUMMARY_METRICS="${SUMMARY_METRICS:-ua,dr_acc,df_acc,val_acc,test_acc,dr_loss,df_loss,val_loss,test_loss,mia}"
SUMMARY_HTML_METRICS="${SUMMARY_HTML_METRICS:-ua,dr_acc,df_acc,val_acc,test_acc,mia}"
SUMMARY_PLOT_SERIES="${SUMMARY_PLOT_SERIES:-main,alt,merge_05,merge_best,retrain,merge_center}"

if [[ -n "${BEST_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${BEST_ENV_PATH}"
fi

if [[ -z "${SALUN_LR}" && -n "${BEST_LR:-}" ]]; then
  SALUN_LR="${BEST_LR}"
fi
if [[ -z "${KEEP_RATIO}" && -n "${BEST_KEEP_RATIO:-}" ]]; then
  KEEP_RATIO="${BEST_KEEP_RATIO}"
fi
if [[ -z "${MASK_PATH}" && -n "${KEEP_RATIO}" ]]; then
  MASK_PATH="runs/${RATIO}/mask_fixed/with_${KEEP_RATIO}.pt"
fi
if [[ -n "${BEST_EPOCH:-}" ]]; then
  SALUN_EPOCHS="${BEST_EPOCH}"
fi

if [[ -z "${SALUN_LR}" ]]; then
  echo "SALUN_LR is required (or provide BEST_ENV_PATH with BEST_LR)." >&2
  exit 1
fi
if [[ -z "${MASK_PATH}" ]]; then
  echo "MASK_PATH is required (or provide KEEP_RATIO/BEST_KEEP_RATIO)." >&2
  exit 1
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${FORGET_INDEX_PATH}" ]]; then
  echo "Forget index file not found: ${FORGET_INDEX_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MASK_PATH}" ]]; then
  echo "Mask path not found: ${MASK_PATH}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
A_DIR="${RUN_ROOT}/salun_A"
B_DIR="${RUN_ROOT}/salun_B"
C_DIR="${RUN_ROOT}/salun_C"
OUT_DIR="${RUN_ROOT}/connectivity_${CONNECTIVITY_MODE}"

CHECKPOINT_ARGS=(
  --checkpoint_every_steps "${SAVE_EVERY_STEPS}"
  --save_checkpoint_step_zero
)
if [[ -n "${CHECKPOINT_STEPS}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint_steps "${CHECKPOINT_STEPS}")
fi

infer_summary_runs_root() {
  local candidate
  candidate="$(dirname "$(dirname "${RUN_ROOT}")")"
  if [[ -d "${candidate}" ]]; then
    echo "${candidate}"
  else
    echo "${RUN_ROOT}"
  fi
}

run_salun() {
  local seed="$1"
  local save_dir="$2"
  python main_forget.py     --arch "${ARCH}"     --dataset "${DATASET}"     --gpu "${GPU}"     --unlearn "${UNLEARN}"     --unlearn_epochs "${SALUN_EPOCHS}"     --unlearn_lr "${SALUN_LR}"     --model_path "${BASE_CKPT}"     --save_dir "${save_dir}"     --mask_path "${MASK_PATH}"     --forget_seed "${FORGET_SEED}"     --forget_index_path "${FORGET_INDEX_PATH}"     --unlearn_seed "${seed}"     --decreasing_lr "${DECREASING_LR}"     "${CHECKPOINT_ARGS[@]}"
}

run_summary() {
  if [[ "${AUTO_SUMMARY}" != "1" ]]; then
    return 0
  fi

  local summary_runs_root="${SUMMARY_RUNS_ROOT}"
  local summary_retrain_root
  local summary_args

  if [[ -z "${summary_runs_root}" ]]; then
    summary_runs_root="$(infer_summary_runs_root)"
  fi
  summary_retrain_root="${SUMMARY_RETRAIN_ROOT:-${summary_runs_root}}"

  summary_args=(
    --runs_root "${summary_runs_root}"
    --retrain_root "${summary_retrain_root}"
    --metrics "${SUMMARY_METRICS}"
    --html_metrics "${SUMMARY_HTML_METRICS}"
    --plot_series "${SUMMARY_PLOT_SERIES}"
  )

  if [[ -n "${SUMMARY_DIR}" ]]; then
    summary_args+=(--summary_dir "${SUMMARY_DIR}")
  fi
  if [[ -n "${SUMMARY_OUTPUT_PREFIX}" ]]; then
    summary_args+=(--output_prefix "${SUMMARY_OUTPUT_PREFIX}")
  fi
  if [[ -n "${SUMMARY_RATIO_RUN_SPECS}" ]]; then
    summary_args+=(--ratio_run_specs "${SUMMARY_RATIO_RUN_SPECS}")
  fi
  if [[ -n "${SUMMARY_PREPEND_RUNS_ROOT}" ]]; then
    summary_args+=(--prepend_runs_root "${SUMMARY_PREPEND_RUNS_ROOT}")
  fi
  if [[ -n "${SUMMARY_PREPEND_RETRAIN_ROOT}" ]]; then
    summary_args+=(--prepend_retrain_root "${SUMMARY_PREPEND_RETRAIN_ROOT}")
  fi
  if [[ -n "${SUMMARY_PREPEND_RATIO_RUN_SPECS}" ]]; then
    summary_args+=(--prepend_ratio_run_specs "${SUMMARY_PREPEND_RATIO_RUN_SPECS}")
  fi
  if [[ -n "${SUMMARY_EPOCH_OFFSET}" ]]; then
    summary_args+=(--epoch_offset "${SUMMARY_EPOCH_OFFSET}")
  fi
  if [[ -n "${SUMMARY_PREPEND_EPOCH_OFFSET}" ]]; then
    summary_args+=(--prepend_epoch_offset "${SUMMARY_PREPEND_EPOCH_OFFSET}")
  fi

  python "${SUMMARY_SCRIPT}" "${summary_args[@]}"
}

run_salun "${UNLEARN_SEED_A}" "${A_DIR}"
run_salun "${UNLEARN_SEED_B}" "${B_DIR}"

if [[ -n "${UNLEARN_SEED_C}" ]]; then
  run_salun "${UNLEARN_SEED_C}" "${C_DIR}"
  if [[ -z "${CONTROL_RUN_DIRS}" ]]; then
    CONTROL_RUN_DIRS="C=${C_DIR}"
  else
    CONTROL_RUN_DIRS="${CONTROL_RUN_DIRS},C=${C_DIR}"
  fi
fi

if [[ "${CONNECTIVITY_MODE}" != "linear" && -z "${CONTROL_RUN_DIRS}" ]]; then
  echo "CONTROL_RUN_DIRS or UNLEARN_SEED_C is required for ${CONNECTIVITY_MODE}." >&2
  exit 1
fi

MODE_CONNECTIVITY_ARGS=(
  --arch "${ARCH}"
  --dataset "${DATASET}"
  --gpu "${GPU}"
  --run_a_dir "${A_DIR}"
  --run_b_dir "${B_DIR}"
  --output_dir "${OUT_DIR}"
  --ratio "${RATIO}"
  --exp_id "${EXP_ID}"
  --pair_id "${PAIR_ID}"
  --forget_seed "${FORGET_SEED}"
  --forget_index_path "${FORGET_INDEX_PATH}"
  --connectivity_mode "${CONNECTIVITY_MODE}"
  --alpha_grid "${ALPHA_GRID}"
  --beta "${BETA}"
  --delta "${DELTA}"
  --refine_radius "${REFINE_RADIUS}"
  --refine_step "${REFINE_STEP}"
  --simplex_weight_step "${SIMPLEX_WEIGHT_STEP}"
  --selection_mode "${SELECTION_MODE}"
  --abs_balance_a "${ABS_BALANCE_A}"
  --abs_balance_b "${ABS_BALANCE_B}"
  --notes "${NOTES}"
)

if [[ -n "${CONTROL_RUN_DIRS}" ]]; then
  MODE_CONNECTIVITY_ARGS+=(--control_run_dirs "${CONTROL_RUN_DIRS}")
fi
if [[ "${ALLOW_SAME_UNLEARN_SEED}" == "1" ]]; then
  MODE_CONNECTIVITY_ARGS+=(--allow_same_unlearn_seed)
fi

python mode_connectivity.py "${MODE_CONNECTIVITY_ARGS[@]}"
run_summary
