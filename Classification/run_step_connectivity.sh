#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

ARCH="${ARCH:-resnet18}"
DATASET="${DATASET:-cifar10}"
UNLEARN="${UNLEARN:-RL}"
EXP_ID="${EXP_ID:-conn_step_v1}"
RATIO="${RATIO:-10}"
RUN_ROOT="${RUN_ROOT:-runs/${RATIO}/step_connectivity}"
BASE_CKPT="${BASE_CKPT:-runs/baseline/0checkpoint.pth.tar}"
FORGET_SEED="${FORGET_SEED:-1}"
FORGET_INDEX_PATH="${FORGET_INDEX_PATH:-runs/${RATIO}/forget_indices.npy}"
UNLEARN_SEED_A="${UNLEARN_SEED_A:-11}"
UNLEARN_SEED_B="${UNLEARN_SEED_B:-22}"
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
PAIR_ID="${PAIR_ID:-A${UNLEARN_SEED_A}_B${UNLEARN_SEED_B}}"
NOTES="${NOTES:-}"
ALLOW_SAME_UNLEARN_SEED="${ALLOW_SAME_UNLEARN_SEED:-0}"
GPU="${GPU:-0}"

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
OUT_DIR="${RUN_ROOT}/connectivity"

CHECKPOINT_ARGS=(
  --checkpoint_every_steps "${SAVE_EVERY_STEPS}"
  --save_checkpoint_step_zero
)
if [[ -n "${CHECKPOINT_STEPS}" ]]; then
  CHECKPOINT_ARGS+=(--checkpoint_steps "${CHECKPOINT_STEPS}")
fi

python main_forget.py   --arch "${ARCH}"   --dataset "${DATASET}"   --gpu "${GPU}"   --unlearn "${UNLEARN}"   --unlearn_epochs "${SALUN_EPOCHS}"   --unlearn_lr "${SALUN_LR}"   --model_path "${BASE_CKPT}"   --save_dir "${A_DIR}"   --mask_path "${MASK_PATH}"   --forget_seed "${FORGET_SEED}"   --forget_index_path "${FORGET_INDEX_PATH}"   --unlearn_seed "${UNLEARN_SEED_A}"   --decreasing_lr "${DECREASING_LR}"   "${CHECKPOINT_ARGS[@]}"

python main_forget.py   --arch "${ARCH}"   --dataset "${DATASET}"   --gpu "${GPU}"   --unlearn "${UNLEARN}"   --unlearn_epochs "${SALUN_EPOCHS}"   --unlearn_lr "${SALUN_LR}"   --model_path "${BASE_CKPT}"   --save_dir "${B_DIR}"   --mask_path "${MASK_PATH}"   --forget_seed "${FORGET_SEED}"   --forget_index_path "${FORGET_INDEX_PATH}"   --unlearn_seed "${UNLEARN_SEED_B}"   --decreasing_lr "${DECREASING_LR}"   "${CHECKPOINT_ARGS[@]}"

STEP_CONNECTIVITY_ARGS=(   --arch "${ARCH}"   --dataset "${DATASET}"   --gpu "${GPU}"   --run_a_dir "${A_DIR}"   --run_b_dir "${B_DIR}"   --output_dir "${OUT_DIR}"   --ratio "${RATIO}"   --exp_id "${EXP_ID}"   --pair_id "${PAIR_ID}"   --forget_seed "${FORGET_SEED}"   --forget_index_path "${FORGET_INDEX_PATH}"   --alpha_grid "${ALPHA_GRID}"   --beta "${BETA}"   --delta "${DELTA}"   --refine_radius "${REFINE_RADIUS}"   --refine_step "${REFINE_STEP}"   --notes "${NOTES}" )

if [[ "${ALLOW_SAME_UNLEARN_SEED}" == "1" ]]; then
  STEP_CONNECTIVITY_ARGS+=(--allow_same_unlearn_seed)
fi

python step_connectivity.py "${STEP_CONNECTIVITY_ARGS[@]}"
