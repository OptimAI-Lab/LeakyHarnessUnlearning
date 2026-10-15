#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# TOOLDELETE-SFT / TOOLDELETE-DPO LAUNCHER
# ============================================================
# Usage:
#   bash run_tooldelete.sh METHOD RUN_NAME [TASK_ARITH_ALPHA] [DPO_BETA]
#
# Examples:
#   bash run_tooldelete.sh sft tooldelete_sft 0.0
#   bash run_tooldelete.sh dpo tooldelete_dpo 1.0 0.1
#
# Paper-specified:
#   LR=1e-5; SFT replaces forget responses with f0(Q);
#   DPO prefers f0(Q) over original on forget and the reverse on retain;
#   final task arithmetic is theta_post + alpha(theta0-thetaR).
#
# Paper does not report epochs, batch size, DPO beta, or numerical alpha.
# Defaults below intentionally match the user's existing xLAM BFCL baseline
# where possible; all are overridable by environment variables.

METHOD="${1:?Usage: bash run_tooldelete.sh METHOD RUN_NAME [TASK_ARITH_ALPHA] [DPO_BETA]}"
RUN_NAME="${2:?Usage: bash run_tooldelete.sh METHOD RUN_NAME [TASK_ARITH_ALPHA] [DPO_BETA]}"
TASK_ARITH_ALPHA="${3:-1.0}"
DPO_BETA="${4:-0.1}"

if [[ "$METHOD" != "sft" && "$METHOD" != "dpo" ]]; then
  echo "METHOD must be 'sft' or 'dpo'" >&2
  exit 2
fi

BFCL_ROOT="${BFCL_ROOT:-$HOME/Desktop/BFCL}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$BFCL_ROOT/tooldelete_1b.py}"
ARITH_SCRIPT="${ARITH_SCRIPT:-$BFCL_ROOT/apply_tooldelete_task_arithmetic.py}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/projects/standard/mhong/shared/hadir/out_dir/bfcl/1B/tooldelete}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

MODEL_NAME="${MODEL_NAME:-Salesforce/xLAM-1b-fc-r}"
TOOL_FREE_MODEL="${TOOL_FREE_MODEL:-deepseek-ai/deepseek-coder-1.3b-instruct}"
FORGET_FILE="${FORGET_FILE:-$BFCL_ROOT/final_forget_train.json}"
RETAIN_FILE="${RETAIN_FILE:-$BFCL_ROOT/final_retain_train_4040.json}"
TOOLFREE_CACHE="${TOOLFREE_CACHE:-$BFCL_ROOT/tooldelete_toolfree_deepseek13b.json}"
DPO_REF_CACHE="${DPO_REF_CACHE:-$BFCL_ROOT/tooldelete_dpo_reference_logps.pt}"

EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
FORGET_COEF="${FORGET_COEF:-1.0}"
RETAIN_COEF="${RETAIN_COEF:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
RETAIN_PER_FORGET="${RETAIN_PER_FORGET:-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
REF_BATCH_SIZE="${REF_BATCH_SIZE:-2}"
DEVICE="${DEVICE:-cuda}"
ARITH_DEVICE="${ARITH_DEVICE:-cuda}"

if [[ ! -f "$TOOLFREE_CACHE" ]]; then
  cat >&2 <<EOF
Missing TOOLDELETE tool-free cache:
  $TOOLFREE_CACHE

Create it once with:
  python $BFCL_ROOT/prepare_tooldelete_toolfree.py \\
    --tool-free-model $TOOL_FREE_MODEL \\
    --root $BFCL_ROOT \\
    --output $TOOLFREE_CACHE
EOF
  exit 3
fi

mkdir -p "$OUTPUT_DIR"

printf '%s\n' \
  "============================================================" \
  "TOOLDELETE-${METHOD^^} TRAINING" \
  "============================================================" \
  "RUN_NAME        : $RUN_NAME" \
  "OUTPUT_DIR      : $OUTPUT_DIR" \
  "TOOL_FREE_MODEL : $TOOL_FREE_MODEL" \
  "EPOCHS          : $EPOCHS" \
  "LR              : $LEARNING_RATE" \
  "DPO_BETA        : $DPO_BETA" \
  "TASK_ALPHA      : $TASK_ARITH_ALPHA" \
  "SEED            : $SEED" \
  "============================================================"

python "$TRAIN_SCRIPT" \
  --method "$METHOD" \
  --model-name "$MODEL_NAME" \
  --root "$BFCL_ROOT" \
  --forget-file "$FORGET_FILE" \
  --retain-file "$RETAIN_FILE" \
  --toolfree-cache "$TOOLFREE_CACHE" \
  --dpo-ref-cache "$DPO_REF_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --forget-coef "$FORGET_COEF" \
  --retain-coef "$RETAIN_COEF" \
  --dpo-beta "$DPO_BETA" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-accum "$GRAD_ACCUM" \
  --retain-per-forget "$RETAIN_PER_FORGET" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --max-length "$MAX_LENGTH" \
  --warmup-ratio "$WARMUP_RATIO" \
  --ref-batch-size "$REF_BATCH_SIZE" \
  --device "$DEVICE"

PRE_DIR="$OUTPUT_DIR/epoch_${EPOCHS}_pre_task_arithmetic"
FINAL_DIR="$OUTPUT_DIR/epoch_${EPOCHS}"

python "$ARITH_SCRIPT" \
  --post-model-path "$PRE_DIR" \
  --tool-free-model "$TOOL_FREE_MODEL" \
  --output-dir "$FINAL_DIR" \
  --alpha "$TASK_ARITH_ALPHA" \
  --random-seed "$SEED" \
  --device "$ARITH_DEVICE"

printf '%s\n' \
  "============================================================" \
  "DONE" \
  "Pre-task-arithmetic checkpoint: $PRE_DIR" \
  "Full TOOLDELETE checkpoint     : $FINAL_DIR" \
  "============================================================"
