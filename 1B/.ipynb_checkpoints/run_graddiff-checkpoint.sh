#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_graddiff.sh RUN_NAME [FORGET_COEF] [RETAIN_COEF]
#
# Example:
#   bash run_graddiff.sh f1_ret1 1.0 1.0

RUN_NAME="${1:?Usage: bash run_graddiff.sh RUN_NAME [FORGET_COEF] [RETAIN_COEF]}"
FORGET_COEF="${2:-1.0}"
RETAIN_COEF="${3:-1.0}"

BFCL_ROOT="${BFCL_ROOT:-$HOME/Desktop/BFCL}"
PY_SCRIPT="${PY_SCRIPT:-$BFCL_ROOT/graddiff_1b.py}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/projects/standard/mhong/shared/hadir/out_dir/bfcl/1B/graddiff}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

MODEL_NAME="${MODEL_NAME:-Salesforce/xLAM-1b-fc-r}"
FORGET_FILE="${FORGET_FILE:-$BFCL_ROOT/final_forget_train.json}"
RETAIN_FILE="${RETAIN_FILE:-$BFCL_ROOT/final_retain_train_4040.json}"

EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
RETAIN_PER_FORGET="${RETAIN_PER_FORGET:-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$OUTPUT_DIR"

python "$PY_SCRIPT" \
  --model-name "$MODEL_NAME" \
  --root "$BFCL_ROOT" \
  --forget-file "$FORGET_FILE" \
  --retain-file "$RETAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --forget-coef "$FORGET_COEF" \
  --retain-coef "$RETAIN_COEF" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-accum "$GRAD_ACCUM" \
  --retain-per-forget "$RETAIN_PER_FORGET" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --max-length "$MAX_LENGTH" \
  --warmup-ratio "$WARMUP_RATIO" \
  --device "$DEVICE"

echo "Saved run: $OUTPUT_DIR"
echo "Final checkpoint: $OUTPUT_DIR/epoch_${EPOCHS}"
