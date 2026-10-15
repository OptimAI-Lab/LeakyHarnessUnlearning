#!/usr/bin/env bash
set -euo pipefail

# Unified ToolDelete launcher for xLAM-2 3B.
#
# SFT:
#   bash run_tooldelete_3b_native.sh sft tooldelete_sft_alpha0
#
# DPO:
#   bash run_tooldelete_3b_native.sh dpo tooldelete_dpo_b01_ep3_alpha0
#
# Defaults:
#   SFT -> 1 epoch
#   DPO -> 3 epochs
#   alpha = 0
#   beta  = 0.1

METHOD="${1:?Usage: bash run_tooldelete_3b.sh METHOD RUN_NAME}"
RUN_NAME="${2:?Usage: bash run_tooldelete_3b.sh METHOD RUN_NAME}"

if [ "$METHOD" != "sft" ] && [ "$METHOD" != "dpo" ]; then
    echo "ERROR: METHOD must be sft or dpo"
    exit 1
fi

BFCL_ROOT="${BFCL_ROOT:-$HOME/Desktop/BFCL}"
PY_SCRIPT="${PY_SCRIPT:-$BFCL_ROOT/tooldelete_3b.py}"

MODEL_NAME="${MODEL_NAME:-Salesforce/xLAM-2-3b-fc-r}"

FORGET_FILE="${FORGET_FILE:-$BFCL_ROOT/final_forget_train.json}"
RETAIN_FILE="${RETAIN_FILE:-$BFCL_ROOT/final_retain_train_4040.json}"
TOOLFREE_CACHE="${TOOLFREE_CACHE:-$BFCL_ROOT/tooldelete_toolfree_deepseek13b.json}"

DPO_REF_CACHE="${DPO_REF_CACHE:-/projects/standard/mhong/shared/hadir/out_dir/bfcl/3B/tooldelete/xlam2_3b_tooldelete_dpo_reference_logps.pt}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/projects/standard/mhong/shared/hadir/out_dir/bfcl/3B/tooldelete}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

if [ "$METHOD" = "sft" ]; then
    EPOCHS="${EPOCHS:-1}"
else
    EPOCHS="${EPOCHS:-3}"
fi

ALPHA="${ALPHA:-0.0}"
DPO_BETA="${DPO_BETA:-0.1}"

FORGET_COEF="${FORGET_COEF:-1.0}"
RETAIN_COEF="${RETAIN_COEF:-1.0}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
RETAIN_PER_FORGET="${RETAIN_PER_FORGET:-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-3072}"
WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
REF_BATCH_SIZE="${REF_BATCH_SIZE:-1}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

if [ "$ALPHA" != "0" ] && [ "$ALPHA" != "0.0" ] && [ "$ALPHA" != "0.00" ]; then
    echo "ERROR: this experiment is alpha=0 only."
    exit 1
fi

if [ -e "$OUTPUT_DIR" ]; then
    echo "ERROR: output already exists:"
    echo "$OUTPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "TOOLDELETE xLAM-2 3B"
echo "============================================================"
echo "METHOD         : $METHOD"
echo "MODEL          : $MODEL_NAME"
echo "OUTPUT         : $OUTPUT_DIR"
echo "EPOCHS         : $EPOCHS"
echo "ALPHA          : $ALPHA"
echo "DPO_BETA       : $DPO_BETA"
echo "LR             : $LEARNING_RATE"
echo "FORGET_COEF    : $FORGET_COEF"
echo "RETAIN_COEF    : $RETAIN_COEF"
echo "TOOLFREE_CACHE : $TOOLFREE_CACHE"
if [ "$METHOD" = "dpo" ]; then
    echo "DPO_REF_CACHE  : $DPO_REF_CACHE"
fi
echo "============================================================"

python "$PY_SCRIPT" \
  --method "$METHOD" \
  --model-name "$MODEL_NAME" \
  --root "$BFCL_ROOT" \
  --forget-file "$FORGET_FILE" \
  --retain-file "$RETAIN_FILE" \
  --toolfree-cache "$TOOLFREE_CACHE" \
  --dpo-ref-cache "$DPO_REF_CACHE" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --alpha "$ALPHA" \
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

echo
echo "Training complete."
echo "Final checkpoint:"
echo "$OUTPUT_DIR/epoch_${EPOCHS}"
