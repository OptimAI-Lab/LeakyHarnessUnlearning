#!/usr/bin/env bash
# Run from the repository root. Extra arguments configure training only.
set -euo pipefail
method=${1:?Usage: bash tool_unlearn/baselines/run.sh METHOD OUT_DIR [training options]}
out=${2:?Specify a new output directory}
shift 2
ACTOR_MODEL=${ACTOR_MODEL:-TangQiaoYu/ToolAlpaca-7B}
case "$method" in
  ga|graddiff|npo|tooldelete_sft|tooldelete_dpo|retain_ft) ;;
  *) echo "Unknown baseline: $method" >&2; exit 2 ;;
esac
if [[ -e "$out" ]]; then
  echo "Choose a new output directory: $out" >&2; exit 2
fi
if [[ "$method" == tooldelete_* ]]; then
  : "${TOOLFREE_MODEL:?Set TOOLFREE_MODEL to the matching pre-tool-training checkpoint}"
fi
mkdir -p "$out/artifacts"
args=(--method "$method" --model_dir "$ACTOR_MODEL")
if [[ "$method" == tooldelete_* ]]; then
  splits=(forget)
  [[ "$method" != tooldelete_dpo ]] || splits+=(retain)
  for split in "${splits[@]}"; do
    python -m tool_unlearn.baselines.prepare_targets generate \
      --input_file "tool_unlearn/data/${split}_train.json" \
      --teacher_model_dir "$TOOLFREE_MODEL" --tokenizer_dir "$ACTOR_MODEL" \
      --output "$out/artifacts/$split.json"
    args+=("--${split}_targets" "$out/artifacts/$split.json")
  done
fi
case "$method" in
  npo|tooldelete_dpo)
    python -m tool_unlearn.baselines.prepare_reference "${args[@]}" \
      --output "$out/artifacts/reference.pt"
    args+=(--reference_cache "$out/artifacts/reference.pt") ;;
esac
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-8}" \
  -m tool_unlearn.baselines.train "${args[@]}" --out_dir "$out/model" \
  --deepspeed tool_unlearn/ds_zero3_adafactor.json "$@"
