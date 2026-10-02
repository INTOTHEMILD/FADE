#!/usr/bin/env bash
# Inference — generate a video with CR-edited backbone + FA-LoRA + C-MoE routing.
#
# Usage:
#   PROMPT="a dog runs on the beach" LORA_DIR=ckpt/lora_imagenette scripts/inference.sh
#
# Env vars:
#   CKPT         CR-edited backbone directory (required unless CKPT_BASE only)
#   LORA_DIR     directory holding routing_config.pt + lora_<concept>.safetensors
#   PROMPT       text prompt (required)
#   OUT          output mp4 path (default: output/<CKPT basename>/<slug>.mp4)
#   FRAME_NUM    frames to render (default: 81)
#   SIZE         resolution (default: 832*480)
#   SEED         RNG seed (default: 42)
#   STEPS        UniPC steps (default: 50)
#   GPUS         CUDA_VISIBLE_DEVICES, single value (default: 0)
#   ORIG_CKPT    unedited Wan2.1 dir; enables Texture-Phase Decay if it exists
#                (default: ckpt/Wan2.1-T2V-1.3B; set ORIG_CKPT= to disable)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CKPT:?set CKPT to a CR-edited backbone dir}"
: "${LORA_DIR:?set LORA_DIR to the Phase 2 output dir}"
: "${PROMPT:?set PROMPT}"

GPUS=${GPUS:-0}
FRAME_NUM=${FRAME_NUM:-81}
SIZE=${SIZE:-832*480}
SEED=${SEED:-42}
STEPS=${STEPS:-50}
ORIG_CKPT=${ORIG_CKPT-ckpt/Wan2.1-T2V-1.3B}

TPD_ARGS=()
if [ -n "$ORIG_CKPT" ] && [ -d "$ORIG_CKPT" ]; then
  TPD_ARGS=(--ts_uce_orig_ckpt "$ORIG_CKPT")
fi

CKPT_BASENAME="$(basename "$CKPT")"
PROMPT_SLUG=$(echo "$PROMPT" \
  | tr '[:upper:]' '[:lower:]' \
  | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g' \
  | cut -c 1-80)
[ -z "$PROMPT_SLUG" ] && PROMPT_SLUG="prompt"

OUT=${OUT:-output/${CKPT_BASENAME}/${PROMPT_SLUG}.mp4}
mkdir -p "$(dirname "$OUT")"

CUDA_VISIBLE_DEVICES="$GPUS" python tools/inference.py \
    --task t2v-1.3B \
    --ckpt_dir "$CKPT" \
    --lora_dir "$LORA_DIR" \
    --prompt "$PROMPT" \
    --out "$OUT" \
    --frame_num "$FRAME_NUM" \
    --size "$SIZE" \
    --seed "$SEED" \
    --num_inference_steps "$STEPS" \
    ${TPD_ARGS[@]+"${TPD_ARGS[@]}"}

echo "[inference] wrote $OUT"
