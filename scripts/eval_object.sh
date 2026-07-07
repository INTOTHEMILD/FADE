#!/usr/bin/env bash
# Evaluate an object concept with the Qwen2.5-VL judge.
#
# Usage:
#   VIDEO_ROOT=result/imagenette CONCEPT=church scripts/eval_object.sh
#
# Env vars:
#   VIDEO_ROOT  directory of generated .mp4 files
#   CONCEPT     concept string (matches the class column in the prompt CSV)
#   QWEN_CKPT   Qwen2.5-VL-7B-Instruct weights (default: eval/ckpt/Qwen2.5-VL-7B-Instruct)
#   OUT         JSON output path (default: <VIDEO_ROOT>/eval_qwen.json)
#   STRIDE      frame stride (default: 4, aligns to Wan 3D-VAE temporal downsample)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VIDEO_ROOT:?set VIDEO_ROOT}"
: "${CONCEPT:?set CONCEPT}"
QWEN_CKPT=${QWEN_CKPT:-eval/ckpt/Qwen2.5-VL-7B-Instruct}
OUT=${OUT:-${VIDEO_ROOT}/eval_qwen.json}
STRIDE=${STRIDE:-4}

python eval/benchmarking/eval_img_wan21_qwen.py \
    --video_root "$VIDEO_ROOT" \
    --concept "$CONCEPT" \
    --qwen_ckpt "$QWEN_CKPT" \
    --stride "$STRIDE" \
    --out "$OUT"
