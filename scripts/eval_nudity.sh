#!/usr/bin/env bash
# Evaluate nudity with the NudeNet detector on the I2P-Video slice.
#
# Usage:
#   VIDEO_ROOT=result/nudity CASE_NAME=fade scripts/eval_nudity.sh
#
# Env vars:
#   VIDEO_ROOT  directory of generated .mp4 files
#   CASE_NAME   subdirectory tag inside VIDEO_ROOT (default: run)
#   OUT_DIR     where to dump per-frame detections and summary
#   STRIDE      frame stride (default: 4)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VIDEO_ROOT:?set VIDEO_ROOT}"
CASE_NAME=${CASE_NAME:-run}
STRIDE=${STRIDE:-4}
OUT_DIR=${OUT_DIR:-${VIDEO_ROOT}/${CASE_NAME}/eval_nudenet}

python eval/benchmarking/nudity_eval_wan21.py \
    --video_root "$VIDEO_ROOT" \
    --case_name "$CASE_NAME" \
    --out_dir "$OUT_DIR" \
    --stride "$STRIDE"
