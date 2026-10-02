#!/usr/bin/env bash
# Evaluate nudity with the NudeNet detector (manifest: prompts/nudity/nudity_unsafe.csv by default).
#
# Usage:
#   VIDEO_ROOT=result/nudity CASE_NAME=fade scripts/eval_nudity.sh
#
# Env vars:
#   VIDEO_ROOT  root of generated videos: {VIDEO_ROOT}/{CASE_NAME}/{class_dir}/{sample}_seed{seed}.mp4
#   MANIFEST_CSV  prompt manifest (default: prompts/nudity/nudity_unsafe.csv)
#   CASE_NAME   subdirectory tag inside VIDEO_ROOT (default: run)
#   OUT_DIR     where to dump per-frame detections and summary
#   STRIDE      frame stride (default: 4)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VIDEO_ROOT:?set VIDEO_ROOT}"
CASE_NAME=${CASE_NAME:-run}
STRIDE=${STRIDE:-4}
MANIFEST_CSV=${MANIFEST_CSV:-prompts/nudity/nudity_unsafe.csv}
OUT_DIR=${OUT_DIR:-${VIDEO_ROOT}/${CASE_NAME}/eval_nudenet}

python eval/benchmarking/nudity_eval_wan21.py \
    --manifest_csv "$MANIFEST_CSV" \
    --video_root "$VIDEO_ROOT" \
    --case_name "$CASE_NAME" \
    --save_dir "$OUT_DIR" \
    --frame_sample_mode stride \
    --frame_stride "$STRIDE"
