#!/usr/bin/env bash
# Evaluate object erasure with the ResNet-50 Imagenette judge.
#
# Usage:
#   VIDEO_ROOT=result/video/imagenette CASE_NAME=fade scripts/eval_object.sh
#
# Env vars:
#   VIDEO_ROOT       root of generated videos, expected layout
#                    {VIDEO_ROOT}/{CASE_NAME}/{class_dir}/{sample}_seed{seed}.mp4
#   CASE_NAME        sub-directory tag identifying the current run (e.g. `fade`, `origin`)
#   CONCEPTS         optional comma-separated concept filter
#                    (default: all classes in the manifest CSV)
#   MANIFEST_CSV     manifest CSV (default: eval/dataset/imagenette_v.csv)
#   LIMIT_PER_CONCEPT  optional cap on samples per concept
#   GPUS             comma-separated GPU list (default: 0)
#   SAVE_DIR         output dir (default: {VIDEO_ROOT}/{CASE_NAME}/eval_resnet)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VIDEO_ROOT:?set VIDEO_ROOT}"
: "${CASE_NAME:?set CASE_NAME}"
MANIFEST_CSV=${MANIFEST_CSV:-eval/dataset/imagenette_v.csv}
GPUS=${GPUS:-0}

EXTRA=()
[ -n "${CONCEPTS:-}" ]          && EXTRA+=(--concepts "$CONCEPTS")
[ -n "${LIMIT_PER_CONCEPT:-}" ] && EXTRA+=(--limit_per_concept "$LIMIT_PER_CONCEPT")
[ -n "${SAVE_DIR:-}" ]          && EXTRA+=(--save_dir "$SAVE_DIR")

python eval/benchmarking/eval_img_wan21.py \
    --manifest_csv "$MANIFEST_CSV" \
    --video_root   "$VIDEO_ROOT" \
    --case_name    "$CASE_NAME" \
    --gpus         "$GPUS" \
    "${EXTRA[@]}"
