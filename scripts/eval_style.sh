#!/usr/bin/env bash
# Evaluate an artistic-style concept: per-frame CLIP similarity to reference art.
#
# Usage:
#   VIDEO_ROOT=result/artists CONCEPT="Van Gogh" scripts/eval_style.sh
#
# Env vars:
#   VIDEO_ROOT  directory of generated .mp4 files
#   CONCEPT     artist name (matches prompt CSV class column)
#   REF_DIR     directory of reference style images
#   OUT         JSON output path (default: <VIDEO_ROOT>/eval_clip.json)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VIDEO_ROOT:?set VIDEO_ROOT}"
: "${CONCEPT:?set CONCEPT}"
: "${REF_DIR:?set REF_DIR to a folder of reference works}"
OUT=${OUT:-${VIDEO_ROOT}/eval_clip.json}

python eval/benchmarking/eval_img_wan21.py \
    --video_root "$VIDEO_ROOT" \
    --concept "$CONCEPT" \
    --ref_dir "$REF_DIR" \
    --judge clip \
    --out "$OUT"
