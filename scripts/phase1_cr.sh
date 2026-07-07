#!/usr/bin/env bash
# Phase 1 — Concept Redirect (CR): closed-form K/V edit over one concept group.
#
# Usage:
#   scripts/phase1_cr.sh imagenette   # 10 Imagenette classes
#   scripts/phase1_cr.sh artists      # 5 painters
#   scripts/phase1_cr.sh nudity       # nudity concept
#
# Env overrides:
#   CKPT_BASE  base Wan2.1 weights (default: ckpt/Wan2.1-T2V-1.3B)
#   OUT_NAME   directory name under ckpt/ for the CR-edited backbone
#   LAMB       ridge weight on |W'-W| (default: 0.5)
#   ES / PS    erase / preserve loss scales (default: 1.0 / 0.1)
set -euo pipefail
cd "$(dirname "$0")/.."

GROUP=${1:?"usage: phase1_cr.sh {imagenette|artists|nudity}"}
CKPT_BASE=${CKPT_BASE:-ckpt/Wan2.1-T2V-1.3B}
LAMB=${LAMB:-0.5}
ES=${ES:-1.0}
PS=${PS:-0.1}

case "$GROUP" in
  imagenette)
    OUT_NAME=${OUT_NAME:-Wan2.1-T2V-1.3B-cr-imagenette}
    EDIT="english springer spaniel;parachute;church;golf ball;garbage truck;cassette player;tench;french horn;chain saw;gas pump"
    PRESERVE="castle;fortress;cabin;school;building;tennis ball;basketball;dump truck;delivery truck;vinyl record;cd player;salmon;trout;trumpet;saxophone;axe;hammer;fuel station"
    ;;
  artists)
    OUT_NAME=${OUT_NAME:-Wan2.1-T2V-1.3B-cr-artists}
    EDIT="Van Gogh;Monet;Picasso;Da Vinci;Salvador Dali"
    PRESERVE="photograph;watercolor;sketch;pencil drawing;realistic painting;cartoon;oil painting;ink drawing;digital art;portrait"
    ;;
  nudity)
    OUT_NAME=${OUT_NAME:-Wan2.1-T2V-1.3B-cr-nudity}
    EDIT="naked person"
    PRESERVE="clothed person;dressed person;person in clothes;person in a coat;person in a dress;person in jeans;person wearing a shirt"
    ;;
  *)
    echo "unknown group: $GROUP (expected: imagenette | artists | nudity)" >&2
    exit 1
    ;;
esac

LOG=${LOG:-logs/phase1_cr_${GROUP}.log}
mkdir -p logs

python cr_edit.py \
    --task t2v-1.3B \
    --ckpt_dir "${CKPT_BASE}" \
    --edit_concepts    "${EDIT}" \
    --guide_concepts   "" \
    --preserve_concepts "${PRESERVE}" \
    --concept_type object \
    --erase_scale "${ES}" --preserve_scale "${PS}" --lamb "${LAMB}" \
    --save_dir ckpt --exp_name "${OUT_NAME}" \
    --link_mode symlink \
    2>&1 | tee "${LOG}"
