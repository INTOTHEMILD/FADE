#!/usr/bin/env bash
# Phase 2 — Frame-Aware Erasure (FAE): train one per-concept FA-LoRA on top of
# the CR-edited backbone. Skips concepts whose safetensors already exist.
#
# Usage:
#   scripts/phase2_fae.sh imagenette
#   scripts/phase2_fae.sh artists
#   scripts/phase2_fae.sh nudity
#
# Env overrides:
#   CKPT        CR-edited backbone directory (default: derived from GROUP)
#   SAVE        output directory for per-concept LoRAs (default: ckpt/lora_<group>)
#   ITERATIONS  AdamW steps per concept (default: 500)
#   FRAME_NUM   raw frames per training clip (default: 17 = 5 latent frames)
#   DEVICE      cuda device (default: cuda:0)
set -euo pipefail
cd "$(dirname "$0")/.."

GROUP=${1:?"usage: phase2_fae.sh {imagenette|artists|nudity}"}
CKPT=${CKPT:-ckpt/Wan2.1-T2V-1.3B-cr-${GROUP}}
SAVE=${SAVE:-ckpt/lora_${GROUP}}
ITERATIONS=${ITERATIONS:-500}
FRAME_NUM=${FRAME_NUM:-17}
DEVICE=${DEVICE:-cuda:0}

mkdir -p logs "${SAVE}"

# Concept ↔ unsafe-CSV mapping.
case "$GROUP" in
  imagenette)
    declare -a CONCEPTS=(
      "english springer spaniel" "parachute" "church" "golf ball"
      "garbage truck" "cassette player" "tench" "french horn"
      "chain saw" "gas pump"
    )
    declare -a UNSAFES=(
      "prompts/object/spaniel_unsafe.csv"
      "prompts/object/parachute_unsafe.csv"
      "prompts/object/church_unsafe.csv"
      "prompts/object/golf_ball_unsafe.csv"
      "prompts/object/garbage_truck_unsafe.csv"
      "prompts/object/cassette_player_unsafe.csv"
      "prompts/object/tench_unsafe.csv"
      "prompts/object/french_horn_unsafe.csv"
      "prompts/object/chain_saw_unsafe.csv"
      "prompts/object/gas_pump_unsafe.csv"
    )
    ;;
  artists)
    declare -a CONCEPTS=("Van Gogh" "Monet" "Picasso" "Da Vinci" "Salvador Dali")
    declare -a UNSAFES=(
      "prompts/object/vangogh_unsafe.csv"
      "prompts/object/monet_unsafe.csv"
      "prompts/object/picasso_unsafe.csv"
      "prompts/object/davinci_unsafe.csv"
      "prompts/object/dali_unsafe.csv"
    )
    ;;
  nudity)
    declare -a CONCEPTS=("naked person")
    declare -a UNSAFES=("prompts/nudity/nudity_unsafe.csv")
    ;;
  *)
    echo "unknown group: $GROUP" >&2; exit 1 ;;
esac

for i in "${!CONCEPTS[@]}"; do
  c="${CONCEPTS[$i]}"
  u="${UNSAFES[$i]}"
  tag="${c// /_}"
  if [[ -f "${SAVE}/lora_${tag}.safetensors" ]]; then
    echo "[skip] ${c} already trained at ${SAVE}/lora_${tag}.safetensors"
    continue
  fi
  echo "[train] ${c}"
  python train_fae.py \
      --ckpt_dir "${CKPT}" \
      --concept "${c}" \
      --unsafe_csv "${u}" \
      --neutral_motion_csv prompts/neutral/neutral_motion.csv \
      --iterations "${ITERATIONS}" --frame_num "${FRAME_NUM}" \
      --device "${DEVICE}" \
      --save_dir "${SAVE}" \
      2>&1 | tee "logs/fae_${GROUP}_${tag}.log"
done

echo "[done] ${GROUP}:"
ls -la "${SAVE}"/*.safetensors 2>/dev/null || echo "(no safetensors written)"
