# FADE — Frame-Aware Diffusion-Transformer-based Multi-Concept Erasure for Video Unlearning

Official implementation of **FADE**, a training-free-plus-lightweight-adapter
framework for erasing multiple target concepts from text-to-video diffusion
models while preserving generative fidelity on unrelated prompts.

FADE is built on top of the [Wan2.1-T2V-1.3B](https://github.com/Wan-Video/Wan2.1)
backbone and has three components:

- **Concept Redirect (CR)** — a closed-form K/V edit augmented by a
  Motion-Paraphrase Embedding on the input axis and a Texture-Phase Decay
  schedule on the timestep axis.
- **Frame-Aware Erasure (FAE)** — a per-concept low-rank adapter (FA-LoRA)
  attached to the FFN path, gated frame-by-frame by a learnable FrameGate MLP
  and trained with a four-term temporal-coherent objective.
- **C-MoE routing** — a token-level max-similarity gate that composes multiple
  FA-LoRAs at inference time without cross-concept interference.

## Repository layout

```
FADE/
├── cr_edit.py                 Phase 1 — Concept Redirect (closed-form K/V edit)
├── train_fae.py               Phase 2 — Frame-Aware Erasure (FA-LoRA training)
├── generate.py                Wan2.1 generation CLI (unedited backbone)
├── wan/                       Wan2.1 backbone (upstream) + FADE modules
│   └── modules/
│       ├── frame_aware_lora.py  FA-LoRA implementation
│       ├── gated_ffn.py         FrameGate MLP
│       ├── lora_router.py       C-MoE token-level routing
│       └── temporal_loss.py     four-term training objective
├── utils/                     ESD helper + LoRA (de)serialisation
├── tools/
│   ├── calibrate_router.py    calibrate (τ_c, T_c) for the C-MoE gate
│   ├── inference.py           FADE inference (CR + FAE + C-MoE)
│   ├── precompute_t5_contexts.py
│   ├── generate_with_t5_cache.py
│   └── configs/               concept + motion-paraphrase JSON registries
├── scripts/                   consolidated bash entry points
├── prompts/                   unsafe / safe / neutral prompt CSVs
├── eval/                      Qwen2.5-VL / CLIP / NudeNet judges
├── docs/                      design notes (K/V vs. FFN, post-MLP anchors)
└── tests/                     unit tests for the FADE modules
```

## Installation

```bash
conda create -n fade python=3.10 -y
conda activate fade
pip install -r requirements.txt
```

The Qwen2.5-VL judge (`eval/benchmarking/eval_img_wan21_qwen.py`) needs
`transformers>=4.49` and is best kept in a separate environment. The NudeNet
detector (`eval/benchmarking/nudity_eval_wan21.py`) and LPIPS
(`eval/benchmarking/eval_temporal_metrics.py`) are optional; install
`nudenet` / `lpips` on demand.

## Model weights

Download the Wan2.1-T2V-1.3B checkpoint from the upstream repository and
place it under `ckpt/`:

```
ckpt/
└── Wan2.1-T2V-1.3B/
    ├── config.json
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── Wan2.1_VAE.pth
    └── diffusion_pytorch_model*.safetensors
```

Weights and generated media are ignored by `.gitignore` and never committed.

## Reproducing the paper

### Phase 1 — Concept Redirect

```bash
# 10-class Imagenette object erasure
scripts/phase1_cr.sh imagenette

# 5-artist style erasure
scripts/phase1_cr.sh artists

# nudity erasure
scripts/phase1_cr.sh nudity
```

Each command runs `cr_edit.py` and writes an edited Wan2.1 checkpoint to
`ckpt/Wan2.1-T2V-1.3B-cr-<group>/`. Symlinks are used for weights that were
not edited (VAE, T5) to avoid duplicating multi-GB files.

### Phase 2 — Frame-Aware Erasure

```bash
scripts/phase2_fae.sh imagenette
scripts/phase2_fae.sh artists
scripts/phase2_fae.sh nudity
```

Trains one FA-LoRA per concept on top of the CR-edited backbone. Per-concept
`.safetensors` land in `ckpt/lora_<group>/`. Skips concepts whose LoRA files
already exist, so the loop can be resumed.

### C-MoE calibration

```bash
python tools/calibrate_router.py \
    --ckpt_dir  ckpt/Wan2.1-T2V-1.3B-cr-imagenette \
    --lora_dir  ckpt/lora_imagenette \
    --unsafe_dir prompts/object \
    --neutral   prompts/neutral/neutral_motion.csv \
    --out       ckpt/lora_imagenette/routing_config.pt
```

Produces a `routing_config.pt` with per-concept anchors and calibrated
`(τ_c, T_c)` gate thresholds, consumed by `tools/inference.py`.

### Inference

```bash
CKPT=ckpt/Wan2.1-T2V-1.3B-cr-imagenette \
LORA_DIR=ckpt/lora_imagenette \
PROMPT="A parachute drifting over a green field" \
scripts/inference.sh
```

Or directly:

```bash
python tools/inference.py \
    --task t2v-1.3B \
    --ckpt_dir ckpt/Wan2.1-T2V-1.3B-cr-imagenette \
    --lora_dir ckpt/lora_imagenette \
    --prompt   "A parachute drifting over a green field" \
    --out      output/fade_parachute.mp4
```

### Evaluation

```bash
# Objects (Qwen2.5-VL judge on Imagenette-10)
VIDEO_ROOT=result/imagenette/church CONCEPT=church scripts/eval_object.sh

# Artistic styles (CLIP similarity to reference works)
VIDEO_ROOT=result/artists/van_gogh CONCEPT="Van Gogh" REF_DIR=data/van_gogh_refs \
    scripts/eval_style.sh

# Nudity (NudeNet on the I2P-Video slice)
VIDEO_ROOT=result/nudity CASE_NAME=fade scripts/eval_nudity.sh
```

## Tests

```bash
pytest tests/
```

Unit tests cover FA-LoRA forward/backward, FrameGate initialisation, the
C-MoE routing scores, the four-term temporal-coherent loss, and the temporal
evaluation utilities.

## Acknowledgements

FADE reuses the Wan2.1-T2V-1.3B backbone released by the
[Wan-Video](https://github.com/Wan-Video/Wan2.1) team under the Apache 2.0
license. See `wan/` and `LICENSE` for the upstream notice.

Prior concept-erasure work directly informing this codebase:

- Gandikota et al. — Erasing Concepts from Diffusion Models (ESD)
- Gandikota et al. — Unified Concept Editing in Diffusion Models (UCE)
- Facchiano et al. — Video Unlearning via Low-Rank Refusal Vector
- Xu et al. — VideoEraser
- Ye et al. — T2VUnlearning

## Citation

The BibTeX entry will be added after the paper is publicly released.

## License

Apache License 2.0. See `LICENSE`.
