#!/usr/bin/env python
"""Phase 2: train per-concept frame-aware FFN LoRA with temporal-coherent ESD loss."""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.cuda.amp as amp
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.wan_esd_utils import (
    compute_seq_len,
    compute_target_shape,
    model_forward,
    partial_denoise,
)
from utils.wan_lora_utils import (
    attach_lora_to_block,
    enable_block_gradient_checkpointing,
    get_lora_params,
    save_lora,
)
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.modules.gated_ffn import GatedFFN
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.temporal_loss import (
    max_frame_leak_loss,
    motion_preserve_loss,
    standard_esd_loss,
    temporal_smooth_loss,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="t2v-1.3B")
    p.add_argument(
        "--ckpt_dir",
        required=True,
        help="Phase 1 init dir (CR-edited Wan)",
    )
    p.add_argument("--concept", required=True)
    p.add_argument("--unsafe_csv", required=True)
    p.add_argument(
        "--neutral_motion_csv",
        default="prompts/neutral/neutral_motion.csv",
    )

    # Loss weights / LoRA shape
    p.add_argument("--r0", type=int, default=4)
    p.add_argument("--r1", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.5, help="L_max-frame-leak weight")
    p.add_argument("--beta", type=float, default=0.5, help="L_motion-preserve weight")
    p.add_argument("--gamma", type=float, default=0.1, help="L_temporal-smooth weight")
    p.add_argument("--eta", type=float, default=1.0, help="ESD negative guidance")
    p.add_argument("--top_k", type=int, default=3)

    # Optimization
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Generation params
    p.add_argument(
        "--frame_num",
        type=int,
        default=33,
        help="Train at 33 frames (T_lat=9); eval at 81 separately",
    )
    p.add_argument("--size", default="832*480")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--timestep_low", type=int, default=5)
    p.add_argument("--timestep_high", type=int, default=None)

    # Misc
    p.add_argument("--device", default="cuda:0",
                   help="trainable model device (does backward)")
    p.add_argument("--base_device", default=None,
                   help="frozen base device. Defaults to --device. Set to a "
                        "second GPU (e.g. cuda:1) when single GPU OOMs.")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="Wrap each block.forward in torch.utils.checkpoint. "
                        "Required for frame_num=33 on 45GB GPUs.")
    p.add_argument("--save_dir", default="ckpt/lora_phase2")
    p.add_argument("--seed", type=int, default=0)
    # Ablation: zero + freeze the frame-aware branch (B1, D1, phi). Reduces
    # the LoRA to ΔW · x = B0 D0 x — frame-agnostic, equivalent to standard
    # LoRA. Used to isolate the contribution of frame-awareness in §5
    # ablations. L_smt is forced to 0 since phi has no trainable parameters.
    p.add_argument("--no_frame_aware", action="store_true",
                   help="Ablation: disable frame-aware branch (B1=D1=0, "
                        "phi frozen). Reduces to a standard LoRA.")
    return p.parse_args()


def _resolve_dtype(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def build_frame_idx_per_token(target_shape, patch_size, seq_len, device):
    """Per-token frame index normalized to [0, 1], length = seq_len.

    Wan patchifies via Conv3d(stride=patch_size); video tokens are flattened in
    (T_p, H_p, W_p) order with H_p = H_lat//patch[1], W_p = W_lat//patch[2].
    Padded tokens beyond the valid prefix get f_norm = 0 (LoRA contribution there
    is irrelevant — they're zero in the input).
    """
    _, T_lat, H_lat, W_lat = target_shape
    pT = T_lat // patch_size[0]
    pH = H_lat // patch_size[1]
    pW = W_lat // patch_size[2]
    n_per_frame = pH * pW
    n_valid = pT * n_per_frame
    f_idx = torch.zeros(seq_len, device=device, dtype=torch.long)
    f_idx[:n_valid] = torch.arange(n_valid, device=device) // n_per_frame
    f_norm = f_idx.float() / max(pT - 1, 1)
    return f_norm


def model_forward_with_lora(
    model, xt, t_at, ctx, seq_len, target_shape, patch_size,
    num_train_timesteps, concept_name, gate_value=1.0,
):
    """Set per-block frame info + active concept gate, then run normal forward.

    During Phase 2 training of a single concept, gate_value is fixed at 1.0 so
    ConceptPhi can learn the per-frame modulation. At inference, the LoRA router
    supplies prompt-conditioned g_c values instead.
    """
    f_norm = build_frame_idx_per_token(target_shape, patch_size, seq_len, xt.device)
    t_norm = float(t_at.item()) / num_train_timesteps
    for blk in model.blocks:
        if isinstance(blk.ffn, GatedFFN):
            blk.ffn.set_frame_time(f_norm, t_norm)
            blk.ffn.set_gates({concept_name: gate_value})
    return model_forward(model, xt, t_at, ctx, seq_len)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = WAN_CONFIGS[args.task]
    device = torch.device(args.device)
    base_device = torch.device(args.base_device) if args.base_device else device
    dtype = _resolve_dtype(args.dtype)
    width, height = SIZE_CONFIGS[args.size]

    # Frozen base = Phase 1 CR-edited Wan (no grad → can sit on a separate GPU)
    logging.info(f"Loading frozen base from {args.ckpt_dir} on {base_device}")
    base = WanModel.from_pretrained(args.ckpt_dir).to(base_device).to(dtype)
    base.eval().requires_grad_(False)

    # Trainable copy with FrameAwareLoRA mounted on each block.ffn
    logging.info(f"Loading trainable copy + attaching LoRA on {device}")
    new = WanModel.from_pretrained(args.ckpt_dir).to(device).to(dtype)
    new.eval().requires_grad_(False)

    d = new.dim
    for blk in new.blocks:
        attach_lora_to_block(
            blk, concept_name=args.concept,
            d_in=d, d_out=d, r0=args.r0, r1=args.r1,
        )
    # LoRA modules were created in fp32; move them to base dtype + device
    new.to(device).to(dtype)

    # Make only LoRA params trainable
    names, lora_params = get_lora_params(new)
    for p in lora_params:
        p.requires_grad_(True)

    if args.no_frame_aware:
        # Zero + freeze the frame-aware branch (B1, D1, phi). This reduces
        # the LoRA to its static B0 D0 path — the frame-agnostic baseline.
        for blk in new.blocks:
            if isinstance(blk.ffn, GatedFFN):
                for lora in blk.ffn.loras.values():
                    lora.B1.weight.data.zero_()
                    lora.D1.weight.data.zero_()
                    lora.B1.weight.requires_grad_(False)
                    lora.D1.weight.requires_grad_(False)
                    for pp in lora.phi.parameters():
                        pp.requires_grad_(False)
        lora_params = [p for p in lora_params if p.requires_grad]
        logging.info("Frame-aware branch DISABLED (B1=D1=0 frozen, phi frozen).")
    logging.info(f"Trainable params: {sum(p.numel() for p in lora_params):,}")

    if args.grad_checkpoint:
        enable_block_gradient_checkpointing(new)
        logging.info("Enabled gradient checkpointing on trainable blocks")

    optim = torch.optim.AdamW(
        lora_params, lr=args.lr, weight_decay=args.weight_decay
    )

    # T5 encoding — load on base_device so we can free it before training proper.
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
        shard_fn=None,
    )
    text_encoder.model.to(base_device).eval()

    unsafe_prompts = pd.read_csv(args.unsafe_csv)["prompt"].tolist()
    neutral_prompts = pd.read_csv(args.neutral_motion_csv)["prompt"].tolist()
    logging.info(
        f"Encoding {len(unsafe_prompts)} unsafe + {len(neutral_prompts)} neutral"
    )

    def _to_cpu(ctx):
        return [t.detach().cpu() for t in ctx]

    with torch.no_grad():
        ctx_unsafe = [_to_cpu(text_encoder([p], base_device)) for p in unsafe_prompts]
        ctx_neutral = [_to_cpu(text_encoder([p], base_device)) for p in neutral_prompts]
        ctx_null = _to_cpu(text_encoder([""], base_device))

    # Aggressively release T5 weights — they aren't needed after this point.
    del text_encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    def _ctx_to(ctx, dev):
        return [t.to(dev, non_blocking=True) for t in ctx]

    target_shape = compute_target_shape(
        z_dim=16,
        vae_stride=cfg.vae_stride,
        size=(width, height),
        frame_num=args.frame_num,
    )
    seq_len = compute_seq_len(target_shape, cfg.patch_size, sp_size=1)
    T_lat = target_shape[1]
    logging.info(f"target_shape={target_shape} seq_len={seq_len} T_lat={T_lat}")

    timestep_high = args.timestep_high or (args.num_inference_steps - 1)

    pbar = tqdm(range(args.iterations), desc=f"LoRA-{args.concept}")
    losses = []
    for it in pbar:
        optim.zero_grad(set_to_none=True)
        run_till = random.randint(args.timestep_low, timestep_high)
        seed = random.randint(0, 2**31 - 1)
        c_u = random.choice(ctx_unsafe)
        c_n = random.choice(ctx_neutral)

        # Move sampled contexts to base/new devices on demand
        c_u_base = _ctx_to(c_u, base_device)
        c_n_base = _ctx_to(c_n, base_device)
        ctx_null_base = _ctx_to(ctx_null, base_device)

        xt_base, t_at_base = partial_denoise(
            base,
            target_shape=target_shape,
            context_cond=c_u_base,
            context_uncond=ctx_null_base,
            guide_scale=args.guide_scale,
            device=base_device,
            seed=seed,
            dtype=dtype,
            seq_len=seq_len,
            run_till_step=run_till,
            num_train_timesteps=cfg.num_train_timesteps,
            num_inference_steps=args.num_inference_steps,
            shift=args.shift,
        )

        # 3 frozen forwards on base (no_grad → low memory pressure)
        with torch.no_grad(), amp.autocast(dtype=dtype):
            n_uncond = model_forward(base, xt_base, t_at_base, ctx_null_base, seq_len)
            n_cond_u = model_forward(base, xt_base, t_at_base, c_u_base, seq_len)
            n_old_n = model_forward(base, xt_base, t_at_base, c_n_base, seq_len)
        eps_target_u = (n_uncond - args.eta * (n_cond_u - n_uncond)).to(device)
        n_old_n_dev = n_old_n.to(device)

        # Move xt + timestep + contexts to trainable device for LoRA forward
        xt_new = xt_base.to(device)
        t_at_new = t_at_base.to(device)
        c_u_new = _ctx_to(c_u, device)
        c_n_new = _ctx_to(c_n, device)

        # 2 trainable forwards on new (LoRA-active)
        with amp.autocast(dtype=dtype):
            n_pred_u = model_forward_with_lora(
                new, xt_new, t_at_new, c_u_new, seq_len,
                target_shape=target_shape, patch_size=cfg.patch_size,
                num_train_timesteps=cfg.num_train_timesteps,
                concept_name=args.concept, gate_value=1.0,
            )
            n_pred_n = model_forward_with_lora(
                new, xt_new, t_at_new, c_n_new, seq_len,
                target_shape=target_shape, patch_size=cfg.patch_size,
                num_train_timesteps=cfg.num_train_timesteps,
                concept_name=args.concept, gate_value=1.0,
            )

        L_esd = standard_esd_loss(n_pred_u, eps_target_u)
        L_max = max_frame_leak_loss(n_pred_u, eps_target_u, k=args.top_k)
        L_mot = motion_preserve_loss(n_pred_n, n_old_n_dev)

        if args.no_frame_aware:
            L_smt = torch.tensor(0.0, device=device)
        else:
            # Temporal smoothness from concept φ at current timestep, T_lat frame slots
            phi = new.blocks[0].ffn.loras[args.concept].phi
            f_norm = torch.linspace(0, 1, T_lat, device=device, dtype=torch.float32)
            t_norm = torch.full_like(
                f_norm, float(t_at_new.item()) / cfg.num_train_timesteps
            )
            phi_per_frame = phi(f_norm, t_norm)
            L_smt = temporal_smooth_loss(phi_per_frame)

        L = (
            L_esd
            + args.alpha * L_max
            + args.beta * L_mot
            + args.gamma * L_smt
        )
        L.backward()
        optim.step()
        losses.append(L.item())

        if (it + 1) % 10 == 0:
            pbar.set_postfix(
                L=f"{L.item():.4f}",
                esd=f"{L_esd.item():.3f}",
                max=f"{L_max.item():.3f}",
                mot=f"{L_mot.item():.3f}",
                smt=f"{L_smt.item():.4f}",
            )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    safe_name = args.concept.replace(" ", "_")
    save_path = os.path.join(args.save_dir, f"lora_{safe_name}.safetensors")
    save_lora(args.concept, new, save_path)
    logging.info(f"Saved LoRA to {save_path}")
    logging.info(
        f"Final loss: {losses[-1]:.4f} | min: {min(losses):.4f}"
    )


if __name__ == "__main__":
    main()
