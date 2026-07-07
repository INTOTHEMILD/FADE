#!/usr/bin/env python
"""Generate one video with mounted concept LoRAs + soft routing."""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import wan
from tools.generate_with_t5_cache import WanT2VFromCache
from utils.wan_lora_utils import attach_lora_to_block, load_lora, load_routing_config
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.modules.gated_ffn import GatedFFN
from wan.modules.lora_router import compute_gates, pool_prompt_context
from wan.utils.utils import cache_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="t2v-1.3B")
    p.add_argument("--ckpt_dir", required=True, help="Phase 1 init dir")
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--lora_dir",
        required=True,
        help="dir holding routing_config.pt + lora_<concept>.safetensors",
    )
    p.add_argument("--out", default="/tmp/sample.mp4")
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--size", default="832*480")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--r0", type=int, default=4)
    p.add_argument("--r1", type=int, default=2)
    p.add_argument("--gate_override", type=float, default=None,
                   help="If set, override all per-concept gates with this scalar")
    p.add_argument("--ctx_cache", default=None,
                   help="Precomputed T5 context .pt; if set, skip T5 loading "
                        "(see tools/generate_with_t5_cache.py). Required for "
                        "safe 4-way parallel inference (B3 in ablation log).")
    p.add_argument("--concepts", default=None,
                   help="Comma-separated subset of concept names to mount. "
                        "If omitted, mounts every concept in routing_config. "
                        "Use for single-concept ablations to bypass router leakage.")
    p.add_argument("--blocks_min", type=int, default=0,
                   help="Mount LoRA only on blocks with index >= this.")
    p.add_argument("--blocks_max", type=int, default=None,
                   help="Mount LoRA only on blocks with index < this. "
                        "None = all blocks. Use to test shallow-only injection "
                        "(e.g. --blocks_max 15 for blocks 0-14).")

    # Time-Scheduled UCE (TS-UCE): blend the UCE-edited cross-attn K/V weights
    # back toward the original Wan weights as denoising progresses, so the
    # full-strength edit is only applied during the structure-defining early
    # timesteps. Eliminates late-timestep texture over-sharpening that arises
    # when static UCE keeps suppressing the K/V column space at fine-detail
    # denoising stages.
    p.add_argument("--ts_uce_orig_ckpt", default=None,
                   help="Path to vanilla Wan ckpt to read W_orig for cross-attn "
                        "K/V. When set, enables TS-UCE: each K/V Linear is "
                        "replaced by a wrapper that lerps between W_orig and "
                        "W_uce (the loaded ckpt) according to ts_alpha schedule.")
    p.add_argument("--ts_alpha_max", type=float, default=1.0,
                   help="Max blend weight on W_uce (full edit) at early "
                        "timesteps. 1.0 = exactly the static UCE edit.")
    p.add_argument("--ts_alpha_min", type=float, default=0.2,
                   help="Min blend weight on W_uce at the final denoising step. "
                        "0.0 → revert to W_orig at t=0; default 0.2 keeps a "
                        "light residual edit through texture refinement.")
    p.add_argument("--ts_split", type=float, default=0.4,
                   help="Normalized timestep below which alpha ramps from "
                        "alpha_max → alpha_min. With t∈[0,1] (1=most noisy), "
                        "t≥ts_split → alpha_max (structure phase); t<ts_split "
                        "→ linear ramp toward alpha_min (texture phase).")
    return p.parse_args()


def _ts_alpha_schedule(t_norm: float, alpha_max: float, alpha_min: float,
                       split: float) -> float:
    """Two-piece schedule on normalized denoising timestep (1=noisy → 0=clean).

    Structure phase  (t ≥ split): full edit (alpha_max).
    Texture phase    (t <  split): linear ramp from alpha_max toward alpha_min.
    """
    if t_norm >= split:
        return alpha_max
    if split <= 0:
        return alpha_min
    frac = max(0.0, t_norm) / split  # 0..1
    return alpha_min + (alpha_max - alpha_min) * frac


class TSLinear(torch.nn.Module):
    """Drop-in replacement for nn.Linear that lerps two weight matrices.

    Output = ((1−α)·W_orig + α·W_uce) · x + b, computed as activation blend
    for memory efficiency. α is set externally before each model forward.
    """

    def __init__(self, w_orig: torch.Tensor, w_uce: torch.Tensor,
                 bias: torch.Tensor | None):
        super().__init__()
        self.weight_orig = torch.nn.Parameter(w_orig, requires_grad=False)
        self.weight_uce = torch.nn.Parameter(w_uce, requires_grad=False)
        self.bias = (torch.nn.Parameter(bias, requires_grad=False)
                     if bias is not None else None)
        self.alpha = 1.0

    def forward(self, x):
        if self.alpha >= 1.0 - 1e-6:
            w = self.weight_uce
        elif self.alpha <= 1e-6:
            w = self.weight_orig
        else:
            w = self.alpha * self.weight_uce + (1.0 - self.alpha) * self.weight_orig
        return torch.nn.functional.linear(x, w, self.bias)


def _build_frame_idx_per_token(target_shape, patch_size, seq_len, device):
    _, T_lat, H_lat, W_lat = target_shape
    pT = T_lat // patch_size[0]
    pH = H_lat // patch_size[1]
    pW = W_lat // patch_size[2]
    n_per_frame = pH * pW
    n_valid = pT * n_per_frame
    f_idx = torch.zeros(seq_len, device=device, dtype=torch.long)
    f_idx[:n_valid] = torch.arange(n_valid, device=device) // n_per_frame
    return f_idx.float() / max(pT - 1, 1)


def _patch_model_for_frame_time(model, num_train_timesteps, patch_size,
                                ts_uce_schedule=None):
    """Wrap model.forward to set f_norm/t_norm on every GatedFFN before each call.

    If ts_uce_schedule is provided, also compute alpha = schedule(t_norm) and
    set it on every TSLinear module in the model — drives Time-Scheduled UCE.
    """
    original_forward = model.forward

    def patched(self, x, t=None, context=None, seq_len=None, **kw):
        if x and isinstance(x, list):
            x0 = x[0]
            target_shape = x0.shape  # [C, T_lat, H_lat, W_lat]
            f_norm = _build_frame_idx_per_token(
                target_shape, patch_size, seq_len, x0.device,
            )
            t_scalar = float(t.flatten()[0].item()) if torch.is_tensor(t) else float(t)
            t_norm = t_scalar / num_train_timesteps
            for blk in self.blocks:
                if isinstance(blk.ffn, GatedFFN):
                    blk.ffn.set_frame_time(f_norm, t_norm)
            if ts_uce_schedule is not None:
                alpha = ts_uce_schedule(t_norm)
                for m in self.modules():
                    if isinstance(m, TSLinear):
                        m.alpha = alpha
        return original_forward(x, t=t, context=context, seq_len=seq_len, **kw)

    model.forward = types.MethodType(patched, model)


def _install_ts_uce(model, orig_ckpt_dir: str, device: torch.device, dtype):
    """Replace each block.cross_attn.{k,v} Linear with TSLinear holding both
    the model's currently-loaded W_uce and the original-Wan W_orig.

    orig_ckpt_dir must be a vanilla Wan ckpt directory (no UCE). We read just
    the cross-attn K/V weights from its safetensors shard.
    """
    from safetensors import safe_open

    needed = set()
    for bi, _ in enumerate(model.blocks):
        needed.add(f"blocks.{bi}.cross_attn.k.weight")
        needed.add(f"blocks.{bi}.cross_attn.v.weight")

    found = {}
    shard = os.path.join(orig_ckpt_dir, "diffusion_pytorch_model.safetensors")
    if not os.path.exists(shard):
        raise FileNotFoundError(
            f"TS-UCE needs vanilla Wan diffusion_pytorch_model.safetensors at "
            f"{shard}; cannot read W_orig.")
    with safe_open(shard, framework="pt") as f:
        for key in f.keys():
            if key in needed:
                found[key] = f.get_tensor(key)
    missing = needed - set(found.keys())
    if missing:
        raise RuntimeError(f"TS-UCE: missing W_orig tensors {missing}")

    n_replaced = 0
    for bi, blk in enumerate(model.blocks):
        for proj in ("k", "v"):
            lin = getattr(blk.cross_attn, proj)
            w_uce = lin.weight.detach().to(device).to(dtype)
            key = f"blocks.{bi}.cross_attn.{proj}.weight"
            w_orig = found[key].to(device).to(dtype)
            if w_orig.shape != w_uce.shape:
                raise RuntimeError(
                    f"shape mismatch on {key}: orig={w_orig.shape} "
                    f"uce={w_uce.shape}")
            bias = lin.bias.detach().to(device).to(dtype) if lin.bias is not None else None
            ts = TSLinear(w_orig, w_uce, bias).to(device).to(dtype)
            setattr(blk.cross_attn, proj, ts)
            n_replaced += 1
    print(f"[ts-uce] replaced {n_replaced} cross-attn projections "
          f"with TSLinear (orig from {orig_ckpt_dir})")


def main():
    args = parse_args()
    cfg = WAN_CONFIGS[args.task]
    device = torch.device(f"cuda:{args.device_id}")

    # Pipeline (handles VAE, T5, model loading, scheduler).
    # When --ctx_cache is set, swap T5 for a disk-backed lookup so we don't
    # pay 12 GB host RAM per parallel worker.
    if args.ctx_cache:
        pipeline = WanT2VFromCache(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            ctx_cache_path=args.ctx_cache,
            device_id=args.device_id,
        )
    else:
        pipeline = wan.WanT2V(
            cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=args.device_id,
        )

    # Mount one LoRA per block per concept; load saved weights
    routing = load_routing_config(
        os.path.join(args.lora_dir, "routing_config.pt"),
        device=device,
    )
    all_concepts = list(routing["anchors"].keys())
    if args.concepts:
        wanted = [c.strip() for c in args.concepts.split(",")]
        active_concepts = wanted[:]
        missing = [c for c in wanted if c not in all_concepts]
        if missing:
            if args.gate_override is None:
                raise ValueError(
                    f"--concepts has unknown names: {missing}; routing config "
                    f"has {all_concepts}. Pass --gate_override to bypass router."
                )
            # gate_override path: synthesize a placeholder anchor so the
            # downstream router code runs without the concept being calibrated.
            # The placeholder is never used since gate_override forces the gate.
            d_anchor = next(iter(routing["anchors"].values())).shape[-1] if routing["anchors"] else 4096
            for m in missing:
                routing["anchors"][m] = torch.zeros(1, d_anchor, device=device)
                routing["tau"][m] = 0.0
                routing["T"][m] = 1.0
        # Drop concepts not requested so the router doesn't see them.
        routing["anchors"] = {c: routing["anchors"][c] for c in active_concepts}
    else:
        active_concepts = all_concepts

    n_blocks = len(pipeline.model.blocks)
    bmin = args.blocks_min
    bmax = args.blocks_max if args.blocks_max is not None else n_blocks
    print(f"[infer] mounting concepts={active_concepts} on blocks [{bmin}, {bmax})")

    d = pipeline.model.dim
    for c_name in active_concepts:
        for bi, blk in enumerate(pipeline.model.blocks):
            if bmin <= bi < bmax:
                attach_lora_to_block(
                    blk, c_name, d_in=d, d_out=d, r0=args.r0, r1=args.r1,
                )
    # Move newly-created LoRA modules to base device + dtype
    base_dtype = next(pipeline.model.parameters()).dtype
    pipeline.model.to(device).to(base_dtype)

    for c_name in active_concepts:
        safe = c_name.replace(" ", "_")
        load_lora(
            c_name, pipeline.model,
            os.path.join(args.lora_dir, f"lora_{safe}.safetensors"),
        )

    # Compute prompt-conditioned routing gates.
    # CachedT5 has model=None; only call .to/.cpu when a real T5 module exists.
    has_real_t5 = getattr(pipeline.text_encoder, "model", None) is not None
    if has_real_t5:
        pipeline.text_encoder.model.to(device)
    h_p = pool_prompt_context(
        pipeline.text_encoder, pipeline.model.text_embedding,
        args.prompt, device,
    )
    if has_real_t5:
        pipeline.text_encoder.model.cpu()
    torch.cuda.empty_cache()

    if args.gate_override is not None:
        gates = {c: args.gate_override for c in routing["anchors"]}
    else:
        gates = compute_gates(h_p, routing["anchors"], routing["tau"], routing["T"])
    print(f"[infer] gates: {gates}")

    for blk in pipeline.model.blocks:
        if isinstance(blk.ffn, GatedFFN):
            blk.ffn.set_gates(gates)

    # Install TS-UCE wrappers on cross-attn K/V if requested.
    ts_uce_schedule = None
    if args.ts_uce_orig_ckpt:
        _install_ts_uce(
            pipeline.model,
            args.ts_uce_orig_ckpt,
            device=device,
            dtype=base_dtype,
        )
        ts_uce_schedule = lambda t_norm: _ts_alpha_schedule(
            t_norm, args.ts_alpha_max, args.ts_alpha_min, args.ts_split,
        )
        print(f"[ts-uce] schedule alpha_max={args.ts_alpha_max} "
              f"alpha_min={args.ts_alpha_min} split={args.ts_split}")

    _patch_model_for_frame_time(
        pipeline.model,
        num_train_timesteps=cfg.num_train_timesteps,
        patch_size=cfg.patch_size,
        ts_uce_schedule=ts_uce_schedule,
    )

    # Generate
    width, height = SIZE_CONFIGS[args.size]
    video = pipeline.generate(
        args.prompt,
        size=(width, height),
        frame_num=args.frame_num,
        shift=args.shift,
        sampling_steps=args.num_inference_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cache_video(
        tensor=video[None],
        save_file=args.out,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"[infer] saved {args.out}")


if __name__ == "__main__":
    main()
