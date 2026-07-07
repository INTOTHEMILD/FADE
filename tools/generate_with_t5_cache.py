#!/usr/bin/env python
"""Drop-in replacement for `python generate.py` that skips T5 entirely.

Reads precomputed T5 contexts from `cache/t5_contexts.pt` (built by
`precompute_t5_contexts.py`) and only loads VAE + WanModel on the GPU.
RAM footprint per worker shrinks from ~25 GB to ~6 GB, so 4 parallel
workers fit on a 141 GB host instead of swapping/OOMing.

CLI mirrors the subset of `generate.py` flags used by the eval harness
(--ckpt_dir, --prompt, --frame_num, --size, --base_seed, --save_file).
LoRA / multi-GPU are not supported here; for those use
`tools/inference.py` or full `generate.py`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan.configs import WAN_CONFIGS
from wan.text2video import WanT2V
from wan.utils.utils import cache_video


class _CachedT5:
    """Drop-in shim for `T5EncoderModel`: returns precomputed contexts.

    `WanT2V.generate()` does:
        context = self.text_encoder([prompt], cpu_device)
        context = [t.to(self.device) for t in context]
    so we mirror that signature: list-in, list-out, returned tensors on cpu.
    """

    def __init__(self, cache_path):
        self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        # Empty `model` attr — `generate()` only touches it when t5_cpu=False.
        self.model = None

    def __call__(self, prompts, device):
        out = []
        for p in prompts:
            if p not in self.cache:
                raise KeyError(
                    f"prompt not in T5 cache: {p!r} — re-run "
                    f"precompute_t5_contexts.py with the right CSVs"
                )
            t = self.cache[p]
            out.append(t.to(device) if device is not None else t)
        return out


class WanT2VFromCache(WanT2V):
    """Subclass that swaps T5 for a cache lookup. Skips the T5 download path
    entirely so we never instantiate the 12 GB encoder weights.
    """

    def __init__(self, config, checkpoint_dir, ctx_cache_path, device_id=0):
        from wan.modules.model import WanModel
        from wan.modules.vae import WanVAE

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = 0
        # `True` keeps generate() on the cpu→device path, where text_encoder
        # is called with cpu device; that branch never references .model.to().
        self.t5_cpu = True

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.text_encoder = _CachedT5(ctx_cache_path)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,
        )

        logging.info(f"Creating WanModel from {checkpoint_dir} (no T5)")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False).to(self.device)

        self.sp_size = 1
        self.sample_neg_prompt = config.sample_neg_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--task", default="t2v-1.3B")
    p.add_argument("--ctx_cache", default="cache/t5_contexts.pt")
    p.add_argument("--prompt", required=True)
    p.add_argument("--save_file", required=True)
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--size", default="832*480")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--sample_shift", type=float, default=8.0)
    p.add_argument("--sample_guide_scale", type=float, default=6.0)
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--device_id", type=int, default=0)
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()
    cfg = WAN_CONFIGS[args.task]
    w, h = (int(x) for x in args.size.split("*"))

    pipe = WanT2VFromCache(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        ctx_cache_path=args.ctx_cache,
        device_id=args.device_id,
    )

    video = pipe.generate(
        input_prompt=args.prompt,
        size=(w, h),
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sampling_steps=args.sampling_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=False,
    )

    Path(args.save_file).parent.mkdir(parents=True, exist_ok=True)
    cache_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logging.info(f"saved {args.save_file}")


if __name__ == "__main__":
    main()
