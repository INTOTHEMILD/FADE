#!/usr/bin/env python
"""Encode all eval prompts (positive + negative) with T5 once, save to disk.

Each parallel inference worker would otherwise re-load T5 (~12 GB RAM each),
overflowing host memory at 4-way parallelism. This pass runs T5 once on a
single GPU, dumps {prompt_str: [L,4096] cpu tensor} to a .pt cache, and
inference workers read from the cache without ever instantiating T5.

Usage:
    python tools/precompute_t5_contexts.py \
        --ckpt_dir ckpt/Wan2.1-T2V-1.3B \
        --csv prompts/object/spaniel_unsafe.csv prompts/object/parachute_unsafe.csv ... \
        --out cache/t5_contexts.pt
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan.configs import WAN_CONFIGS
from wan.modules.t5 import T5EncoderModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="Original Wan2.1 ckpt dir (T5 + tokenizer live here)")
    p.add_argument("--task", default="t2v-1.3B")
    p.add_argument("--csv", nargs="+", required=True,
                   help="One or more prompt CSVs with a 'prompt' column")
    p.add_argument("--out", default="cache/t5_contexts.pt")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = WAN_CONFIGS[args.task]
    device = torch.device(args.device)

    prompts = {""}
    prompts.add(cfg.sample_neg_prompt)
    for path in args.csv:
        df = pd.read_csv(path)
        for p in df["prompt"].tolist():
            prompts.add(str(p))
    prompts = sorted(prompts)
    print(f"[t5-cache] encoding {len(prompts)} unique prompts")

    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
    )
    text_encoder.model.to(device).eval()

    cache = {}
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ctx = text_encoder([p], device)[0]
            cache[p] = ctx.cpu()
            if (i + 1) % 25 == 0 or i == 0:
                print(f"[t5-cache] {i+1}/{len(prompts)}  L={ctx.shape[0]}  '{p[:50]}...'")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[t5-cache] saved {len(cache)} entries to {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
