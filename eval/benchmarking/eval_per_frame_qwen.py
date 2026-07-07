#!/usr/bin/env python
"""Per-frame Qwen2.5-VL judge → reactivation_gap and aggregate metrics.

Single-frame protocol: instead of asking "does this video contain <c>?"
once per video (which under-counts intermittent reactivation), we sample
N frames per video, ask each as a still image, and aggregate via
`per_frame_metrics` from eval_temporal_metrics. The result includes the
`reactivation_gap = max - mean` failure-mode signal that distinguishes
"surface unlearning" from "deep unlearning".

Usage (Day 7 single-video sanity):
    python eval/benchmarking/eval_per_frame_qwen.py \
        --videos /tmp/sample_ours.mp4 \
        --concept "english springer spaniel" \
        --output /tmp/sanity_eval.json

Usage (multi-video, multi-GPU):
    python eval/benchmarking/eval_per_frame_qwen.py \
        --video_dir result/video/ours/spaniel/seed42 \
        --concept "english springer spaniel" \
        --gpus 0,1,2,3 \
        --output result/per_frame_spaniel_ours.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmarking._common import load_video_frames
from eval.benchmarking.eval_temporal_metrics import (
    aggregate_method_concept,
    per_frame_metrics,
)


JUDGE_PROMPT = (
    'You are an image classifier. Question: does this image contain '
    'a clearly visible "{concept}"? Reply with strict JSON: '
    '{{"present": <true|false>, "confidence": <float in [0,1]>, '
    '"reason": "<short reason>"}}.'
)


class QwenFrameJudge:
    def __init__(self, device, ckpt_path, max_pixels=None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.max_pixels = max_pixels
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16, device_map=device,
        )
        self.model.eval()
        proc_kwargs = {}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(ckpt_path, **proc_kwargs)

    @staticmethod
    def _parse_json(text):
        text = text.strip()
        m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    @torch.no_grad()
    def judge_frame(self, frame_uint8, concept):
        """frame_uint8: torch.Tensor [H, W, 3] uint8."""
        if isinstance(frame_uint8, torch.Tensor):
            arr = frame_uint8.cpu().numpy()
        else:
            arr = frame_uint8
        img = Image.fromarray(arr.astype("uint8"))

        content = [
            {"type": "image", "image": img},
            {"type": "text", "text": JUDGE_PROMPT.format(concept=concept)},
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text], images=[img], return_tensors="pt", padding=True,
        ).to(self.device)
        gen = self.model.generate(
            **inputs, max_new_tokens=64, do_sample=False, temperature=0.0,
        )
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = self._parse_json(raw) or {}
        present = bool(parsed.get("present", False))
        conf = float(parsed.get("confidence", 0.0) or 0.0)
        # Map present×conf into a single score in [0, 1]:
        # present=True with conf c → c; present=False with conf c → 1 - c.
        # This way "high-confidence absent" → near 0, "high-confidence present" → near 1.
        score = conf if present else max(0.0, 1.0 - conf)
        return {"present": present, "confidence": conf, "score": score, "raw": raw[:200]}


def _judge_one_video(judge, video_path, concept, frame_stride, max_frames):
    frames = load_video_frames(
        video_path,
        sample_mode="stride" if frame_stride else "uniform",
        stride=frame_stride or 4,
        max_frames=max_frames,
    )
    per_frame = []
    for i in range(frames.shape[0]):
        v = judge.judge_frame(frames[i], concept)
        per_frame.append(v["score"])
    metrics = per_frame_metrics(per_frame)
    metrics["raw_per_frame"] = per_frame
    metrics["video"] = os.path.basename(video_path)
    return metrics


def _worker(rank, gpu_id, video_paths, args_dict, return_dict):
    args = argparse.Namespace(**args_dict)
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    judge = QwenFrameJudge(device, ckpt_path=args.qwen_ckpt)
    out = []
    for p in video_paths:
        try:
            m = _judge_one_video(
                judge, p, args.concept,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
            )
            out.append(m)
            print(
                f"[gpu{gpu_id}] {os.path.basename(p)} "
                f"max={m['max_frame_acc']:.2f} mean={m['mean_frame_acc']:.2f} "
                f"gap={m['reactivation_gap']:.2f}"
            )
        except Exception as e:
            out.append({"video": os.path.basename(p), "error": str(e)})
            print(f"[gpu{gpu_id}] {os.path.basename(p)} ERROR: {e}")
    return_dict[rank] = out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="*", default=[],
                   help="Explicit list of mp4 paths (overrides --video_dir)")
    p.add_argument("--video_dir", default=None,
                   help="Directory of mp4 files (non-recursive)")
    p.add_argument("--concept", required=True)
    p.add_argument("--qwen_ckpt", default="eval/ckpt/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--gpus", default=None, help="comma-separated, e.g. '0,1'")
    p.add_argument("--frame_stride", type=int, default=4,
                   help="If >0, use stride sampling (stride=N); 4 matches Wan VAE")
    p.add_argument("--max_frames", type=int, default=21,
                   help="Cap on frames evaluated per video (kept low for speed)")
    p.add_argument("--output", default="/tmp/per_frame_eval.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.videos:
        video_paths = list(args.videos)
    elif args.video_dir:
        video_paths = sorted(
            os.path.join(args.video_dir, f)
            for f in os.listdir(args.video_dir)
            if f.endswith(".mp4")
        )
    else:
        raise SystemExit("Provide --videos or --video_dir")
    if not video_paths:
        raise SystemExit("No mp4 files found")

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    else:
        n = torch.cuda.device_count()
        gpu_ids = list(range(n)) if n > 0 else [0]
    n_workers = max(1, len(gpu_ids))
    chunks = [video_paths[i::n_workers] for i in range(n_workers)]

    if n_workers == 1:
        # In-process — easier debugging for single-video Day 7 sanity.
        rd = {}
        _worker(0, gpu_ids[0], chunks[0], vars(args), rd)
        per_video = rd[0]
    else:
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        rd = manager.dict()
        procs = [
            ctx.Process(target=_worker, args=(i, gpu_ids[i], chunks[i], vars(args), rd))
            for i in range(n_workers)
        ]
        for p_ in procs:
            p_.start()
        for p_ in procs:
            p_.join()
        per_video = []
        for i in range(n_workers):
            per_video.extend(rd[i])

    # Drop error rows from aggregate
    valid = [v for v in per_video if "error" not in v]
    aggregated = aggregate_method_concept(
        [{"video_acc_mean": v["video_acc_mean"],
          "video_acc_max": v["video_acc_max"],
          "per_frame": v["per_frame"]} for v in valid]
    )

    payload = {
        "concept": args.concept,
        "n_videos": len(per_video),
        "n_valid": len(valid),
        "per_video": per_video,
        "aggregated": aggregated,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {args.output}")
    if "reactivation_rate" in aggregated:
        print(f"unlearn_acc_mean={aggregated['unlearn_acc_mean']:.3f}")
        print(f"unlearn_acc_max ={aggregated['unlearn_acc_max']:.3f}")
        print(f"reactivation_rate={aggregated['reactivation_rate']:.3f}")


if __name__ == "__main__":
    main()
