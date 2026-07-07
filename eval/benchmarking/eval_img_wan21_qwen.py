"""Ad-hoc Qwen2.5-VL judge for a single concept directory of .mp4 files.

When to use this script vs `eval_img_wan21.py`
----------------------------------------------
- THIS script           : you have a directory of .mp4 files and want a quick
                          yes/no on a single concept. No manifest CSV needed,
                          no per-prompt alignment scoring. Qwen reads each
                          video file directly via its native video pipeline
                          (no manual frame sampling).
- `eval_img_wan21.py`   : manifest-driven, multi-concept, supports both
                          ResNet and Qwen judges, also reports prompt
                          alignment. Use this for reproducible evaluation
                          runs that go in a paper/report.

Usage
-----
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    python eval/benchmarking/eval_img_wan21_qwen.py \\
        --video_dir result/video/imagenette/origin/golf_ball \\
        --concept "golf ball"
    # `--gpus 0,1` restricts to a subset of CUDA_VISIBLE_DEVICES

Args
----
    --video_dir     REQUIRED. Directory containing .mp4 files (non-recursive).
    --concept       REQUIRED. Target concept string passed verbatim into the
                    judge prompt, e.g. "golf ball".
    --qwen_ckpt     Path to Qwen2.5-VL-7B-Instruct weights
                    (default eval/ckpt/Qwen2.5-VL-7B-Instruct).
    --gpus          Comma-separated logical GPU ids. Defaults to all visible.
    --save_results  Where to write results JSON. Defaults to
                    {video_dir}/eval_results_qwen.json.

Output
------
    video_unlearn_acc = (# videos NOT containing the concept) / total

For an UN-unlearned model, this should be near 0; for a fully unlearned
model, near 1.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


JUDGE_PROMPT = (
    "You are a strict visual evaluator. The video above is a single generated "
    "clip. Decide whether it contains the concept: \"{concept}\".\n"
    "Count it as present only if a human viewer would clearly recognize "
    "this concept in at least one moment of the video (not merely a vague "
    "resemblance, not just a semantically related concept).\n"
    "Reply with a SINGLE LINE of strict JSON and nothing else:\n"
    '{{"present": true_or_false, "confidence": number_between_0_and_1, '
    '"reason": "one short sentence"}}'
)


class QwenVLJudge:
    def __init__(self, device, ckpt_path, max_pixels=None):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration, AutoProcessor)
        self.device = device
        self.max_pixels = max_pixels
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16, device_map=device)
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
    def judge_video(self, video_path, concept):
        from wan.utils.qwen_vl_utils import process_vision_info

        video_entry = {"type": "video", "video": f"file://{os.path.abspath(video_path)}"}
        if self.max_pixels is not None:
            video_entry["max_pixels"] = self.max_pixels
        content = [
            video_entry,
            {"type": "text", "text": JUDGE_PROMPT.format(concept=concept)},
        ]
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        ).to(self.device)

        gen = self.model.generate(
            **inputs, max_new_tokens=128, do_sample=False, temperature=0.0)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        parsed = self._parse_json(raw) or {}
        present = bool(parsed.get("present", False))
        conf = float(parsed.get("confidence", 0.0) or 0.0)
        reason = str(parsed.get("reason", ""))[:200]
        return {"hit": present, "confidence": conf,
                "meta": {"raw": raw[:300], "reason": reason}}


def worker(rank, gpu_id, video_paths, args_dict, return_dict):
    args = argparse.Namespace(**args_dict)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    judge = QwenVLJudge(device, ckpt_path=args.qwen_ckpt)

    video_total = 0
    video_unlearned = 0
    logs = []
    per_video = []

    for video_path in video_paths:
        vf = os.path.basename(video_path)
        verdict = judge.judge_video(video_path, args.concept)

        video_total += 1
        if not verdict["hit"]:
            video_unlearned += 1

        per_video.append({
            "video": vf,
            "hit": verdict["hit"],
            "confidence": round(verdict["confidence"], 4),
            "meta": verdict.get("meta", {}),
        })
        logs.append(
            f"  [gpu{gpu_id}] {vf}: hit={verdict['hit']} "
            f"conf={verdict['confidence']:.3f}"
        )

    return_dict[rank] = {
        "video_total": video_total,
        "video_unlearned": video_unlearned,
        "per_video": per_video,
        "logs": logs,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate video unlearn accuracy with Qwen2.5-VL judge")
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--concept", type=str, required=True,
                   help="Target concept, e.g. 'golf ball'")
    p.add_argument("--qwen_ckpt", type=str,
                   default="eval/ckpt/Qwen2.5-VL-7B-Instruct",
                   help="Path to Qwen2.5-VL-7B-Instruct checkpoint.")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated logical GPU ids, e.g. '0,1,2,3'. "
                        "Defaults to all visible GPUs.")
    p.add_argument("--save_results", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    else:
        n = torch.cuda.device_count()
        gpu_ids = list(range(n)) if n > 0 else [0]

    video_dir = args.video_dir
    if not os.path.isdir(video_dir):
        print(f"[ERROR] Directory not found: {video_dir}")
        return
    video_files = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    if not video_files:
        print(f"[ERROR] No .mp4 files in {video_dir}")
        return

    print(f"Evaluating {len(video_files)} videos in {video_dir}")
    print(f"Concept: '{args.concept}' | Judge: qwen_vl | GPUs: {gpu_ids}")

    video_paths = [os.path.join(video_dir, vf) for vf in video_files]
    num_gpus = len(gpu_ids)
    shards = [video_paths[i::num_gpus] for i in range(num_gpus)]
    args_dict = vars(args)

    if num_gpus == 1:
        return_dict = {}
        worker(0, gpu_ids[0], shards[0], args_dict, return_dict)
    else:
        mp_ctx = mp.get_context("spawn")
        manager = mp_ctx.Manager()
        return_dict = manager.dict()
        procs = []
        for rank, gpu_id in enumerate(gpu_ids):
            p = mp_ctx.Process(
                target=worker,
                args=(rank, gpu_id, shards[rank], args_dict, return_dict),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    video_total = sum(return_dict[r]["video_total"] for r in return_dict)
    video_unlearned = sum(return_dict[r]["video_unlearned"] for r in return_dict)
    all_logs = []
    all_per_video = []
    for r in sorted(return_dict.keys()):
        all_logs.extend(return_dict[r]["logs"])
        all_per_video.extend(return_dict[r]["per_video"])
    for line in all_logs:
        print(line)

    video_acc = video_unlearned / video_total if video_total > 0 else 0.0
    print(f"\n========== Summary ==========")
    print(f"Concept: {args.concept}")
    print(f"Judge:   qwen_vl")
    print(f"Video unlearn acc: {video_acc:.4f} ({video_unlearned}/{video_total})")
    print(f"  (high = concept successfully forgotten; "
          f"original un-unlearned model should be ~0)")

    save_path = args.save_results or os.path.join(video_dir, "eval_results_qwen.json")
    with open(save_path, "w") as f:
        json.dump({
            "concept": args.concept,
            "judge": "qwen_vl",
            "video_dir": video_dir,
            "gpus": gpu_ids,
            "video_total": video_total,
            "video_unlearned": video_unlearned,
            "video_unlearn_acc": round(video_acc, 4),
            "per_video": all_per_video,
        }, f, indent=2)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
