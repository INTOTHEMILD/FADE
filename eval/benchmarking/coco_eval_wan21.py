"""COCO CLIP-similarity evaluation for Wan2.1 generated videos.

Companion to `coco_generator_wan21.py`. **Eval-only**: does not generate any
video. Prompt alignment is measured as CLIP cosine similarity between
sampled frames and the original text prompt, averaged to a per-video score
and finally to a dataset-level mean — mirroring the SD-side
`sd/eval_coco_sd.py` contract but adapted to video.

Expected layout (matches the generator):

    {video_root}/{case_name}/{case_number}_seed{evaluation_seed}.mp4
    {video_root}/{case_name}/manifest.csv   (preferred manifest source)

Manifest resolution order (first match wins):
    1. `--manifest_csv` if explicitly passed.
    2. `{video_root}/{case_name}/manifest.csv` (written by the generator).
    3. Global fallback `eval/dataset/coco_30k.csv`.

Only the first `--limit` rows (if set) are scored.

Frame sampling is done by the shared `_common.load_video_frames`, which is
identical to what `eval_img_wan21.py` and `nudity_eval_wan21.py` use — that
way CLIP-over-video numbers live in the same "sampled-frame space" as the
concept-unlearn and nudity numbers and remain cross-comparable.

Typical usage
-------------
    # Score 500 videos from a generated case, 4 GPUs:
    python eval/benchmarking/coco_eval_wan21.py \\
        --case_name origin_coco --limit 500 --gpus 0,1,2,3

    # Custom CLIP backbone:
    python eval/benchmarking/coco_eval_wan21.py \\
        --case_name origin_coco \\
        --clip_model openai/clip-vit-large-patch14

Outputs (under `--save_dir`, defaults to
`{video_root}/{case_name}/eval_clip/`):
    config.json        Resolved CLI args.
    summary.json       Aggregate counters + mean CLIP score.
    per_video.jsonl    One line per video (score + per-frame scores).
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp

from _common import (
    load_flat_manifest,
    load_video_frames,
    resolve_video_path,
)


DEFAULT_VIDEO_ROOT = "result/video/coco"
DEFAULT_FALLBACK_MANIFEST = "eval/dataset/coco_30k.csv"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class CLIPVideoScorer:
    """Frame-wise CLIP cosine similarity, averaged to one score per video.

    Per-video score = mean over sampled frames of cos(image_emb, text_emb).
    This mirrors the SD-side image-level CLIP score ( `eval_coco_sd.py`) but
    lifted to a video by averaging across frames. We intentionally do NOT
    take the max / min because COCO alignment is about typical scene content,
    not peak matching.
    """

    def __init__(self, device, clip_model_id):
        # Lazy imports so the main process can spin up even without CLIP
        # deps available, matching the pattern used by Qwen in
        # `eval_img_wan21.py`.
        from transformers import AutoProcessor, AutoTokenizer, CLIPModel

        self.device = device
        self.model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(clip_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model_id)

    @torch.no_grad()
    def score(self, frames_uint8, prompt):
        # `frames_uint8` is (N, H, W, 3); pass PIL images to the HF processor
        # so we reuse its resize/normalize pipeline unchanged.
        from PIL import Image

        frames_pil = [Image.fromarray(f.numpy()) for f in frames_uint8]
        image_inputs = self.processor(images=frames_pil, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_inputs = self.tokenizer(
            [prompt], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        sims = (image_features @ text_features.T).squeeze(1)  # (N,)
        per_frame = [float(s) for s in sims.tolist()]
        mean_sim = float(sims.mean().item())
        return {
            "status": "ok",
            "video_score": mean_sim,
            "per_frame_scores": per_frame,
        }


def resolve_manifest(args):
    """Pick the manifest source, preferring run-local over the global CSV."""
    if args.manifest_csv:
        return Path(args.manifest_csv)
    per_run = Path(args.video_root) / args.case_name / "manifest.csv"
    if per_run.is_file():
        return per_run
    return Path(DEFAULT_FALLBACK_MANIFEST)


def build_eval_items(args):
    manifest = resolve_manifest(args)
    records = load_flat_manifest(manifest, limit=args.limit)
    items = []
    for record in records:
        video_path = resolve_video_path(
            record,
            video_root=Path(args.video_root),
            case_name=args.case_name,
            path_template=args.path_template,
        )
        item = dict(record)
        item["video_path"] = str(video_path)
        item["video_exists"] = video_path.is_file()
        items.append(item)
    return items, manifest


def worker(rank, gpu_id, items, args_dict, return_dict):
    """Score one shard on one GPU."""
    args = argparse.Namespace(**args_dict)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    scorer = CLIPVideoScorer(device, args.clip_model)

    logs = []
    per_video = []
    counters = {
        "expected_total": len(items),
        "found_total": 0,
        "evaluated_total": 0,
        "missing_total": 0,
        "decode_error_total": 0,
        "scorer_error_total": 0,
    }
    score_sum = 0.0
    score_count = 0

    for item in items:
        result = {
            "case_number": item["case_number"],
            "evaluation_seed": item["evaluation_seed"],
            "prompt": item["prompt"],
            "video_path": item["video_path"],
        }

        if not item["video_exists"]:
            counters["missing_total"] += 1
            result.update({"status": "missing_video", "video_score": None, "per_frame_scores": []})
            per_video.append(result)
            logs.append(f"[gpu{gpu_id}] missing {item['video_path']}")
            continue

        counters["found_total"] += 1
        try:
            frames = load_video_frames(
                item["video_path"],
                sample_mode=args.frame_sample_mode,
                stride=args.frame_stride,
                max_frames=args.max_frames,
            )
        except Exception as exc:
            counters["decode_error_total"] += 1
            result.update(
                {"status": "decode_error", "video_score": None, "per_frame_scores": [], "error": str(exc)}
            )
            per_video.append(result)
            logs.append(f"[gpu{gpu_id}] decode_error {item['video_path']}: {exc}")
            continue

        try:
            verdict = scorer.score(frames, item["prompt"])
        except Exception as exc:
            counters["scorer_error_total"] += 1
            result.update(
                {"status": "scorer_error", "video_score": None, "per_frame_scores": [], "error": str(exc)}
            )
            per_video.append(result)
            logs.append(f"[gpu{gpu_id}] scorer_error {item['video_path']}: {exc}")
            continue

        counters["evaluated_total"] += 1
        score_sum += verdict["video_score"]
        score_count += 1
        result.update(
            {
                "status": verdict["status"],
                "video_score": verdict["video_score"],
                "per_frame_scores": verdict["per_frame_scores"],
            }
        )
        per_video.append(result)
        logs.append(
            f"[gpu{gpu_id}] case={item['case_number']} score={verdict['video_score']:.4f} "
            f"frames={len(verdict['per_frame_scores'])}"
        )

    return_dict[rank] = {
        "counters": counters,
        "per_video": per_video,
        "logs": logs,
        "score_sum": score_sum,
        "score_count": score_count,
    }


def aggregate_results(return_dict):
    all_logs = []
    all_per_video = []
    counters = defaultdict(int)
    score_sum = 0.0
    score_count = 0

    for rank in sorted(return_dict.keys()):
        payload = return_dict[rank]
        all_logs.extend(payload["logs"])
        all_per_video.extend(payload["per_video"])
        for key, value in payload["counters"].items():
            counters[key] += value
        score_sum += payload["score_sum"]
        score_count += payload["score_count"]

    return counters, all_logs, all_per_video, score_sum, score_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="COCO CLIP similarity evaluation for Wan2.1 videos."
    )
    parser.add_argument("--manifest_csv", type=str, default=None,
                        help="Prompt CSV. Defaults to {video_root}/{case_name}/manifest.csv, "
                             "then eval/dataset/coco_30k.csv if that is missing.")
    parser.add_argument("--video_root", type=str, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of manifest rows to evaluate.")
    parser.add_argument(
        "--path_template",
        type=str,
        default="{video_root}/{case_name}/{case_number}_seed{evaluation_seed}.mp4",
        help="Filename template used to resolve each CSV row to a video path.",
    )
    parser.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--frame_sample_mode", type=str, default="uniform",
                        choices=["uniform", "stride", "headtail"])
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated CUDA device indices. Defaults to all visible GPUs.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Defaults to {video_root}/{case_name}/eval_clip.")
    return parser.parse_args()


def main():
    args = parse_args()
    items, manifest = build_eval_items(args)
    if not items:
        raise SystemExit("No manifest rows matched the requested filters.")

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    else:
        count = torch.cuda.device_count()
        gpu_ids = list(range(count)) if count > 0 else [0]

    save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else Path(args.video_root) / args.case_name / "eval_clip"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest        : {manifest}")
    print(f"Evaluating rows : {len(items)}")
    print(f"Case            : {args.case_name}")
    print(f"CLIP model      : {args.clip_model}")
    print(f"GPUs            : {gpu_ids}")

    shards = [items[i::len(gpu_ids)] for i in range(len(gpu_ids))]
    args_dict = vars(args)

    if len(gpu_ids) == 1:
        return_dict = {}
        worker(0, gpu_ids[0], shards[0], args_dict, return_dict)
    else:
        mp_ctx = mp.get_context("spawn")
        manager = mp_ctx.Manager()
        return_dict = manager.dict()
        procs = []
        for rank, gpu_id in enumerate(gpu_ids):
            proc = mp_ctx.Process(
                target=worker,
                args=(rank, gpu_id, shards[rank], args_dict, return_dict),
            )
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()

    counters, logs, per_video, score_sum, score_count = aggregate_results(return_dict)
    for line in logs:
        print(line)

    average_similarity = score_sum / score_count if score_count else None

    summary = {
        "manifest_csv": str(manifest),
        "video_root": args.video_root,
        "case_name": args.case_name,
        "clip_model": args.clip_model,
        "gpus": gpu_ids,
        "frame_sample_mode": args.frame_sample_mode,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "counters": dict(counters),
        "average_similarity": average_similarity,
        "evaluated_total": score_count,
    }

    with open(save_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(args_dict, handle, indent=2, ensure_ascii=False)
    with open(save_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(save_dir / "per_video.jsonl", "w", encoding="utf-8") as handle:
        for row in per_video:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n========== Summary ==========")
    print(f"Evaluated videos    : {score_count}")
    print(
        f"Average CLIP sim    : "
        f"{average_similarity if average_similarity is not None else 'n/a'}"
    )
    print(f"Missing videos      : {counters['missing_total']}")
    print(f"Decode errors       : {counters['decode_error_total']}")
    print(f"Scorer errors       : {counters['scorer_error_total']}")
    print(f"Results saved to    : {save_dir}")


if __name__ == "__main__":
    main()
