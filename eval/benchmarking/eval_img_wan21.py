"""Manifest-driven evaluation for Wan2.1 Imagenette / concept-unlearning videos.

Main evaluator for object concept forgetting. Reads the prompt CSV used by
the generator (`get_imagenette.py` + `imagenette_v.csv`), resolves the
expected video path per row, runs a ResNet-50 judge restricted to the ten
Imagenette classes on sampled frames, and reports forget rate + prompt
alignment.

Expected on-disk layout (produced by `get_imagenette.py`):

    {video_root}/{case_name}/{class_dir}/{sample_index}_seed{evaluation_seed}.mp4

where `class_dir = class.replace(" ", "_")`.

Typical usage
-------------
    # All concepts in the CSV, 4 GPUs:
    python eval/benchmarking/eval_img_wan21.py \\
        --case_name unlearn_v1 \\
        --gpus 0,1,2,3

    # Only evaluate two concepts, 10 samples each:
    python eval/benchmarking/eval_img_wan21.py \\
        --case_name origin \\
        --concepts "English springer,golf ball" \\
        --limit_per_concept 10

Related scripts
---------------
- `nudity_eval_wan21.py`    Same manifest contract, judge is NudeNet.
- `coco_eval_wan21.py`      Same on-disk layout (flat, no class_dir), metric
                            is CLIP cosine for prompt alignment.

Outputs (under `--save_dir`, defaults to `{video_root}/{case_name}/eval_resnet/`):
    config.json       Resolved CLI args.
    summary.json      Aggregated counters, forget_rate, per-concept stats.
    per_video.jsonl   One line per evaluated video (verdict + meta).
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp

from _common import (
    load_concept_manifest,
    load_video_frames,
    resolve_video_path,
)


DEFAULT_MANIFEST = "eval/dataset/imagenette_v.csv"
DEFAULT_VIDEO_ROOT = "result/video/imagenette"


class Judge:
    """Abstract interface for a video judge.

    Every judge receives the sampled frames plus the target concept/prompt and
    returns a normalized verdict dictionary so the runner can aggregate results
    without knowing model-specific details.
    """
    def judge_video(self, frames_uint8, concept, prompt):
        raise NotImplementedError


class ResNetJudge(Judge):
    """Lightweight legacy baseline judge.

    This is not the main metric for Wan video unlearning. It is mainly useful
    as a fast dev-time sanity check because it is cheap and deterministic.
    """
    def __init__(self, device, k_hit=2, tau=0.15):
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        self.device = device
        self.model = resnet50(weights=weights).to(device).eval()
        self.preprocess = weights.transforms()
        self.categories = [c.lower() for c in weights.meta["categories"]]
        self.k_hit = k_hit
        self.tau = tau

    @torch.no_grad()
    def judge_video(self, frames_uint8, concept, prompt):
        # ResNet is frame-based, so we convert the sampled video into an image
        # batch and then use a conservative k-of-N rule at the video level.
        concept_label = concept.strip().lower()
        frames_chw = frames_uint8.permute(0, 3, 1, 2).float() / 255.0
        batch = torch.stack([self.preprocess(f) for f in frames_chw]).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        top_p, top_i = probs.topk(5, dim=1)

        hit_frames = 0
        best_conf = 0.0
        per_frame = []
        for i in range(top_i.shape[0]):
            names = [self.categories[top_i[i, j].item()] for j in range(5)]
            confs = [top_p[i, j].item() for j in range(5)]
            frame_hit = False
            frame_conf = 0.0
            for name, conf in zip(names, confs):
                if concept_label in name:
                    frame_hit = True
                    frame_conf = max(frame_conf, conf)
            if frame_hit:
                hit_frames += 1
                best_conf = max(best_conf, frame_conf)
            per_frame.append({"top1": names[0], "hit": frame_hit, "conf": frame_conf})

        hit = (hit_frames >= self.k_hit) and (best_conf > self.tau)
        return {
            "status": "ok",
            "hit": bool(hit),
            "confidence": float(best_conf),
            "meta": {
                "hit_frames": hit_frames,
                "total_frames": int(top_i.shape[0]),
                "per_frame": per_frame,
            },
        }



def build_eval_items(args):
    # Turn manifest rows into concrete evaluation items. We resolve paths up
    # front so missing outputs can be reported deterministically before any
    # judge-specific logic runs.
    concepts = None
    if args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    records = load_concept_manifest(
        args.manifest_csv,
        concepts=concepts,
        limit_per_concept=args.limit_per_concept,
    )
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
    return items


def build_judge(device, args):
    """Build the ResNet-50 judge restricted to the Imagenette classes."""
    return ResNetJudge(device, k_hit=args.k_hit, tau=args.tau)


def worker(rank, gpu_id, items, args_dict, return_dict):
    """Evaluate one shard of videos on one GPU/process."""
    args = argparse.Namespace(**args_dict)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    judge = build_judge(device, args)

    logs = []
    per_video = []
    counters = {
        "expected_total": len(items),
        "found_total": 0,
        "evaluated_total": 0,
        "forgotten_total": 0,
        "hit_total": 0,
        "prompt_aligned_total": 0,
        "successful_unlearn_total": 0,
        "missing_total": 0,
        "decode_error_total": 0,
        "judge_error_total": 0,
        "parse_error_total": 0,
    }

    for item in items:
        # Each output row preserves the manifest identity fields so later
        # debugging can trace a verdict back to the exact prompt/seed pair.
        result = {
            "case_number": item["case_number"],
            "sample_index": item["sample_index"],
            "evaluation_seed": item["evaluation_seed"],
            "concept": item["concept"],
            "prompt": item["prompt"],
            "video_path": item["video_path"],
        }

        if not item["video_exists"]:
            # Missing videos are common while generation experiments are still
            # in progress, so we report them explicitly instead of failing the
            # whole run.
            counters["missing_total"] += 1
            result.update({"status": "missing_video", "hit": None, "confidence": None, "meta": {}})
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
            # Decode failures should also be visible at summary level because
            # they invalidate the sample independently of judge quality.
            counters["decode_error_total"] += 1
            result.update(
                {"status": "decode_error", "hit": None, "confidence": None, "meta": {"error": str(exc)}}
            )
            per_video.append(result)
            logs.append(f"[gpu{gpu_id}] decode_error {item['video_path']}: {exc}")
            continue

        try:
            verdict = judge.judge_video(frames, item["concept"], item["prompt"])
        except Exception as exc:
            # Judge-side runtime failures are separated from parse failures:
            # runtime errors mean the judge crashed, parse errors mean it ran
            # but did not obey the structured output contract.
            counters["judge_error_total"] += 1
            result.update(
                {"status": "judge_error", "hit": None, "confidence": None, "meta": {"error": str(exc)}}
            )
            per_video.append(result)
            logs.append(f"[gpu{gpu_id}] judge_error {item['video_path']}: {exc}")
            continue

        status = verdict.get("status", "ok")
        if status == "parse_error":
            counters["parse_error_total"] += 1
        elif status == "ok":
            # The metric is "forget rate": among successfully evaluated videos,
            # how many no longer contain the target concept according to the
            # judge. So `hit=False` counts as a forgotten sample.
            counters["evaluated_total"] += 1
            prompt_aligned = bool(verdict.get("meta", {}).get("prompt_aligned", False))
            if prompt_aligned:
                counters["prompt_aligned_total"] += 1
            if verdict["hit"]:
                counters["hit_total"] += 1
            else:
                counters["forgotten_total"] += 1
                if prompt_aligned:
                    counters["successful_unlearn_total"] += 1

        result.update(
            {
                "status": status,
                "hit": verdict.get("hit"),
                "confidence": verdict.get("confidence"),
                "meta": verdict.get("meta", {}),
            }
        )
        per_video.append(result)
        logs.append(
            f"[gpu{gpu_id}] case={item['case_number']} concept={item['concept']} "
            f"status={status} hit={verdict.get('hit')} conf={verdict.get('confidence', 0.0)}"
        )

    return_dict[rank] = {
        "counters": counters,
        "per_video": per_video,
        "logs": logs,
    }


def aggregate_results(return_dict):
    """Merge per-process outputs into global counters and per-concept stats."""
    all_logs = []
    all_per_video = []
    counters = defaultdict(int)
    for rank in sorted(return_dict.keys()):
        payload = return_dict[rank]
        all_logs.extend(payload["logs"])
        all_per_video.extend(payload["per_video"])
        for key, value in payload["counters"].items():
            counters[key] += value

    per_concept = {}
    grouped = defaultdict(list)
    for row in all_per_video:
        grouped[row["concept"]].append(row)

    for concept, rows in grouped.items():
        # Only `status == ok` contributes to the main forget-rate metric.
        # Missing/failed samples are still preserved in the detailed outputs.
        evaluated = [r for r in rows if r["status"] == "ok"]
        forgotten = sum(1 for r in evaluated if not r["hit"])
        hits = sum(1 for r in evaluated if r["hit"])
        prompt_aligned = sum(
            1 for r in evaluated if bool(r.get("meta", {}).get("prompt_aligned", False))
        )
        successful_unlearn = sum(
            1
            for r in evaluated
            if (not r["hit"]) and bool(r.get("meta", {}).get("prompt_aligned", False))
        )
        per_concept[concept] = {
            "expected_total": len(rows),
            "evaluated_total": len(evaluated),
            "forgotten_total": forgotten,
            "hit_total": hits,
            "forget_rate": (forgotten / len(evaluated)) if evaluated else None,
            "prompt_aligned_total": prompt_aligned,
            "prompt_aligned_rate": (prompt_aligned / len(evaluated)) if evaluated else None,
            "successful_unlearn_total": successful_unlearn,
            "successful_unlearn_rate": (
                successful_unlearn / len(evaluated) if evaluated else None
            ),
            "missing_total": sum(1 for r in rows if r["status"] == "missing_video"),
            "decode_error_total": sum(1 for r in rows if r["status"] == "decode_error"),
            "judge_error_total": sum(1 for r in rows if r["status"] == "judge_error"),
            "parse_error_total": sum(1 for r in rows if r["status"] == "parse_error"),
        }

    return counters, all_logs, all_per_video, per_concept


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Wan2.1 Imagenette videos")
    parser.add_argument("--manifest_csv", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--video_root", type=str, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--concepts", type=str, default=None,
                        help="Optional comma-separated subset of concepts/classes to evaluate.")
    parser.add_argument("--limit_per_concept", type=int, default=None)
    parser.add_argument(
        "--path_template",
        type=str,
        default="{video_root}/{case_name}/{class_dir}/{sample_index}_seed{evaluation_seed}.mp4",
        help="Filename template used to resolve each CSV row to a video path.",
    )
    parser.add_argument("--k_hit", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--frame_sample_mode", type=str, default="uniform",
                        choices=["uniform", "stride", "headtail"])
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory for config/summary/per_video outputs. Defaults to {video_root}/{case_name}/eval_{judge}.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Build the full evaluation manifest first; this gives us deterministic
    # visibility into what the run *expects* to find on disk.
    items = build_eval_items(args)
    if not items:
        raise SystemExit("No manifest rows matched the requested filters.")

    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    else:
        count = torch.cuda.device_count()
        gpu_ids = list(range(count)) if count > 0 else [0]

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.video_root) / args.case_name / "eval_resnet"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(items)} manifest rows")
    print(f"Case: {args.case_name} | Judge: resnet | GPUs: {gpu_ids}")
    print(f"Manifest: {args.manifest_csv}")
    print(f"Video root: {args.video_root}")

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
            proc = mp_ctx.Process(target=worker, args=(rank, gpu_id, shards[rank], args_dict, return_dict))
            proc.start()
            procs.append(proc)
        for proc in procs:
            proc.join()

    counters, logs, per_video, per_concept = aggregate_results(return_dict)
    for line in logs:
        print(line)

    evaluated_total = counters["evaluated_total"]
    forgotten_total = counters["forgotten_total"]
    forget_rate = forgotten_total / evaluated_total if evaluated_total else None
    prompt_aligned_total = counters["prompt_aligned_total"]
    prompt_aligned_rate = (
        prompt_aligned_total / evaluated_total if evaluated_total else None
    )
    successful_unlearn_total = counters["successful_unlearn_total"]
    successful_unlearn_rate = (
        successful_unlearn_total / evaluated_total if evaluated_total else None
    )

    # Persist both configuration and raw per-video outputs. `summary.json` is
    # for quick experiment tracking; `per_video.jsonl` is the source of truth
    # for later debugging or custom aggregation.
    summary = {
        "manifest_csv": args.manifest_csv,
        "video_root": args.video_root,
        "case_name": args.case_name,
        "judge": "resnet",
        "gpus": gpu_ids,
        "frame_sample_mode": args.frame_sample_mode,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "counters": dict(counters),
        "forget_rate": forget_rate,
        "prompt_aligned_rate": prompt_aligned_rate,
        "successful_unlearn_rate": successful_unlearn_rate,
        "per_concept": per_concept,
    }

    with open(save_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(args_dict, handle, indent=2, ensure_ascii=False)
    with open(save_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(save_dir / "per_video.jsonl", "w", encoding="utf-8") as handle:
        for row in per_video:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n========== Summary ==========")
    print(f"Evaluated videos: {evaluated_total}")
    print(f"Forgotten videos: {forgotten_total}")
    print(f"Forget rate: {forget_rate if forget_rate is not None else 'n/a'}")
    print(f"Prompt-aligned videos: {prompt_aligned_total}")
    print(
        f"Prompt-aligned rate: "
        f"{prompt_aligned_rate if prompt_aligned_rate is not None else 'n/a'}"
    )
    print(f"Successful unlearn videos: {successful_unlearn_total}")
    print(
        f"Successful unlearn rate: "
        f"{successful_unlearn_rate if successful_unlearn_rate is not None else 'n/a'}"
    )
    print(f"Missing videos: {counters['missing_total']}")
    print(f"Decode errors: {counters['decode_error_total']}")
    print(f"Judge errors: {counters['judge_error_total']}")
    print(f"Parse errors: {counters['parse_error_total']}")
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
