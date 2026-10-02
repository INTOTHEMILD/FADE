"""Wan2.1 video generator for i2p nudity / safety evaluation.

Companion to `nudity_eval_wan21.py`. Generation-only: pulls prompts from the
i2p benchmark (`AIML-TUDA/i2p`), filters to the categories the user enabled,
runs Wan2.1 T2V, and writes videos plus a manifest CSV that the eval script
can consume unchanged.

Layout (same convention as `get_imagenette.py`; class_dir collapses both
spaces AND dashes, since i2p categories include "self-harm" / "illegal
activity"):

    {save_dir}/{case_name}/{class_dir}/{sample_index}_seed{evaluation_seed}.mp4
    {save_dir}/{case_name}/manifest.csv

Manifest CSV columns: `case_number,prompt,class,evaluation_seed`.

Typical usage
-------------
    # 50 sexual-category videos with Wan-1.3B, 4 GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
        eval/benchmarking/nudity_generator_wan21.py \\
            --ckpt_dir ckpt/Wan2.1-T2V-1.3B \\
            --case_name origin_i2p \\
            --number 50 \\
            --sexual True \\
            --dit_fsdp --t5_fsdp --ulysses_size 4

    # Enable multiple i2p categories:
    ... --case_name origin_i2p_full \\
        --number 200 \\
        --sexual True --violence True --shocking True

Then score with:

    python eval/benchmarking/nudity_eval_wan21.py \\
        --manifest_csv result/video/nudity/origin_i2p/manifest.csv \\
        --video_root result/video/nudity \\
        --case_name origin_i2p

Runtime args (shared with the other generators) come from
`_wan_args.add_wan_runtime_args`. Script-specific args below.
Each i2p category has its own boolean flag; default is only `--sexual=True`.
"""

import argparse
import csv
import logging
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist

# Make the repository root importable when run as `python eval/benchmarking/<script>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video, str2bool

from _wan_args import add_wan_runtime_args


# i2p's `categories` column is a comma-separated string drawn from this fixed
# vocabulary. Argparse flag names use underscores so `self-harm` / `illegal
# activity` become `--self_harm` / `--illegal_activity`.
I2P_CATEGORIES = [
    "hate",
    "harassment",
    "violence",
    "self-harm",
    "sexual",
    "shocking",
    "illegal activity",
]


def _category_to_flag(category):
    return category.replace("-", "_").replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Wan2.1 videos for i2p nudity/safety evaluation."
    )
    add_wan_runtime_args(parser)

    # Output.
    parser.add_argument("--save_dir", type=str, default="result/video/nudity")
    parser.add_argument("--case_name", type=str, required=True,
                        help="Sub-directory under save_dir, e.g. 'origin_i2p'.")

    # i2p selection.
    parser.add_argument("--i2p_repo", type=str, default="AIML-TUDA/i2p",
                        help="HF dataset repo id (should not normally change).")
    parser.add_argument("--i2p_split", type=str, default="train")
    parser.add_argument("--hard_only", type=str2bool, default=False,
                        help="If True, keep only rows where i2p `hard == 1`.")
    parser.add_argument("--number", type=int, required=True,
                        help="Total number of videos to generate (across all "
                             "enabled categories, filtered in dataset order).")
    parser.add_argument("--seed_fallback", type=int, default=0,
                        help="Seed to use if an i2p row has no usable sd_seed.")

    # One boolean per i2p category. Default: only --sexual is enabled.
    for cat in I2P_CATEGORIES:
        parser.add_argument(
            f"--{_category_to_flag(cat)}",
            type=str2bool,
            default=(cat == "sexual"),
            help=f"If True, include i2p category `{cat}`.",
        )

    return parser.parse_args()


def enabled_categories(args):
    return [cat for cat in I2P_CATEGORIES if getattr(args, _category_to_flag(cat))]


def row_categories(row):
    # i2p stores categories as a comma-separated string like "sexual, shocking".
    raw = row.get("categories") or ""
    return [c.strip() for c in raw.split(",") if c.strip()]


def select_i2p_records(args, enabled):
    """Return an ordered list of records for generation.

    Each record is assigned to exactly one `class` (the first enabled category
    in its i2p `categories` string) so the downstream eval can report per-class
    numbers cleanly. Rows are kept in their native dataset order so the
    selection is reproducible across runs.
    """
    from datasets import load_dataset

    ds = load_dataset(args.i2p_repo, split=args.i2p_split)
    enabled_set = set(enabled)

    selected = []
    for global_idx, row in enumerate(ds):
        if args.hard_only and int(row.get("hard", 0)) != 1:
            continue
        cats = row_categories(row)
        match = next((c for c in cats if c in enabled_set), None)
        if match is None:
            continue
        try:
            seed = int(row.get("sd_seed"))
        except (TypeError, ValueError):
            seed = int(args.seed_fallback)
        selected.append(
            {
                "global_idx": global_idx,
                "prompt": row["prompt"],
                "class": match,
                "evaluation_seed": seed,
            }
        )
        if len(selected) >= args.number:
            break

    return selected


def write_manifest(manifest_path, records_with_sample_index):
    """Emit a CSV compatible with `nudity_eval_wan21.py`.

    `case_number` is the original i2p row index so traceability back to the
    benchmark is preserved; `sample_index` (per-class 0-based) is implicit and
    reconstructed by the eval script.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_number", "prompt", "class", "evaluation_seed"])
        for rec in records_with_sample_index:
            writer.writerow(
                [rec["global_idx"], rec["prompt"], rec["class"], rec["evaluation_seed"]]
            )


def main():
    args = parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            rank=rank, world_size=world_size,
        )
    else:
        assert not (args.t5_fsdp or args.dit_fsdp)
        assert not (args.ulysses_size > 1 or args.ring_size > 1)

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size
        from xfuser.core.distributed import (
            init_distributed_environment, initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    enabled = enabled_categories(args)
    if not enabled:
        logging.error(
            "No i2p category flags enabled. Pass at least one, e.g. --sexual True."
        )
        return

    logging.info(f"Enabled i2p categories : {enabled}")
    logging.info(f"Hard-only filter      : {args.hard_only}")
    logging.info(f"Number to generate    : {args.number}")

    records = select_i2p_records(args, enabled)
    if not records:
        logging.error(
            f"i2p selection produced 0 records. "
            f"Categories={enabled}, hard_only={args.hard_only}."
        )
        return

    # Assign per-class sample_index in dataset-traversal order. This matches
    # how `nudity_eval_wan21.py` recomputes `sample_index` from the manifest.
    per_class_counter = defaultdict(int)
    for rec in records:
        cls = rec["class"]
        rec["sample_index"] = per_class_counter[cls]
        per_class_counter[cls] += 1

    class_counts = {cls: per_class_counter[cls] for cls in per_class_counter}
    logging.info(f"Per-class counts       : {class_counts}")

    case_root = Path(args.save_dir) / args.case_name
    case_root.mkdir(parents=True, exist_ok=True)

    # Only rank 0 writes the manifest; all ranks share the same records so the
    # eval-side filename template (`{sample_index}_seed{evaluation_seed}.mp4`)
    # is identical across ranks.
    if rank == 0:
        manifest_path = case_root / "manifest.csv"
        write_manifest(manifest_path, records)
        logging.info(f"Wrote manifest -> {manifest_path}")

    # Build Wan2.1 pipeline once.
    cfg = WAN_CONFIGS[args.task]
    logging.info("Creating WanT2V pipeline.")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    total = len(records)
    for count, rec in enumerate(records, start=1):
        class_dir = case_root / rec["class"].replace(" ", "_").replace("-", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

        save_path = class_dir / f"{rec['sample_index']}_seed{rec['evaluation_seed']}.mp4"
        if save_path.is_file():
            # Re-runnable: skip already-generated videos so interrupted runs
            # can be resumed without wasting GPU hours.
            logging.info(f"[{count}/{total}] skip (exists): {save_path}")
            continue

        logging.info(
            f"[{count}/{total}] class={rec['class']} "
            f"idx={rec['sample_index']} seed={rec['evaluation_seed']} "
            f"prompt={rec['prompt']}"
        )

        video = wan_t2v.generate(
            rec["prompt"],
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=rec["evaluation_seed"],
            offload_model=args.offload_model,
        )

        if rank == 0:
            logging.info(f"Saving video -> {save_path}")
            cache_video(
                tensor=video[None],
                save_file=str(save_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

    logging.info("All done.")


if __name__ == "__main__":
    main()
