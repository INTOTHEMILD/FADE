"""Wan2.1 video generator for COCO CLIP-similarity evaluation.

Companion to `coco_eval_wan21.py`. Mirrors the split used for i2p/nudity and
imagenette: this script only generates videos + a manifest CSV. Scoring is
run separately by `coco_eval_wan21.py` over the produced files.

Default manifest is `eval/dataset/coco_30k.csv` whose columns are
`case_number, source, prompt, evaluation_seed, coco_id`. COCO has no class
field, so videos are written FLAT under a per-run `case_name` directory
(no `{class_dir}`):

    {save_dir}/{case_name}/{case_number}_seed{evaluation_seed}.mp4
    {save_dir}/{case_name}/manifest.csv

Typical usage
-------------
    # Generate 500 COCO-aligned videos with the original Wan-1.3B:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\
        eval/benchmarking/coco_generator_wan21.py \\
            --ckpt_dir ckpt/Wan2.1-T2V-1.3B \\
            --case_name origin_coco \\
            --number 500 \\
            --dit_fsdp --t5_fsdp --ulysses_size 4

    # Resume from row 500 and generate the next 500:
    ... --case_name origin_coco_part2 --offset 500 --number 500

Runtime args (shared with the other generators) come from
`_wan_args.add_wan_runtime_args`. Script-specific args below.
"""

import argparse
import csv
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist

# Make the repository root importable when run as `python eval/benchmarking/<script>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video

from _wan_args import add_wan_runtime_args


DEFAULT_CSV = "eval/dataset/coco_30k.csv"
DEFAULT_SAVE_DIR = "result/video/coco"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Wan2.1 videos from a COCO-style prompt CSV."
    )
    add_wan_runtime_args(parser)

    # I/O.
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV,
                        help="CSV with at least `case_number, prompt, evaluation_seed`.")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--case_name", type=str, required=True,
                        help="Sub-directory under save_dir, e.g. 'origin_coco'.")
    parser.add_argument("--number", type=int, required=True,
                        help="Total videos to generate (takes the first N CSV rows in file order).")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many rows before counting toward --number.")
    parser.add_argument("--seed_fallback", type=int, default=0,
                        help="Seed to use if a CSV row has no usable evaluation_seed.")

    return parser.parse_args()


def load_prompt_records(csv_path, number, offset):
    """Read CSV rows in file order, skipping `offset` and capping at `number`.

    Keeps the original `case_number` (global row id in the benchmark), because
    the eval side uses it as the on-disk filename key.
    """
    required = {"case_number", "prompt", "evaluation_seed"}
    records = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path}: expected columns containing {sorted(required)}, "
                f"got {reader.fieldnames!r}"
            )
        skipped = 0
        for row in reader:
            if skipped < offset:
                skipped += 1
                continue
            try:
                seed = int(row["evaluation_seed"])
            except (TypeError, ValueError):
                seed = None
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue
            records.append(
                {
                    "case_number": int(row["case_number"]),
                    "prompt": prompt,
                    "evaluation_seed": seed,
                }
            )
            if len(records) >= number:
                break
    return records


def write_manifest(manifest_path, records, seed_fallback):
    """Copy the generated subset into a manifest the eval script can consume."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_number", "prompt", "evaluation_seed"])
        for rec in records:
            seed = rec["evaluation_seed"]
            if seed is None:
                seed = seed_fallback
            writer.writerow([rec["case_number"], rec["prompt"], seed])


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

    records = load_prompt_records(args.csv_path, args.number, args.offset)
    if not records:
        logging.error(f"No rows selected from {args.csv_path} (offset={args.offset}, number={args.number}).")
        return

    logging.info(f"CSV              : {args.csv_path}")
    logging.info(f"Rows selected    : {len(records)} (offset={args.offset}, number={args.number})")

    case_root = Path(args.save_dir) / args.case_name
    case_root.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        manifest_path = case_root / "manifest.csv"
        write_manifest(manifest_path, records, args.seed_fallback)
        logging.info(f"Wrote manifest -> {manifest_path}")

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
        seed = rec["evaluation_seed"] if rec["evaluation_seed"] is not None else args.seed_fallback
        save_path = case_root / f"{rec['case_number']}_seed{seed}.mp4"
        if save_path.is_file():
            # Re-runnable: skip already-generated videos so interrupted runs
            # can resume without wasting GPU hours.
            logging.info(f"[{count}/{total}] skip (exists): {save_path}")
            continue

        logging.info(
            f"[{count}/{total}] case={rec['case_number']} seed={seed} "
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
            seed=seed,
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
