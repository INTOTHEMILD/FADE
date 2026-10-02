# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan2.1 generator for Imagenette concept-unlearn videos.

Companion to `eval_img_wan21.py`. Reads `eval/dataset/imagenette_v.csv`
(columns: case_number, prompt, class, evaluation_seed) and writes videos
under `{save_dir}/{case_name}/{class_dir}/{sample_index}_seed{seed}.mp4`,
where `class_dir = class.replace(" ", "_")` and `sample_index` is the row's
position within its class after any filtering.

To evaluate a CR-edited backbone, point `--ckpt_dir` at the Phase 1 output
directory (`ckpt/Wan2.1-T2V-1.3B-cr-<group>/`). To evaluate an ESD baseline,
pass `--esd_ckpt` alongside the original Wan checkpoint.

Typical usage
-------------
    # Generate 10 videos each for 2 concepts with the ORIGINAL Wan-1.3B model
    # on 8 GPUs:
    GPUS=0,1,2,3,4,5,6,7 ; NPROC=8
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC \\
        eval/benchmarking/get_imagenette.py \\
            --ckpt_dir ckpt/Wan2.1-T2V-1.3B \\
            --case_name origin \\
            --concept_list "English springer" "golf ball" \\
            --num_per_concept 10 \\
            --dit_fsdp --t5_fsdp --ulysses_size $NPROC

    # Same, but with the CR-edited backbone:
    ... --case_name fade_cr \\
        --ckpt_dir ckpt/Wan2.1-T2V-1.3B-cr-imagenette

GPU-count / num_heads divisibility: see scripts/inference.sh
(1.3B: 1/2/3/4/6/12; 14B: 1/2/4/5/8/10/20/40).

Runtime args (shared with the nudity/coco generators) come from
`_wan_args.add_wan_runtime_args`. Script-specific args below.
"""

import argparse
import csv
import logging
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist

# Make the repository root importable when run as `python eval/benchmarking/<script>.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video

from _wan_args import add_wan_runtime_args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch generate videos from imagenette_v.csv")
    add_wan_runtime_args(parser)
    parser.add_argument(
        "--csv_path", type=str,
        default="eval/dataset/imagenette_v.csv",
        help="Prompt CSV with case_number,prompt,class,evaluation_seed.")
    parser.add_argument(
        "--save_dir", type=str,
        default="result/video/imagenette",
        help="Output root; videos go under {save_dir}/{case_name}/{class_dir}/.")
    parser.add_argument(
        "--case_name", type=str, required=True,
        help="Sub-directory name under save_dir, e.g. 'origin', 'unlearn_v1'.")
    parser.add_argument(
        "--concept_list", type=str, nargs='+', required=True,
        help="Concepts to generate, e.g. --concept_list 'English springer' 'church'.")
    parser.add_argument(
        "--num_per_concept", type=int, default=10,
        help="Number of videos to generate per concept.")
    parser.add_argument(
        "--esd_ckpt", type=str, default=None,
        help="Optional ESD safetensors (trained subset of WanModel parameters, "
             "e.g. cross_attn.q/k/v/o for esd-x). Requires --ckpt_dir to point "
             "at the original Wan checkpoint directory. Not supported with "
             "--dit_fsdp.")
    return parser.parse_args()


def load_csv(csv_path, concept_list, num_per_concept):
    """Load CSV and return {concept: [(prompt, seed), ...]} filtered by concept_list."""
    concept_data = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            concept = row['class']
            if concept in concept_list:
                concept_data[concept].append((row['prompt'], int(row['evaluation_seed'])))

    # Truncate each concept to num_per_concept
    for concept in list(concept_data.keys()):
        concept_data[concept] = concept_data[concept][:num_per_concept]

    return concept_data


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
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp)
        assert not (args.ulysses_size > 1 or args.ring_size > 1)

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size
        from xfuser.core.distributed import (
            init_distributed_environment, initialize_model_parallel)
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size)

    # Load CSV data
    concept_data = load_csv(args.csv_path, args.concept_list, args.num_per_concept)

    if not concept_data:
        logging.error(f"No matching concepts found. Available concepts in CSV, "
                      f"requested: {args.concept_list}")
        return

    logging.info(f"Concepts to generate: {list(concept_data.keys())}")
    for c, items in concept_data.items():
        logging.info(f"  {c}: {len(items)} videos")

    # Build pipeline once
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

    # Overlay ESD baseline weights if requested.
    # NOTE: dit_fsdp is not supported (sharded params block a plain in-place
    # copy). Use single-GPU or non-FSDP for ESD inference.
    if args.esd_ckpt:
        assert not args.dit_fsdp, \
            "--esd_ckpt cannot be combined with --dit_fsdp; run without dit_fsdp."
        from utils.wan_esd_utils import apply_esd_state
        logging.info(f"Applying ESD weights from {args.esd_ckpt}")
        apply_esd_state(wan_t2v.model, args.esd_ckpt)

    # Generate videos
    total = sum(len(v) for v in concept_data.values())
    count = 0
    for concept, items in concept_data.items():
        # Create concept directory
        concept_dir = os.path.join(args.save_dir, args.case_name, concept.replace(" ", "_"))
        os.makedirs(concept_dir, exist_ok=True)

        for idx, (prompt, seed) in enumerate(items):
            count += 1
            logging.info(f"[{count}/{total}] Concept: {concept} | "
                         f"idx: {idx} | seed: {seed} | prompt: {prompt}")

            video = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=seed,
                offload_model=args.offload_model)

            if rank == 0:
                save_path = os.path.join(concept_dir, f"{idx}_seed{seed}.mp4")
                logging.info(f"Saving video to {save_path}")
                cache_video(
                    tensor=video[None],
                    save_file=save_path,
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1))

    logging.info("All done.")


if __name__ == "__main__":
    main()
