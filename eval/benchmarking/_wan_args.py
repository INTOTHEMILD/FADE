"""Shared argparse helpers for Wan2.1 generation scripts.

`get_imagenette.py`, `coco_generator_wan21.py`, and `nudity_generator_wan21.py`
all share the same ~15 Wan runtime arguments (model size, sampler, FSDP, etc.).
Declaring them once here means:

1. Changing a default (e.g. `--sample_steps` from 50 to 40) only needs one edit.
2. All three generators stay in lockstep on argument names — evaluation scripts
   and shell wrappers never have to branch on which generator wrote a run.

Usage in a generator script
---------------------------
    import argparse
    from _wan_args import add_wan_runtime_args

    parser = argparse.ArgumentParser()
    add_wan_runtime_args(parser)
    # ...then add script-specific I/O / dataset / case_name args...
    args = parser.parse_args()

Args contributed
----------------
    --task                  Wan config name (t2v-1.3B default).
    --size                  Output size, e.g. "832*480".
    --frame_num             Number of video frames.
    --ckpt_dir              (required) Path to Wan model weights.
    --offload_model         Offload weights between modules (bool/None).
    --sample_steps          Diffusion steps.
    --sample_shift          Sampler shift parameter.
    --sample_solver         "unipc" or "dpm++".
    --sample_guide_scale    Classifier-free guidance scale.
    --t5_fsdp, --dit_fsdp   Shard T5 / DiT across ranks (FSDP).
    --t5_cpu                Keep T5 on CPU.
    --ulysses_size, --ring_size
                            Sequence-parallel dims (must satisfy
                            ulysses_size * ring_size == world_size).
"""

from __future__ import annotations

from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import str2bool


def add_wan_runtime_args(parser):
    """Add the shared Wan2.1 generation runtime args to `parser` in-place.

    Kept flat (no subgroup) so `--help` stays compatible with the pre-merge
    generator scripts.
    """
    parser.add_argument(
        "--task", type=str, default="t2v-1.3B",
        choices=list(WAN_CONFIGS.keys()),
        help="Wan pipeline config name.",
    )
    parser.add_argument(
        "--size", type=str, default="832*480",
        choices=list(SIZE_CONFIGS.keys()),
        help="Output frame size.",
    )
    parser.add_argument(
        "--frame_num", type=int, default=81,
        help="Number of frames to generate per video.",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="Path to the Wan checkpoint directory.",
    )
    parser.add_argument(
        "--offload_model", type=str2bool, default=None,
        help="Offload model weights between modules. None = auto (False for "
             "multi-rank, True for single-rank).",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=50,
        help="Number of diffusion sampling steps.",
    )
    parser.add_argument(
        "--sample_shift", type=float, default=5.0,
        help="Noise schedule shift parameter passed to the sampler.",
    )
    parser.add_argument(
        "--sample_solver", type=str, default="unipc",
        choices=["unipc", "dpm++"],
        help="Diffusion sampler choice.",
    )
    parser.add_argument(
        "--sample_guide_scale", type=float, default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--t5_fsdp", action="store_true", default=False,
        help="FSDP-shard the T5 encoder across ranks.",
    )
    parser.add_argument(
        "--t5_cpu", action="store_true", default=False,
        help="Keep the T5 encoder on CPU (saves GPU memory).",
    )
    parser.add_argument(
        "--dit_fsdp", action="store_true", default=False,
        help="FSDP-shard the DiT across ranks.",
    )
    parser.add_argument(
        "--ulysses_size", type=int, default=1,
        help="Sequence-parallel Ulysses degree.",
    )
    parser.add_argument(
        "--ring_size", type=int, default=1,
        help="Sequence-parallel Ring degree. Must satisfy "
             "ulysses_size * ring_size == world_size when >1.",
    )
