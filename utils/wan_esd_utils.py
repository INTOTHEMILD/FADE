"""ESD helpers for Wan2.1 T2V.

Companion to ``esd_wan21.py``. Mirrors the SD ESD helpers
(``DFM/esd_2025/utils/sd_utils.py`` + ``esd_base/esd_sd.py``) but adapted to
Wan's flow-matching scheduler, list-of-tensor I/O and DiT cross-attn naming.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.cuda.amp as amp
from safetensors.torch import load_file, save_file

from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ---------------------------------------------------------------------------
# Small geometry / shape utilities
# ---------------------------------------------------------------------------

@contextmanager
def noop_no_sync():
    yield


def compute_target_shape(z_dim: int, vae_stride: Sequence[int],
                         size: Tuple[int, int], frame_num: int) -> Tuple[int, int, int, int]:
    """Replicates the latent shape calc inside ``WanT2V.generate``.

    Args:
        z_dim: VAE latent channel count (``vae.model.z_dim``).
        vae_stride: ``(t_stride, h_stride, w_stride)``.
        size: ``(width, height)`` -- same convention as Wan's ``size`` arg.
        frame_num: pixel-space frame count, must be ``4n+1``.

    Returns:
        ``(C_lat, F_lat, H_lat, W_lat)``
    """
    assert (frame_num - 1) % vae_stride[0] == 0, (
        f"frame_num-1 must be divisible by vae_stride[0]={vae_stride[0]}, "
        f"got frame_num={frame_num}"
    )
    width, height = size
    return (
        z_dim,
        (frame_num - 1) // vae_stride[0] + 1,
        height // vae_stride[1],
        width // vae_stride[2],
    )


def compute_seq_len(target_shape: Sequence[int], patch_size: Sequence[int],
                    sp_size: int = 1) -> int:
    """Replicates ``seq_len`` calc inside ``WanT2V.generate``."""
    return math.ceil(
        (target_shape[2] * target_shape[3]) /
        (patch_size[1] * patch_size[2]) *
        target_shape[1] / sp_size
    ) * sp_size


# ---------------------------------------------------------------------------
# Forward / partial-denoise wrappers
# ---------------------------------------------------------------------------

def model_forward(model: WanModel, xt: torch.Tensor, t_scalar: torch.Tensor,
                  context: List[torch.Tensor], seq_len: int) -> torch.Tensor:
    """Run a single-sample forward through ``WanModel``.

    Args:
        model: WanModel (any of base/esd).
        xt:    latent of shape ``[C, F, H, W]`` (no batch dim).
        t_scalar: 0-d tensor, the diffusion timestep value.
        context:  list ``[Tensor[L, 4096]]`` from the T5 encoder.
        seq_len:  the precomputed padded sequence length.

    Returns:
        Tensor of shape ``[C, F, H, W]`` -- the model's prediction for ``xt``.
    """
    out = model([xt], t=torch.stack([t_scalar]),
                context=context, seq_len=seq_len)
    return out[0]


def _build_fresh_scheduler(num_train_timesteps: int, num_inference_steps: int,
                           shift: float, device: torch.device
                           ) -> FlowUniPCMultistepScheduler:
    """Allocate a stateless scheduler. UniPC is multistep, so reusing one across
    ESD iterations leaves stale ``model_outputs`` / ``this_order`` and crashes
    on the second call -- ``WanT2V.generate`` itself rebuilds per call."""
    sched = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1, use_dynamic_shifting=False,
    )
    sched.set_timesteps(num_inference_steps, device=device, shift=shift)
    return sched


@torch.no_grad()
def partial_denoise(model: WanModel, target_shape: Sequence[int],
                    context_cond: List[torch.Tensor],
                    context_uncond: List[torch.Tensor],
                    guide_scale: float, device: torch.device, seed: int,
                    dtype: torch.dtype, seq_len: int,
                    run_till_step: int,
                    num_train_timesteps: int,
                    num_inference_steps: int,
                    shift: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partially denoise a fresh noise sample, mirroring ``WanT2V.generate``.

    The base model is used to produce a half-denoised latent ``x_t`` at
    timestep ``scheduler.timesteps[run_till_step]``. This is the Wan analogue
    of SD ESD's ``pipe(...) ... run_till_timestep`` call.

    A fresh ``FlowUniPCMultistepScheduler`` is built each call because UniPC
    is a multistep solver -- reusing one accumulates ``model_outputs`` /
    ``this_order`` state that breaks the next iteration.

    Args:
        run_till_step: number of solver steps to take. ``0`` means we return
            pure noise at ``scheduler.timesteps[0]``; ``num_inference_steps-1``
            is the last in-flight step.

    Returns:
        ``(xt, t)`` -- ``xt`` shape ``[C, F, H, W]``; ``t`` is a 0-d tensor.
    """
    scheduler = _build_fresh_scheduler(
        num_train_timesteps=num_train_timesteps,
        num_inference_steps=num_inference_steps,
        shift=shift,
        device=device,
    )
    timesteps = scheduler.timesteps
    n_steps = len(timesteps)
    run_till_step = max(0, min(int(run_till_step), n_steps - 1))

    # Per-iteration generator gives us reproducibility without touching globals.
    g = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(*target_shape, dtype=torch.float32,
                        device=device, generator=g)
    latents = [noise]

    with amp.autocast(dtype=dtype):
        for i in range(run_till_step):
            t = timesteps[i]
            ncond = model_forward(model, latents[0], t, context_cond, seq_len)
            nuncond = model_forward(model, latents[0], t, context_uncond, seq_len)
            noise_pred = nuncond + guide_scale * (ncond - nuncond)
            step_out = scheduler.step(
                noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                return_dict=False, generator=g,
            )[0]
            latents = [step_out.squeeze(0)]

    return latents[0].detach(), timesteps[run_till_step]


# ---------------------------------------------------------------------------
# Trainable parameter selection
# ---------------------------------------------------------------------------

# Modules we never want to fine-tune via ESD on Wan, regardless of `train_method`.
# These are load-bearing for *any* generation -- corrupting them kills the model
# entirely instead of just removing a concept.
_ESD_HARD_SKIP_MODULES = (
    "text_embedding.",   # T5 -> DiT bridge MLP (only path between 4096 and dim)
    "time_embedding.",   # diffusion-step conditioning
    "time_projection.",  # AdaLN modulation generator
    "head.",             # final unpatchify head
    "patch_embedding.",  # 3D conv input projection (Conv3d, but be explicit)
    "img_emb.",          # i2v / flf2v CLIP projector
)

_VALID_TRAIN_METHODS = ("esd-x", "esd-x-strict", "esd-u", "esd-all")


def _is_hard_skip(name: str) -> bool:
    return any(name.startswith(prefix) or f".{prefix}" in name
               for prefix in _ESD_HARD_SKIP_MODULES)


def get_esd_trainable_parameters(esd_model: WanModel,
                                 train_method: str = "esd-x"
                                 ) -> Tuple[List[str], List[torch.nn.Parameter]]:
    """Return ``(names, params)`` of the modules we want to fine-tune.

    Mirrors ``DFM/esd_2025/esd_base/esd_sd.py::get_esd_trainable_parameters``
    but rewritten for Wan's DiT naming:

    +------------------+------------------------------------------------------+
    | train_method     | targets (substrings on the *module* name)            |
    +==================+======================================================+
    | ``esd-x``        | every ``Linear`` whose module path contains          |
    |                  | ``cross_attn`` (Wan's text cross-attn)               |
    +------------------+------------------------------------------------------+
    | ``esd-x-strict`` | only ``cross_attn.k`` and ``cross_attn.v``           |
    +------------------+------------------------------------------------------+
    | ``esd-u``        | every ``Linear`` *outside* ``cross_attn`` (i.e.      |
    |                  | self-attn projections + FFN), minus the hard-skip    |
    |                  | list above                                           |
    +------------------+------------------------------------------------------+
    | ``esd-all``      | every ``Linear``, minus hard-skip                    |
    +------------------+------------------------------------------------------+
    """
    if train_method not in _VALID_TRAIN_METHODS:
        raise ValueError(
            f"train_method='{train_method}' not in {_VALID_TRAIN_METHODS}"
        )

    names: List[str] = []
    params: List[torch.nn.Parameter] = []

    for mod_name, module in esd_model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        in_cross = "cross_attn" in mod_name
        # cross_attn includes the QK norms (WanRMSNorm) but those aren't Linear,
        # so the isinstance check above already drops them.

        if train_method == "esd-x" and not in_cross:
            continue
        if train_method == "esd-x-strict" and not (
            mod_name.endswith("cross_attn.k") or mod_name.endswith("cross_attn.v")
        ):
            continue
        if train_method == "esd-u" and in_cross:
            continue
        # esd-all: take the module if not in hard-skip

        if train_method in ("esd-u", "esd-all") and _is_hard_skip(mod_name):
            continue

        for pn, p in module.named_parameters(recurse=False):
            full = f"{mod_name}.{pn}"
            names.append(full)
            params.append(p)

    return names, params


# ---------------------------------------------------------------------------
# Save / load adapter-style ESD checkpoints
# ---------------------------------------------------------------------------

def save_esd_state(names: Iterable[str], params: Iterable[torch.nn.Parameter],
                   save_path: str) -> None:
    """Write the trained subset of weights to ``save_path`` as safetensors."""
    state = {n: p.detach().cpu().contiguous()
             for n, p in zip(names, params)}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    save_file(state, save_path)


def apply_esd_state(model: WanModel, ckpt_path: str) -> None:
    """Load an ESD safetensors and copy_ each tensor into ``model``."""
    state = load_file(ckpt_path)
    name_to_param = dict(model.named_parameters())
    missing = [n for n in state if n not in name_to_param]
    if missing:
        raise KeyError(
            f"{len(missing)} ESD parameters not found in model "
            f"(first few: {missing[:3]})"
        )
    with torch.no_grad():
        for n, t in state.items():
            p = name_to_param[n]
            p.data.copy_(t.to(p.dtype).to(p.device))
