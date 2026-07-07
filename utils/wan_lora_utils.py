"""Helpers for Phase 2 LoRA training: mounting GatedFFN, save/load LoRA state."""
from pathlib import Path

import torch
import torch.utils.checkpoint as _checkpoint
from safetensors.torch import load_file, save_file

from wan.modules.frame_aware_lora import FrameAwareLoRA
from wan.modules.gated_ffn import GatedFFN


def attach_lora_to_block(
    block,
    concept_name: str,
    d_in: int,
    d_out: int,
    r0: int = 4,
    r1: int = 2,
) -> FrameAwareLoRA:
    """Wrap block.ffn in a GatedFFN with one LoRA for `concept_name`.

    Idempotent: if block.ffn is already a GatedFFN, just register the new LoRA.
    Returns the freshly-created FrameAwareLoRA.
    """
    lora = FrameAwareLoRA(d_in=d_in, d_out=d_out, r0=r0, r1=r1)
    if isinstance(block.ffn, GatedFFN):
        block.ffn.loras[concept_name] = lora
    else:
        original = block.ffn
        block.ffn = GatedFFN(
            original_ffn=original,
            concept_loras={concept_name: lora},
            anchors={},
        )
    return lora


def get_lora_params(model) -> tuple:
    """Return (names, params) for all LoRA-side params (LoRAs + φ MLPs)."""
    names, params = [], []
    for n, p in model.named_parameters():
        if "loras." in n or ".phi." in n:
            names.append(n)
            params.append(p)
    return names, params


def freeze_base(model):
    """Freeze every non-LoRA param in `model` (i.e. all original Wan weights)."""
    for n, p in model.named_parameters():
        if "loras." in n or ".phi." in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def save_lora(concept_name: str, model, save_path: str):
    """Save the LoRA params (and φ MLP) for one concept to a safetensors file."""
    state = {
        n: p.detach().cpu().contiguous()
        for n, p in model.named_parameters()
        if f"loras.{concept_name}." in n
    }
    if not state:
        raise ValueError(f"No LoRA params found for concept '{concept_name}'")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, save_path)


def load_lora(concept_name: str, model, load_path: str):
    """Load LoRA params for `concept_name` into the corresponding GatedFFN slots."""
    state = load_file(load_path)
    loaded = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if f"loras.{concept_name}." in n and n in state:
                p.copy_(state[n].to(p.device).to(p.dtype))
                loaded += 1
    if loaded == 0:
        raise RuntimeError(
            f"load_lora({concept_name}, {load_path}): no params matched. "
            "Did you call attach_lora_to_block first?"
        )
    return loaded


def enable_block_gradient_checkpointing(model):
    """Wrap each block.forward with torch.utils.checkpoint.

    Block kwargs (e, seq_lens, grid_sizes, freqs, context, context_lens) are
    captured via closure rather than passed through checkpoint — they're
    constant from the trainer's perspective (frozen base contributions or
    encoder outputs), so no need to differentiate through them. Only x flows
    through the checkpoint boundary, which is enough for LoRA-side backward.

    Trade-off: ~30% slower iter for ~2× activation memory savings.
    """
    for blk in model.blocks:
        orig_forward = blk.forward

        def make_ckpt_forward(orig):
            def ckpt_forward(x, **kwargs):
                def run(x_in):
                    return orig(x_in, **kwargs)
                return _checkpoint.checkpoint(run, x, use_reentrant=False)
            return ckpt_forward

        blk.forward = make_ckpt_forward(orig_forward)


def load_routing_config(path: str, device=None) -> dict:
    """Load routing config saved by tools/calibrate_router.py.

    Returns dict with keys: anchors, tau, T (concepts → tensor / float).
    """
    cfg = torch.load(path, map_location="cpu")
    if device is not None:
        cfg["anchors"] = {c: a.to(device) for c, a in cfg["anchors"].items()}
    return cfg
