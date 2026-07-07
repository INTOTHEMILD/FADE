"""Token-level max-similarity routing for multi-concept LoRA gating.

Replaces single-vector cosine routing — pooled prompt vectors get diluted by
scene/function words, especially for object-in-motion concepts (spaniel,
parachute, church). Token-level matching takes the *maximum* sim over any
(prompt_token × anchor_token) pair, so a single concept-bearing token in the
prompt is enough to trigger routing regardless of prompt length or scene.

Anchor for concept c: T5 token embeddings of the concept string itself,
excluding `</s>`. ~3-5 tokens per concept (e.g. "english springer spaniel" →
4 tokens). Prompt is tokenized the same way.

g_c = sigmoid((max_token_sim − τ_c) / T_c).
"""
from __future__ import annotations

import math

import torch


def _drop_eos(t5_out: torch.Tensor) -> torch.Tensor:
    """Drop the trailing `</s>` that Wan's T5 wrapper appends; keep all content tokens."""
    if t5_out.shape[0] >= 2:
        return t5_out[:-1]
    return t5_out


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-row unit-norm. x: [..., D]."""
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def compute_concept_anchor(
    concept: str,
    templates: list,
    text_encoder,
    text_embedding,
    device,
    global_mean: torch.Tensor = None,
) -> torch.Tensor:
    """Token-level anchor: T5 token embeddings of concept string alone.

    Returns [L_c, 4096] unit-normed per-row (excluding `</s>`). Templates and
    `text_embedding` parameters are accepted for API compatibility but unused
    — encoding the bare concept string maximizes lexical signal.
    """
    del text_embedding, templates
    t5_out = text_encoder([concept], device)[0]   # [L_c+1, 4096]
    tokens = _drop_eos(t5_out).float()
    if global_mean is not None:
        tokens = tokens - global_mean.to(tokens.device).unsqueeze(0)
    return _l2_normalize(tokens)


def compute_gates(
    h_p: torch.Tensor,
    anchors: dict,
    tau: dict,
    T: dict,
    g_min: float = 0.05,
) -> dict:
    """g_c = sigmoid((max_sim − τ_c) / T_c) for token-level max-sim routing.

    h_p: [L_p, 4096] unit-normed prompt tokens.
    anchors: {c → [L_c, 4096]} unit-normed concept tokens.
    g_min: hard cutoff — any gate below this is short-circuited to 0 so the
        corresponding LoRA contribution is fully dropped (matches paper §3.4 +
        Algorithm 1 active-set definition). Default 0.05.
    Returns: {c → float in [0, 1]}.
    """
    gates = {}
    for c, anchor in anchors.items():
        # [L_p, D] @ [D, L_c] → [L_p, L_c]
        sim = h_p @ anchor.t().to(h_p.device)
        max_sim = float(sim.max())
        g = 1.0 / (1.0 + math.exp(-(max_sim - tau[c]) / T[c]))
        gates[c] = 0.0 if g < g_min else g
    return gates


@torch.no_grad()
def pool_prompt_context(
    text_encoder,
    text_embedding,
    prompt: str,
    device,
    global_mean: torch.Tensor = None,
) -> torch.Tensor:
    """prompt → all T5 content tokens (no `</s>`) → unit-normed per row.

    Returns [L_p, 4096]. The function is misnamed historically, it no longer
    pools. Kept for backward compat with calibrate_router / inference.
    """
    del text_embedding
    t5_out = text_encoder([prompt], device)[0]
    tokens = _drop_eos(t5_out).float()
    if global_mean is not None:
        tokens = tokens - global_mean.to(tokens.device).unsqueeze(0)
    return _l2_normalize(tokens)


@torch.no_grad()
def estimate_global_mean(
    text_encoder,
    prompts: list,
    device,
) -> torch.Tensor:
    """Mean per-token T5 embedding over a corpus of prompts (4096-d).

    Token-level max-sim usually doesn't need centering — the discriminative
    signal lives in specific token directions, not in the global angle. Kept
    for ablation purposes; pass `global_mean=None` to disable.
    """
    accum = None
    n_tokens = 0
    for p in prompts:
        t5_out = text_encoder([p], device)[0]
        toks = _drop_eos(t5_out).float()
        if accum is None:
            accum = toks.sum(dim=0)
        else:
            accum = accum + toks.sum(dim=0)
        n_tokens += toks.shape[0]
    return accum / max(n_tokens, 1)
