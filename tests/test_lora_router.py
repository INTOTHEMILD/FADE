from unittest.mock import MagicMock

import torch

from wan.modules.lora_router import (
    compute_concept_anchor,
    compute_gates,
    pool_prompt_context,
)


def test_compute_concept_anchor_token_level():
    """Anchor returns [L_c, 4096] unit-normed per row (T5 tokens minus </s>)."""
    text_encoder = MagicMock(side_effect=lambda prompts, dev: [torch.randn(5, 4096)])
    text_embedding = torch.nn.Linear(4096, 1536)  # accepted but ignored
    h = compute_concept_anchor(
        concept="dog",
        templates=["a photo of <c>"],
        text_encoder=text_encoder,
        text_embedding=text_embedding,
        device=torch.device("cpu"),
    )
    # 5 input tokens minus </s> → 4 content tokens
    assert h.shape == (4, 4096), f"expected [4,4096], got {h.shape}"
    # Per-row unit norm
    assert torch.allclose(h.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_pool_prompt_context_token_level():
    text_encoder = MagicMock(side_effect=lambda prompts, dev: [torch.randn(7, 4096)])
    text_embedding = torch.nn.Linear(4096, 1536)
    h = pool_prompt_context(
        text_encoder=text_encoder,
        text_embedding=text_embedding,
        prompt="a dog running",
        device=torch.device("cpu"),
    )
    # 7 tokens minus </s> → 6 rows
    assert h.shape == (6, 4096)
    assert torch.allclose(h.norm(dim=-1), torch.ones(6), atol=1e-5)


def test_compute_gates_token_max_sim_high():
    """One prompt token aligned with one anchor token → max_sim=1 → g_c≈1."""
    D = 4096
    h_p = torch.zeros(3, D); h_p[1, 0] = 1.0          # token 1 has axis-0 component
    anchor_c1 = torch.zeros(2, D); anchor_c1[0, 0] = 1.0  # anchor[0] also axis-0
    anchor_c2 = torch.zeros(2, D); anchor_c2[0, 1] = 1.0  # anchor on axis-1, no overlap
    anchors = {"c1": anchor_c1, "c2": anchor_c2}
    tau = {"c1": 0.3, "c2": 0.3}
    T = {"c1": 0.05, "c2": 0.05}
    gates = compute_gates(h_p, anchors, tau, T)
    assert gates["c1"] > 0.95, f"high token-match should give high g_c, got {gates['c1']}"
    assert gates["c2"] < 0.05, f"no token-match should give low g_c, got {gates['c2']}"


def test_compute_gates_no_dilution_by_other_tokens():
    """Adding many noise tokens must NOT dilute the one matching token."""
    D = 4096
    h_p = torch.zeros(50, D)
    h_p[7, 0] = 1.0  # only token 7 matches anchor; rest are zero
    anchor = torch.zeros(1, D); anchor[0, 0] = 1.0
    gates = compute_gates(
        h_p,
        {"c": anchor},
        {"c": 0.3}, {"c": 0.05},
    )
    assert gates["c"] > 0.95, (
        f"max-sim should ignore non-matching tokens, got {gates['c']}"
    )
