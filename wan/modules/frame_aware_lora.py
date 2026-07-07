import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptPhi(nn.Module):
    """Per-concept (frame, timestep)-conditioned gate for frame-aware LoRA.

    Sigmoid output in [0,1]^r1. We init the bias at 0.0 (sigmoid → 0.5,
    high-slope region) so fc2.bias has live gradient from step 0; the
    earlier −3.0 init landed phi in sigmoid's saturated tail (slope ≈
    0.045), which combined with B1=0 init froze phi for the entire run
    (see ablation_log § C3 for the gradient-deadlock derivation).
    """

    def __init__(self, r1: int = 2, hidden: int = 32, init_bias: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden)
        self.fc2 = nn.Linear(hidden, r1)
        nn.init.constant_(self.fc2.bias, init_bias)

    def _pos_enc(self, f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.sin(2 * math.pi * f),
                torch.cos(2 * math.pi * f),
                torch.sin(2 * math.pi * t),
                torch.cos(2 * math.pi * t),
                torch.sin(4 * math.pi * f),
                torch.cos(4 * math.pi * f),
            ],
            dim=-1,
        )

    def forward(self, f_norm: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        w_dtype = self.fc1.weight.dtype
        enc = self._pos_enc(f_norm.to(w_dtype), t_norm.to(w_dtype))
        h = F.gelu(self.fc1(enc))
        return torch.sigmoid(self.fc2(h))


class FrameAwareLoRA(nn.Module):
    """Frame-aware LoRA: ΔW · x = B₀ D₀ x + B₁ · (φ(f,t) ⊙ D₁ x).

    Two parallel paths:
      - rank-r0 frame-agnostic (always on)
      - rank-r1 frame-aware, gated per-token by ConceptPhi(f, t)
    Init produces zero output (B₀=B₁=0) so the wrapped FFN is unchanged at start.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        r0: int = 4,
        r1: int = 2,
        phi: ConceptPhi | None = None,
    ):
        super().__init__()
        self.r0, self.r1 = r0, r1
        self.D0 = nn.Linear(d_in, r0, bias=False)
        self.B0 = nn.Linear(r0, d_out, bias=False)
        self.D1 = nn.Linear(d_in, r1, bias=False)
        self.B1 = nn.Linear(r1, d_out, bias=False)
        nn.init.kaiming_uniform_(self.D0.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.D1.weight, a=math.sqrt(5))
        # B0 stays zero-init: keeps the agnostic-branch ΔW = 0 at start.
        # B1 is small-kaiming (1% of full kaiming) instead of zero so that
        # ∂L/∂φ via the aware-branch chain `B1 · D1·x` is non-zero from the
        # first step. Init magnitude (~5e-5 per ΔW element) is well below
        # the post-training B0·D0 scale, so the base model is not perturbed
        # at step 0 (see ablation_log § C3 for the derivation).
        nn.init.zeros_(self.B0.weight)
        nn.init.kaiming_uniform_(self.B1.weight, a=math.sqrt(5))
        self.B1.weight.data.mul_(0.01)
        self.phi = phi if phi is not None else ConceptPhi(r1=r1)

    def forward(
        self,
        x: torch.Tensor,
        f_norm: torch.Tensor,
        t_norm: torch.Tensor,
    ) -> torch.Tensor:
        agnostic = self.B0(self.D0(x))
        d1x = self.D1(x)
        gate = self.phi(f_norm, t_norm)
        # gate shape must broadcast with d1x: both end in r1 dim
        aware = self.B1(d1x * gate)
        return agnostic + aware
