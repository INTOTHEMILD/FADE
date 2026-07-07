import torch
import torch.nn as nn


class GatedFFN(nn.Module):
    """Frozen FFN + per-concept FrameAwareLoRA paths gated by per-prompt scalars.

    Wan's blocks call `self.ffn(x)` positionally with x only, so per-token frame
    index and per-call timestep are stashed via set_frame_time(). Per-concept
    routing scalars come from set_gates(); g_c < SKIP_THRESHOLD short-circuits.
    """

    SKIP_THRESHOLD = 0.5

    def __init__(self, original_ffn: nn.Module, concept_loras: dict, anchors: dict):
        super().__init__()
        self.ffn = original_ffn
        self.loras = nn.ModuleDict(concept_loras)
        self.anchors = anchors
        self._gates: dict = {}
        self._f_norm = None
        self._t_norm = 0.0

    def set_gates(self, gates: dict):
        self._gates = gates

    def set_frame_time(self, f_norm: torch.Tensor, t_norm: float):
        """Stash per-token f_norm [seq_len] and scalar t_norm for next forward."""
        self._f_norm = f_norm
        self._t_norm = float(t_norm)

    def forward(self, x):
        out = self.ffn(x)
        if not self.loras or not self._gates:
            return out

        f = self._f_norm
        if f is None or x.dim() < 2 or f.shape[0] != x.shape[1]:
            return out  # frame info missing or shape mismatch → skip LoRA

        t = torch.full_like(f, self._t_norm)
        # Broadcast f, t to [B, N]
        f_b = f.unsqueeze(0).expand(x.shape[0], -1)
        t_b = t.unsqueeze(0).expand(x.shape[0], -1)
        for c_name, lora in self.loras.items():
            g = float(self._gates.get(c_name, 0.0))
            if g < self.SKIP_THRESHOLD:
                continue
            out = out + g * lora(x, f_b, t_b)
        return out
