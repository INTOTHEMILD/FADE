import torch

from wan.modules.frame_aware_lora import FrameAwareLoRA
from wan.modules.gated_ffn import GatedFFN


def _set_frame_time(gffn, n_tokens, t_norm=0.5):
    f = torch.linspace(0, 1, n_tokens)
    gffn.set_frame_time(f, t_norm)


def test_gated_ffn_passthrough_no_loras():
    """No LoRAs → output equals original FFN."""
    inner = torch.nn.Linear(8, 32)
    gffn = GatedFFN(inner, concept_loras={}, anchors={})
    x = torch.randn(1, 5, 8)
    _set_frame_time(gffn, 5)
    out = gffn(x)
    expected = inner(x)
    assert torch.allclose(out, expected), "passthrough broken"


def test_gated_ffn_passthrough_no_frame_info():
    """LoRAs registered but no frame info set → still passthrough."""
    inner = torch.nn.Linear(8, 32)
    lora = FrameAwareLoRA(d_in=8, d_out=32, r0=2, r1=2)
    with torch.no_grad():
        lora.B0.weight.fill_(1.0)
    gffn = GatedFFN(inner, concept_loras={"c1": lora}, anchors={"c1": torch.zeros(8)})
    gffn.set_gates({"c1": 1.0})
    # NO set_frame_time
    x = torch.randn(1, 5, 8)
    out = gffn(x)
    assert torch.allclose(out, inner(x), atol=1e-6), "missing frame info should bypass LoRA"


def test_gated_ffn_skip_below_threshold():
    inner = torch.nn.Linear(8, 32)
    lora = FrameAwareLoRA(d_in=8, d_out=32, r0=2, r1=2)
    with torch.no_grad():
        lora.B0.weight.fill_(1.0)
    gffn = GatedFFN(inner, concept_loras={"c1": lora}, anchors={"c1": torch.zeros(8)})
    x = torch.randn(1, 3, 8)
    _set_frame_time(gffn, 3)
    gffn.set_gates({"c1": 0.01})
    out_skipped = gffn(x)
    gffn.set_gates({"c1": 0.5})
    out_active = gffn(x)
    assert not torch.allclose(out_skipped, out_active), "threshold skip broken"
    assert torch.allclose(out_skipped, inner(x), atol=1e-6), "below-threshold leaked"


def test_gated_ffn_additive_multiple_concepts():
    inner = torch.nn.Linear(8, 32)
    l1 = FrameAwareLoRA(d_in=8, d_out=32, r0=2, r1=2)
    l2 = FrameAwareLoRA(d_in=8, d_out=32, r0=2, r1=2)
    with torch.no_grad():
        l1.B0.weight.normal_(std=0.01)
        l2.B0.weight.normal_(std=0.01)
    gffn = GatedFFN(
        inner,
        concept_loras={"c1": l1, "c2": l2},
        anchors={"c1": torch.zeros(8), "c2": torch.zeros(8)},
    )
    x = torch.randn(1, 3, 8)
    _set_frame_time(gffn, 3)
    gffn.set_gates({"c1": 0.5, "c2": 0.0})
    out_c1_only = gffn(x)
    gffn.set_gates({"c1": 0.0, "c2": 0.5})
    out_c2_only = gffn(x)
    gffn.set_gates({"c1": 0.5, "c2": 0.5})
    out_both = gffn(x)
    base = inner(x)
    expected = base + (out_c1_only - base) + (out_c2_only - base)
    assert torch.allclose(out_both, expected, atol=1e-5)


def test_gated_ffn_frame_aware_changes_with_phi():
    """Different t_norm should produce different outputs once LoRA is non-zero."""
    inner = torch.nn.Linear(8, 32)
    lora = FrameAwareLoRA(d_in=8, d_out=32, r0=2, r1=2)
    with torch.no_grad():
        lora.B1.weight.normal_(std=0.5)  # frame-aware path active
    gffn = GatedFFN(inner, concept_loras={"c1": lora}, anchors={"c1": torch.zeros(8)})
    gffn.set_gates({"c1": 1.0})
    x = torch.randn(1, 5, 8)

    f = torch.linspace(0, 1, 5)
    gffn.set_frame_time(f, t_norm=0.0)
    out_t0 = gffn(x)
    gffn.set_frame_time(f, t_norm=0.9)
    out_t1 = gffn(x)
    assert not torch.allclose(out_t0, out_t1, atol=1e-5), "t_norm change had no effect"
