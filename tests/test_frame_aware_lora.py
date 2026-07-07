import torch

from wan.modules.frame_aware_lora import ConceptPhi, FrameAwareLoRA


def test_concept_phi_shape():
    phi = ConceptPhi(r1=2, hidden=32)
    f_norm = torch.tensor([0.0, 0.5, 1.0])
    t_norm = torch.tensor([0.0, 0.5, 1.0])
    out = phi(f_norm, t_norm)
    assert out.shape == (3, 2), f"expected [3, 2], got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "phi outputs must be in [0,1]"


def test_concept_phi_default_init_bias_is_zero():
    """Default init_bias must be 0.0 → gate ≈ sigmoid(0) = 0.5 in high-slope
    region. Earlier −3 default landed phi in sigmoid's saturated tail
    (slope ≈ 0.045), causing the training deadlock documented in
    ablation_log § C3. Regression guard.
    """
    phi = ConceptPhi(r1=2, hidden=32)
    assert torch.allclose(
        phi.fc2.bias, torch.zeros_like(phi.fc2.bias)
    ), f"default fc2.bias must be 0.0, got {phi.fc2.bias.tolist()}"


def test_concept_phi_explicit_low_init_still_supported():
    """Caller can still request the low-gate init explicitly for ablation."""
    phi = ConceptPhi(r1=2, hidden=32, init_bias=-3.0)
    f = torch.tensor([0.5])
    t = torch.tensor([0.5])
    out = phi(f, t)
    assert (out < 0.2).all(), f"explicit low-bias init gate should be low, got {out}"


def test_concept_phi_token_dim():
    """ConceptPhi must accept [B, N] inputs and return [B, N, r1]."""
    phi = ConceptPhi(r1=2, hidden=32)
    f = torch.rand(2, 100)
    t = torch.rand(2, 100)
    out = phi(f, t)
    assert out.shape == (2, 100, 2)


def test_frame_aware_lora_init_output_negligible():
    """B0 stays zero-init (agnostic branch = 0), but B1 is small-kaiming so
    the aware branch is non-zero from step 0 — needed to keep ∂L/∂φ alive
    (see ablation_log § C3). Init output should still be negligible
    relative to typical FFN outputs (< 1e-3 RMS), not strictly zero.
    """
    lora = FrameAwareLoRA(d_in=1536, d_out=6144, r0=4, r1=2)
    x = torch.randn(2, 100, 1536)
    f = torch.zeros(2, 100)
    t = torch.zeros(2, 100)
    out = lora(x, f, t)
    assert out.shape == (2, 100, 6144)
    rms = out.pow(2).mean().sqrt().item()
    # Threshold: B1 small-kaiming × 0.5 phi × D1 kaiming gives rms ≈ 1.5e-3.
    # The bound is "much smaller than typical FFN output rms ≈ 1", not strictly zero.
    assert rms < 5e-3, f"init LoRA output should be small, got rms={rms}"
    # B0 must still be exactly zero so the agnostic branch contributes nothing
    assert torch.all(lora.B0.weight == 0), "B0 must remain zero-init"


def test_frame_aware_lora_b1_nonzero_init_unlocks_phi_gradient():
    """Regression guard for ablation_log § C3 (ConceptPhi gradient deadlock).

    With B1 zero-init (the original buggy state), ∂L/∂fc2.bias ≡ 0 because
    the aware-branch gradient chain `∂L/∂φ = (∂L/∂aware) · B1 · D1·x`
    multiplies by B1=0. With the fixed small-kaiming init, fc2.bias must
    receive non-zero gradient on a non-trivial loss.
    """
    lora = FrameAwareLoRA(d_in=64, d_out=128, r0=2, r1=2)
    x = torch.randn(1, 4, 64)
    f = torch.linspace(0, 1, 4).unsqueeze(0)
    t = torch.full((1, 4), 0.5)

    # Fixed init: fc2.bias must get non-zero gradient.
    out = lora(x, f, t)
    out.sum().backward()
    assert lora.phi.fc2.bias.grad.abs().max() > 1e-6, (
        "fc2.bias gradient is zero — phi cannot train; revert means lock 1 is back"
    )

    # Counterfactual: zero out B1 (the old buggy state) and re-check.
    lora.zero_grad()
    with torch.no_grad():
        lora.B1.weight.zero_()
    out = lora(x, f, t)
    out.sum().backward()
    assert lora.phi.fc2.bias.grad.abs().max() < 1e-9, (
        "with B1=0 (old init), fc2.bias must have ~zero gradient — if not, "
        "the deadlock derivation in ablation_log § C3 is wrong"
    )


def test_frame_aware_lora_param_count():
    lora = FrameAwareLoRA(d_in=1536, d_out=6144, r0=4, r1=2)
    n = sum(p.numel() for p in lora.parameters())
    # 7680*r0 + 7680*r1 + small phi MLP ≈ 30720 + 15360 + 290 ≈ 46370
    assert 40000 < n < 60000, f"unexpected param count {n}"


def test_frame_aware_lora_nonzero_after_train_step():
    """After a fake training step, LoRA output should be non-zero."""
    lora = FrameAwareLoRA(d_in=64, d_out=128, r0=2, r1=2)
    # Manually push B0/B1 off zero
    with torch.no_grad():
        lora.B0.weight.normal_(std=0.01)
        lora.B1.weight.normal_(std=0.01)
    x = torch.randn(1, 4, 64)
    f = torch.tensor([[0.0, 0.25, 0.5, 0.75]])
    t = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    out = lora(x, f, t)
    assert out.shape == (1, 4, 128)
    assert not torch.allclose(out, torch.zeros_like(out))
