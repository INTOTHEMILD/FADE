import torch

from wan.modules.temporal_loss import (
    max_frame_leak_loss,
    motion_preserve_loss,
    standard_esd_loss,
    temporal_smooth_loss,
)


def test_max_frame_leak_topk_3():
    """Loss should equal mean of top-3 worst per-frame MSE residuals."""
    T_lat = 5
    eps_pred = torch.zeros(1, 1, T_lat, 1, 1)
    eps_targ = torch.zeros(1, 1, T_lat, 1, 1)
    for f, val in enumerate([1, 2, 3, 4, 5]):
        eps_pred[0, 0, f, 0, 0] = val
    # per-frame squared diff: [1, 4, 9, 16, 25]; top-3 = [9, 16, 25]; mean = 50/3
    L = max_frame_leak_loss(eps_pred, eps_targ, k=3)
    assert torch.allclose(L, torch.tensor(50.0 / 3.0), atol=1e-3), f"got {L}"


def test_max_frame_leak_k_capped_at_T():
    """If k > T_lat, k should be clamped to T_lat (and result == mean over all)."""
    eps_pred = torch.tensor([[[[[1.0]], [[2.0]]]]])  # [1,1,2,1,1]
    eps_targ = torch.zeros_like(eps_pred)
    L = max_frame_leak_loss(eps_pred, eps_targ, k=10)
    assert torch.allclose(L, torch.tensor(2.5), atol=1e-5)


def test_max_frame_leak_4d_input():
    """4D [C, T, H, W] (Wan single-sample shape) must work."""
    T_lat = 5
    eps_pred = torch.zeros(1, T_lat, 1, 1)
    eps_targ = torch.zeros(1, T_lat, 1, 1)
    for f, val in enumerate([1, 2, 3, 4, 5]):
        eps_pred[0, f, 0, 0] = val
    L = max_frame_leak_loss(eps_pred, eps_targ, k=3)
    assert torch.allclose(L, torch.tensor(50.0 / 3.0), atol=1e-3)


def test_temporal_smooth_zero_for_constant_phi():
    phi_per_frame = torch.ones(21, 2)
    L = temporal_smooth_loss(phi_per_frame)
    assert L.item() < 1e-9


def test_temporal_smooth_nonzero_for_varying_phi():
    phi = torch.zeros(5, 2)
    phi[2, 0] = 1.0
    L = temporal_smooth_loss(phi)
    assert L.item() > 0


def test_motion_preserve_zero_for_identical():
    eps = torch.randn(2, 3, 9, 4, 4)
    L = motion_preserve_loss(eps, eps.clone())
    assert L.item() < 1e-9


def test_standard_esd_matches_mse():
    pred = torch.randn(1, 4, 9, 8, 8)
    targ = torch.randn(1, 4, 9, 8, 8)
    L = standard_esd_loss(pred, targ)
    expected = (pred - targ).pow(2).mean()
    assert torch.allclose(L, expected)


def test_total_loss_weighted_sum():
    eps_pred = torch.randn(1, 16, 9, 60, 104)
    eps_targ = torch.randn(1, 16, 9, 60, 104)
    eps_neu_n = torch.randn(1, 16, 9, 60, 104)
    eps_neu_o = torch.randn(1, 16, 9, 60, 104)
    phi = torch.rand(9, 2)
    L = (
        standard_esd_loss(eps_pred, eps_targ)
        + 0.5 * max_frame_leak_loss(eps_pred, eps_targ, k=3)
        + 0.5 * motion_preserve_loss(eps_neu_n, eps_neu_o)
        + 0.1 * temporal_smooth_loss(phi)
    )
    assert L.item() > 0
    assert L.requires_grad is False
