import torch


def standard_esd_loss(eps_pred: torch.Tensor, eps_target: torch.Tensor) -> torch.Tensor:
    """Plain MSE used in classic ESD."""
    return (eps_pred - eps_target).pow(2).mean()


def max_frame_leak_loss(
    eps_pred: torch.Tensor,
    eps_target: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """Top-k mean of per-frame MSE residuals; addresses (β) reactivation.

    Accepts 4D [C, T, H, W] (single-sample, as Wan returns) or 5D [B, C, T, H, W].
    T_lat is always the antepenultimate dim. Per-frame loss is averaged over all
    other dims; the top-k worst frames are then averaged.
    """
    diff = (eps_pred - eps_target).pow(2)
    if diff.dim() < 3:
        raise ValueError(f"need at least 3 dims (T,H,W); got {diff.shape}")
    t_axis = diff.dim() - 3
    reduce_dims = tuple(d for d in range(diff.dim()) if d != t_axis)
    L_per = diff.mean(dim=reduce_dims)
    k_eff = min(k, L_per.shape[0])
    return L_per.topk(k_eff).values.mean()


def motion_preserve_loss(
    eps_new_neutral: torch.Tensor,
    eps_old_neutral: torch.Tensor,
) -> torch.Tensor:
    """MSE on neutral motion-rich prompts; addresses (α) quality drift."""
    return (eps_new_neutral - eps_old_neutral).pow(2).mean()


def temporal_smooth_loss(phi_per_frame: torch.Tensor) -> torch.Tensor:
    """Mean ‖φ(f)−φ(f+1)‖² along frame axis; regularizes ConceptPhi.

    phi_per_frame shape: [T_lat, r1] (one timestep slice).
    """
    if phi_per_frame.shape[0] < 2:
        return torch.tensor(0.0, device=phi_per_frame.device)
    diff = phi_per_frame[1:] - phi_per_frame[:-1]
    return diff.pow(2).mean()
