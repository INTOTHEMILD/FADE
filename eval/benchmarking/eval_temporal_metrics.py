"""Per-frame max-protocol + reactivation_gap + warping + T-LPIPS.

Heavy deps (lpips, torchvision.models.optical_flow) are imported lazily so
unit tests on the pure-numpy aggregation helpers run without them installed.
"""
from __future__ import annotations

import numpy as np
import torch


TAU = 0.5


def per_frame_metrics(per_frame_conf) -> dict:
    """Per-video metrics from a 1-D list/array of per-frame hit confidences."""
    s = np.asarray(per_frame_conf, dtype=np.float64)
    return {
        "per_frame": s.tolist(),
        "mean_frame_acc": float(s.mean()),
        "max_frame_acc": float(s.max()),
        "video_acc_mean": int(float(s.mean()) > TAU),
        "video_acc_max": int(float(s.max()) > TAU),
        "reactivation_gap": float(s.max() - s.mean()),
    }


def aggregate_method_concept(video_metrics: list) -> dict:
    """Aggregate a list of per-video metrics into one (method, concept) row."""
    n = len(video_metrics)
    if n == 0:
        return {"n_videos": 0}
    acc_mean = [v["video_acc_mean"] for v in video_metrics]
    acc_max = [v["video_acc_max"] for v in video_metrics]
    per_frame_arr = np.asarray([v["per_frame"] for v in video_metrics])
    return {
        "n_videos": n,
        "unlearn_acc_mean": float(np.mean(acc_mean)),
        "unlearn_acc_max": float(np.mean(acc_max)),
        "mean_max_gap": float(np.mean(acc_max) - np.mean(acc_mean)),
        "reactivation_rate": float(
            np.mean([(am == 0) and (amx == 1) for am, amx in zip(acc_mean, acc_max)])
        ),
        "per_frame_curve": per_frame_arr.mean(axis=0).tolist(),
    }


def _build_warp_grid(flow: torch.Tensor) -> torch.Tensor:
    """Pixel-space optical flow → grid_sample [-1, 1] coords."""
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=flow.device),
        torch.arange(W, device=flow.device),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=0).float()
    new = base.unsqueeze(0) + flow
    new[:, 0] = 2 * new[:, 0] / max(W - 1, 1) - 1
    new[:, 1] = 2 * new[:, 1] / max(H - 1, 1) - 1
    return new.permute(0, 2, 3, 1)


def warping_error(frames: torch.Tensor) -> float:
    """RAFT-flow warping error (mean squared) between consecutive frames.

    frames: [N, 3, H, W] in [0, 1]. Returns scalar.
    """
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    raft = raft_large(weights=weights, progress=False).eval().to(frames.device)
    transforms = weights.transforms()
    f1 = frames[:-1]
    f2 = frames[1:]
    f1_t, f2_t = transforms(f1, f2)
    with torch.no_grad():
        flow = raft(f1_t, f2_t)[-1]
    grid = _build_warp_grid(flow)
    warped = torch.nn.functional.grid_sample(f1, grid, align_corners=True)
    return float(((warped - f2) ** 2).mean())


def temporal_lpips(frames: torch.Tensor) -> float:
    """LPIPS between consecutive frames. frames: [N, 3, H, W] in [-1, 1]."""
    import lpips

    net = lpips.LPIPS(net="alex").eval().to(frames.device)
    f1 = frames[:-1]
    f2 = frames[1:]
    with torch.no_grad():
        d = net(f1, f2)
    return float(d.mean())
