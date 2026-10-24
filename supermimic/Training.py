"""Small training/reset operations independent of the Isaac Lab runtime."""

from __future__ import annotations

import torch


@torch.no_grad()
def update_bin_returns(bin_G: torch.Tensor, bins: torch.Tensor, returns: torch.Tensor, decay: float = 0.8) -> None:
    """Update each observed bin once, using the mean of its finished episodes.

    Repeated advanced-index assignment is not a reduction: several environments
    can finish in the same motion bin on a single simulation step. Aggregate first
    so the EMA does not depend on which duplicate assignment happens last.
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be in [0, 1]")
    if bin_G.ndim != 1 or bins.ndim != 1 or returns.shape != bins.shape:
        raise ValueError("bin_G, bins and returns must be matching one-dimensional inputs")
    if bins.numel() == 0:
        return
    observed, inverse = torch.unique(bins, sorted=True, return_inverse=True)
    sums = torch.zeros(observed.numel(), dtype=bin_G.dtype, device=bin_G.device)
    sums.scatter_add_(0, inverse, returns.to(dtype=bin_G.dtype))
    counts = torch.bincount(inverse, minlength=observed.numel()).to(bin_G.dtype)
    bin_G[observed] = decay * bin_G[observed] + (1.0 - decay) * sums / counts


def warmup_steps_needed(buffer_size: int, capacity: int, wm_window: int, warmup_steps: int) -> int:
    """Number of newly collected time steps needed before model sampling.

    Checkpoints do not contain the replay buffer. The same rule therefore applies
    to fresh runs and resumed runs, regardless of the restored iteration number.
    A requested warmup larger than capacity is capped; a window larger than the
    buffer is impossible and is rejected rather than entering training.
    """
    if wm_window < 1:
        raise ValueError("wm_window must be positive")
    if capacity < wm_window:
        raise ValueError("replay capacity must be at least wm_window")
    if not 0 <= buffer_size <= capacity:
        raise ValueError("buffer_size must lie between zero and capacity")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    target = min(capacity, max(wm_window, warmup_steps))
    return max(0, target - buffer_size)


@torch.no_grad()
def terrain_reset_mask(root_xy: torch.Tensor, body_xy: torch.Tensor, boxes, margin: float = 0.35) -> torch.Tensor:
    """Conservatively keep reference frames away from raised box footprints.

    Each root is surrounded by a square covering the maximum horizontal reach
    of its tracked body centers plus ``margin`` metres for body geometry and reset
    perturbations. Reject intersections with every raised box, even if an arm is
    above the box. This is a conservative frame filter, not a mesh collision test.
    It never modifies reference poses, velocities or cached forward kinematics.
    """
    if root_xy.ndim != 2 or root_xy.shape[-1] != 2 or body_xy.ndim != 3 or body_xy.shape[0] != root_xy.shape[0] or body_xy.shape[-1] != 2:
        raise ValueError("root_xy must be [frames,2] and body_xy [frames,bodies,2]")
    if margin < 0 or body_xy.shape[1] == 0:
        raise ValueError("margin must be nonnegative and at least one tracked body is required")
    safe = torch.isfinite(root_xy).all(-1) & torch.isfinite(body_xy).all(dim=(-1, -2))
    if not boxes:
        return safe
    radius = torch.linalg.vector_norm(body_xy - root_xy[:, None, :], dim=-1).amax(-1) + margin
    for cx, cy, sx, sy, height in boxes:
        if height <= 0:
            continue
        intersects = ((root_xy[:, 0] - cx).abs() <= sx / 2 + radius) & ((root_xy[:, 1] - cy).abs() <= sy / 2 + radius)
        safe &= ~intersects
    return safe


@torch.no_grad()
def build_reset_candidates(bin_start: torch.Tensor, bin_clip_end: torch.Tensor, safe_frames: torch.Tensor,
                           bin_size: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """Build packed valid frames per bin; exclude each clip's last frame.

    Returns ``(frames[num_bins, bin_size], counts[num_bins])``. Only each row's
    first ``counts[b]`` entries are meaningful. Precomputation replaces unbounded
    rejection loops when some or all reference poses overlap an obstacle.
    """
    if bin_size < 1 or bin_start.ndim != 1 or bin_clip_end.shape != bin_start.shape or safe_frames.ndim != 1:
        raise ValueError("invalid reset candidate dimensions or bin_size")
    if safe_frames.numel() == 0:
        raise ValueError("No safe reference reset frames: motion library is empty")
    frames = bin_start[:, None] + torch.arange(bin_size, device=bin_start.device)
    valid = (frames >= 0) & (frames < bin_clip_end[:, None]) & (frames < safe_frames.numel())
    valid &= safe_frames[frames.clamp(0, safe_frames.numel() - 1)]
    counts = valid.sum(-1)
    if not bool((counts > 0).any()):
        raise ValueError("No safe reference reset frames remain for this terrain; use flat terrain or compatible motions")
    packed = torch.where(valid, frames, safe_frames.numel()).sort(dim=-1).values
    packed = torch.where(packed < safe_frames.numel(), packed, 0)
    return packed, counts
