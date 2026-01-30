"""Packed2-friendly metrics for RL."""

from __future__ import annotations

from typing import Any

from gbxcule.rl.packed_pixels import get_diff_lut


def _require_torch() -> Any:
    import importlib

    return importlib.import_module("torch")


def packed_l1_distance(  # type: ignore[no-untyped-def]
    packed_obs,
    packed_goal,
):
    """Compute normalized L1 distance between packed2 observations and goal."""
    torch = _require_torch()
    if packed_obs.dtype is not torch.uint8:
        raise ValueError("packed_obs must be uint8")
    if packed_goal.dtype is not torch.uint8:
        raise ValueError("packed_goal must be uint8")
    if packed_obs.ndim != 4:
        raise ValueError("packed_obs must be [N, 1, H, W_bytes]")
    if packed_goal.ndim == 3:
        packed_goal = packed_goal.unsqueeze(0)
    if packed_goal.ndim != 4:
        raise ValueError("packed_goal must be [1, 1, H, W_bytes] or [1, H, W_bytes]")
    if packed_goal.shape[1] != 1:
        raise ValueError("packed_goal must have channel dim = 1")
    if packed_goal.shape[2:] != packed_obs.shape[2:]:
        raise ValueError("packed_goal spatial shape mismatch")

    diff_lut = get_diff_lut(device=packed_obs.device)
    diff = diff_lut[packed_obs.to(torch.int64), packed_goal.to(torch.int64)]
    diff_sum = diff.to(torch.float32).sum(dim=(1, 2, 3))
    num_pixels = int(packed_obs.shape[2] * packed_obs.shape[3] * 4)
    return diff_sum / float(num_pixels * 3)
