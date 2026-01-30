"""Pixel-only goal matching utilities (torch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.goal_match. Install with `uv sync`."
        ) from exc


@dataclass(frozen=True)
class GoalMatchConfig:
    """Goal matching configuration."""

    tau: float
    k_consecutive: int


@dataclass(frozen=True)
class RewardShapingConfig:
    """Reward shaping configuration."""

    step_cost: float
    alpha: float
    goal_bonus: float


def _validate_goal_shapes(frame, goal) -> None:  # type: ignore[no-untyped-def]
    if frame.ndim not in (3, 4):
        raise ValueError(f"frame must be 3D or 4D, got shape {tuple(frame.shape)}")
    if goal.ndim not in (2, 3):
        raise ValueError(f"goal must be 2D or 3D, got shape {tuple(goal.shape)}")
    if frame.ndim == 3 and goal.ndim != 2:
        raise ValueError("goal must be 2D when frame is 3D")
    if frame.ndim == 4 and goal.ndim == 3 and frame.shape[1] != goal.shape[0]:
        raise ValueError(
            f"goal stack depth {goal.shape[0]} does not match frame {frame.shape[1]}"
        )
    if frame.ndim == 4 and goal.ndim == 2:
        return
    if frame.ndim == 3 and goal.ndim == 2:
        return


def _select_frame_for_goal(frame, goal):  # type: ignore[no-untyped-def]
    _validate_goal_shapes(frame, goal)
    if frame.ndim == 4 and goal.ndim == 2:
        return frame[:, -1], goal
    return frame, goal


def compute_dist_l1(  # type: ignore[no-untyped-def]
    frame, goal, *, denom: float = 3.0
):
    """Compute normalized mean L1 distance between pixels and goal."""
    torch = _require_torch()
    if frame.dtype is not torch.uint8:
        raise ValueError("frame must be uint8")
    if goal.dtype is not torch.uint8:
        raise ValueError("goal must be uint8")
    frame_sel, goal_sel = _select_frame_for_goal(frame, goal)
    if goal_sel.ndim == frame_sel.ndim - 1:
        goal_sel = goal_sel.unsqueeze(0)
    diff = (frame_sel.to(torch.float32) - goal_sel.to(torch.float32)).abs()
    axes = tuple(range(1, diff.ndim))
    dist = diff.mean(dim=axes) / float(denom)
    return dist


def update_consecutive(  # type: ignore[no-untyped-def]
    consec, dist, *, tau: float
):
    """Update consecutive-match counter based on thresholded distance."""
    torch = _require_torch()
    if consec.dtype is not torch.int32:
        raise ValueError("consec must be int32")
    if dist.ndim != 1:
        raise ValueError("dist must be 1D (per-env)")
    if consec.shape != dist.shape:
        raise ValueError("consec and dist must have matching shape")
    zeros = torch.zeros_like(consec)
    return torch.where(dist < float(tau), consec + 1, zeros)


def compute_done(consec, *, k_consecutive: int):  # type: ignore[no-untyped-def]
    """Return done mask when consecutive count reaches K."""
    torch = _require_torch()
    if consec.dtype is not torch.int32:
        raise ValueError("consec must be int32")
    if k_consecutive < 1:
        raise ValueError("k_consecutive must be >= 1")
    return consec >= int(k_consecutive)


def compute_reward(  # type: ignore[no-untyped-def]
    prev_dist,
    dist,
    done,
    cfg: RewardShapingConfig,
):
    """Compute shaped reward from distance delta and done bonus."""
    torch = _require_torch()
    if prev_dist.dtype is not torch.float32:
        raise ValueError("prev_dist must be float32")
    if dist.dtype is not torch.float32:
        raise ValueError("dist must be float32")
    if done.dtype is not torch.bool:
        raise ValueError("done must be bool")
    if prev_dist.shape != dist.shape or prev_dist.shape != done.shape:
        raise ValueError("prev_dist, dist, and done must have matching shape")
    base = float(cfg.step_cost) + float(cfg.alpha) * (prev_dist - dist)
    bonus = done.to(torch.float32) * float(cfg.goal_bonus)
    return base + bonus


def compute_trunc(  # type: ignore[no-untyped-def]
    episode_step,
    *,
    max_steps: int,
):
    """Return trunc mask when episode_step reaches max_steps."""
    torch = _require_torch()
    if episode_step.dtype is not torch.int32:
        raise ValueError("episode_step must be int32")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    return episode_step >= int(max_steps)
