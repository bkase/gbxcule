"""Stage-goal distance helpers for multi-stage RL."""

from __future__ import annotations


def compute_stage_goal_distance(obs_u8, stage_goals_f, env_stages):  # type: ignore[no-untyped-def]
    """Compute per-env distance to current stage goal.

    Args:
        obs_u8: uint8 tensor shaped [N, 1, 72, 80].
        stage_goals_f: float tensor shaped [S, 1, 72, 80].
        env_stages: int tensor shaped [N] with values in [0, S-1].
    """
    obs_f = obs_u8.to(dtype=stage_goals_f.dtype)
    env_goals = stage_goals_f[env_stages]
    diff = (obs_f - env_goals).abs()
    return diff.mean(dim=(1, 2, 3)) / 3.0
