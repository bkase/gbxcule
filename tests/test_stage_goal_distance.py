from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.stage_goal_distance import compute_stage_goal_distance  # noqa: E402


def _ref_distance(obs_u8, goals_f, env_stages):  # type: ignore[no-untyped-def]
    num_stages = int(goals_f.shape[0])
    out = torch.zeros((obs_u8.shape[0],), dtype=torch.float32)
    for stage_idx in range(num_stages):
        mask = env_stages == stage_idx
        if torch.any(mask):
            diff = (obs_u8[mask].float() - goals_f[stage_idx]).abs()
            out[mask] = diff.mean(dim=(1, 2, 3)) / 3.0
    return out


def test_compute_stage_goal_distance_matches_reference() -> None:
    torch.manual_seed(0)
    num_envs = 8
    num_stages = 3
    obs = torch.randint(0, 4, (num_envs, 1, 72, 80), dtype=torch.uint8)
    goals = torch.randint(0, 4, (num_stages, 1, 72, 80), dtype=torch.uint8).float()
    env_stages = torch.randint(0, num_stages, (num_envs,), dtype=torch.int64)

    expected = _ref_distance(obs, goals, env_stages)
    got = compute_stage_goal_distance(obs, goals, env_stages)
    assert torch.allclose(got, expected)
