from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.goal_match import (  # noqa: E402
    RewardShapingConfig,
    compute_done,
    compute_reward,
    compute_trunc,
    update_consecutive,
)


def test_compute_reward_delta() -> None:
    cfg = RewardShapingConfig(step_cost=-0.01, alpha=1.0, goal_bonus=10.0)
    prev = torch.tensor([0.5, 0.2], dtype=torch.float32)
    dist = torch.tensor([0.4, 0.25], dtype=torch.float32)
    done = torch.tensor([False, False])
    reward = compute_reward(prev, dist, done, cfg)
    expected = torch.tensor([0.09, -0.06], dtype=torch.float32)
    assert torch.allclose(reward, expected, atol=1e-6)


def test_compute_reward_done_bonus() -> None:
    cfg = RewardShapingConfig(step_cost=-0.01, alpha=1.0, goal_bonus=10.0)
    prev = torch.tensor([0.5], dtype=torch.float32)
    dist = torch.tensor([0.4], dtype=torch.float32)
    done = torch.tensor([True])
    reward = compute_reward(prev, dist, done, cfg)
    expected = torch.tensor([10.09], dtype=torch.float32)
    assert torch.allclose(reward, expected, atol=1e-6)


def test_compute_trunc_boundary() -> None:
    steps = torch.tensor([0, 4, 5], dtype=torch.int32)
    trunc = compute_trunc(steps, max_steps=5)
    assert trunc.tolist() == [False, False, True]


def test_consecutive_done_integration() -> None:
    dist = torch.tensor([0.10, 0.02, 0.03], dtype=torch.float32)
    consec = torch.zeros((3,), dtype=torch.int32)
    consec = update_consecutive(consec, dist, tau=0.05)
    done = compute_done(consec, k_consecutive=1)
    assert done.tolist() == [False, True, True]


def test_dtype_validation() -> None:
    cfg = RewardShapingConfig(step_cost=-0.01, alpha=1.0, goal_bonus=1.0)
    prev = torch.tensor([0.1], dtype=torch.float64)
    dist = torch.tensor([0.1], dtype=torch.float32)
    done = torch.tensor([False])
    with pytest.raises(ValueError):
        compute_reward(prev, dist, done, cfg)


def test_shape_validation() -> None:
    cfg = RewardShapingConfig(step_cost=-0.01, alpha=1.0, goal_bonus=1.0)
    prev = torch.tensor([0.1, 0.2], dtype=torch.float32)
    dist = torch.tensor([0.1], dtype=torch.float32)
    done = torch.tensor([False], dtype=torch.bool)
    with pytest.raises(ValueError):
        compute_reward(prev, dist, done, cfg)
