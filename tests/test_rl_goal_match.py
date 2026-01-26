from __future__ import annotations

import pytest

from gbxcule.rl.goal_match import compute_dist_l1, compute_done, update_consecutive

torch = pytest.importorskip("torch")


def test_dist_last_frame() -> None:
    frame = torch.zeros((2, 4, 2, 2), dtype=torch.uint8)
    goal = torch.zeros((2, 2), dtype=torch.uint8)
    frame[0, -1] = 3
    dist = compute_dist_l1(frame, goal)
    assert dist.shape == (2,)
    assert dist[0].item() == pytest.approx(1.0)
    assert dist[1].item() == pytest.approx(0.0)


def test_dist_stack() -> None:
    frame = torch.zeros((1, 2, 2, 2), dtype=torch.uint8)
    goal = torch.zeros((2, 2, 2), dtype=torch.uint8)
    frame[0, 0] = 3
    goal[1] = 3
    dist = compute_dist_l1(frame, goal)
    assert dist.shape == (1,)
    assert dist[0].item() == pytest.approx(1.0)


def test_consecutive_and_done() -> None:
    consec = torch.tensor([1, 1], dtype=torch.int32)
    dist = torch.tensor([0.1, 0.01], dtype=torch.float32)
    updated = update_consecutive(consec, dist, tau=0.05)
    assert updated.tolist() == [0, 2]
    done = compute_done(updated, k_consecutive=2)
    assert done.tolist() == [False, True]


def test_shape_validation_errors() -> None:
    frame = torch.zeros((2, 2), dtype=torch.uint8)
    goal = torch.zeros((2, 2), dtype=torch.uint8)
    with pytest.raises(ValueError):
        compute_dist_l1(frame, goal)
