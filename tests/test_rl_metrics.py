from __future__ import annotations

import torch

from gbxcule.rl.metrics import (
    MetricsAccumulator,
    compute_action_hist,
    compute_dist_percentiles,
    compute_entropy_from_logits,
)


def test_compute_action_hist() -> None:
    actions = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
    hist = compute_action_hist(actions, num_actions=4)
    assert hist.tolist() == [1, 2, 1, 0]


def test_compute_dist_percentiles() -> None:
    dist = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0])
    expected = torch.quantile(dist, torch.tensor([0.1, 0.5, 0.9]))
    actual = compute_dist_percentiles(dist)
    assert torch.allclose(actual, expected)


def test_compute_entropy_from_logits() -> None:
    logits = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    entropy = compute_entropy_from_logits(logits)
    assert entropy.shape == (2,)
    assert entropy[0] > entropy[1]


def test_metrics_accumulator_basic() -> None:
    acc = MetricsAccumulator(num_envs=2, num_actions=3, device="cpu")
    reward = torch.tensor([1.0, 2.0])
    done = torch.tensor([True, False])
    trunc = torch.tensor([False, False])
    reset_mask = done | trunc
    dist = torch.tensor([0.2, 0.8])
    actions = torch.tensor([0, 2], dtype=torch.int64)
    logits = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]])
    values = torch.tensor([0.1, 0.2])

    acc.update(
        reward=reward,
        done=done,
        trunc=trunc,
        reset_mask=reset_mask,
        dist=dist,
        actions=actions,
        logits=logits,
        values=values,
    )
    record = acc.as_record()
    assert record["reward_mean"] == 1.5
    assert record["done_rate"] == 0.5
    assert record["trunc_rate"] == 0.0
    assert record["reset_rate"] == 0.5
    assert record["action_hist"] == [1, 0, 1]
