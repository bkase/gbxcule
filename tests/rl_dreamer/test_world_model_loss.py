from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.dists import (  # noqa: E402
    BernoulliSafeMode,
    MSEDistribution,
    TwoHotEncodingDistribution,
)
from gbxcule.rl.dreamer_v3.world_model import reconstruction_loss  # noqa: E402


def _dummy_dists(t: int, b: int, bins: int):
    obs = torch.zeros((t, b, 1, 2, 2), dtype=torch.float32)
    po = {"pixels": MSEDistribution(obs, dims=3)}
    reward_logits = torch.zeros((t, b, bins), dtype=torch.float32)
    pr = TwoHotEncodingDistribution(reward_logits, dims=1, low=-20.0, high=20.0)
    pc = torch.distributions.Independent(
        BernoulliSafeMode(logits=torch.zeros((t, b, 1))), 1
    )
    return po, pr, pc, obs


def test_kl_stopgrad_paths() -> None:
    torch.manual_seed(0)
    t, b, stoch, discrete = 3, 2, 2, 3
    priors = torch.randn((t, b, stoch, discrete), requires_grad=True)
    posteriors = torch.randn((t, b, stoch, discrete), requires_grad=True)
    po, pr, pc, obs = _dummy_dists(t, b, bins=7)
    rewards = torch.zeros((t, b, 1), dtype=torch.float32)
    continues = torch.ones((t, b, 1), dtype=torch.float32)

    loss, *_ = reconstruction_loss(
        po,
        {"pixels": obs},
        pr,
        rewards,
        priors,
        posteriors,
        kl_dynamic=1.0,
        kl_representation=0.0,
        kl_free_nats=0.0,
        kl_regularizer=1.0,
        pc=pc,
        continue_targets=continues,
        continue_scale_factor=1.0,
    )
    loss.backward()
    assert posteriors.grad is None or torch.allclose(
        posteriors.grad, torch.zeros_like(posteriors.grad)
    )

    priors.grad = None
    posteriors.grad = None
    loss, *_ = reconstruction_loss(
        po,
        {"pixels": obs},
        pr,
        rewards,
        priors,
        posteriors,
        kl_dynamic=0.0,
        kl_representation=1.0,
        kl_free_nats=0.0,
        kl_regularizer=1.0,
        pc=pc,
        continue_targets=continues,
        continue_scale_factor=1.0,
    )
    loss.backward()
    assert priors.grad is None or torch.allclose(
        priors.grad, torch.zeros_like(priors.grad)
    )


def test_free_nats_and_scaling() -> None:
    t, b, stoch, discrete = 2, 1, 2, 3
    priors = torch.zeros((t, b, stoch, discrete), requires_grad=True)
    posteriors = torch.zeros((t, b, stoch, discrete), requires_grad=True)
    po, pr, pc, obs = _dummy_dists(t, b, bins=7)
    rewards = torch.zeros((t, b, 1), dtype=torch.float32)
    continues = torch.ones((t, b, 1), dtype=torch.float32)

    loss, kl, kl_loss, *_ = reconstruction_loss(
        po,
        {"pixels": obs},
        pr,
        rewards,
        priors,
        posteriors,
        kl_dynamic=0.5,
        kl_representation=0.1,
        kl_free_nats=1.0,
        kl_regularizer=2.0,
        pc=pc,
        continue_targets=continues,
        continue_scale_factor=2.0,
    )
    expected_kl_loss = 1.0 * (0.5 + 0.1)
    assert torch.allclose(kl, torch.zeros_like(kl))
    assert torch.allclose(kl_loss, torch.tensor(expected_kl_loss))
    assert loss.item() >= expected_kl_loss
