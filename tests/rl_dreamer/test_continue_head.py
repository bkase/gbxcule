from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.dists import (  # noqa: E402
    BernoulliSafeMode,
    MSEDistribution,
    TwoHotEncodingDistribution,
)
from gbxcule.rl.dreamer_v3.world_model import reconstruction_loss  # noqa: E402


def test_continue_scale_factor_applied() -> None:
    t, b = 2, 1
    obs = torch.zeros((t, b, 1, 2, 2), dtype=torch.float32)
    po = {"pixels": MSEDistribution(obs, dims=3)}
    pr = TwoHotEncodingDistribution(
        torch.zeros((t, b, 7)), dims=1, low=-20.0, high=20.0
    )
    pc = torch.distributions.Independent(
        BernoulliSafeMode(logits=torch.zeros((t, b, 1))), 1
    )
    rewards = torch.zeros((t, b, 1), dtype=torch.float32)
    continues = torch.ones((t, b, 1), dtype=torch.float32)
    priors = torch.zeros((t, b, 2, 3))
    posteriors = torch.zeros((t, b, 2, 3))

    _, _, _, _, _, cont1 = reconstruction_loss(
        po,
        {"pixels": obs},
        pr,
        rewards,
        priors,
        posteriors,
        kl_dynamic=0.0,
        kl_representation=0.0,
        kl_free_nats=0.0,
        kl_regularizer=1.0,
        pc=pc,
        continue_targets=continues,
        continue_scale_factor=1.0,
    )
    _, _, _, _, _, cont2 = reconstruction_loss(
        po,
        {"pixels": obs},
        pr,
        rewards,
        priors,
        posteriors,
        kl_dynamic=0.0,
        kl_representation=0.0,
        kl_free_nats=0.0,
        kl_regularizer=1.0,
        pc=pc,
        continue_targets=continues,
        continue_scale_factor=2.0,
    )
    assert torch.allclose(cont2, cont1 * 2.0)
