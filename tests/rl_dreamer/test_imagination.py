from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.behavior import Actor, Critic  # noqa: E402
from gbxcule.rl.dreamer_v3.heads import ContinueHead, RewardHead  # noqa: E402
from gbxcule.rl.dreamer_v3.imagination import imagine_rollout  # noqa: E402
from gbxcule.rl.dreamer_v3.rssm import build_rssm  # noqa: E402


def _build_components():
    torch.manual_seed(0)
    rssm = build_rssm(
        action_dim=3,
        embed_dim=4,
        stochastic_size=3,
        discrete_size=4,
        recurrent_state_size=5,
        dense_units=8,
        hidden_size=8,
        unimix=0.0,
        layer_norm_eps=1e-3,
        activation="torch.nn.SiLU",
        learnable_initial_recurrent_state=True,
        hafner_init=False,
        rnn_dtype=torch.float32,
    )
    latent_dim = 3 * 4 + 5
    reward_model = RewardHead(input_dim=latent_dim, bins=5, mlp_layers=1, dense_units=8)
    continue_model = ContinueHead(input_dim=latent_dim, mlp_layers=1, dense_units=8)
    actor = Actor(
        latent_state_size=latent_dim,
        actions_dim=(3,),
        is_continuous=False,
        distribution_cfg={"type": "discrete"},
        dense_units=8,
        mlp_layers=2,
        unimix=0.0,
    )
    critic = Critic(latent_state_size=latent_dim, bins=5, dense_units=8, mlp_layers=2)
    return rssm, actor, critic, reward_model, continue_model


def test_imagine_rollout_shapes() -> None:
    rssm, actor, critic, reward_model, continue_model = _build_components()
    T, B = 2, 3
    posteriors = torch.zeros(T, B, 3, 4, dtype=torch.float32)
    posteriors[..., 0] = 1.0
    recurrent_states = torch.zeros(T, B, 5, dtype=torch.float32)
    terminated = torch.zeros(T, B, 1, dtype=torch.float32)
    true_continue = 1 - terminated

    out = imagine_rollout(
        rssm=rssm,
        actor=actor,
        critic=critic,
        reward_model=reward_model,
        continue_model=continue_model,
        posteriors=posteriors,
        recurrent_states=recurrent_states,
        true_continue=true_continue,
        horizon=3,
        gamma=0.9,
        greedy=True,
        sample_state=False,
    )

    batch = T * B
    assert out.latent_states.shape == (4, batch, 3 * 4 + 5)
    assert out.actions.shape == (4, batch, 3)
    assert out.rewards.shape == (4, batch, 1)
    assert out.values.shape == (4, batch, 1)
    assert out.continues.shape == (4, batch, 1)
    assert out.discounts.shape == (4, batch, 1)
    assert torch.all(out.continues[0] == 1.0)
    assert torch.all((out.continues == 0) | (out.continues == 1))
