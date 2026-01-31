from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.behavior import (  # noqa: E402
    Actor,
    Critic,
    behavior_losses,
    behavior_step,
)
from gbxcule.rl.dreamer_v3.dists import TwoHotEncodingDistribution  # noqa: E402
from gbxcule.rl.dreamer_v3.heads import ContinueHead, RewardHead  # noqa: E402
from gbxcule.rl.dreamer_v3.return_ema import ReturnEMA  # noqa: E402
from gbxcule.rl.dreamer_v3.rssm import build_rssm  # noqa: E402


def _build_components():
    torch.manual_seed(1)
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
    target_critic = Critic(
        latent_state_size=latent_dim, bins=5, dense_units=8, mlp_layers=2
    )
    target_critic.load_state_dict(critic.state_dict())
    return rssm, actor, critic, target_critic, reward_model, continue_model


def _make_batch(T: int, B: int):
    posteriors = torch.zeros(T, B, 3, 4, dtype=torch.float32)
    posteriors[..., 0] = 1.0
    recurrent_states = torch.zeros(T, B, 5, dtype=torch.float32)
    terminated = torch.zeros(T, B, 1, dtype=torch.float32)
    return posteriors, recurrent_states, terminated


def test_behavior_losses_finite() -> None:
    rssm, actor, critic, target_critic, reward_model, continue_model = (
        _build_components()
    )
    posteriors, recurrent_states, terminated = _make_batch(2, 3)
    moments = ReturnEMA(decay=0.99, percentiles=(0.05, 0.95), max_value=1.0)
    losses = behavior_losses(
        rssm=rssm,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        reward_model=reward_model,
        continue_model=continue_model,
        posteriors=posteriors,
        recurrent_states=recurrent_states,
        terminated=terminated,
        horizon=3,
        gamma=0.9,
        lmbda=0.95,
        moments=moments,
        greedy=True,
        sample_state=False,
    )
    assert torch.isfinite(losses.policy_loss).item()
    assert torch.isfinite(losses.value_loss).item()

    qv = TwoHotEncodingDistribution(
        critic(losses.imagination.latent_states.detach()[:-1]), dims=1
    )
    target_mean = TwoHotEncodingDistribution(
        target_critic(losses.imagination.latent_states.detach()[:-1]), dims=1
    ).mean
    expected = -qv.log_prob(losses.lambda_values.detach())
    expected = expected - qv.log_prob(target_mean.detach())
    expected = torch.mean(expected * losses.discount[:-1].squeeze(-1))
    assert torch.allclose(losses.value_loss, expected, atol=1e-5, rtol=1e-5)


def test_behavior_step_updates_actor_critic_only() -> None:
    rssm, actor, critic, target_critic, reward_model, continue_model = (
        _build_components()
    )
    posteriors, recurrent_states, terminated = _make_batch(2, 3)
    moments = ReturnEMA(decay=0.99, percentiles=(0.05, 0.95), max_value=1.0)

    actor_before = [p.detach().clone() for p in actor.parameters()]
    critic_before = [p.detach().clone() for p in critic.parameters()]
    reward_before = [p.detach().clone() for p in reward_model.parameters()]

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    behavior_step(
        rssm=rssm,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        reward_model=reward_model,
        continue_model=continue_model,
        posteriors=posteriors,
        recurrent_states=recurrent_states,
        terminated=terminated,
        horizon=3,
        gamma=0.9,
        lmbda=0.95,
        moments=moments,
        actor_optimizer=actor_opt,
        critic_optimizer=critic_opt,
        target_update_freq=1,
        tau=0.5,
        step=0,
        greedy=True,
        sample_state=False,
    )

    assert any(
        not torch.allclose(b, a)
        for b, a in zip(actor_before, actor.parameters(), strict=True)
    )
    assert any(
        not torch.allclose(b, a)
        for b, a in zip(critic_before, critic.parameters(), strict=True)
    )
    assert all(
        torch.allclose(b, a)
        for b, a in zip(reward_before, reward_model.parameters(), strict=True)
    )
    for p, tp in zip(critic.parameters(), target_critic.parameters(), strict=True):
        assert torch.allclose(p, tp)


def test_advantage_normalization_stable() -> None:
    moments_a = ReturnEMA(decay=0.99, percentiles=(0.05, 0.95), max_value=1.0)
    moments_b = ReturnEMA(decay=0.99, percentiles=(0.05, 0.95), max_value=1.0)
    lambda_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).view(-1, 1, 1)
    baseline = lambda_values * 0.5
    scale = 10.0
    lambda_scaled = lambda_values * scale
    baseline_scaled = baseline * scale

    for _ in range(100):
        moments_a.update(lambda_values)
        moments_b.update(lambda_scaled)

    offset_a, invscale_a = moments_a.update(lambda_values)
    adv_a = (lambda_values - offset_a) / invscale_a - (baseline - offset_a) / invscale_a

    offset_b, invscale_b = moments_b.update(lambda_scaled)
    adv_b = (lambda_scaled - offset_b) / invscale_b - (
        baseline_scaled - offset_b
    ) / invscale_b

    assert torch.allclose(adv_a, adv_b, atol=1e-5, rtol=1e-5)
