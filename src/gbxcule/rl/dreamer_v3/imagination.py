"""Imagination rollout utilities for Dreamer v3 behavior learning (M5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gbxcule.rl.dreamer_v3 import dists as dv3_dists


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


@dataclass
class ImaginationOutput:
    latent_states: Any
    actions: Any
    rewards: Any
    values: Any
    continues: Any
    discounts: Any


def compute_discounts(continues, *, gamma: float):  # type: ignore[no-untyped-def]
    """Compute discount weights (sheeprl parity).

    discount = cumprod(continues * gamma) / gamma
    """
    torch = _require_torch()
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    return torch.cumprod(continues * gamma, dim=0) / gamma


def _flatten_time_batch(x):  # type: ignore[no-untyped-def]
    if x.dim() < 2:
        raise ValueError("input must be at least 2D time-major [T, B, ...]")
    t, b = x.shape[:2]
    return x.reshape(1, t * b, *x.shape[2:])


def imagine_rollout(
    *,
    rssm,  # type: ignore[no-untyped-def]
    actor,  # type: ignore[no-untyped-def]
    critic,  # type: ignore[no-untyped-def]
    reward_model,  # type: ignore[no-untyped-def]
    continue_model,  # type: ignore[no-untyped-def]
    posteriors,  # type: ignore[no-untyped-def]
    recurrent_states,  # type: ignore[no-untyped-def]
    true_continue,  # type: ignore[no-untyped-def]
    horizon: int,
    gamma: float,
    reward_low: float = -20.0,
    reward_high: float = 20.0,
    value_low: float = -20.0,
    value_high: float = 20.0,
    greedy: bool = False,
    generator=None,  # type: ignore[no-untyped-def]
    sample_state: bool = True,
) -> ImaginationOutput:
    """Roll out imagined trajectories (H+1) starting from posterior states.

    Args:
        posteriors: [T, B, stoch, discrete]
        recurrent_states: [T, B, recurrent]
        true_continue: [1, T*B, 1] or [T, B, 1]
    """
    torch = _require_torch()
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    prior = _flatten_time_batch(posteriors).detach()
    recurrent = _flatten_time_batch(recurrent_states).detach()
    prior_flat = prior.view(prior.shape[0], prior.shape[1], -1)
    latent_state = torch.cat((prior_flat, recurrent), dim=-1)

    batch = latent_state.shape[1]
    latent_dim = latent_state.shape[-1]

    latent_states = torch.empty(
        horizon + 1,
        batch,
        latent_dim,
        device=latent_state.device,
        dtype=latent_state.dtype,
    )
    actions = None
    latent_states[0] = latent_state.squeeze(0)

    actions_dim = actor.actions_dim
    total_action_dim = int(sum(actions_dim))
    actions = torch.empty(
        horizon + 1,
        batch,
        total_action_dim,
        device=latent_state.device,
        dtype=latent_state.dtype,
    )

    action, _ = actor(latent_state.detach(), greedy=greedy, generator=generator)
    actions[0] = torch.cat(action, dim=-1).squeeze(0)

    current_prior = prior_flat
    current_recurrent = recurrent

    for i in range(1, horizon + 1):
        imagined_prior, current_recurrent = rssm.imagination(
            current_prior,
            current_recurrent,
            actions[i - 1 : i],
            sample_state=sample_state,
        )
        current_prior = imagined_prior.view(1, batch, -1)
        latent_state = torch.cat((current_prior, current_recurrent), dim=-1)
        latent_states[i] = latent_state.squeeze(0)
        action, _ = actor(latent_state.detach(), greedy=greedy, generator=generator)
        actions[i] = torch.cat(action, dim=-1).squeeze(0)

    reward_logits = reward_model(latent_states)
    rewards = dv3_dists.TwoHotEncodingDistribution(
        reward_logits, dims=1, low=reward_low, high=reward_high
    ).mean
    value_logits = critic(latent_states)
    values = dv3_dists.TwoHotEncodingDistribution(
        value_logits, dims=1, low=value_low, high=value_high
    ).mean

    if continue_model is None:
        continues = torch.ones_like(rewards)
    else:
        continue_logits = continue_model(latent_states)
        continues = dv3_dists.BernoulliSafeMode(logits=continue_logits).mode
    if true_continue.dim() == 2:
        true_continue = true_continue.unsqueeze(-1)
    if true_continue.dim() == 3 and true_continue.shape[0] != 1:
        true_continue = _flatten_time_batch(true_continue)
    continues = torch.cat((true_continue.to(continues.dtype), continues[1:]), dim=0)
    discounts = compute_discounts(continues, gamma=gamma)

    return ImaginationOutput(
        latent_states=latent_states,
        actions=actions,
        rewards=rewards,
        values=values,
        continues=continues,
        discounts=discounts,
    )
