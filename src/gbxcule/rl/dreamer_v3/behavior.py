"""Behavior learning components for Dreamer v3 (M5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from gbxcule.rl.dreamer_v3 import dists as dv3_dists
from gbxcule.rl.dreamer_v3.imagination import imagine_rollout
from gbxcule.rl.dreamer_v3.init import init_weights, uniform_init_weights
from gbxcule.rl.dreamer_v3.mlp import MLP, LayerNorm
from gbxcule.rl.dreamer_v3.return_ema import ReturnEMA
from gbxcule.rl.dreamer_v3.returns import lambda_returns


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


class OneHotCategoricalST:
    """One-hot categorical with straight-through samples and generator support."""

    def __init__(self, logits):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        self.logits = logits
        self.probs = torch.softmax(logits, dim=-1)
        self._log_probs = torch.log_softmax(logits, dim=-1)

    def rsample(self, *, generator=None):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        probs = self.probs
        flat = probs.reshape(-1, probs.shape[-1])
        if generator is None:
            raise ValueError("generator must be provided for stochastic sampling")
        idx = torch.multinomial(flat, 1, generator=generator)
        one_hot = torch.zeros_like(flat).scatter_(1, idx, 1.0).reshape_as(probs)
        return one_hot + probs - probs.detach()

    @property
    def mode(self):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        idx = torch.argmax(self.probs, dim=-1)
        return torch.nn.functional.one_hot(idx, num_classes=self.probs.shape[-1]).to(
            self.probs.dtype
        )

    def log_prob(self, value):  # type: ignore[no-untyped-def]
        return (value * self._log_probs).sum(dim=-1)

    def entropy(self):  # type: ignore[no-untyped-def]
        return -(self.probs * self._log_probs).sum(dim=-1)


class Actor(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        latent_state_size: int,
        actions_dim: Sequence[int],
        is_continuous: bool,
        distribution_cfg: dict[str, Any] | None = None,
        init_std: float = 0.0,
        min_std: float = 1.0,
        max_std: float = 1.0,
        dense_units: int = 1024,
        activation: Any = None,
        mlp_layers: int = 5,
        layer_norm_cls: Any = LayerNorm,
        layer_norm_kw: dict[str, Any] | None = None,
        unimix: float = 0.01,
        action_clip: float = 1.0,
    ) -> None:
        super().__init__()
        torch = _require_torch()
        if distribution_cfg is None:
            distribution_cfg = {}
        dist_type = distribution_cfg.get("type", "auto")
        dist_type = dist_type.lower() if isinstance(dist_type, str) else "auto"
        if dist_type not in (
            "auto",
            "normal",
            "tanh_normal",
            "scaled_normal",
            "discrete",
        ):
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        if dist_type == "discrete" and is_continuous:
            raise ValueError("Cannot use discrete distribution for continuous actions")
        if dist_type == "auto":
            dist_type = "scaled_normal" if is_continuous else "discrete"

        self.distribution = dist_type
        self.actions_dim = tuple(int(v) for v in actions_dim)
        self.is_continuous = bool(is_continuous)
        self.init_std = float(init_std)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self._unimix = float(unimix)
        self._action_clip = float(action_clip)

        if activation is None:
            activation = torch.nn.SiLU
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=None,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=activation,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw or {"eps": 1e-3},
            bias=layer_norm_cls in (None, torch.nn.Identity),
            flatten_dim=None,
        )
        if self.is_continuous:
            total = int(sum(self.actions_dim))
            self.mlp_heads = torch.nn.ModuleList(
                [torch.nn.Linear(dense_units, total * 2)]
            )
        else:
            self.mlp_heads = torch.nn.ModuleList(
                [
                    torch.nn.Linear(dense_units, action_dim)
                    for action_dim in self.actions_dim
                ]
            )

        self.apply(init_weights)
        if not self.is_continuous:
            for head in self.mlp_heads:
                head.apply(uniform_init_weights(1.0))

    def _uniform_mix(self, logits):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        from torch.distributions.utils import probs_to_logits

        if self._unimix <= 0.0:
            return logits
        probs = torch.softmax(logits, dim=-1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        return probs_to_logits(probs, is_binary=False)

    def forward(
        self,
        state,
        greedy: bool = False,
        mask: dict[str, Any] | None = None,
        *,
        generator=None,  # type: ignore[no-untyped-def]
    ):
        torch = _require_torch()
        out = self.model(state)
        pre_dist = [head(out) for head in self.mlp_heads]

        if self.is_continuous:
            mean, std = torch.chunk(pre_dist[0], 2, dim=-1)
            if self.distribution == "tanh_normal":
                mean = 5 * torch.tanh(mean / 5)
                std = torch.nn.functional.softplus(std + self.init_std) + self.min_std
                dist = torch.distributions.Normal(mean, std)
                dist = torch.distributions.Independent(
                    torch.distributions.TransformedDistribution(
                        dist, torch.distributions.TanhTransform()
                    ),
                    1,
                )
            elif self.distribution == "normal":
                dist = torch.distributions.Independent(
                    torch.distributions.Normal(mean, std), 1
                )
            elif self.distribution == "scaled_normal":
                std = (self.max_std - self.min_std) * torch.sigmoid(
                    std + self.init_std
                ) + self.min_std
                dist = torch.distributions.Independent(
                    torch.distributions.Normal(torch.tanh(mean), std), 1
                )
            else:
                raise ValueError(f"Unsupported distribution: {self.distribution}")

            if greedy:
                sample = dist.sample((100,))
                log_prob = dist.log_prob(sample)
                actions = sample[log_prob.argmax(0)].view(*sample.shape[1:])
            else:
                actions = dist.rsample()
            if self._action_clip > 0.0:
                clip = torch.full_like(actions, self._action_clip)
                actions = (
                    actions * (clip / torch.maximum(clip, torch.abs(actions))).detach()
                )
            return (actions,), (dist,)

        actions = []
        dists = []
        for logits in pre_dist:
            logits = self._uniform_mix(logits)
            dist = OneHotCategoricalST(logits)
            dists.append(dist)
            action = dist.mode if greedy else dist.rsample(generator=generator)
            actions.append(action)

        return tuple(actions), tuple(dists)


class Critic(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        latent_state_size: int,
        bins: int,
        dense_units: int = 1024,
        activation: Any = None,
        mlp_layers: int = 5,
        layer_norm_cls: Any = LayerNorm,
        layer_norm_kw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        torch = _require_torch()
        if bins < 2:
            raise ValueError("bins must be >= 2")
        if activation is None:
            activation = torch.nn.SiLU
        self.model = MLP(
            input_dims=latent_state_size,
            output_dim=bins,
            hidden_sizes=[dense_units] * mlp_layers,
            activation=activation,
            layer_norm_cls=layer_norm_cls,
            layer_norm_kw=layer_norm_kw
            or {"eps": 1e-3, "normalized_shape": dense_units},
            bias=layer_norm_cls in (None, torch.nn.Identity),
            flatten_dim=None,
        )
        self.apply(init_weights)
        self.model.model[-1].apply(uniform_init_weights(0.0))

    def forward(self, latent_states):  # type: ignore[no-untyped-def]
        return self.model(latent_states)


@dataclass
class BehaviorLosses:
    policy_loss: Any
    value_loss: Any
    entropy: Any
    lambda_values: Any
    advantage: Any
    discount: Any
    imagination: Any


def _resolve_true_continue(continues, terminated):  # type: ignore[no-untyped-def]
    if continues is not None:
        return continues
    if terminated is None:
        raise KeyError("behavior losses require 'continues' or 'terminated'")
    return 1 - terminated


def behavior_losses(
    *,
    rssm,  # type: ignore[no-untyped-def]
    actor: Actor,
    critic: Critic,
    target_critic: Critic,
    reward_model,  # type: ignore[no-untyped-def]
    continue_model,  # type: ignore[no-untyped-def]
    posteriors,  # type: ignore[no-untyped-def]
    recurrent_states,  # type: ignore[no-untyped-def]
    continues=None,  # type: ignore[no-untyped-def]
    terminated=None,  # type: ignore[no-untyped-def]
    horizon: int,
    gamma: float,
    lmbda: float,
    moments: ReturnEMA,
    reward_low: float = -20.0,
    reward_high: float = 20.0,
    value_low: float = -20.0,
    value_high: float = 20.0,
    ent_coef: float = 0.0,
    greedy: bool = False,
    generator=None,  # type: ignore[no-untyped-def]
    sample_state: bool = True,
) -> BehaviorLosses:
    torch = _require_torch()
    true_continue = _resolve_true_continue(continues, terminated)

    rollout = imagine_rollout(
        rssm=rssm,
        actor=actor,
        critic=critic,
        reward_model=reward_model,
        continue_model=continue_model,
        posteriors=posteriors,
        recurrent_states=recurrent_states,
        true_continue=true_continue,
        horizon=horizon,
        gamma=gamma,
        reward_low=reward_low,
        reward_high=reward_high,
        value_low=value_low,
        value_high=value_high,
        greedy=greedy,
        generator=generator,
        sample_state=sample_state,
    )

    rewards = rollout.rewards[1:]
    values = rollout.values[1:]
    continues_gamma = rollout.continues[1:] * gamma
    lambda_values = lambda_returns(rewards, values, continues_gamma, lmbda=lmbda)

    baseline = rollout.values[:-1]
    offset, invscale = moments.update(lambda_values)
    normed_lambda = (lambda_values - offset) / invscale
    normed_baseline = (baseline - offset) / invscale
    advantage = normed_lambda - normed_baseline

    _, policies = actor(rollout.latent_states.detach(), greedy=True)
    action_splits = torch.split(rollout.actions, actor.actions_dim, dim=-1)
    entropy = torch.zeros_like(rollout.discounts)
    if actor.is_continuous:
        entropy = ent_coef * policies[0].entropy().unsqueeze(-1)
        objective = advantage
    else:
        log_probs = torch.stack(
            [
                dist.log_prob(action.detach()).unsqueeze(-1)[:-1]
                for dist, action in zip(policies, action_splits, strict=True)
            ],
            dim=-1,
        ).sum(dim=-1)
        objective = log_probs * advantage.detach()
        entropy = (
            ent_coef
            * torch.stack([dist.entropy() for dist in policies], dim=-1).sum(dim=-1)
        ).unsqueeze(-1)

    policy_loss = -torch.mean(
        rollout.discounts[:-1].detach() * (objective + entropy[:-1])
    )

    qv = dv3_dists.TwoHotEncodingDistribution(
        critic(rollout.latent_states.detach()[:-1]),
        dims=1,
        low=value_low,
        high=value_high,
    )
    target_mean = dv3_dists.TwoHotEncodingDistribution(
        target_critic(rollout.latent_states.detach()[:-1]),
        dims=1,
        low=value_low,
        high=value_high,
    ).mean
    value_loss = -qv.log_prob(lambda_values.detach())
    value_loss = value_loss - qv.log_prob(target_mean.detach())
    value_loss = torch.mean(value_loss * rollout.discounts[:-1].squeeze(-1))

    return BehaviorLosses(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        lambda_values=lambda_values,
        advantage=advantage,
        discount=rollout.discounts,
        imagination=rollout,
    )


def behavior_step(
    *,
    rssm,  # type: ignore[no-untyped-def]
    actor: Actor,
    critic: Critic,
    target_critic: Critic,
    reward_model,  # type: ignore[no-untyped-def]
    continue_model,  # type: ignore[no-untyped-def]
    posteriors,  # type: ignore[no-untyped-def]
    recurrent_states,  # type: ignore[no-untyped-def]
    continues=None,  # type: ignore[no-untyped-def]
    terminated=None,  # type: ignore[no-untyped-def]
    horizon: int,
    gamma: float,
    lmbda: float,
    moments: ReturnEMA,
    actor_optimizer=None,  # type: ignore[no-untyped-def]
    critic_optimizer=None,  # type: ignore[no-untyped-def]
    actor_clip_grad: float | None = None,
    critic_clip_grad: float | None = None,
    target_update_freq: int | None = None,
    tau: float | None = None,
    step: int | None = None,
    reward_low: float = -20.0,
    reward_high: float = 20.0,
    value_low: float = -20.0,
    value_high: float = 20.0,
    ent_coef: float = 0.0,
    greedy: bool = False,
    generator=None,  # type: ignore[no-untyped-def]
    sample_state: bool = True,
):
    torch = _require_torch()
    losses = behavior_losses(
        rssm=rssm,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        reward_model=reward_model,
        continue_model=continue_model,
        posteriors=posteriors,
        recurrent_states=recurrent_states,
        continues=continues,
        terminated=terminated,
        horizon=horizon,
        gamma=gamma,
        lmbda=lmbda,
        moments=moments,
        reward_low=reward_low,
        reward_high=reward_high,
        value_low=value_low,
        value_high=value_high,
        ent_coef=ent_coef,
        greedy=greedy,
        generator=generator,
        sample_state=sample_state,
    )

    if actor_optimizer is not None:
        actor_optimizer.zero_grad(set_to_none=True)
        losses.policy_loss.backward()
        if actor_clip_grad is not None and actor_clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), actor_clip_grad)
        actor_optimizer.step()

    if critic_optimizer is not None:
        critic_optimizer.zero_grad(set_to_none=True)
        losses.value_loss.backward()
        if critic_clip_grad is not None and critic_clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_clip_grad)
        critic_optimizer.step()

    if (
        target_update_freq is not None
        and tau is not None
        and step is not None
        and target_update_freq > 0
    ):
        from gbxcule.rl.dreamer_v3.targets import maybe_update_target

        maybe_update_target(
            critic, target_critic, tau, step=step, update_freq=target_update_freq
        )

    return losses
