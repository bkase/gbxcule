"""World model forward pass and reconstruction loss for Dreamer v3 (M4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gbxcule.rl.dreamer_v3 import dists as dv3_dists
from gbxcule.rl.dreamer_v3.decoders import MultiDecoder
from gbxcule.rl.dreamer_v3.encoders import Packed2PixelEncoder
from gbxcule.rl.dreamer_v3.rssm import RSSM


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


@dataclass
class WorldModelOutput:
    embedded_obs: Any
    posteriors: Any
    priors: Any
    posterior_logits: Any
    prior_logits: Any
    recurrent_states: Any
    latent_states: Any
    reconstructions: dict[str, Any]
    reward_logits: Any
    continue_logits: Any | None


@dataclass
class ReconstructionMetrics:
    loss: Any
    kl: Any
    kl_loss: Any
    reward_loss: Any
    observation_loss: Any
    continue_loss: Any


def reconstruction_loss(
    po: dict[str, Any],
    observations: dict[str, Any],
    pr: Any,
    rewards: Any,
    priors_logits: Any,
    posteriors_logits: Any,
    *,
    kl_dynamic: float = 0.5,
    kl_representation: float = 0.1,
    kl_free_nats: float = 1.0,
    kl_regularizer: float = 1.0,
    pc: Any | None = None,
    continue_targets: Any | None = None,
    continue_scale_factor: float = 1.0,
):
    torch = _require_torch()
    from torch.distributions import Independent, OneHotCategoricalStraightThrough
    from torch.distributions.kl import kl_divergence

    reward_loss = -pr.log_prob(rewards)
    observation_loss = torch.zeros_like(reward_loss)
    for key in po:
        observation_loss = observation_loss - po[key].log_prob(observations[key])
    dyn_loss = kl = kl_divergence(
        Independent(
            OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1
        ),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
    )
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)
    repr_loss = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
    )
    repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)
    kl_loss = dyn_loss + repr_loss
    if pc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continue_targets)
    else:
        continue_loss = torch.zeros_like(reward_loss)
    total_loss = (
        kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss
    ).mean()
    return (
        total_loss,
        kl.mean(),
        kl_loss.mean(),
        reward_loss.mean(),
        observation_loss.mean(),
        continue_loss.mean(),
    )


class WorldModel(_require_torch().nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        *,
        encoder: Any,
        rssm: RSSM,
        observation_model: MultiDecoder,
        reward_model: Any,
        continue_model: Any | None,
        cnn_keys: list[str],
        mlp_keys: list[str],
        reward_low: float = -20.0,
        reward_high: float = 20.0,
        continue_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.continue_model = continue_model
        self.cnn_keys = list(cnn_keys)
        self.mlp_keys = list(mlp_keys)
        self.reward_low = float(reward_low)
        self.reward_high = float(reward_high)
        self.continue_scale_factor = float(continue_scale_factor)
        self._cnn_encoder_expects_packed2 = isinstance(encoder, Packed2PixelEncoder)
        if not self._cnn_encoder_expects_packed2 and hasattr(encoder, "cnn_encoder"):
            self._cnn_encoder_expects_packed2 = isinstance(
                encoder.cnn_encoder, Packed2PixelEncoder
            )

    def prepare_obs(
        self, obs: dict[str, Any], *, obs_format: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        torch = _require_torch()
        if not isinstance(obs, dict):
            raise TypeError("obs must be a dict")
        encoder_obs: dict[str, Any] = dict(obs)
        loss_obs: dict[str, Any] = dict(obs)
        for key in self.cnn_keys:
            value = obs[key]
            if obs_format == "packed2":
                loss_obs[key] = Packed2PixelEncoder.unpack_and_normalize(value)
                if not self._cnn_encoder_expects_packed2:
                    encoder_obs[key] = loss_obs[key]
            else:
                if value.dtype is torch.uint8:
                    value = value.to(torch.float32) / 255.0 - 0.5
                loss_obs[key] = value
                encoder_obs[key] = value
        return encoder_obs, loss_obs

    def _prepare_actions(self, actions, *, action_dim: int | None):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        if actions.ndim == 2:
            if action_dim is None:
                raise ValueError("action_dim must be provided for integer actions")
            actions = torch.nn.functional.one_hot(
                actions.to(torch.int64), action_dim
            ).to(torch.float32)
        else:
            actions = actions.to(torch.float32)
        return actions

    def forward(
        self,
        obs: dict[str, Any],
        actions,
        is_first,
        *,
        action_dim: int | None = None,
    ) -> WorldModelOutput:
        torch = _require_torch()
        actions = self._prepare_actions(actions, action_dim=action_dim)
        embedded_obs = self.encoder(obs)
        t, b = actions.shape[:2]
        recurrent_state, posterior = self.rssm.get_initial_states((1, b))
        posteriors = []
        priors = []
        posteriors_logits = []
        priors_logits = []
        recurrent_states = []
        for idx in range(t):
            recurrent_state, posterior, prior, posterior_logits, prior_logits = (
                self.rssm.dynamic(
                    posterior,
                    recurrent_state,
                    actions[idx : idx + 1],
                    embedded_obs[idx : idx + 1],
                    is_first[idx : idx + 1],
                )
            )
            recurrent_states.append(recurrent_state.squeeze(0))
            posteriors.append(posterior.squeeze(0))
            priors.append(prior.squeeze(0))
            posteriors_logits.append(posterior_logits.squeeze(0))
            priors_logits.append(prior_logits.squeeze(0))
        recurrent_states_t = torch.stack(recurrent_states)
        posteriors_t = torch.stack(posteriors)
        priors_t = torch.stack(priors)
        posteriors_logits_t = torch.stack(posteriors_logits)
        priors_logits_t = torch.stack(priors_logits)
        stoch = posteriors_t.shape[-2]
        discrete = posteriors_t.shape[-1]
        posteriors_logits_t = posteriors_logits_t.view(t, b, stoch, discrete)
        priors_logits_t = priors_logits_t.view(t, b, stoch, discrete)
        latent_states = torch.cat(
            (posteriors_t.view(t, b, -1), recurrent_states_t), dim=-1
        )
        reconstructions = self.observation_model(latent_states)
        reward_logits = self.reward_model(latent_states)
        continue_logits = None
        if self.continue_model is not None:
            continue_logits = self.continue_model(latent_states)
        return WorldModelOutput(
            embedded_obs=embedded_obs,
            posteriors=posteriors_t,
            priors=priors_t,
            posterior_logits=posteriors_logits_t,
            prior_logits=priors_logits_t,
            recurrent_states=recurrent_states_t,
            latent_states=latent_states,
            reconstructions=reconstructions,
            reward_logits=reward_logits,
            continue_logits=continue_logits,
        )

    def build_distributions(
        self, outputs: WorldModelOutput
    ) -> tuple[dict[str, Any], Any, Any | None]:
        torch = _require_torch()
        po: dict[str, Any] = {}
        for key in self.cnn_keys:
            recon = outputs.reconstructions[key]
            po[key] = dv3_dists.MSEDistribution(recon, dims=len(recon.shape[2:]))
        for key in self.mlp_keys:
            recon = outputs.reconstructions[key]
            po[key] = dv3_dists.SymlogDistribution(recon, dims=len(recon.shape[2:]))
        pr = dv3_dists.TwoHotEncodingDistribution(
            outputs.reward_logits,
            dims=1,
            low=self.reward_low,
            high=self.reward_high,
        )
        pc = None
        if outputs.continue_logits is not None:
            pc = torch.distributions.Independent(
                dv3_dists.BernoulliSafeMode(logits=outputs.continue_logits), 1
            )
        return po, pr, pc

    def loss(
        self,
        outputs: WorldModelOutput,
        observations: dict[str, Any],
        rewards: Any,
        continue_targets: Any | None,
        *,
        kl_dynamic: float,
        kl_representation: float,
        kl_free_nats: float,
        kl_regularizer: float,
    ) -> ReconstructionMetrics:
        po, pr, pc = self.build_distributions(outputs)
        total, kl, kl_loss, reward_loss, obs_loss, cont_loss = reconstruction_loss(
            po,
            observations,
            pr,
            rewards,
            outputs.prior_logits,
            outputs.posterior_logits,
            kl_dynamic=kl_dynamic,
            kl_representation=kl_representation,
            kl_free_nats=kl_free_nats,
            kl_regularizer=kl_regularizer,
            pc=pc,
            continue_targets=continue_targets,
            continue_scale_factor=self.continue_scale_factor,
        )
        return ReconstructionMetrics(
            loss=total,
            kl=kl,
            kl_loss=kl_loss,
            reward_loss=reward_loss,
            observation_loss=obs_loss,
            continue_loss=cont_loss,
        )
