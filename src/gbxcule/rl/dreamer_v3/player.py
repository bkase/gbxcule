"""Actor core helpers for Dreamer v3 (player-side inference)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


@dataclass
class DreamerActorState:
    recurrent: Any
    posterior: Any
    prev_action: Any


class DreamerActorCore:
    """Actor core for Dreamer v3 (packed2, discrete actions)."""

    def __init__(
        self,
        *,
        encoder,
        rssm,
        actor,
        action_dim: int,
        greedy: bool = False,
    ) -> None:
        torch = _require_torch()
        self.model_encoder = encoder
        self.model_rssm = rssm
        self.model_actor = actor

        self.player_encoder = copy.deepcopy(encoder).to(torch.device("cuda"))
        self.player_rssm = copy.deepcopy(rssm).to(torch.device("cuda"))
        self.player_actor = copy.deepcopy(actor).to(torch.device("cuda"))
        self.player_actor.eval()
        self.player_encoder.eval()
        self.player_rssm.eval()

        self.action_dim = int(action_dim)
        self.greedy = bool(greedy)

    def sync_player(self) -> None:
        with _require_torch().no_grad():
            self.player_encoder.load_state_dict(self.model_encoder.state_dict())
            self.player_rssm.load_state_dict(self.model_rssm.state_dict())
            self.player_actor.load_state_dict(self.model_actor.state_dict())

    def init_state(self, num_envs: int, device):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        recurrent, posterior = self.player_rssm.get_initial_states((num_envs,))
        prev_action = torch.zeros(
            (num_envs, self.action_dim),
            device=device,
            dtype=torch.float32,
        )
        return DreamerActorState(
            recurrent=recurrent,
            posterior=posterior,
            prev_action=prev_action,
        )

    def act(self, obs, is_first, state, *, generator):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        with torch.no_grad():
            if not isinstance(obs, dict):
                raise TypeError("DreamerActorCore.act expects dict observations.")
            obs_dict = {key: value.unsqueeze(0) for key, value in obs.items()}
            embedded = self.player_encoder(obs_dict)
            recurrent, posterior, _, _, _ = self.player_rssm.dynamic(
                state.posterior.unsqueeze(0),
                state.recurrent.unsqueeze(0),
                state.prev_action.unsqueeze(0),
                embedded,
                is_first.unsqueeze(0),
                sample_state=True,
            )
            recurrent = recurrent.squeeze(0)
            posterior = posterior.squeeze(0)
            latent = torch.cat(
                (posterior.view(posterior.shape[0], -1), recurrent), dim=-1
            )

            actions, _ = self.player_actor(
                latent, greedy=self.greedy, generator=generator
            )
            action_onehot = torch.cat(actions, dim=-1)
            action_idx = torch.argmax(action_onehot, dim=-1).to(torch.int32)
            next_state = DreamerActorState(
                recurrent=recurrent,
                posterior=posterior,
                prev_action=action_onehot.detach(),
            )
            return action_idx, next_state


class ConstantActorCore:
    """Actor core that returns a fixed action (standing-still mode)."""

    def __init__(self, *, action: int) -> None:
        self._action = int(action)

    def sync_player(self) -> None:
        return None

    def init_state(self, num_envs: int, device):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        return torch.zeros((num_envs, 1), device=device)

    def act(self, obs, is_first, state, *, generator):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        _ = obs
        _ = is_first
        _ = generator
        actions = torch.full(
            (state.shape[0],),
            self._action,
            device=state.device,
            dtype=torch.int32,
        )
        return actions, state


__all__ = ["ConstantActorCore", "DreamerActorCore", "DreamerActorState"]
