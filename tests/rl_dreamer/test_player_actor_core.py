"""Tests for DreamerActorCore dict-observation handling."""

from __future__ import annotations

import pytest

from gbxcule.rl.dreamer_v3.player import ConstantActorCore, DreamerActorCore

torch = pytest.importorskip("torch")


class _StubEncoder:
    def __init__(self) -> None:
        self.last_obs = None

    def __call__(self, obs_dict):
        self.last_obs = obs_dict
        sample = next(iter(obs_dict.values()))
        time_dim, batch_dim = sample.shape[0], sample.shape[1]
        return torch.zeros((time_dim, batch_dim, 4), dtype=torch.float32)

    def to(self, device):  # noqa: ARG002 - match torch module API
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return None


class _StubRssm:
    def __init__(self, state_dim: int = 3) -> None:
        self.state_dim = int(state_dim)

    def get_initial_states(self, batch_shape):
        num_envs = int(batch_shape[0])
        recurrent = torch.zeros((num_envs, self.state_dim), dtype=torch.float32)
        posterior = torch.zeros((num_envs, self.state_dim), dtype=torch.float32)
        return recurrent, posterior

    def dynamic(
        self,
        posterior,
        recurrent,
        prev_action,
        embedded,
        is_first,
        sample_state: bool = True,
    ):
        _ = prev_action
        _ = embedded
        _ = is_first
        _ = sample_state
        return recurrent, posterior, None, None, None

    def to(self, device):  # noqa: ARG002 - match torch module API
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return None


class _StubActor:
    def __init__(self, action_dim: int) -> None:
        self.action_dim = int(action_dim)

    def __call__(self, latent, greedy: bool = False, generator=None):
        _ = greedy
        _ = generator
        batch_dim = latent.shape[0]
        actions = torch.zeros((batch_dim, self.action_dim), device=latent.device)
        actions[:, 0] = 1.0
        return [actions], None

    def to(self, device):  # noqa: ARG002 - match torch module API
        return self

    def eval(self):
        return self

    def state_dict(self):
        return {}

    def load_state_dict(self, _state):
        return None


def test_dreamer_actor_core_dict_obs_adds_time_dim():
    encoder = _StubEncoder()
    rssm = _StubRssm(state_dim=2)
    actor = _StubActor(action_dim=3)
    core = DreamerActorCore(
        encoder=encoder,
        rssm=rssm,
        actor=actor,
        action_dim=3,
        greedy=True,
    )

    num_envs = 2
    obs = {
        "pixels": torch.zeros((num_envs, 1, 72, 20), dtype=torch.uint8),
        "senses": torch.zeros((num_envs, 8), dtype=torch.float32),
    }
    is_first = torch.ones((num_envs,), dtype=torch.bool)
    state = core.init_state(num_envs, device=torch.device("cpu"))

    actions, next_state = core.act(obs, is_first, state, generator=None)

    seen = core.player_encoder.last_obs
    assert seen is not None
    assert seen["pixels"].shape == (1, num_envs, 1, 72, 20)
    assert seen["senses"].shape == (1, num_envs, 8)
    assert actions.shape == (num_envs,)
    assert actions.dtype == torch.int32
    assert next_state.prev_action.shape == (num_envs, 3)


def test_constant_actor_core_accepts_dict_obs():
    actor = ConstantActorCore(action=2)
    num_envs = 3
    state = actor.init_state(num_envs, device=torch.device("cpu"))
    obs = {"pixels": torch.zeros((num_envs, 1, 1, 1), dtype=torch.uint8)}
    is_first = torch.zeros((num_envs,), dtype=torch.bool)

    actions, next_state = actor.act(obs, is_first, state, generator=None)

    assert actions.shape == (num_envs,)
    assert actions.dtype == torch.int32
    assert torch.all(actions == 2)
    assert next_state is state
