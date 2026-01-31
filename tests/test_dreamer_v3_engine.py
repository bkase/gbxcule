from __future__ import annotations

import pytest

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES
from gbxcule.rl.dreamer_v3.async_dreamer_v3_engine import AsyncDreamerV3Engine
from gbxcule.rl.dreamer_v3.config import DreamerEngineConfig
from gbxcule.rl.dreamer_v3.scheduler import Ratio

torch = pytest.importorskip("torch")


class _ToyEnv:
    def __init__(self, num_envs: int, obs_shape: tuple[int, ...], device: str) -> None:
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.device = torch.device(device)
        self._steps = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self._obs = torch.zeros(
            (num_envs, *obs_shape), dtype=torch.uint8, device=self.device
        )

    def reset(self, seed: int | None = None):  # type: ignore[no-untyped-def]
        _ = seed
        self._steps.zero_()
        self._obs.zero_()
        return self._obs

    def step(self, actions):  # type: ignore[no-untyped-def]
        self._steps = self._steps + 1
        fill = (
            (actions % 4)
            .to(torch.uint8)
            .view(self.num_envs, *([1] * len(self.obs_shape)))
        )
        self._obs = fill.expand_as(self._obs).clone()
        reward = self._obs.to(torch.float32).mean(dim=tuple(range(1, self._obs.ndim)))
        terminated = self._steps >= 3
        truncated = self._steps >= 5
        return self._obs, reward, terminated, truncated, {}

    def reset_mask(self, mask):  # type: ignore[no-untyped-def]
        if mask.dtype is not torch.bool:
            mask = mask.to(torch.bool)
        self._steps = torch.where(mask, torch.zeros_like(self._steps), self._steps)
        mask_view = mask.view(self.num_envs, *([1] * len(self.obs_shape)))
        self._obs = torch.where(mask_view, torch.zeros_like(self._obs), self._obs)

    def close(self) -> None:
        return None


class _ToyActorCore:
    def __init__(
        self, obs_shape: tuple[int, ...], num_actions: int, device: str
    ) -> None:
        obs_size = 1
        for dim in obs_shape:
            obs_size *= dim
        self.model = torch.nn.Linear(obs_size, num_actions).to(device)
        self.player = torch.nn.Linear(obs_size, num_actions).to(device)
        self.sync_player()
        self._obs_size = obs_size

    def init_state(self, num_envs: int, device):  # type: ignore[no-untyped-def]
        return torch.zeros((num_envs, 1), device=device)

    def act(self, obs, is_first, state, *, generator):  # type: ignore[no-untyped-def]
        _ = is_first
        flat = obs.to(torch.float32).view(obs.shape[0], -1)
        logits = self.player(flat)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(
            1
        )
        return actions.to(torch.int32), state

    def sync_player(self) -> None:
        for p_player, p_model in zip(
            self.player.parameters(), self.model.parameters(), strict=True
        ):
            p_player.data.copy_(p_model.data)


def _make_update_fns(actor_core: _ToyActorCore, device: str):
    optimizer = torch.optim.Adam(actor_core.model.parameters(), lr=1e-3)

    def world_model_update(batch):  # type: ignore[no-untyped-def]
        loss = batch.obs.to(torch.float32).mean()
        return {"loss_world": loss}

    def behavior_update(batch):  # type: ignore[no-untyped-def]
        optimizer.zero_grad(set_to_none=True)
        flat = batch.obs[-1].to(torch.float32).view(batch.obs.shape[1], -1)
        logits = actor_core.model(flat)
        loss = logits.mean()
        loss.backward()
        optimizer.step()
        return {"loss_behavior": loss}

    return world_model_update, behavior_update


def test_ratio_scheduler() -> None:
    ratio = Ratio(0.5, pretrain_steps=0)
    assert ratio(0) == 0
    assert ratio(10) == 5
    assert ratio(12) == 1

    ratio = Ratio(1.0, pretrain_steps=4)
    assert ratio(2) == 2


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        DreamerEngineConfig(seq_len=1)
    with pytest.raises(ValueError):
        DreamerEngineConfig(commit_stride=0)
    with pytest.raises(ValueError):
        DreamerEngineConfig(min_ready_steps=1, seq_len=2)
    with pytest.raises(ValueError):
        DreamerEngineConfig(safety_margin=1, seq_len=2)


def test_cpu_engine_smoke() -> None:
    cfg = DreamerEngineConfig(
        num_envs=4,
        obs_shape=(1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
        replay_capacity=64,
        seq_len=2,
        batch_size=4,
        steps_per_rollout=4,
        commit_stride=2,
        replay_ratio=1.0,
        learning_starts=0,
        safety_margin=2,
        device="cpu",
        debug=True,
    )
    env = _ToyEnv(cfg.num_envs, cfg.obs_shape, cfg.device)
    actor_core = _ToyActorCore(cfg.obs_shape, num_actions=5, device=cfg.device)
    world_model_update, behavior_update = _make_update_fns(actor_core, cfg.device)
    engine = AsyncDreamerV3Engine(
        cfg,
        env=env,
        actor_core=actor_core,
        world_model_update=world_model_update,
        behavior_update=behavior_update,
    )
    metrics = engine.run(num_iterations=3)
    assert metrics["train_steps"] > 0
    assert metrics["committed_t"] >= 0
    engine.close()


def test_starvation_guard() -> None:
    cfg = DreamerEngineConfig(
        num_envs=2,
        obs_shape=(1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
        replay_capacity=32,
        seq_len=2,
        batch_size=2,
        steps_per_rollout=1,
        commit_stride=1,
        replay_ratio=1.0,
        learning_starts=0,
        safety_margin=2,
        device="cpu",
    )
    env = _ToyEnv(cfg.num_envs, cfg.obs_shape, cfg.device)
    actor_core = _ToyActorCore(cfg.obs_shape, num_actions=3, device=cfg.device)
    world_model_update, behavior_update = _make_update_fns(actor_core, cfg.device)
    engine = AsyncDreamerV3Engine(
        cfg,
        env=env,
        actor_core=actor_core,
        world_model_update=world_model_update,
        behavior_update=behavior_update,
    )
    metrics = engine.run(num_iterations=6)
    assert metrics["train_steps"] > 0
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_engine_smoke() -> None:
    cfg = DreamerEngineConfig(
        num_envs=2,
        obs_shape=(1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
        replay_capacity=32,
        seq_len=2,
        batch_size=2,
        steps_per_rollout=2,
        commit_stride=1,
        replay_ratio=1.0,
        learning_starts=0,
        safety_margin=2,
        device="cuda",
        debug=True,
    )
    env = _ToyEnv(cfg.num_envs, cfg.obs_shape, cfg.device)
    actor_core = _ToyActorCore(cfg.obs_shape, num_actions=4, device=cfg.device)
    world_model_update, behavior_update = _make_update_fns(actor_core, cfg.device)
    engine = AsyncDreamerV3Engine(
        cfg,
        env=env,
        actor_core=actor_core,
        world_model_update=world_model_update,
        behavior_update=behavior_update,
    )
    metrics = engine.run(num_iterations=3)
    assert metrics["train_steps"] > 0
    engine.close()
