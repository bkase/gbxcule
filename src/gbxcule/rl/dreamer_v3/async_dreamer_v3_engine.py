"""Async Dreamer v3 engine (CPU/CUDA)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from gbxcule.rl.dreamer_v3.config import DreamerEngineConfig
from gbxcule.rl.dreamer_v3.replay import ReplayRing
from gbxcule.rl.dreamer_v3.replay_commit import ReplayCommitManager
from gbxcule.rl.dreamer_v3.replay_cuda import ReplayRingCUDA
from gbxcule.rl.dreamer_v3.scheduler import Ratio
from gbxcule.rl.failfast import assert_device, assert_finite, assert_shape


def _require_torch():  # type: ignore[no-untyped-def]
    import importlib

    return importlib.import_module("torch")


@dataclass
class DreamerBatch:
    obs: Any
    action: Any
    reward: Any
    is_first: Any
    continues: Any
    episode_id: Any
    start_times: Any
    env_indices: Any


class ActorCore(Protocol):
    def init_state(self, num_envs: int, device) -> Any: ...

    def act(self, obs, is_first, state, *, generator) -> tuple[Any, Any]: ...

    def sync_player(self) -> None: ...


UpdateFn = Callable[[DreamerBatch], dict[str, Any]]


class AsyncDreamerV3Engine:
    """Async Dreamer v3 engine with replay-ratio scheduling."""

    def __init__(
        self,
        config: DreamerEngineConfig,
        *,
        env: Any,
        actor_core: ActorCore,
        world_model_update: UpdateFn,
        behavior_update: UpdateFn,
        experiment: Any | None = None,
    ) -> None:
        torch = _require_torch()
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for Dreamer engine")

        self._torch = torch
        self.config = config
        self.env = env
        self.actor_core = actor_core
        self.world_model_update = world_model_update
        self.behavior_update = behavior_update
        self.experiment = experiment
        self.device = torch.device(config.device)

        if self.device.type == "cuda":
            self.replay = ReplayRingCUDA(
                capacity=config.replay_capacity,
                num_envs=config.num_envs,
                device=config.device,
                obs_shape=config.obs_shape,  # type: ignore[arg-type]
            )
        else:
            self.replay = ReplayRing(
                capacity=config.replay_capacity,
                num_envs=config.num_envs,
                device=config.device,
                obs_shape=config.obs_shape,  # type: ignore[arg-type]
            )
        self.commit = ReplayCommitManager(
            commit_stride=config.commit_stride,
            safety_margin=config.safety_margin,
            device=config.device,
        )
        self._ratio = Ratio(config.replay_ratio, config.pretrain_steps)
        self._policy_steps = 0
        self._train_steps = 0
        self._prefill_steps = config.learning_starts - int(config.learning_starts > 0)

        self._obs = self._env_reset()
        self._is_first = torch.ones(
            (config.num_envs,), dtype=torch.bool, device=self.device
        )
        self._episode_id = torch.zeros(
            (config.num_envs,), dtype=torch.int32, device=self.device
        )
        self._actor_state = actor_core.init_state(config.num_envs, self.device)

        self._sample_gen = torch.Generator(device=self.device)
        self._sample_gen.manual_seed(config.seed)
        self._action_gen = torch.Generator(device=self.device)
        self._action_gen.manual_seed(config.seed + 1)

        self._policy_version = 0
        self._last_sync_version = -1
        self._wall_start = time.time()
        if self.device.type == "cuda":
            self._actor_stream = torch.cuda.Stream()
            self._learner_stream = torch.cuda.Stream()
            self._policy_ready_event = torch.cuda.Event()
            with torch.cuda.stream(self._learner_stream):
                self._sync_actor_weights(stream=self._learner_stream)
            torch.cuda.synchronize()
        else:
            self._actor_stream = None
            self._learner_stream = None
            self._policy_ready_event = None
            self._sync_actor_weights(stream=None)

        self._checked_shapes = False

    def _env_reset(self):  # type: ignore[no-untyped-def]
        if hasattr(self.env, "reset_torch"):
            obs = self.env.reset_torch(seed=self.config.seed)
        else:
            try:
                obs = self.env.reset(seed=self.config.seed)
            except TypeError:
                obs = self.env.reset()
        return obs

    def _env_step(self, actions):  # type: ignore[no-untyped-def]
        if hasattr(self.env, "step_torch"):
            return self.env.step_torch(actions)
        return self.env.step(actions)

    def _maybe_reset_env(self, reset_mask) -> None:  # type: ignore[no-untyped-def]
        if hasattr(self.env, "reset_mask"):
            self.env.reset_mask(reset_mask)

    def _sync_actor_weights(self, stream=None) -> None:  # type: ignore[no-untyped-def]
        self.actor_core.sync_player()
        self._policy_version += 1
        self._last_sync_version = self._policy_version
        if stream is not None and self._policy_ready_event is not None:
            self._policy_ready_event.record(stream)

    def _assert_shapes(self) -> None:
        if self._checked_shapes or not self.config.debug:
            return
        if self.device.type == "cuda":
            if self._obs.device.type != "cuda":
                raise AssertionError("obs expected device cuda")
        else:
            assert_device(self._obs, self.device, "obs", self.experiment, None)
        assert_shape(
            self._obs,
            (self.config.num_envs, *self.config.obs_shape),
            "obs",
            self.experiment,
            None,
        )
        self._checked_shapes = True

    def _actor_rollout(self, *, stream=None) -> int:  # type: ignore[no-untyped-def]
        torch = self._torch
        cfg = self.config
        steps = int(cfg.steps_per_rollout)
        for _ in range(steps):
            self._assert_shapes()
            actions, self._actor_state = self.actor_core.act(
                self._obs,
                self._is_first,
                self._actor_state,
                generator=self._action_gen,
            )
            next_obs, reward, terminated, truncated, _ = self._env_step(actions)
            continues = (~terminated).to(dtype=torch.float32)
            self.replay.push_step(
                obs=self._obs,
                action=actions,
                reward=reward,
                is_first=self._is_first,
                continue_=continues,
                episode_id=self._episode_id,
                terminated=terminated,
                truncated=truncated,
            )
            self.commit.mark_written(self.replay.total_steps - 1, stream=stream)

            reset_mask = terminated | truncated
            self._episode_id = self._episode_id + reset_mask.to(torch.int32)
            self._is_first = reset_mask
            self._maybe_reset_env(reset_mask)
            self._obs = next_obs

        self._policy_steps += cfg.num_envs * steps
        return steps

    def _replay_ready(self) -> bool:
        if self.commit.committed_t < 0:
            return False
        start_time = self.replay.total_steps - self.replay.size
        committed_available = self.commit.committed_t - start_time + 1
        if committed_available < int(self.config.min_ready_steps):
            return False
        max_time = self.commit.safe_max_t()
        if max_time < start_time:
            return False
        return (max_time - start_time + 1) >= int(self.config.seq_len)

    def _max_sample_time(self) -> int:
        return int(self.commit.safe_max_t())

    def _sample_batch(self) -> dict[str, Any]:
        max_time = self._max_sample_time()
        if max_time < 0:
            raise ValueError("no committed samples ready")
        exclude_head = isinstance(self.replay, ReplayRingCUDA)
        return self.replay.sample_sequences(
            batch=self.config.batch_size,
            seq_len=self.config.seq_len,
            gen=self._sample_gen,
            committed_t=self.commit.committed_t,
            safety_margin=self.commit.safety_margin,
            exclude_head=exclude_head,
            return_indices=True,
        )

    def _make_batch(self, sample: dict[str, Any]) -> DreamerBatch:
        return DreamerBatch(
            obs=sample["obs"],
            action=sample["action"],
            reward=sample["reward"],
            is_first=sample["is_first"],
            continues=sample["continue"],
            episode_id=sample["episode_id"],
            start_times=sample.get("meta", {}).get("start_offset"),
            env_indices=sample.get("meta", {}).get("env_idx"),
        )

    def _learner_updates(self, *, stream=None) -> tuple[int, dict[str, float]]:  # type: ignore[no-untyped-def]
        cfg = self.config
        if self._policy_steps < cfg.learning_starts:
            return 0, {}
        ratio_steps = max(0, self._policy_steps - self._prefill_steps)
        updates = self._ratio(ratio_steps)
        if cfg.max_learner_steps_per_tick is not None:
            updates = min(updates, cfg.max_learner_steps_per_tick)

        if updates < 1:
            return 0, {}
        if not self._replay_ready():
            return 0, {}

        losses: dict[str, float] = {}
        updates_done = 0
        for _ in range(updates):
            if not self._replay_ready():
                break
            sample = self._sample_batch()
            batch = self._make_batch(sample)
            wm_metrics = self.world_model_update(batch)
            beh_metrics = self.behavior_update(batch)
            loss_total = 0.0
            for key, value in {**wm_metrics, **beh_metrics}.items():
                if not isinstance(value, self._torch.Tensor):
                    continue
                loss_val = float(value.detach().mean().item())
                losses[key] = losses.get(key, 0.0) + loss_val
                if "loss" in key:
                    loss_total += loss_val
            if cfg.debug:
                total_t = self._torch.tensor(loss_total, device=self.device)
                assert_finite(total_t, "loss_total", self.experiment, None)
            updates_done += 1
            self._train_steps += 1

        if updates_done > 0:
            for key in list(losses.keys()):
                losses[key] /= float(updates_done)
            self._sync_actor_weights(stream=stream)
            if self._policy_ready_event is not None:
                assert self._last_sync_version == self._policy_version, (
                    "policy_ready_event recorded before sync"
                )
        return updates_done, losses

    def _metrics(
        self,
        *,
        actor_ms: float,
        learner_ms: float,
        total_ms: float,
        updates: int,
        loss_metrics: dict[str, float],
    ) -> dict[str, Any]:
        replay_ratio_actual = (
            float(self._train_steps) / float(self._policy_steps)
            if self._policy_steps > 0
            else 0.0
        )
        start_time = self.replay.total_steps - self.replay.size
        if self.commit.committed_t >= 0:
            committed_available = self.commit.committed_t - start_time + 1
        else:
            committed_available = 0
        committed_available = max(0, int(committed_available))
        overlap_eff = (actor_ms + learner_ms) / total_ms if total_ms > 0 else 0.0
        env_steps_iter = int(self.config.num_envs * self.config.steps_per_rollout)
        sps = float(env_steps_iter) / (total_ms / 1000.0) if total_ms > 0 else 0.0
        train_sps = float(updates) / (learner_ms / 1000.0) if learner_ms > 0 else 0.0
        metrics = {
            "env_steps": int(self._policy_steps),
            "train_steps": int(self._train_steps),
            "opt_steps": int(self._train_steps),
            "wall_time_s": float(time.time() - self._wall_start),
            "updates": int(updates),
            "replay_ratio": float(replay_ratio_actual),
            "replay_ratio_actual": float(replay_ratio_actual),
            "replay_ratio_target": float(self.config.replay_ratio),
            "replay_size": int(self.replay.size),
            "commit_stride": int(self.config.commit_stride),
            "ready_steps": int(committed_available),
            "min_ready_steps": int(self.config.min_ready_steps),
            "seq_len": int(self.config.seq_len),
            "sps": float(sps),
            "train_sps": float(train_sps),
            "actor_ms": float(actor_ms),
            "learner_ms": float(learner_ms),
            "total_ms": float(total_ms),
            "overlap_eff": float(overlap_eff),
            "committed_t": int(self.commit.committed_t),
        }
        metrics.update(loss_metrics)
        return metrics

    def _run_cpu_iteration(self) -> dict[str, Any]:
        start = time.perf_counter()
        actor_start = time.perf_counter()
        self._actor_rollout(stream=None)
        actor_end = time.perf_counter()
        updates, loss_metrics = self._learner_updates(stream=None)
        learner_end = time.perf_counter()
        total_ms = (learner_end - start) * 1000.0
        actor_ms = (actor_end - actor_start) * 1000.0
        learner_ms = (learner_end - actor_end) * 1000.0
        return self._metrics(
            actor_ms=actor_ms,
            learner_ms=learner_ms,
            total_ms=total_ms,
            updates=updates,
            loss_metrics=loss_metrics,
        )

    def _run_cuda_iteration(self) -> dict[str, Any]:
        torch = self._torch
        actor_stream = self._actor_stream
        learner_stream = self._learner_stream
        policy_ready_event = self._policy_ready_event
        assert actor_stream is not None
        assert learner_stream is not None
        assert policy_ready_event is not None

        actor_start = torch.cuda.Event(enable_timing=True)
        actor_end = torch.cuda.Event(enable_timing=True)
        learner_start = torch.cuda.Event(enable_timing=True)
        learner_end = torch.cuda.Event(enable_timing=True)
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)

        default_stream = torch.cuda.current_stream()
        total_start.record(default_stream)

        with torch.cuda.stream(learner_stream):
            learner_start.record(learner_stream)
            updates, loss_metrics = self._learner_updates(stream=learner_stream)
            learner_end.record(learner_stream)

        with torch.cuda.stream(actor_stream):
            actor_stream.wait_event(policy_ready_event)
            actor_start.record(actor_stream)
            self._actor_rollout(stream=actor_stream)
            actor_end.record(actor_stream)

        default_stream.wait_event(actor_end)
        default_stream.wait_event(learner_end)
        total_end.record(default_stream)
        torch.cuda.synchronize()

        actor_ms = float(actor_start.elapsed_time(actor_end))
        learner_ms = float(learner_start.elapsed_time(learner_end))
        total_ms = float(total_start.elapsed_time(total_end))
        return self._metrics(
            actor_ms=actor_ms,
            learner_ms=learner_ms,
            total_ms=total_ms,
            updates=updates,
            loss_metrics=loss_metrics,
        )

    def run(self, *, num_iterations: int = 1) -> dict[str, Any]:
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        metrics: dict[str, Any] = {}
        for _ in range(int(num_iterations)):
            if self.device.type == "cuda":
                metrics = self._run_cuda_iteration()
            else:
                metrics = self._run_cpu_iteration()
            if self.experiment is not None:
                self.experiment.log_metrics(metrics)
        return metrics

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


__all__ = ["AsyncDreamerV3Engine", "DreamerBatch", "DreamerEngineConfig"]
