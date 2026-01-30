"""Async PPO engine for reusable benchmarking and training."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend


def _require_torch():  # type: ignore[no-untyped-def]
    import importlib

    return importlib.import_module("torch")


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


@dataclass(frozen=True)
class AsyncPPOEngineConfig:
    rom_path: str
    state_path: str
    goal_dir: str
    device: str = "cuda"
    obs_format: str = "u8"
    num_envs: int = 1024
    frames_per_step: int = 24
    release_after_frames: int = 8
    steps_per_rollout: int = 32
    updates: int = 4
    ppo_epochs: int = 1
    minibatch_size: int = 32768
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 0.5
    seed: int = 1234
    max_steps: int = 3000
    tau: float = 0.05
    step_cost: float = -0.01
    alpha: float = 1.0
    goal_bonus: float = 10.0
    obs_dim: int = 32


class AsyncPPOEngine:
    """Async PPO engine with double-buffered rollouts."""

    def __init__(
        self, config: AsyncPPOEngineConfig, *, experiment: Any | None = None
    ) -> None:
        torch = _require_torch()
        device = config.device
        obs_format = config.obs_format
        if obs_format not in ("u8", "packed2"):
            raise ValueError("obs_format must be 'u8' or 'packed2'")
        if obs_format == "packed2" and device != "cuda":
            raise ValueError("packed2 currently requires CUDA device")
        if device == "cuda":
            if not _cuda_available():
                raise RuntimeError("CUDA not available for AsyncPPOEngine.")
            if not torch.cuda.is_available():
                raise RuntimeError("torch CUDA not available for AsyncPPOEngine.")
        elif device != "cpu":
            raise ValueError(f"Unsupported device: {device}")

        from gbxcule.core.reset_cache import ResetCache
        from gbxcule.rl.async_ppo import AsyncPPOBufferManager
        from gbxcule.rl.goal_template import load_goal_template
        from gbxcule.rl.models import PixelActorCriticCNN
        from gbxcule.rl.rollout import RolloutBuffer

        self.config = config
        self._torch = torch
        self.experiment = experiment
        self.obs_format = obs_format
        if device == "cuda":
            backend = WarpVecCudaBackend(
                str(Path(config.rom_path)),
                num_envs=config.num_envs,
                frames_per_step=config.frames_per_step,
                release_after_frames=config.release_after_frames,
                obs_dim=config.obs_dim,
                render_pixels=True,
                render_pixels_packed=False,
                render_on_step=False,
            )
        else:
            backend = WarpVecCpuBackend(
                str(Path(config.rom_path)),
                num_envs=config.num_envs,
                frames_per_step=config.frames_per_step,
                release_after_frames=config.release_after_frames,
                obs_dim=config.obs_dim,
                render_pixels=True,
                render_pixels_packed=False,
                render_on_step=False,
            )
        self.backend = backend
        backend.reset(seed=config.seed)
        backend.load_state_file(str(Path(config.state_path)), env_idx=0)
        self.reset_cache = ResetCache.from_backend(backend, env_idx=0)
        if device == "cuda":
            all_mask = torch.ones(config.num_envs, dtype=torch.uint8, device="cuda")
            self.reset_cache.apply_mask_torch(all_mask)
        else:
            import numpy as np

            all_mask = np.ones((config.num_envs,), dtype=bool)
            self.reset_cache.apply_mask_np(all_mask)

        template, _ = load_goal_template(
            Path(config.goal_dir),
            action_codec_id=backend.action_codec.id,
            frames_per_step=None,
            release_after_frames=None,
            stack_k=1,
            dist_metric=None,
            pipeline_version=None,
            obs_format="packed2" if obs_format == "packed2" else None,
        )
        goal_np = template.squeeze(0) if template.ndim == 3 else template
        goal_t = torch.tensor(goal_np, device=device, dtype=torch.uint8)
        if goal_t.ndim == 3:
            goal_t = goal_t.unsqueeze(0)
        self.goal = goal_t
        self.goal_f = self.goal.to(dtype=torch.float32) if obs_format == "u8" else None

        self.actor_model = PixelActorCriticCNN(
            num_actions=backend.num_actions,
            in_frames=1,
            input_format=obs_format,
        ).to(device)
        self.learner_model = PixelActorCriticCNN(
            num_actions=backend.num_actions,
            in_frames=1,
            input_format=obs_format,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.learner_model.parameters(), lr=config.lr)

        buffer_count = 2 if device == "cuda" else 1
        obs_steps = config.steps_per_rollout + 1 if obs_format == "packed2" else None
        self.rollout_buffers = [
            RolloutBuffer(
                steps=config.steps_per_rollout,
                num_envs=config.num_envs,
                stack_k=1,
                obs_format=obs_format,
                obs_steps=obs_steps,
                device=device,
            )
            for _ in range(buffer_count)
        ]

        if device == "cuda":
            self.manager = AsyncPPOBufferManager(num_buffers=2)
            self.actor_stream = torch.cuda.Stream()
            self.learner_stream = torch.cuda.Stream()
            self.policy_ready_event = torch.cuda.Event()
        else:
            self.manager = None
            self.actor_stream = None
            self.learner_stream = None
            self.policy_ready_event = None

        if device == "cuda":
            backend_cuda = cast(WarpVecCudaBackend, backend)
            assert self.actor_stream is not None
            assert self.learner_stream is not None
            assert self.policy_ready_event is not None
            with torch.cuda.stream(self.actor_stream):
                backend_cuda.render_pixels_snapshot_torch()
                self.obs = backend_cuda.pixels_torch().unsqueeze(1)
                self.episode_steps = torch.zeros(
                    config.num_envs, dtype=torch.int32, device=device
                )
                self.prev_dist = torch.ones(
                    config.num_envs, dtype=torch.float32, device=device
                )
            torch.cuda.synchronize()
            with torch.cuda.stream(self.learner_stream):
                self._sync_actor_weights()
                self.policy_ready_event.record(self.learner_stream)
        else:
            backend.render_pixels_snapshot()
            self.obs = backend.pixels_torch().unsqueeze(1)
            self.episode_steps = torch.zeros(
                config.num_envs, dtype=torch.int32, device=device
            )
            self.prev_dist = torch.ones(
                config.num_envs, dtype=torch.float32, device=device
            )
            self._sync_actor_weights()

    def _sync_actor_weights(self) -> None:
        for p_actor, p_learner in zip(
            self.actor_model.parameters(),
            self.learner_model.parameters(),
            strict=True,
        ):
            p_actor.data.copy_(p_learner.data)

    def run(self, *, updates: int | None = None) -> dict[str, Any]:
        from gbxcule.rl.ppo import (
            compute_gae,
            logprob_from_logits,
            ppo_update_minibatch,
        )

        torch = self._torch
        cfg = self.config
        update_count = int(updates) if updates is not None else int(cfg.updates)
        if update_count < 1:
            raise ValueError("updates must be >= 1")

        if cfg.device == "cpu":
            return self._run_cpu(
                update_count,
                compute_gae=compute_gae,
                logprob_from_logits=logprob_from_logits,
                ppo_update_minibatch=ppo_update_minibatch,
            )
        if cfg.obs_format == "packed2":
            return self._run_cuda_packed(
                update_count,
                compute_gae=compute_gae,
                logprob_from_logits=logprob_from_logits,
                ppo_update_minibatch=ppo_update_minibatch,
            )

        backend = cast(WarpVecCudaBackend, self.backend)
        manager = self.manager
        actor_stream = self.actor_stream
        learner_stream = self.learner_stream
        policy_ready_event = self.policy_ready_event
        assert manager is not None
        assert actor_stream is not None
        assert learner_stream is not None
        assert policy_ready_event is not None

        actor_events: list[tuple[Any, Any]] = []
        learner_events: list[tuple[Any, Any]] = []
        total_events: list[tuple[Any, Any]] = []
        default_stream = torch.cuda.current_stream()

        start = time.perf_counter()
        for update_idx in range(update_count):
            buf_idx = update_idx % 2
            prev_idx = (update_idx - 1) % 2

            actor_start = torch.cuda.Event(enable_timing=True)
            actor_end = torch.cuda.Event(enable_timing=True)
            learner_start = torch.cuda.Event(enable_timing=True)
            learner_end = torch.cuda.Event(enable_timing=True)
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record(default_stream)

            with torch.cuda.stream(learner_stream):
                learner_start.record(learner_stream)
                if update_idx > 0:
                    manager.wait_ready(prev_idx, learner_stream)
                    rollout = self.rollout_buffers[prev_idx]
                    with torch.no_grad():
                        _, last_value = self.learner_model(self.obs)
                    advantages, returns = compute_gae(
                        rollout.rewards,
                        rollout.values,
                        rollout.dones,
                        last_value,
                        gamma=cfg.gamma,
                        gae_lambda=cfg.gae_lambda,
                    )
                    batch = rollout.as_batch(flatten_obs=True)
                    ppo_update_minibatch(
                        model=self.learner_model,
                        optimizer=self.optimizer,
                        obs=batch["obs_u8"],
                        actions=batch["actions"],
                        old_logprobs=batch["logprobs"],
                        returns=returns.reshape(-1),
                        advantages=advantages.reshape(-1),
                        clip=cfg.clip,
                        value_coef=cfg.value_coef,
                        entropy_coef=cfg.entropy_coef,
                        ppo_epochs=cfg.ppo_epochs,
                        minibatch_size=cfg.minibatch_size,
                        grad_clip=cfg.grad_clip,
                    )
                    self._sync_actor_weights()
                    policy_ready_event.record(learner_stream)
                    manager.mark_free(prev_idx, learner_stream)
                learner_end.record(learner_stream)

            with torch.cuda.stream(actor_stream):
                manager.wait_free(buf_idx, actor_stream)
                actor_stream.wait_event(policy_ready_event)
                rollout = self.rollout_buffers[buf_idx]
                rollout.reset()
                actor_start.record(actor_stream)
                for _ in range(cfg.steps_per_rollout):
                    with torch.no_grad():
                        logits, values = self.actor_model(self.obs)
                        actions_i64 = torch.multinomial(
                            torch.softmax(logits, dim=-1), num_samples=1
                        ).squeeze(1)
                        logprobs = logprob_from_logits(logits, actions_i64)
                    actions = actions_i64.to(torch.int32)
                    backend.step_torch(actions)
                    backend.render_pixels_snapshot_torch()
                    next_obs = backend.pixels_torch().unsqueeze(1)
                    self.episode_steps = self.episode_steps + 1

                    diff = torch.abs(next_obs.float() - self.goal_f)
                    curr_dist = diff.mean(dim=(1, 2, 3)) / 3.0
                    done = curr_dist < cfg.tau
                    trunc = self.episode_steps >= cfg.max_steps
                    reward = torch.full((cfg.num_envs,), cfg.step_cost, device="cuda")
                    reward += cfg.alpha * (self.prev_dist - curr_dist)
                    reward[done] += cfg.goal_bonus
                    reset_mask = done | trunc

                    rollout.add(
                        self.obs,
                        actions,
                        reward,
                        reset_mask,
                        values.detach(),
                        logprobs.detach(),
                    )

                    self.reset_cache.apply_mask_torch(reset_mask.to(torch.uint8))
                    self.episode_steps = torch.where(
                        reset_mask,
                        torch.zeros_like(self.episode_steps),
                        self.episode_steps,
                    )
                    if reset_mask.any().item():
                        backend.render_pixels_snapshot_torch()
                        next_obs = backend.pixels_torch().unsqueeze(1)
                        curr_dist = (
                            torch.abs(next_obs.float() - self.goal_f).mean(
                                dim=(1, 2, 3)
                            )
                            / 3.0
                        )
                    self.prev_dist = curr_dist
                    self.obs = next_obs

                actor_end.record(actor_stream)
                manager.mark_ready(buf_idx, actor_stream, policy_version=update_idx)

            default_stream.wait_event(actor_end)
            default_stream.wait_event(learner_end)
            total_end.record(default_stream)
            actor_events.append((actor_start, actor_end))
            learner_events.append((learner_start, learner_end))
            total_events.append((total_start, total_end))

        with torch.cuda.stream(learner_stream):
            last_idx = (update_count - 1) % 2
            manager.wait_ready(last_idx, learner_stream)
            rollout = self.rollout_buffers[last_idx]
            with torch.no_grad():
                _, last_value = self.learner_model(self.obs)
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.dones,
                last_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )
            batch = rollout.as_batch(flatten_obs=True)
            ppo_update_minibatch(
                model=self.learner_model,
                optimizer=self.optimizer,
                obs=batch["obs_u8"],
                actions=batch["actions"],
                old_logprobs=batch["logprobs"],
                returns=returns.reshape(-1),
                advantages=advantages.reshape(-1),
                clip=cfg.clip,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
                ppo_epochs=cfg.ppo_epochs,
                minibatch_size=cfg.minibatch_size,
                grad_clip=cfg.grad_clip,
            )
            self._sync_actor_weights()
            policy_ready_event.record(learner_stream)
            manager.mark_free(last_idx, learner_stream)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        env_steps = cfg.num_envs * cfg.steps_per_rollout * update_count
        sps = env_steps / elapsed if elapsed > 0 else 0.0

        t_actor_ms = 0.0
        t_learner_ms = 0.0
        t_total_ms = 0.0
        timed_updates = 0
        for idx in range(update_count):
            if idx == 0:
                continue
            a_start, a_end = actor_events[idx]
            l_start, l_end = learner_events[idx]
            t_start, t_end = total_events[idx]
            t_actor_ms += float(a_start.elapsed_time(a_end))
            t_learner_ms += float(l_start.elapsed_time(l_end))
            t_total_ms += float(t_start.elapsed_time(t_end))
            timed_updates += 1

        if timed_updates > 0:
            t_actor_ms /= timed_updates
            t_learner_ms /= timed_updates
            t_total_ms /= timed_updates

        overlap_eff = (
            (t_actor_ms + t_learner_ms) / t_total_ms if t_total_ms > 0 else 0.0
        )

        metrics = {
            "num_envs": int(cfg.num_envs),
            "frames_per_step": int(cfg.frames_per_step),
            "release_after_frames": int(cfg.release_after_frames),
            "steps_per_rollout": int(cfg.steps_per_rollout),
            "updates": int(update_count),
            "ppo_epochs": int(cfg.ppo_epochs),
            "minibatch_size": int(cfg.minibatch_size),
            "env_steps": int(env_steps),
            "elapsed_s": float(elapsed),
            "sps": float(sps),
            "t_actor_rollout_ms": float(t_actor_ms),
            "t_learner_update_ms": float(t_learner_ms),
            "t_total_update_ms": float(t_total_ms),
            "overlap_efficiency": float(overlap_eff),
        }
        if self.experiment is not None:
            self.experiment.log_metrics(metrics)
        return metrics

    def _run_cpu(
        self,
        update_count: int,
        *,
        compute_gae,
        logprob_from_logits,
        ppo_update_minibatch,
    ) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        import numpy as np

        torch = self._torch
        cfg = self.config
        t_actor_ms = 0.0
        t_learner_ms = 0.0
        t_total_ms = 0.0

        start = time.perf_counter()
        for _ in range(update_count):
            rollout = self.rollout_buffers[0]
            rollout.reset()
            update_start = time.perf_counter()
            actor_start = time.perf_counter()
            for _ in range(cfg.steps_per_rollout):
                with torch.no_grad():
                    logits, values = self.actor_model(self.obs)
                    actions_i64 = torch.multinomial(
                        torch.softmax(logits, dim=-1), num_samples=1
                    ).squeeze(1)
                    logprobs = logprob_from_logits(logits, actions_i64)
                actions = actions_i64.to(torch.int32).cpu().numpy()
                self.backend.step(actions)
                self.backend.render_pixels_snapshot()
                next_obs = self.backend.pixels_torch().unsqueeze(1)
                self.episode_steps = self.episode_steps + 1

                diff = torch.abs(next_obs.float() - self.goal_f)
                curr_dist = diff.mean(dim=(1, 2, 3)) / 3.0
                done = curr_dist < cfg.tau
                trunc = self.episode_steps >= cfg.max_steps
                reward = torch.full((cfg.num_envs,), cfg.step_cost, device="cpu")
                reward += cfg.alpha * (self.prev_dist - curr_dist)
                reward[done] += cfg.goal_bonus
                reset_mask = done | trunc

                rollout.add(
                    self.obs,
                    actions_i64.to(torch.int32),
                    reward,
                    reset_mask,
                    values.detach(),
                    logprobs.detach(),
                )

                self.reset_cache.apply_mask_np(
                    np.asarray(reset_mask.cpu().numpy(), dtype=bool)
                )
                self.episode_steps = torch.where(
                    reset_mask,
                    torch.zeros_like(self.episode_steps),
                    self.episode_steps,
                )
                if reset_mask.any().item():
                    self.backend.render_pixels_snapshot()
                    next_obs = self.backend.pixels_torch().unsqueeze(1)
                    curr_dist = (
                        torch.abs(next_obs.float() - self.goal_f).mean(dim=(1, 2, 3))
                        / 3.0
                    )
                self.prev_dist = curr_dist
                self.obs = next_obs

            actor_end = time.perf_counter()
            learner_start = time.perf_counter()
            with torch.no_grad():
                _, last_value = self.learner_model(self.obs)
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.dones,
                last_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )
            batch = rollout.as_batch(flatten_obs=True)
            ppo_update_minibatch(
                model=self.learner_model,
                optimizer=self.optimizer,
                obs=batch["obs_u8"],
                actions=batch["actions"],
                old_logprobs=batch["logprobs"],
                returns=returns.reshape(-1),
                advantages=advantages.reshape(-1),
                clip=cfg.clip,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
                ppo_epochs=cfg.ppo_epochs,
                minibatch_size=cfg.minibatch_size,
                grad_clip=cfg.grad_clip,
            )
            self._sync_actor_weights()
            learner_end = time.perf_counter()
            update_end = learner_end

            t_actor_ms += (actor_end - actor_start) * 1000.0
            t_learner_ms += (learner_end - learner_start) * 1000.0
            t_total_ms += (update_end - update_start) * 1000.0

        elapsed = time.perf_counter() - start
        env_steps = cfg.num_envs * cfg.steps_per_rollout * update_count
        sps = env_steps / elapsed if elapsed > 0 else 0.0

        t_actor_ms /= update_count
        t_learner_ms /= update_count
        t_total_ms /= update_count
        overlap_eff = (
            (t_actor_ms + t_learner_ms) / t_total_ms if t_total_ms > 0 else 0.0
        )

        metrics = {
            "num_envs": int(cfg.num_envs),
            "frames_per_step": int(cfg.frames_per_step),
            "release_after_frames": int(cfg.release_after_frames),
            "steps_per_rollout": int(cfg.steps_per_rollout),
            "updates": int(update_count),
            "ppo_epochs": int(cfg.ppo_epochs),
            "minibatch_size": int(cfg.minibatch_size),
            "env_steps": int(env_steps),
            "elapsed_s": float(elapsed),
            "sps": float(sps),
            "t_actor_rollout_ms": float(t_actor_ms),
            "t_learner_update_ms": float(t_learner_ms),
            "t_total_update_ms": float(t_total_ms),
            "overlap_efficiency": float(overlap_eff),
        }
        if self.experiment is not None:
            self.experiment.log_metrics(metrics)
        return metrics

    def _run_cuda_packed(
        self,
        update_count: int,
        *,
        compute_gae,
        logprob_from_logits,
        ppo_update_minibatch,
    ) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        from gbxcule.rl.packed_metrics import packed_l1_distance

        torch = self._torch
        cfg = self.config
        backend = cast(WarpVecCudaBackend, self.backend)
        manager = self.manager
        actor_stream = self.actor_stream
        learner_stream = self.learner_stream
        policy_ready_event = self.policy_ready_event
        assert manager is not None
        assert actor_stream is not None
        assert learner_stream is not None
        assert policy_ready_event is not None

        actor_events: list[tuple[Any, Any]] = []
        learner_events: list[tuple[Any, Any]] = []
        total_events: list[tuple[Any, Any]] = []
        default_stream = torch.cuda.current_stream()

        start = time.perf_counter()
        for update_idx in range(update_count):
            buf_idx = update_idx % 2
            prev_idx = (update_idx - 1) % 2

            actor_start = torch.cuda.Event(enable_timing=True)
            actor_end = torch.cuda.Event(enable_timing=True)
            learner_start = torch.cuda.Event(enable_timing=True)
            learner_end = torch.cuda.Event(enable_timing=True)
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record(default_stream)

            with torch.cuda.stream(learner_stream):
                learner_start.record(learner_stream)
                if update_idx > 0:
                    manager.wait_ready(prev_idx, learner_stream)
                    rollout = self.rollout_buffers[prev_idx]
                    with torch.no_grad():
                        _, last_value = self.learner_model(self.obs)
                    advantages, returns = compute_gae(
                        rollout.rewards,
                        rollout.values,
                        rollout.dones,
                        last_value,
                        gamma=cfg.gamma,
                        gae_lambda=cfg.gae_lambda,
                    )
                    batch = rollout.as_batch(flatten_obs=True)
                    ppo_update_minibatch(
                        model=self.learner_model,
                        optimizer=self.optimizer,
                        obs=batch["obs_u8"],
                        actions=batch["actions"],
                        old_logprobs=batch["logprobs"],
                        returns=returns.reshape(-1),
                        advantages=advantages.reshape(-1),
                        clip=cfg.clip,
                        value_coef=cfg.value_coef,
                        entropy_coef=cfg.entropy_coef,
                        ppo_epochs=cfg.ppo_epochs,
                        minibatch_size=cfg.minibatch_size,
                        grad_clip=cfg.grad_clip,
                    )
                    self._sync_actor_weights()
                    policy_ready_event.record(learner_stream)
                    manager.mark_free(prev_idx, learner_stream)
                learner_end.record(learner_stream)

            with torch.cuda.stream(actor_stream):
                manager.wait_free(buf_idx, actor_stream)
                actor_stream.wait_event(policy_ready_event)
                rollout = self.rollout_buffers[buf_idx]
                rollout.reset()
                actor_start.record(actor_stream)

                slot0 = rollout.obs_slot(0)
                backend.render_pixels_snapshot_packed_to_torch(slot0, 0)
                obs = slot0
                self.obs = obs
                self.prev_dist = packed_l1_distance(obs, self.goal)

                for t in range(cfg.steps_per_rollout):
                    with torch.no_grad():
                        logits, values = self.actor_model(obs)
                        actions_i64 = torch.multinomial(
                            torch.softmax(logits, dim=-1), num_samples=1
                        ).squeeze(1)
                        logprobs = logprob_from_logits(logits, actions_i64)
                    actions = actions_i64.to(torch.int32)
                    backend.step_torch(actions)
                    next_slot = rollout.obs_slot(t + 1)
                    backend.render_pixels_snapshot_packed_to_torch(next_slot, 0)
                    self.episode_steps = self.episode_steps + 1

                    curr_dist = packed_l1_distance(next_slot, self.goal)
                    done = curr_dist < cfg.tau
                    trunc = self.episode_steps >= cfg.max_steps
                    reward = torch.full((cfg.num_envs,), cfg.step_cost, device="cuda")
                    reward += cfg.alpha * (self.prev_dist - curr_dist)
                    reward[done] += cfg.goal_bonus
                    reset_mask = done | trunc

                    rollout.set_step_fields(
                        t,
                        actions,
                        reward,
                        reset_mask,
                        values.detach(),
                        logprobs.detach(),
                    )

                    self.reset_cache.apply_mask_torch(reset_mask.to(torch.uint8))
                    self.episode_steps = torch.where(
                        reset_mask,
                        torch.zeros_like(self.episode_steps),
                        self.episode_steps,
                    )
                    if reset_mask.any().item():
                        backend.render_pixels_snapshot_packed_to_torch(next_slot, 0)
                        curr_dist = packed_l1_distance(next_slot, self.goal)

                    self.prev_dist = curr_dist
                    obs = next_slot

                self.obs = obs
                actor_end.record(actor_stream)
                manager.mark_ready(buf_idx, actor_stream, policy_version=update_idx)

            default_stream.wait_event(actor_end)
            default_stream.wait_event(learner_end)
            total_end.record(default_stream)
            actor_events.append((actor_start, actor_end))
            learner_events.append((learner_start, learner_end))
            total_events.append((total_start, total_end))

        with torch.cuda.stream(learner_stream):
            last_idx = (update_count - 1) % 2
            manager.wait_ready(last_idx, learner_stream)
            rollout = self.rollout_buffers[last_idx]
            with torch.no_grad():
                _, last_value = self.learner_model(self.obs)
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.dones,
                last_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )
            batch = rollout.as_batch(flatten_obs=True)
            ppo_update_minibatch(
                model=self.learner_model,
                optimizer=self.optimizer,
                obs=batch["obs_u8"],
                actions=batch["actions"],
                old_logprobs=batch["logprobs"],
                returns=returns.reshape(-1),
                advantages=advantages.reshape(-1),
                clip=cfg.clip,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
                ppo_epochs=cfg.ppo_epochs,
                minibatch_size=cfg.minibatch_size,
                grad_clip=cfg.grad_clip,
            )
            self._sync_actor_weights()
            policy_ready_event.record(learner_stream)
            manager.mark_free(last_idx, learner_stream)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        env_steps = cfg.num_envs * cfg.steps_per_rollout * update_count
        sps = env_steps / elapsed if elapsed > 0 else 0.0

        t_actor_ms = 0.0
        t_learner_ms = 0.0
        t_total_ms = 0.0
        timed_updates = 0
        for idx in range(update_count):
            if idx == 0:
                continue
            a_start, a_end = actor_events[idx]
            l_start, l_end = learner_events[idx]
            t_start, t_end = total_events[idx]
            t_actor_ms += float(a_start.elapsed_time(a_end))
            t_learner_ms += float(l_start.elapsed_time(l_end))
            t_total_ms += float(t_start.elapsed_time(t_end))
            timed_updates += 1

        if timed_updates > 0:
            t_actor_ms /= timed_updates
            t_learner_ms /= timed_updates
            t_total_ms /= timed_updates

        overlap_eff = (
            (t_actor_ms + t_learner_ms) / t_total_ms if t_total_ms > 0 else 0.0
        )

        metrics = {
            "num_envs": int(cfg.num_envs),
            "frames_per_step": int(cfg.frames_per_step),
            "release_after_frames": int(cfg.release_after_frames),
            "steps_per_rollout": int(cfg.steps_per_rollout),
            "updates": int(update_count),
            "ppo_epochs": int(cfg.ppo_epochs),
            "minibatch_size": int(cfg.minibatch_size),
            "env_steps": int(env_steps),
            "elapsed_s": float(elapsed),
            "sps": float(sps),
            "t_actor_rollout_ms": float(t_actor_ms),
            "t_learner_update_ms": float(t_learner_ms),
            "t_total_update_ms": float(t_total_ms),
            "overlap_efficiency": float(overlap_eff),
        }
        if self.experiment is not None:
            self.experiment.log_metrics(metrics)
        return metrics

    def close(self) -> None:
        self.backend.close()
