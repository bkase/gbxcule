"""Async PPO engine for reusable benchmarking and training."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

    def __init__(self, config: AsyncPPOEngineConfig) -> None:
        if not _cuda_available():
            raise RuntimeError("CUDA not available for AsyncPPOEngine.")
        torch = _require_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA not available for AsyncPPOEngine.")

        from gbxcule.backends.warp_vec import WarpVecCudaBackend
        from gbxcule.core.reset_cache import ResetCache
        from gbxcule.rl.async_ppo import AsyncPPOBufferManager
        from gbxcule.rl.goal_template import load_goal_template
        from gbxcule.rl.models import PixelActorCriticCNN
        from gbxcule.rl.rollout import RolloutBuffer

        self.config = config
        self._torch = torch
        self.backend = WarpVecCudaBackend(
            str(Path(config.rom_path)),
            num_envs=config.num_envs,
            frames_per_step=config.frames_per_step,
            release_after_frames=config.release_after_frames,
            obs_dim=config.obs_dim,
            render_pixels=False,
            render_pixels_packed=False,
        )
        self.backend.reset(seed=config.seed)
        self.backend.load_state_file(str(Path(config.state_path)), env_idx=0)
        self.reset_cache = ResetCache.from_backend(self.backend, env_idx=0)
        all_mask = torch.ones(config.num_envs, dtype=torch.uint8, device="cuda")
        self.reset_cache.apply_mask_torch(all_mask)

        template, _ = load_goal_template(
            Path(config.goal_dir),
            action_codec_id=self.backend.action_codec.id,
            frames_per_step=None,
            release_after_frames=None,
            stack_k=1,
            dist_metric=None,
            pipeline_version=None,
        )
        goal_np = template.squeeze(0) if template.ndim == 3 else template
        self.goal = torch.tensor(goal_np, device="cuda", dtype=torch.uint8).unsqueeze(0)
        self.goal_f = self.goal.to(dtype=torch.float32)

        self.actor_model = PixelActorCriticCNN(
            num_actions=self.backend.num_actions, in_frames=1
        ).to("cuda")
        self.learner_model = PixelActorCriticCNN(
            num_actions=self.backend.num_actions, in_frames=1
        ).to("cuda")
        self.optimizer = torch.optim.Adam(self.learner_model.parameters(), lr=config.lr)

        self.rollout_buffers = [
            RolloutBuffer(
                steps=config.steps_per_rollout,
                num_envs=config.num_envs,
                stack_k=1,
                device="cuda",
            )
            for _ in range(2)
        ]

        self.manager = AsyncPPOBufferManager(num_buffers=2)
        self.actor_stream = torch.cuda.Stream()
        self.learner_stream = torch.cuda.Stream()
        self.policy_ready_event = torch.cuda.Event()

        with torch.cuda.stream(self.actor_stream):
            self.backend.render_pixels_snapshot_torch()
            self.obs = self.backend.pixels_torch().unsqueeze(1)
            self.episode_steps = torch.zeros(
                config.num_envs, dtype=torch.int32, device="cuda"
            )
            self.prev_dist = torch.ones(
                config.num_envs, dtype=torch.float32, device="cuda"
            )
        torch.cuda.synchronize()

        with torch.cuda.stream(self.learner_stream):
            self._sync_actor_weights()
            self.policy_ready_event.record(self.learner_stream)

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

        start = time.perf_counter()
        for update_idx in range(update_count):
            buf_idx = update_idx % 2
            prev_idx = (update_idx - 1) % 2

            with torch.cuda.stream(self.learner_stream):
                if update_idx > 0:
                    self.manager.wait_ready(prev_idx, self.learner_stream)
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
                    self.policy_ready_event.record(self.learner_stream)
                    self.manager.mark_free(prev_idx, self.learner_stream)

            with torch.cuda.stream(self.actor_stream):
                self.manager.wait_free(buf_idx, self.actor_stream)
                self.actor_stream.wait_event(self.policy_ready_event)
                rollout = self.rollout_buffers[buf_idx]
                rollout.reset()
                for _ in range(cfg.steps_per_rollout):
                    with torch.no_grad():
                        logits, values = self.actor_model(self.obs)
                        actions_i64 = torch.multinomial(
                            torch.softmax(logits, dim=-1), num_samples=1
                        ).squeeze(1)
                        logprobs = logprob_from_logits(logits, actions_i64)
                    actions = actions_i64.to(torch.int32)
                    self.backend.step_torch(actions)
                    self.backend.render_pixels_snapshot_torch()
                    next_obs = self.backend.pixels_torch().unsqueeze(1)
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
                    self.backend.render_pixels_snapshot_torch()
                    next_obs = self.backend.pixels_torch().unsqueeze(1)
                    curr_dist = (
                        torch.abs(next_obs.float() - self.goal_f).mean(dim=(1, 2, 3))
                        / 3.0
                    )
                    self.prev_dist = curr_dist
                    self.obs = next_obs

                self.manager.mark_ready(
                    buf_idx, self.actor_stream, policy_version=update_idx
                )

        with torch.cuda.stream(self.learner_stream):
            last_idx = (update_count - 1) % 2
            self.manager.wait_ready(last_idx, self.learner_stream)
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
            self.policy_ready_event.record(self.learner_stream)
            self.manager.mark_free(last_idx, self.learner_stream)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        env_steps = cfg.num_envs * cfg.steps_per_rollout * update_count
        sps = env_steps / elapsed if elapsed > 0 else 0.0

        return {
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
        }

    def close(self) -> None:
        self.backend.close()
