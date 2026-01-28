"""Torch-facing pixels-only RL env with reward/done/trunc + autoreset."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from gbxcule.backends.warp_vec import WarpVecCudaBackend
from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W
from gbxcule.core.reset_cache import ResetCache
from gbxcule.rl.goal_match import (
    RewardShapingConfig,
    compute_dist_l1,
    compute_done,
    compute_reward,
    compute_trunc,
    update_consecutive,
)
from gbxcule.rl.goal_template import load_goal_template


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.pokered_pixels_goal_env. "
            "Install with `uv sync --group rl`."
        ) from exc


class PokeredPixelsGoalEnv:
    """Pixels-only RL env wrapper with reward shaping + autoreset."""

    def __init__(
        self,
        rom_path: str,
        *,
        goal_dir: str,
        state_path: str | None = None,
        num_envs: int = 1,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        stack_k: int | None = None,
        action_codec: str | None = None,
        max_steps: int = 512,
        step_cost: float = -0.01,
        alpha: float = 1.0,
        goal_bonus: float = 10.0,
        tau: float | None = None,
        k_consecutive: int | None = None,
        force_lcdc_on_render: bool = True,
        skip_reset_if_empty: bool = False,
        info_mode: str = "full",
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        torch = _require_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA torch is required for PokeredPixelsGoalEnv.")

        backend_kwargs: dict[str, Any] = {}
        if action_codec is not None:
            backend_kwargs["action_codec"] = action_codec

        self.backend = WarpVecCudaBackend(
            rom_path,
            num_envs=num_envs,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=32,
            render_pixels=True,
            force_lcdc_on_render=force_lcdc_on_render,
            **backend_kwargs,
        )

        action_codec_id = self.backend.action_codec.id
        template, meta = load_goal_template(
            Path(goal_dir),
            action_codec_id=action_codec_id,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            stack_k=stack_k,
            dist_metric=None,
            pipeline_version=None,
        )
        if stack_k is None:
            stack_k = meta.stack_k
        if stack_k < 1:
            raise ValueError(f"stack_k must be >= 1, got {stack_k}")

        self._torch = torch
        self.num_envs = num_envs
        self.stack_k = int(stack_k)
        self.max_steps = int(max_steps)
        self._state_path = str(Path(state_path)) if state_path is not None else None

        self._goal = torch.tensor(template, device="cuda", dtype=torch.uint8)
        self._tau = float(tau) if tau is not None else float(meta.tau)
        self._k_consecutive = (
            int(k_consecutive) if k_consecutive is not None else int(meta.k_consecutive)
        )
        self._reward_cfg = RewardShapingConfig(
            step_cost=step_cost, alpha=alpha, goal_bonus=goal_bonus
        )
        self._skip_reset_if_empty = bool(skip_reset_if_empty)
        if info_mode not in {"full", "stats", "none"}:
            raise ValueError(f"info_mode must be full, stats, or none, got {info_mode}")
        self._info_mode = info_mode

        self._pix = None
        self._stack = None
        self._start_stack = None
        self._start_dist = None
        self._prev_dist = None
        self._consec_match = None
        self._episode_step = None
        self._reset_cache = None

    @property
    def pixels(self):  # type: ignore[no-untyped-def]
        if self._pix is None:
            self._pix = self.backend.pixels_torch()
        return self._pix

    @property
    def obs(self):  # type: ignore[no-untyped-def]
        if self._stack is None:
            raise RuntimeError("Call reset() before accessing obs.")
        return self._stack

    def reset(self, seed: int | None = None):  # type: ignore[no-untyped-def]
        """Reset all envs to the cached start snapshot."""
        torch = self._torch
        self._pix = None
        self.backend.reset(seed=seed)
        if self._state_path is not None:
            self.backend.load_state_file(self._state_path, env_idx=0)

        self.backend.render_pixels_snapshot_torch()
        pix = self.pixels

        if self._stack is None:
            self._stack = torch.empty(
                (self.num_envs, self.stack_k, DOWNSAMPLE_H, DOWNSAMPLE_W),
                dtype=torch.uint8,
                device=pix.device,
            )
        if self.stack_k == 1:
            self._stack[:, 0].copy_(pix)
        else:
            for k in range(self.stack_k):
                self._stack[:, k].copy_(pix)

        if self._reset_cache is None:
            self._reset_cache = ResetCache.from_backend(self.backend, env_idx=0)

        mask_all = torch.ones((self.num_envs,), device="cuda", dtype=torch.uint8)
        self._reset_cache.apply_mask_torch(mask_all)

        self._start_stack = self._stack[0].clone()
        self._stack[:] = self._start_stack

        dist = compute_dist_l1(self._stack, self._goal)
        self._start_dist = dist.clone()

        if self._prev_dist is None:
            self._prev_dist = dist.clone()
        else:
            self._prev_dist.copy_(dist)

        if self._consec_match is None:
            self._consec_match = torch.zeros(
                (self.num_envs,), dtype=torch.int32, device="cuda"
            )
        else:
            self._consec_match.zero_()

        if self._episode_step is None:
            self._episode_step = torch.zeros(
                (self.num_envs,), dtype=torch.int32, device="cuda"
            )
        else:
            self._episode_step.zero_()

        return self._stack

    def step(self, actions):  # type: ignore[no-untyped-def]
        """Step all envs with reward/done/trunc + autoreset."""
        if self._stack is None or self._episode_step is None:
            raise RuntimeError("Call reset() before step().")
        if self._reset_cache is None or self._start_dist is None:
            raise RuntimeError("Reset cache not initialized. Call reset() first.")

        torch = self._torch
        tensor_type = getattr(torch, "Tensor", None)
        if tensor_type is None or not isinstance(actions, tensor_type):
            raise TypeError("actions must be a torch.Tensor")
        if actions.device.type != "cuda":
            raise ValueError("actions must be a CUDA tensor")
        if actions.ndim != 1 or int(actions.shape[0]) != self.num_envs:
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if actions.dtype is not torch.int32:
            raise ValueError("actions must have dtype torch.int32 (no implicit cast)")

        self.backend.step_torch(actions)
        self.backend.render_pixels_snapshot_torch()

        pix = self.pixels
        if self.stack_k == 1:
            self._stack[:, 0].copy_(pix)
        else:
            self._stack[:, :-1].copy_(self._stack[:, 1:].clone())
            self._stack[:, -1].copy_(pix)

        self._episode_step.add_(1)

        dist = compute_dist_l1(self._stack, self._goal)
        self._consec_match = update_consecutive(self._consec_match, dist, tau=self._tau)
        done = compute_done(self._consec_match, k_consecutive=self._k_consecutive)
        trunc = compute_trunc(self._episode_step, max_steps=self.max_steps)
        reward = compute_reward(self._prev_dist, dist, done, self._reward_cfg)

        reset_mask = done | trunc
        if self._skip_reset_if_empty:
            if torch.any(reset_mask):
                self._reset_cache.apply_mask_torch(reset_mask)

                zeros_i32 = torch.zeros_like(self._episode_step)
                self._episode_step = torch.where(
                    reset_mask, zeros_i32, self._episode_step
                )
                zeros_consec = torch.zeros_like(self._consec_match)
                self._consec_match = torch.where(
                    reset_mask, zeros_consec, self._consec_match
                )
                self._prev_dist = torch.where(reset_mask, self._start_dist, dist)

                if self._start_stack is not None:
                    self._stack[reset_mask] = self._start_stack
        else:
            self._reset_cache.apply_mask_torch(reset_mask)

            zeros_i32 = torch.zeros_like(self._episode_step)
            self._episode_step = torch.where(reset_mask, zeros_i32, self._episode_step)
            zeros_consec = torch.zeros_like(self._consec_match)
            self._consec_match = torch.where(
                reset_mask, zeros_consec, self._consec_match
            )
            self._prev_dist = torch.where(reset_mask, self._start_dist, dist)

            if self._start_stack is not None:
                self._stack[reset_mask] = self._start_stack

        if self._info_mode == "full":
            info = {"dist": dist, "reset_mask": reset_mask}
        elif self._info_mode == "stats":
            dist_f = dist.to(torch.float32)
            quantiles = torch.tensor([0.1, 0.5, 0.9], device=dist.device)
            q_vals = torch.quantile(dist_f, quantiles)
            info = {
                "dist_mean": float(dist_f.mean().item()),
                "dist_p10": float(q_vals[0].item()),
                "dist_p50": float(q_vals[1].item()),
                "dist_p90": float(q_vals[2].item()),
                "done_rate": float(done.to(torch.float32).mean().item()),
                "trunc_rate": float(trunc.to(torch.float32).mean().item()),
                "reset_rate": float(reset_mask.to(torch.float32).mean().item()),
            }
        else:
            info = {}
        return self._stack, reward, done, trunc, info

    def close(self) -> None:
        self.backend.close()
        self._pix = None
        self._stack = None
        self._start_stack = None
        self._start_dist = None
        self._prev_dist = None
        self._consec_match = None
        self._episode_step = None
        self._reset_cache = None
