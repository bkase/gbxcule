"""Rollout buffer for PPO (torch)."""

from __future__ import annotations

from typing import Any

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W, DOWNSAMPLE_W_BYTES


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl. Install with `uv sync`."
        ) from exc


class RolloutBuffer:  # type: ignore[no-any-unimported]
    """Preallocated rollout buffer (GPU-friendly)."""

    def __init__(
        self,
        *,
        steps: int,
        num_envs: int,
        stack_k: int,
        obs_format: str = "u8",
        height: int = DOWNSAMPLE_H,
        width: int = DOWNSAMPLE_W,
        device: str = "cuda",
    ) -> None:
        torch = _require_torch()
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if stack_k < 1:
            raise ValueError("stack_k must be >= 1")
        if height < 1 or width < 1:
            raise ValueError("height/width must be >= 1")

        self._torch = torch
        if obs_format not in ("u8", "packed2"):
            raise ValueError("obs_format must be 'u8' or 'packed2'")

        if obs_format == "packed2":
            if width == DOWNSAMPLE_W:
                width = DOWNSAMPLE_W_BYTES
            if width != DOWNSAMPLE_W_BYTES:
                raise ValueError("packed2 width must equal DOWNSAMPLE_W_BYTES")

        self.steps = int(steps)
        self.num_envs = int(num_envs)
        self.stack_k = int(stack_k)
        self.obs_format = obs_format
        self.height = int(height)
        self.width = int(width)
        self.device = torch.device(device)

        self.obs_u8 = torch.empty(
            (self.steps, self.num_envs, self.stack_k, self.height, self.width),
            dtype=torch.uint8,
            device=self.device,
        )
        self.actions = torch.empty(
            (self.steps, self.num_envs),
            dtype=torch.int32,
            device=self.device,
        )
        self.rewards = torch.empty(
            (self.steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.dones = torch.empty(
            (self.steps, self.num_envs),
            dtype=torch.bool,
            device=self.device,
        )
        self.values = torch.empty(
            (self.steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.logprobs = torch.empty(
            (self.steps, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    @property
    def obs(self):  # type: ignore[no-untyped-def]
        return self.obs_u8

    def reset(self) -> None:
        self._step = 0

    def add(  # type: ignore[no-untyped-def]
        self,
        obs_u8,
        actions,
        rewards,
        dones,
        values,
        logprobs,
    ) -> None:
        torch = self._torch
        if self._step >= self.steps:
            raise RuntimeError("rollout buffer is full")
        if obs_u8.shape != (self.num_envs, self.stack_k, self.height, self.width):
            raise ValueError("obs_u8 shape mismatch")
        if obs_u8.dtype is not torch.uint8:
            raise ValueError("obs_u8 must be uint8")
        if actions.shape != (self.num_envs,):
            raise ValueError("actions shape mismatch")
        if actions.dtype is not torch.int32:
            raise ValueError("actions must be int32")
        if rewards.shape != (self.num_envs,):
            raise ValueError("rewards shape mismatch")
        if rewards.dtype is not torch.float32:
            raise ValueError("rewards must be float32")
        if dones.shape != (self.num_envs,):
            raise ValueError("dones shape mismatch")
        if dones.dtype is not torch.bool:
            raise ValueError("dones must be bool")
        if values.shape != (self.num_envs,):
            raise ValueError("values shape mismatch")
        if values.dtype is not torch.float32:
            raise ValueError("values must be float32")
        if logprobs.shape != (self.num_envs,):
            raise ValueError("logprobs shape mismatch")
        if logprobs.dtype is not torch.float32:
            raise ValueError("logprobs must be float32")

        self.obs_u8[self._step].copy_(obs_u8)
        self.actions[self._step].copy_(actions)
        self.rewards[self._step].copy_(rewards)
        self.dones[self._step].copy_(dones)
        self.values[self._step].copy_(values)
        self.logprobs[self._step].copy_(logprobs)
        self._step += 1

    def obs_slot(self, step_idx: int):  # type: ignore[no-untyped-def]
        if step_idx < 0 or step_idx >= self.steps:
            raise ValueError("step_idx out of range")
        return self.obs_u8[step_idx]

    def set_step_fields(  # type: ignore[no-untyped-def]
        self,
        step_idx: int,
        actions,
        rewards,
        dones,
        values,
        logprobs,
    ) -> None:
        torch = self._torch
        if step_idx < 0 or step_idx >= self.steps:
            raise ValueError("step_idx out of range")
        if actions.shape != (self.num_envs,):
            raise ValueError("actions shape mismatch")
        if actions.dtype is not torch.int32:
            raise ValueError("actions must be int32")
        if rewards.shape != (self.num_envs,):
            raise ValueError("rewards shape mismatch")
        if rewards.dtype is not torch.float32:
            raise ValueError("rewards must be float32")
        if dones.shape != (self.num_envs,):
            raise ValueError("dones shape mismatch")
        if dones.dtype is not torch.bool:
            raise ValueError("dones must be bool")
        if values.shape != (self.num_envs,):
            raise ValueError("values shape mismatch")
        if values.dtype is not torch.float32:
            raise ValueError("values must be float32")
        if logprobs.shape != (self.num_envs,):
            raise ValueError("logprobs shape mismatch")
        if logprobs.dtype is not torch.float32:
            raise ValueError("logprobs must be float32")

        self.actions[step_idx].copy_(actions)
        self.rewards[step_idx].copy_(rewards)
        self.dones[step_idx].copy_(dones)
        self.values[step_idx].copy_(values)
        self.logprobs[step_idx].copy_(logprobs)

    def as_batch(self, *, flatten_obs: bool = True):  # type: ignore[no-untyped-def]
        if self._step != self.steps:
            raise RuntimeError("rollout buffer is not full")
        batch = self.steps * self.num_envs
        obs = (
            self.obs_u8.reshape(batch, self.stack_k, self.height, self.width)
            if flatten_obs
            else self.obs_u8
        )
        return {
            "obs_u8": obs,
            "actions": self.actions.reshape(batch),
            "rewards": self.rewards.reshape(batch),
            "dones": self.dones.reshape(batch),
            "values": self.values.reshape(batch),
            "logprobs": self.logprobs.reshape(batch),
        }
