"""CUDA replay ingestion helpers (Dreamer v3 M6)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gbxcule.rl.dreamer_v3.replay_commit import ReplayCommitManager
from gbxcule.rl.dreamer_v3.replay_cuda import ReplayRingCUDA


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


class ReplayIngestorCUDA:  # type: ignore[no-any-unimported]
    """Sheeprl-aligned replay ingestion helper."""

    def __init__(
        self,
        ring: ReplayRingCUDA,
        commit: ReplayCommitManager,
    ) -> None:
        torch = _require_torch()
        if ring.device != commit.device:
            raise ValueError("ring and commit manager must share the same device")
        self._torch = torch
        self.ring = ring
        self.commit = commit
        self._episode_id = torch.zeros(
            (ring.num_envs,), dtype=torch.int32, device=ring.device
        )
        self._pending_reward = None
        self._pending_terminated = None
        self._pending_truncated = None
        self._pending_is_first = None
        self._pending_continue = None
        self._pending_ready = False

    @property
    def episode_id(self):  # type: ignore[no-untyped-def]
        return self._episode_id

    def _coerce(  # type: ignore[no-untyped-def]
        self,
        value,
        *,
        dtype,
        name: str,
    ):
        torch = self._torch
        if value is None:
            raise ValueError(f"{name} must be provided")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if value.device != self.ring.device and not (
            value.device.type == "cuda" and self.ring.device.type == "cuda"
        ):
            raise ValueError(f"{name} must be on {self.ring.device}")
        if value.dtype is not dtype:
            raise ValueError(f"{name} must be {dtype}")
        if value.shape != (self.ring.num_envs,):
            raise ValueError(f"{name} shape mismatch")
        return value

    def start(  # type: ignore[no-untyped-def]
        self,
        render_fn: Callable[[Any], None],
        *,
        reward=None,
        terminated=None,
        truncated=None,
        is_first=None,
        episode_id=None,
    ) -> None:
        """Render the first observation and stage metadata for the next commit."""
        torch = self._torch
        if reward is None:
            reward = torch.zeros(
                (self.ring.num_envs,), dtype=torch.float32, device=self.ring.device
            )
        if terminated is None:
            terminated = torch.zeros(
                (self.ring.num_envs,), dtype=torch.bool, device=self.ring.device
            )
        if truncated is None:
            truncated = torch.zeros(
                (self.ring.num_envs,), dtype=torch.bool, device=self.ring.device
            )
        if is_first is None:
            is_first = torch.ones(
                (self.ring.num_envs,), dtype=torch.bool, device=self.ring.device
            )
        if episode_id is None:
            episode_id = self._episode_id
        else:
            episode_id = self._coerce(episode_id, dtype=torch.int32, name="episode_id")
            self._episode_id = episode_id.clone()

        reward = self._coerce(reward, dtype=torch.float32, name="reward")
        terminated = self._coerce(terminated, dtype=torch.bool, name="terminated")
        truncated = self._coerce(truncated, dtype=torch.bool, name="truncated")
        is_first = self._coerce(is_first, dtype=torch.bool, name="is_first")

        continue_ = torch.where(
            terminated, torch.zeros_like(reward), torch.ones_like(reward)
        )

        slot = self.ring.obs_slot(self.ring.head)
        render_fn(slot)

        self._pending_reward = reward
        self._pending_terminated = terminated
        self._pending_truncated = truncated
        self._pending_is_first = is_first
        self._pending_continue = continue_
        self._pending_ready = True

    def commit_action(self, action, *, stream=None) -> int:  # type: ignore[no-untyped-def]
        """Finalize the current observation row by writing its action + metadata."""
        if not self._pending_ready:
            raise RuntimeError(
                "No pending observation. Call start() or set_next_obs()."
            )
        torch = self._torch
        if not isinstance(action, torch.Tensor):
            raise ValueError("action must be a torch.Tensor")
        if action.device != self.ring.device:
            raise ValueError("action device mismatch")
        if action.dtype is not torch.int32:
            raise ValueError("action must be int32")
        if action.shape != (self.ring.num_envs,):
            raise ValueError("action shape mismatch")

        idx = self.ring.push_step(
            obs=None,
            action=action,
            reward=self._pending_reward,
            is_first=self._pending_is_first,
            continue_=self._pending_continue,
            episode_id=self._episode_id,
            terminated=self._pending_terminated,
            truncated=self._pending_truncated,
        )
        self.commit.mark_written(self.ring.total_steps - 1, stream=stream)
        self._pending_ready = False
        return idx

    def set_next_obs(  # type: ignore[no-untyped-def]
        self,
        render_fn: Callable[[Any], None],
        *,
        reward,
        terminated,
        truncated,
        reset_mask=None,
    ) -> None:
        """Stage metadata for the next observation and render it into the ring."""
        torch = self._torch
        reward = self._coerce(reward, dtype=torch.float32, name="reward")
        terminated = self._coerce(terminated, dtype=torch.bool, name="terminated")
        truncated = self._coerce(truncated, dtype=torch.bool, name="truncated")
        if reset_mask is None:
            reset_mask = terminated | truncated
        else:
            reset_mask = self._coerce(reset_mask, dtype=torch.bool, name="reset_mask")

        self._episode_id = torch.where(
            reset_mask,
            self._episode_id + 1,
            self._episode_id,
        )
        is_first = reset_mask
        continue_ = torch.where(
            terminated, torch.zeros_like(reward), torch.ones_like(reward)
        )

        slot = self.ring.obs_slot(self.ring.head)
        render_fn(slot)

        self._pending_reward = reward
        self._pending_terminated = terminated
        self._pending_truncated = truncated
        self._pending_is_first = is_first
        self._pending_continue = continue_
        self._pending_ready = True

    def insert_reset_obs(  # type: ignore[no-untyped-def]
        self,
        render_fn: Callable[[Any], None],
        *,
        reset_mask,
        stream=None,
    ) -> int:
        """Insert an explicit reset observation row (optional)."""
        reset_mask = self._coerce(reset_mask, dtype=self._torch.bool, name="reset_mask")
        self._episode_id = self._torch.where(
            reset_mask,
            self._episode_id + 1,
            self._episode_id,
        )
        reward = self._torch.zeros(
            (self.ring.num_envs,), dtype=self._torch.float32, device=self.ring.device
        )
        terminated = self._torch.zeros(
            (self.ring.num_envs,), dtype=self._torch.bool, device=self.ring.device
        )
        truncated = self._torch.zeros(
            (self.ring.num_envs,), dtype=self._torch.bool, device=self.ring.device
        )
        is_first = reset_mask
        continue_ = self._torch.ones_like(reward)
        action = self._torch.zeros(
            (self.ring.num_envs,), dtype=self._torch.int32, device=self.ring.device
        )

        slot = self.ring.obs_slot(self.ring.head)
        render_fn(slot)
        idx = self.ring.push_step(
            obs=None,
            action=action,
            reward=reward,
            is_first=is_first,
            continue_=continue_,
            episode_id=self._episode_id,
            terminated=terminated,
            truncated=truncated,
        )
        self.commit.mark_written(self.ring.total_steps - 1, stream=stream)
        return idx
