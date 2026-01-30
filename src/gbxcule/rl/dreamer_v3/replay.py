"""ReplayRing for Dreamer v3 (time-major, device-agnostic)."""

from __future__ import annotations

from typing import Any

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - exercised in runtime
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


def _require_generator(gen: Any) -> None:
    torch = _require_torch()
    if gen is None:
        raise ValueError("gen must be a torch.Generator")
    if not isinstance(gen, torch.Generator):
        raise ValueError("gen must be a torch.Generator")


class ReplayRing:  # type: ignore[no-any-unimported]
    """Time-major replay ring with deterministic sampling and continuity checks."""

    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        device: str = "cpu",
        obs_shape: tuple[int, int, int] = (1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
        debug_checks: bool = False,
    ) -> None:
        torch = _require_torch()
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if len(obs_shape) != 3:
            raise ValueError("obs_shape must be (C, H, W)")

        self._torch = torch
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.device = torch.device(device)
        self.debug_checks = bool(debug_checks)

        c, h, w = self.obs_shape
        if c != 1:
            raise ValueError("obs_shape channels must be 1 for packed2")
        if h != DOWNSAMPLE_H or w != DOWNSAMPLE_W_BYTES:
            raise ValueError("obs_shape must match packed2 geometry")

        self.obs = torch.empty(
            (self.capacity, self.num_envs, c, h, w),
            dtype=torch.uint8,
            device=self.device,
        )
        self.action = torch.empty(
            (self.capacity, self.num_envs),
            dtype=torch.int32,
            device=self.device,
        )
        self.reward = torch.empty(
            (self.capacity, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.is_first = torch.empty(
            (self.capacity, self.num_envs),
            dtype=torch.bool,
            device=self.device,
        )
        self.continues = torch.empty(
            (self.capacity, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        self.episode_id = torch.empty(
            (self.capacity, self.num_envs),
            dtype=torch.int32,
            device=self.device,
        )

        self._head = 0
        self._size = 0
        self._total_steps = 0

    @property
    def head(self) -> int:
        return self._head

    @property
    def size(self) -> int:
        return self._size

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def __len__(self) -> int:
        return self._size

    def obs_slot(self, step_idx: int):  # type: ignore[no-untyped-def]
        if step_idx < 0 or step_idx >= self.capacity:
            raise ValueError("step_idx out of range")
        return self.obs[step_idx]

    def push_step(  # type: ignore[no-untyped-def]
        self,
        *,
        obs=None,
        action,
        reward,
        is_first,
        continue_,
        episode_id,
        validate_args: bool = True,
    ) -> int:
        torch = self._torch
        if validate_args:
            if obs is not None:
                if obs.shape != (self.num_envs, *self.obs_shape):
                    raise ValueError("obs shape mismatch")
                if obs.dtype is not torch.uint8:
                    raise ValueError("obs must be uint8")
                if obs.device != self.device:
                    raise ValueError("obs device mismatch")
            if action.shape != (self.num_envs,):
                raise ValueError("action shape mismatch")
            if action.dtype is not torch.int32:
                raise ValueError("action must be int32")
            if reward.shape != (self.num_envs,):
                raise ValueError("reward shape mismatch")
            if reward.dtype is not torch.float32:
                raise ValueError("reward must be float32")
            if is_first.shape != (self.num_envs,):
                raise ValueError("is_first shape mismatch")
            if is_first.dtype is not torch.bool:
                raise ValueError("is_first must be bool")
            if continue_.shape != (self.num_envs,):
                raise ValueError("continue shape mismatch")
            if continue_.dtype is not torch.float32:
                raise ValueError("continue must be float32")
            if episode_id.shape != (self.num_envs,):
                raise ValueError("episode_id shape mismatch")
            if episode_id.dtype is not torch.int32:
                raise ValueError("episode_id must be int32")
            if action.device != self.device:
                raise ValueError("action device mismatch")
            if reward.device != self.device:
                raise ValueError("reward device mismatch")
            if is_first.device != self.device:
                raise ValueError("is_first device mismatch")
            if continue_.device != self.device:
                raise ValueError("continue device mismatch")
            if episode_id.device != self.device:
                raise ValueError("episode_id device mismatch")

        idx = self._head
        if obs is not None:
            self.obs[idx].copy_(obs)
        self.action[idx].copy_(action)
        self.reward[idx].copy_(reward)
        self.is_first[idx].copy_(is_first)
        self.continues[idx].copy_(continue_)
        self.episode_id[idx].copy_(episode_id)

        self._head = (self._head + 1) % self.capacity
        self._total_steps += 1
        if self._size < self.capacity:
            self._size += 1

        if self.debug_checks:
            self.check_invariants()
        return idx

    def _chronological_indices(self):  # type: ignore[no-untyped-def]
        torch = self._torch
        if self._size == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        if self._size < self.capacity:
            return torch.arange(self._size, device=self.device, dtype=torch.int64)
        base = torch.arange(self._size, device=self.device, dtype=torch.int64)
        return (base + self._head) % self.capacity

    def check_invariants(self, *, eps: float = 1e-6) -> None:
        if self._size <= 1:
            return
        torch = self._torch
        idxes = self._chronological_indices()
        epi = self.episode_id.index_select(0, idxes)
        is_first = self.is_first.index_select(0, idxes)
        cont = self.continues.index_select(0, idxes)

        epi_next = epi[1:]
        epi_prev = epi[:-1]
        is_first_next = is_first[1:]

        ok_continuity = (epi_next == epi_prev) | is_first_next
        if not torch.all(ok_continuity):
            raise ValueError("episode_id continuity violated")

        ok_increment = (~is_first_next) | (epi_next == epi_prev + 1)
        if not torch.all(ok_increment):
            raise ValueError("episode_id increment violated at is_first")

        cont_valid = (cont - 0.0).abs().le(eps) | (cont - 1.0).abs().le(eps)
        if not torch.all(cont_valid):
            raise ValueError("continue values must be 0.0 or 1.0")

    def sample_sequences(  # type: ignore[no-untyped-def]
        self,
        *,
        batch: int,
        seq_len: int,
        gen,
        return_indices: bool = False,
    ):
        torch = self._torch
        _require_generator(gen)
        if batch < 1:
            raise ValueError("batch must be >= 1")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self._size < seq_len:
            raise ValueError("not enough samples to draw sequence")

        max_start = self._size - seq_len
        env_idx = torch.randint(
            0,
            self.num_envs,
            (batch,),
            device=self.device,
            generator=gen,
            dtype=torch.int64,
        )
        if max_start == 0:
            start_offsets = torch.zeros((batch,), device=self.device, dtype=torch.int64)
        else:
            start_offsets = torch.randint(
                0,
                max_start + 1,
                (batch,),
                device=self.device,
                generator=gen,
                dtype=torch.int64,
            )

        t_offsets = torch.arange(seq_len, device=self.device, dtype=torch.int64)
        time_offsets = start_offsets.view(1, -1) + t_offsets.view(-1, 1)
        if self._size < self.capacity:
            time_idx = time_offsets
        else:
            time_idx = (time_offsets + self._head) % self.capacity

        env_idx_grid = env_idx.view(1, -1).expand(seq_len, batch)

        obs = self.obs[time_idx, env_idx_grid]
        action = self.action[time_idx, env_idx_grid]
        reward = self.reward[time_idx, env_idx_grid]
        is_first = self.is_first[time_idx, env_idx_grid]
        continues = self.continues[time_idx, env_idx_grid]
        episode_id = self.episode_id[time_idx, env_idx_grid]

        out = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "is_first": is_first,
            "continue": continues,
            "episode_id": episode_id,
        }
        if return_indices:
            out["meta"] = {
                "env_idx": env_idx,
                "start_offset": start_offsets,
                "time_idx": time_idx,
            }
        return out
