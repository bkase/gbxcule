"""CUDA replay ring for Dreamer v3 (M6)."""

from __future__ import annotations

from typing import Any

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


def _require_generator(gen: Any) -> None:
    torch = _require_torch()
    if gen is None:
        raise ValueError("gen must be a torch.Generator")
    if not isinstance(gen, torch.Generator):
        raise ValueError("gen must be a torch.Generator")


def _device_matches(tensor_device, target_device) -> bool:  # type: ignore[no-untyped-def]
    return tensor_device == target_device or (
        tensor_device.type == target_device.type and tensor_device.type == "cuda"
    )


class ReplayRingCUDA:  # type: ignore[no-any-unimported]
    """Time-major CUDA replay ring with commit-aware sampling."""

    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        device: str = "cuda",
        obs_shape: tuple[int, int, int] = (1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
        obs_spec: dict[str, tuple[tuple[int, ...], Any]] | None = None,
        debug_checks: bool = False,
    ) -> None:
        torch = _require_torch()
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        self._torch = torch
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.debug_checks = bool(debug_checks)

        self._obs_is_dict = obs_spec is not None
        if obs_spec is not None:
            if not isinstance(obs_spec, dict) or not obs_spec:
                raise ValueError("obs_spec must be a non-empty dict")
            required = {"pixels", "senses"}
            if set(obs_spec.keys()) != required:
                raise ValueError("obs_spec must have pixels and senses keys")
            normalized: dict[str, tuple[tuple[int, ...], Any]] = {}
            for key, spec in obs_spec.items():
                if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                    raise ValueError("obs_spec entries must be (shape, dtype)")
                shape, dtype = spec
                if not isinstance(shape, (tuple, list)):
                    raise ValueError("obs_spec shape must be tuple")
                shape = tuple(int(x) for x in shape)
                if key == "pixels":
                    if dtype is not torch.uint8:
                        raise ValueError("pixels obs_spec dtype must be uint8")
                    if shape != (1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES):
                        raise ValueError("pixels obs_spec must match packed2 geometry")
                elif key == "senses":
                    if dtype is not torch.float32:
                        raise ValueError("senses obs_spec dtype must be float32")
                    if len(shape) != 1:
                        raise ValueError("senses obs_spec must be 1D")
                normalized[key] = (shape, dtype)
            self.obs_spec = normalized
            self.obs_shape = None
        else:
            if len(obs_shape) != 3:
                raise ValueError("obs_shape must be (C, H, W)")
            self.obs_shape = tuple(int(x) for x in obs_shape)
            c, h, w = self.obs_shape
            if c != 1:
                raise ValueError("obs_shape channels must be 1 for packed2")
            if h != DOWNSAMPLE_H or w != DOWNSAMPLE_W_BYTES:
                raise ValueError("obs_shape must match packed2 geometry")
            self.obs_spec = None

        if self._obs_is_dict:
            self.obs = {
                key: torch.empty(
                    (self.capacity, self.num_envs, *shape),
                    dtype=dtype,
                    device=self.device,
                )
                for key, (shape, dtype) in self.obs_spec.items()
            }
        else:
            c, h, w = self.obs_shape
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
        self.terminated = torch.empty(
            (self.capacity, self.num_envs),
            dtype=torch.bool,
            device=self.device,
        )
        self.truncated = torch.empty(
            (self.capacity, self.num_envs),
            dtype=torch.bool,
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
        if self._obs_is_dict:
            return {key: self.obs[key][step_idx] for key in self.obs_spec}
        return self.obs[step_idx]

    def assert_alignment(self, *, alignment: int = 256) -> None:
        if self.device.type != "cuda":
            return
        if self._obs_is_dict:
            ptr = int(self.obs["pixels"].data_ptr())
        else:
            ptr = int(self.obs.data_ptr())
        if ptr % alignment != 0:
            raise ValueError(f"obs data_ptr not {alignment}-byte aligned")

    def push_step(  # type: ignore[no-untyped-def]
        self,
        *,
        obs=None,
        action,
        reward,
        is_first,
        continue_,
        episode_id,
        terminated=None,
        truncated=None,
        validate_args: bool = True,
    ) -> int:
        torch = self._torch
        if validate_args:
            if obs is not None:
                if self._obs_is_dict:
                    if not isinstance(obs, dict):
                        raise ValueError("obs must be a dict when obs_spec is set")
                    if set(obs.keys()) != set(self.obs_spec.keys()):
                        raise ValueError("obs dict keys mismatch")
                    for key, (shape, dtype) in self.obs_spec.items():
                        value = obs[key]
                        if value.shape != (self.num_envs, *shape):
                            raise ValueError("obs shape mismatch")
                        if value.dtype is not dtype:
                            raise ValueError("obs dtype mismatch")
                        if not _device_matches(value.device, self.device):
                            raise ValueError("obs device mismatch")
                else:
                    if isinstance(obs, dict):
                        raise ValueError("obs must be a tensor when obs_spec is unset")
                    if obs.shape != (self.num_envs, *self.obs_shape):
                        raise ValueError("obs shape mismatch")
                    if obs.dtype is not torch.uint8:
                        raise ValueError("obs must be uint8")
                    if not _device_matches(obs.device, self.device):
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
            if not _device_matches(action.device, self.device):
                raise ValueError("action device mismatch")
            if not _device_matches(reward.device, self.device):
                raise ValueError("reward device mismatch")
            if not _device_matches(is_first.device, self.device):
                raise ValueError("is_first device mismatch")
            if not _device_matches(continue_.device, self.device):
                raise ValueError("continue device mismatch")
            if not _device_matches(episode_id.device, self.device):
                raise ValueError("episode_id device mismatch")
            if terminated is not None:
                if terminated.shape != (self.num_envs,):
                    raise ValueError("terminated shape mismatch")
                if terminated.dtype is not torch.bool:
                    raise ValueError("terminated must be bool")
                if not _device_matches(terminated.device, self.device):
                    raise ValueError("terminated device mismatch")
            if truncated is not None:
                if truncated.shape != (self.num_envs,):
                    raise ValueError("truncated shape mismatch")
                if truncated.dtype is not torch.bool:
                    raise ValueError("truncated must be bool")
                if not _device_matches(truncated.device, self.device):
                    raise ValueError("truncated device mismatch")

        idx = self._head
        if obs is not None:
            if self._obs_is_dict:
                for key in self.obs_spec:
                    self.obs[key][idx].copy_(obs[key])
            else:
                self.obs[idx].copy_(obs)
        self.action[idx].copy_(action)
        self.reward[idx].copy_(reward)
        self.is_first[idx].copy_(is_first)
        self.continues[idx].copy_(continue_)
        self.episode_id[idx].copy_(episode_id)
        if terminated is None:
            self.terminated[idx].zero_()
        else:
            self.terminated[idx].copy_(terminated)
        if truncated is None:
            self.truncated[idx].zero_()
        else:
            self.truncated[idx].copy_(truncated)

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
        committed_t: int | None = None,
        safety_margin: int = 0,
        exclude_head: bool = True,
        return_indices: bool = False,
    ):
        torch = self._torch
        _require_generator(gen)
        if batch < 1:
            raise ValueError("batch must be >= 1")
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if safety_margin < 0:
            raise ValueError("safety_margin must be >= 0")
        if self._size < seq_len:
            raise ValueError("not enough samples to draw sequence")

        latest = self._total_steps - 1
        if latest < 0:
            raise ValueError("replay is empty")
        if committed_t is None:
            committed_t = latest
        if committed_t > latest:
            committed_t = latest

        earliest = self._total_steps - self._size
        if exclude_head and self._size == self.capacity:
            earliest += 1

        safe_max = committed_t - safety_margin
        min_start = earliest
        max_start = safe_max - seq_len + 1
        if max_start < min_start:
            raise ValueError("not enough committed samples to draw sequence")

        env_idx = torch.randint(
            0,
            self.num_envs,
            (batch,),
            device=self.device,
            generator=gen,
            dtype=torch.int64,
        )
        if max_start == min_start:
            start_offsets = torch.full(
                (batch,),
                min_start,
                device=self.device,
                dtype=torch.int64,
            )
        else:
            start_offsets = torch.randint(
                min_start,
                max_start + 1,
                (batch,),
                device=self.device,
                generator=gen,
                dtype=torch.int64,
            )

        t_offsets = torch.arange(seq_len, device=self.device, dtype=torch.int64)
        time_offsets = start_offsets.view(1, -1) + t_offsets.view(-1, 1)
        time_idx = time_offsets % self.capacity
        env_idx_grid = env_idx.view(1, -1).expand(seq_len, batch)

        if self._obs_is_dict:
            obs = {
                key: self.obs[key][time_idx, env_idx_grid]
                for key in self.obs_spec
            }
        else:
            obs = self.obs[time_idx, env_idx_grid]
        action = self.action[time_idx, env_idx_grid]
        reward = self.reward[time_idx, env_idx_grid]
        is_first = self.is_first[time_idx, env_idx_grid]
        continues = self.continues[time_idx, env_idx_grid]
        episode_id = self.episode_id[time_idx, env_idx_grid]
        terminated = self.terminated[time_idx, env_idx_grid]
        truncated = self.truncated[time_idx, env_idx_grid]

        out = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "is_first": is_first,
            "continue": continues,
            "episode_id": episode_id,
            "terminated": terminated,
            "truncated": truncated,
        }
        if return_indices:
            out["meta"] = {
                "env_idx": env_idx,
                "start_offset": start_offsets,
                "time_idx": time_idx,
            }
        return out
