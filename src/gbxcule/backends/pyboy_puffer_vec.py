"""PufferLib-vectorized PyBoy backend (CPU baseline)."""

from __future__ import annotations

import os
from typing import Any

from gbxcule.backends.common import (
    ArraySpec,
    Device,
    NDArrayBool,
    NDArrayF32,
    NDArrayI32,
    action_codec_spec,
    as_i32_actions,
    resolve_action_codec,
)
from gbxcule.backends.pyboy_gym_env import PyBoyGymEnv
from gbxcule.core.action_schedule import validate_schedule


def _require_pufferlib() -> tuple[Any, Any, Any]:
    try:
        import pufferlib
        from pufferlib import vector as puffer_vector
        from pufferlib.emulation import GymnasiumPufferEnv
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pufferlib is required for pyboy_puffer_vec. "
            "Install with the puffer dependency group."
        ) from exc
    return pufferlib, puffer_vector, GymnasiumPufferEnv


def _resolve_num_workers(num_envs: int, requested: int | None) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError(f"num_workers must be >= 1, got {requested}")
        if num_envs % requested != 0:
            raise ValueError(
                f"num_workers ({requested}) must divide num_envs ({num_envs})"
            )
        return requested

    max_workers = min(num_envs, os.cpu_count() or 1)
    for workers in range(max_workers, 0, -1):
        if num_envs % workers == 0:
            return workers
    return 1


class PyBoyPufferVecBackend:
    """PufferLib vectorized backend for PyBoy micro-ROMs."""

    name: str = "pyboy_puffer_vec"
    device: Device = "cpu"

    def __init__(
        self,
        rom_path: str,
        *,
        num_envs: int = 1,
        num_workers: int | None = None,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        obs_dim: int = 32,
        action_codec: str = "pokemonred_puffer_v1",
        vec_backend: str = "puffer_mp_sync",
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        validate_schedule(frames_per_step, release_after_frames)
        if frames_per_step < 1:
            raise ValueError("frames_per_step must be >= 1")

        self.num_envs = num_envs
        self._frames_per_step = frames_per_step
        self._release_after_frames = release_after_frames
        self._obs_dim = obs_dim
        self._action_codec = resolve_action_codec(action_codec)
        self.action_codec = action_codec_spec(action_codec)
        self.num_actions = self._action_codec.num_actions

        self.action_spec = ArraySpec(
            shape=(self.num_envs,),
            dtype="int32",
            meaning=(
                f"action index [0, {self.num_actions}) "
                f"({self.action_codec.name}@{self.action_codec.version})"
            ),
        )
        self.obs_spec = ArraySpec(
            shape=(self.num_envs, obs_dim),
            dtype="float32",
            meaning="normalized register features",
        )

        _, puffer_vector, GymnasiumPufferEnv = _require_pufferlib()

        vec_backend_normalized = vec_backend.lower()
        if vec_backend_normalized in {"puffer_serial", "serial"}:
            backend_cls = puffer_vector.Serial
            self.vec_backend = "puffer_serial"
            effective_num_workers = None
        elif vec_backend_normalized in {
            "puffer_mp_sync",
            "mp_sync",
            "multiprocessing",
        }:
            backend_cls = puffer_vector.Multiprocessing
            self.vec_backend = "puffer_mp_sync"
            effective_num_workers = _resolve_num_workers(num_envs, num_workers)
        else:
            raise ValueError(f"Unknown vec_backend: {vec_backend}")

        self.num_workers = effective_num_workers

        def env_creator() -> Any:
            return GymnasiumPufferEnv(
                env=PyBoyGymEnv(
                    rom_path,
                    frames_per_step=frames_per_step,
                    release_after_frames=release_after_frames,
                    obs_dim=obs_dim,
                    action_codec=action_codec,
                )
            )

        kwargs: dict[str, Any] = {
            "backend": backend_cls,
            "num_envs": num_envs,
            "batch_size": num_envs,
            "zero_copy": True,
        }
        if backend_cls is puffer_vector.Multiprocessing:
            kwargs["num_workers"] = effective_num_workers

        self._vecenv = puffer_vector.make(env_creator, **kwargs)

    def reset(self, seed: int | None = None) -> tuple[NDArrayF32, dict[str, Any]]:
        if seed is None:
            seed = 0
        obs, info = self._vecenv.reset(seed=seed)
        return obs, info if isinstance(info, dict) else {}

    def step(
        self, actions: NDArrayI32
    ) -> tuple[NDArrayF32, NDArrayF32, NDArrayBool, NDArrayBool, dict[str, Any]]:
        actions = as_i32_actions(actions, self.num_envs)
        obs, reward, done, trunc, info = self._vecenv.step(actions)
        return obs, reward, done, trunc, info if isinstance(info, dict) else {}

    def get_cpu_state(self, env_idx: int) -> dict[str, Any]:
        raise NotImplementedError("pyboy_puffer_vec does not expose per-env CPU state")

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        raise NotImplementedError("pyboy_puffer_vec does not expose per-env memory")

    def close(self) -> None:
        self._vecenv.close()
