"""Warp-based backend (CPU bring-up)."""

from __future__ import annotations

from typing import Any

import numpy as np

from gbxcule.backends.common import (
    NUM_ACTIONS,
    ArraySpec,
    CpuState,
    Device,
    NDArrayBool,
    NDArrayF32,
    NDArrayI32,
    as_i32_actions,
    empty_obs,
    flags_from_f,
)
from gbxcule.kernels.cpu_step import (
    get_inc_counter_kernel,
    get_warp,
    warmup_warp_cpu,
)


class WarpVecCpuBackend:
    """Warp-based vectorized backend (CPU bring-up)."""

    name: str = "warp_vec_cpu"
    device: Device = "cpu"

    def __init__(
        self,
        rom_path: str,
        *,
        num_envs: int = 1,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        obs_dim: int = 32,
        base_seed: int | None = None,
    ) -> None:
        """Initialize the warp_vec CPU backend."""
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        self._rom_path = rom_path
        self._initialized = False
        self._frames_per_step = frames_per_step
        self._release_after_frames = release_after_frames
        self._base_seed = base_seed

        self._wp = get_warp()
        self._device = self._wp.get_device("cpu")
        warmup_warp_cpu()
        self._kernel = get_inc_counter_kernel()
        self._counter = None

        self.action_spec = ArraySpec(
            shape=(num_envs,),
            dtype="int32",
            meaning="Action index per environment",
        )
        self.obs_spec = ArraySpec(
            shape=(num_envs, obs_dim),
            dtype="float32",
            meaning="Observation vector per environment",
        )

    def reset(self, seed: int | None = None) -> tuple[NDArrayF32, dict[str, Any]]:
        """Reset all environments.

        Args:
            seed: Optional seed (ignored in stub).

        Returns:
            Tuple of (observations, info).
        """
        self._initialized = True
        self._counter = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        obs = empty_obs(self.num_envs, self._obs_dim)
        return obs, {"seed": seed}

    def step(
        self, actions: NDArrayI32
    ) -> tuple[NDArrayF32, NDArrayF32, NDArrayBool, NDArrayBool, dict[str, Any]]:
        """Step all environments forward.

        Args:
            actions: Actions array, shape (num_envs,).

        Returns:
            Tuple of (obs, reward, done, trunc, info).
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        actions = as_i32_actions(actions, self.num_envs)
        invalid_mask = (actions < 0) | (actions >= NUM_ACTIONS)
        if np.any(invalid_mask):
            bad = int(actions[invalid_mask][0])
            raise ValueError(f"Action {bad} out of range [0, {NUM_ACTIONS})")

        if self._counter is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        self._wp.launch(
            self._kernel,
            dim=self.num_envs,
            inputs=[self._counter],
            device=self._device,
        )
        self._wp.synchronize()

        obs = empty_obs(self.num_envs, self._obs_dim)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=np.bool_)
        trunc = np.zeros((self.num_envs,), dtype=np.bool_)

        return obs, reward, done, trunc, {}

    def get_cpu_state(self, env_idx: int) -> CpuState:
        """Get CPU register state for verification."""
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if not self._initialized or self._counter is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        counter = int(self._counter.numpy()[env_idx])
        pc = (0x100 + counter) & 0xFFFF
        sp = 0xFFFE
        a = counter & 0xFF
        f = ((counter & 0xF) << 4) & 0xF0
        flags = flags_from_f(f)

        return CpuState(
            pc=pc,
            sp=sp,
            a=a,
            f=f,
            b=0,
            c=0,
            d=0,
            e=0,
            h=0,
            l=0,
            flags=flags,
            instr_count=counter,
            cycle_count=counter * 4,
        )

    def close(self) -> None:
        """Close the backend and release resources."""
        self._initialized = False
        self._counter = None


class WarpVecBackend(WarpVecCpuBackend):
    """Alias for warp_vec backend name."""

    name: str = "warp_vec"
