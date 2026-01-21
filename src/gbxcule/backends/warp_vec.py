"""Warp-based backend (CPU-debug stub + GPU).

This module provides a stub implementation of the warp_vec backend for M0.
The stub returns predictable (wrong) CPU state to enable verify mode testing.
Real GPU implementation will be added in later stories.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gbxcule.backends.common import (
    ArraySpec,
    CpuFlags,
    CpuState,
    Device,
    NDArrayBool,
    NDArrayF32,
    NDArrayI32,
)


class WarpVecBackend:
    """Warp-based vectorized backend (stub for M0).

    This stub returns predictable (wrong) CPU state for verification testing.
    The real implementation will use Warp kernels for GPU-accelerated emulation.
    """

    name: str = "warp_vec"
    device: Device = "cpu"  # Stub runs on CPU

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
        """Initialize the warp_vec stub backend.

        Args:
            rom_path: Path to ROM file (ignored in stub).
            num_envs: Number of environments.
            frames_per_step: Frames per step (ignored in stub).
            release_after_frames: Frames before button release (ignored in stub).
            obs_dim: Observation dimension.
            base_seed: Optional base seed (ignored in stub).
        """
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        self._rom_path = rom_path
        self._step_count = 0
        self._initialized = False

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
        self._step_count = 0
        obs = np.zeros((self.num_envs, self._obs_dim), dtype=np.float32)
        return obs, {"seed": seed, "stub": True}

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
            raise RuntimeError("Backend not initialized - call reset() first")

        self._step_count += 1

        # Return zeros for all outputs
        obs = np.zeros((self.num_envs, self._obs_dim), dtype=np.float32)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=bool)
        trunc = np.zeros((self.num_envs,), dtype=bool)

        return obs, reward, done, trunc, {"step": self._step_count, "stub": True}

    def get_cpu_state(self, env_idx: int) -> CpuState:
        """Get CPU register state for verification.

        The stub returns predictable (wrong) state: all zeros except PC increments
        with step count. This ensures verification will detect a mismatch.

        Args:
            env_idx: Environment index.

        Returns:
            CpuState with stub values.

        Raises:
            ValueError: If env_idx is out of range.
        """
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")

        # Return predictable (wrong) values
        # PC increments with step count to make mismatches obvious
        flags: CpuFlags = {"z": 0, "n": 0, "h": 0, "c": 0}

        return CpuState(
            pc=self._step_count,  # Obviously wrong - should match actual emulation
            sp=0xFFFE,
            a=0,
            f=0,
            b=0,
            c=0,
            d=0,
            e=0,
            h=0,
            l=0,
            flags=flags,
            instr_count=None,
            cycle_count=None,
        )

    def close(self) -> None:
        """Close the backend and release resources."""
        self._initialized = False
