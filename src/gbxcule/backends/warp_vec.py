"""Warp-based backend (CPU bring-up).

M2 Workstream 2: ROM loading + ABI v0 memory model.
"""

from __future__ import annotations

from pathlib import Path
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
from gbxcule.core import abi
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
        self._rom_path = Path(rom_path)
        self._rom_bytes = self._rom_path.read_bytes()
        if len(self._rom_bytes) > abi.MEM_SIZE:
            raise ValueError(
                f"ROM too large for ABI v0 memory: {len(self._rom_bytes)} > "
                f"{abi.MEM_SIZE}"
            )
        self._rom_u8 = np.frombuffer(self._rom_bytes, dtype=np.uint8)
        self._rom_len = int(self._rom_u8.size)
        self._initialized = False
        self._frames_per_step = frames_per_step
        self._release_after_frames = release_after_frames
        self._base_seed = base_seed

        self._wp = get_warp()
        self._device = self._wp.get_device("cpu")
        warmup_warp_cpu()
        self._kernel = get_inc_counter_kernel()
        self._mem = None
        self._pc = None
        self._sp = None
        self._a = None
        self._f = None
        self._b = None
        self._c = None
        self._d = None
        self._e = None
        self._h = None
        self._l = None
        self._instr_count = None
        self._cycle_count = None

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
        self._mem = self._wp.zeros(
            self.num_envs * abi.MEM_SIZE, dtype=self._wp.uint8, device=self._device
        )
        self._pc = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint16, device=self._device
        )
        self._sp = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint16, device=self._device
        )
        self._a = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._f = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._b = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._c = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._d = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._e = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._h = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._l = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._instr_count = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._cycle_count = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )

        # Deterministic (wrong) reset state for bring-up.
        self._pc.numpy()[:] = 0x0100
        self._sp.numpy()[:] = 0xFFFE

        # Load ROM bytes into memory prefix for every environment.
        mem_np = self._mem.numpy()
        for env_idx in range(self.num_envs):
            start = env_idx * abi.MEM_SIZE
            mem_np[start : start + self._rom_len] = self._rom_u8

        obs = empty_obs(self.num_envs, self._obs_dim)
        return obs, {"seed": seed, "rom_len": self._rom_len}

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

        if (
            self._instr_count is None
            or self._cycle_count is None
            or self._pc is None
            or self._sp is None
            or self._a is None
            or self._f is None
        ):
            raise RuntimeError("Backend not initialized. Call reset() first.")

        self._wp.launch(
            self._kernel,
            dim=self.num_envs,
            inputs=[self._instr_count],
            device=self._device,
        )
        self._wp.synchronize()

        instr_np = self._instr_count.numpy()
        self._cycle_count.numpy()[:] = instr_np * 4
        self._pc.numpy()[:] = (0x100 + instr_np) & 0xFFFF
        self._sp.numpy()[:] = 0xFFFE
        self._a.numpy()[:] = instr_np & 0xFF
        self._f.numpy()[:] = ((instr_np & 0xF) << 4) & 0xF0

        obs = empty_obs(self.num_envs, self._obs_dim)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=np.bool_)
        trunc = np.zeros((self.num_envs,), dtype=np.bool_)

        return obs, reward, done, trunc, {}

    def get_cpu_state(self, env_idx: int) -> CpuState:
        """Get CPU register state for verification."""
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if (
            not self._initialized
            or self._pc is None
            or self._sp is None
            or self._a is None
            or self._f is None
            or self._b is None
            or self._c is None
            or self._d is None
            or self._e is None
            or self._h is None
            or self._l is None
            or self._instr_count is None
            or self._cycle_count is None
        ):
            raise RuntimeError("Backend not initialized. Call reset() first.")

        pc = int(self._pc.numpy()[env_idx])
        sp = int(self._sp.numpy()[env_idx])
        a = int(self._a.numpy()[env_idx])
        f = int(self._f.numpy()[env_idx])
        b = int(self._b.numpy()[env_idx])
        c = int(self._c.numpy()[env_idx])
        d = int(self._d.numpy()[env_idx])
        e = int(self._e.numpy()[env_idx])
        h = int(self._h.numpy()[env_idx])
        l = int(self._l.numpy()[env_idx])  # noqa: E741 - canonical register name
        instr_count = int(self._instr_count.numpy()[env_idx])
        cycle_count = int(self._cycle_count.numpy()[env_idx])
        flags = flags_from_f(f)

        return CpuState(
            pc=pc,
            sp=sp,
            a=a,
            f=f,
            b=b,
            c=c,
            d=d,
            e=e,
            h=h,
            l=l,
            flags=flags,
            instr_count=instr_count,
            cycle_count=cycle_count,
        )

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        """Debug helper: read memory bytes in [lo, hi)."""
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if self._mem is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if lo < 0 or hi < 0 or lo > abi.MEM_SIZE or hi > abi.MEM_SIZE or lo > hi:
            raise ValueError(f"Invalid lo/hi: lo={lo} hi={hi}")
        mem_np = self._mem.numpy()
        start = env_idx * abi.MEM_SIZE + lo
        end = env_idx * abi.MEM_SIZE + hi
        return mem_np[start:end].tobytes()

    def write_memory(self, env_idx: int, addr: int, data: bytes) -> None:
        """Debug helper: write bytes starting at addr."""
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if self._mem is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if addr < 0 or addr > abi.MEM_SIZE:
            raise ValueError(f"addr must be in [0, {abi.MEM_SIZE}], got {addr}")
        end = addr + len(data)
        if end > abi.MEM_SIZE:
            raise ValueError(
                f"write exceeds memory: addr={addr} len={len(data)} end={end} > "
                f"{abi.MEM_SIZE}"
            )
        mem_np = self._mem.numpy()
        start = env_idx * abi.MEM_SIZE + addr
        mem_np[start : start + len(data)] = np.frombuffer(data, dtype=np.uint8)

    def close(self) -> None:
        """Close the backend and release resources."""
        self._initialized = False
        self._mem = None
        self._pc = None
        self._sp = None
        self._a = None
        self._f = None
        self._b = None
        self._c = None
        self._d = None
        self._e = None
        self._h = None
        self._l = None
        self._instr_count = None
        self._cycle_count = None


class WarpVecBackend(WarpVecCpuBackend):
    """Alias for warp_vec backend name."""

    name: str = "warp_vec"
