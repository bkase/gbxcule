"""Warp-based backends (CPU + CUDA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gbxcule.backends.common import (
    DEFAULT_ACTION_CODEC_ID,
    ArraySpec,
    CpuState,
    Device,
    NDArrayBool,
    NDArrayF32,
    NDArrayI32,
    action_codec_spec,
    as_i32_actions,
    empty_obs,
    flags_from_f,
    resolve_action_codec,
)
from gbxcule.core.abi import MEM_SIZE
from gbxcule.kernels.cpu_step import (
    get_cpu_step_kernel,
    get_warp,
    warmup_warp_cpu,
    warmup_warp_cuda,
)

BOOTROM_PATH = Path("bench/roms/bootrom_fast_dmg.bin")


class WarpVecBaseBackend:
    """Warp-based vectorized backend (shared base)."""

    def __init__(
        self,
        rom_path: str,
        *,
        num_envs: int = 1,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        obs_dim: int = 32,
        base_seed: int | None = None,
        device: Device,
        device_name: str,
    ) -> None:
        """Initialize the warp_vec backend."""
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        self._rom_path = rom_path
        self._initialized = False
        self._frames_per_step = frames_per_step
        self._release_after_frames = release_after_frames
        self._base_seed = base_seed
        self.device = device
        self._device_name = device_name
        self._sync_after_step = True
        self._action_codec = resolve_action_codec(DEFAULT_ACTION_CODEC_ID)
        self.action_codec = action_codec_spec(DEFAULT_ACTION_CODEC_ID)
        self.num_actions = self._action_codec.num_actions

        self._wp = get_warp()
        self._device = self._wp.get_device(self._device_name)
        self._warmup()
        self._kernel = get_cpu_step_kernel()

        self._mem = None
        self._pc = None
        self._sp = None
        self._a = None
        self._b = None
        self._c = None
        self._d = None
        self._e = None
        self._h = None
        self._l = None
        self._f = None
        self._instr_count = None
        self._cycle_count = None
        self._cycle_in_frame = None

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

    def _warmup(self) -> None:
        if self.device == "cpu":
            warmup_warp_cpu()
        else:
            warmup_warp_cuda(self._device_name)

    def _synchronize(self) -> None:
        if self._sync_after_step:
            self._wp.synchronize()

    def _load_rom_bytes(self) -> bytes:
        rom_path = Path(self._rom_path)
        if not rom_path.exists():
            raise FileNotFoundError(f"ROM file not found: {rom_path}")
        rom_bytes = rom_path.read_bytes()
        if len(rom_bytes) > MEM_SIZE:
            raise ValueError(f"ROM too large: {len(rom_bytes)} bytes")
        return rom_bytes

    def _load_bootrom_bytes(self) -> bytes:
        if not BOOTROM_PATH.exists():
            raise FileNotFoundError(
                "Boot ROM not found: "
                f"{BOOTROM_PATH}. Expected repo-local fast boot ROM."
            )
        bootrom = BOOTROM_PATH.read_bytes()
        if len(bootrom) != 0x100:
            raise ValueError(f"Boot ROM must be 256 bytes, got {len(bootrom)}")
        return bootrom

    def reset(self, seed: int | None = None) -> tuple[NDArrayF32, dict[str, Any]]:
        """Reset all environments.

        Args:
            seed: Optional seed (ignored in CPU backend).

        Returns:
            Tuple of (observations, info).
        """
        rom_bytes = self._load_rom_bytes()
        bootrom_bytes = self._load_bootrom_bytes()

        mem_np = np.zeros(self.num_envs * MEM_SIZE, dtype=np.uint8)
        for env_idx in range(self.num_envs):
            base = env_idx * MEM_SIZE
            mem_np[base : base + len(rom_bytes)] = np.frombuffer(
                rom_bytes, dtype=np.uint8
            )
            mem_np[base : base + len(bootrom_bytes)] = np.frombuffer(
                bootrom_bytes, dtype=np.uint8
            )

        self._mem = self._wp.array(mem_np, dtype=self._wp.uint8, device=self._device)
        self._pc = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._sp = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._a = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._b = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._c = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._d = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._e = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._h = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._l = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._f = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._instr_count = self._wp.zeros(
            self.num_envs, dtype=self._wp.int64, device=self._device
        )
        self._cycle_count = self._wp.zeros(
            self.num_envs, dtype=self._wp.int64, device=self._device
        )
        self._cycle_in_frame = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )

        self._initialized = True
        obs = empty_obs(self.num_envs, self._obs_dim)
        return obs, {"seed": seed}

    def step(
        self, actions: NDArrayI32
    ) -> tuple[NDArrayF32, NDArrayF32, NDArrayBool, NDArrayBool, dict[str, Any]]:
        """Step all environments forward."""
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        actions = as_i32_actions(actions, self.num_envs)
        invalid_mask = (actions < 0) | (actions >= self.num_actions)
        if np.any(invalid_mask):
            bad = int(actions[invalid_mask][0])
            raise ValueError(f"Action {bad} out of range [0, {self.num_actions})")

        if self._mem is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        self._wp.launch(
            self._kernel,
            dim=self.num_envs,
            inputs=[
                self._mem,
                self._pc,
                self._sp,
                self._a,
                self._b,
                self._c,
                self._d,
                self._e,
                self._h,
                self._l,
                self._f,
                self._instr_count,
                self._cycle_count,
                self._cycle_in_frame,
                int(self._frames_per_step),
            ],
            device=self._device,
        )
        self._synchronize()

        obs = empty_obs(self.num_envs, self._obs_dim)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=np.bool_)
        trunc = np.zeros((self.num_envs,), dtype=np.bool_)

        return obs, reward, done, trunc, {}

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        """Read memory slice [lo, hi) for a given env."""
        if self._mem is None or not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if lo < 0 or hi < 0 or lo > MEM_SIZE or hi > MEM_SIZE or lo > hi:
            raise ValueError(f"Invalid memory range: lo={lo} hi={hi}")
        base = env_idx * MEM_SIZE
        mem_np = self._mem.numpy()
        return mem_np[base + lo : base + hi].tobytes()

    def write_memory(self, env_idx: int, addr: int, data: bytes) -> None:
        """Write bytes into memory starting at addr for a given env."""
        if self._mem is None or not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if addr < 0 or addr + len(data) > MEM_SIZE:
            raise ValueError(f"Write out of range: addr={addr} len={len(data)}")
        base = env_idx * MEM_SIZE
        mem_np = self._mem.numpy()
        mem_np[base + addr : base + addr + len(data)] = np.frombuffer(
            data, dtype=np.uint8
        )

    def get_cpu_state(self, env_idx: int) -> CpuState:
        """Get CPU register state for verification."""
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if not self._initialized or self._pc is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if self.device == "cuda":
            self._wp.synchronize()

        pc = int(self._pc.numpy()[env_idx]) & 0xFFFF
        sp = int(self._sp.numpy()[env_idx]) & 0xFFFF
        a = int(self._a.numpy()[env_idx]) & 0xFF
        b = int(self._b.numpy()[env_idx]) & 0xFF
        c = int(self._c.numpy()[env_idx]) & 0xFF
        d = int(self._d.numpy()[env_idx]) & 0xFF
        e = int(self._e.numpy()[env_idx]) & 0xFF
        h = int(self._h.numpy()[env_idx]) & 0xFF
        l_reg = int(self._l.numpy()[env_idx]) & 0xFF
        f = int(self._f.numpy()[env_idx]) & 0xF0
        flags = flags_from_f(f)
        instr_count = int(self._instr_count.numpy()[env_idx])
        cycle_count = int(self._cycle_count.numpy()[env_idx])

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
            l=l_reg,
            flags=flags,
            instr_count=instr_count,
            cycle_count=cycle_count,
        )

    def close(self) -> None:
        """Close the backend and release resources."""
        self._initialized = False
        self._mem = None
        self._pc = None
        self._sp = None
        self._a = None
        self._b = None
        self._c = None
        self._d = None
        self._e = None
        self._h = None
        self._l = None
        self._f = None
        self._instr_count = None
        self._cycle_count = None
        self._cycle_in_frame = None


class WarpVecCpuBackend(WarpVecBaseBackend):
    """Warp-based vectorized backend (CPU)."""

    name: str = "warp_vec_cpu"

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
        super().__init__(
            rom_path,
            num_envs=num_envs,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=obs_dim,
            base_seed=base_seed,
            device="cpu",
            device_name="cpu",
        )


class WarpVecCudaBackend(WarpVecBaseBackend):
    """Warp-based vectorized backend (CUDA)."""

    name: str = "warp_vec_cuda"

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
        super().__init__(
            rom_path,
            num_envs=num_envs,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            obs_dim=obs_dim,
            base_seed=base_seed,
            device="cuda",
            device_name="cuda:0",
        )
        self._sync_after_step = False
        self._mem_readback = None
        self._mem_readback_capacity = 0

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        if self._mem is None or not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if lo < 0 or hi < 0 or lo > MEM_SIZE or hi > MEM_SIZE or lo > hi:
            raise ValueError(f"Invalid memory range: lo={lo} hi={hi}")
        count = hi - lo
        if count == 0:
            return b""
        if self._mem_readback is None or self._mem_readback_capacity < count:
            self._mem_readback = self._wp.empty(
                count, dtype=self._wp.uint8, device="cpu", pinned=True
            )
            self._mem_readback_capacity = count
        base = env_idx * MEM_SIZE
        self._wp.copy(
            self._mem_readback,
            self._mem,
            dest_offset=0,
            src_offset=base + lo,
            count=count,
        )
        self._wp.synchronize()
        return self._mem_readback.numpy()[:count].tobytes()

    def write_memory(self, env_idx: int, addr: int, data: bytes) -> None:
        raise NotImplementedError(
            "CUDA write_memory is not implemented yet (debug-only for CPU)."
        )

    def close(self) -> None:
        super().close()
        self._mem_readback = None
        self._mem_readback_capacity = 0


class WarpVecBackend(WarpVecCpuBackend):
    """Alias for warp_vec backend name."""

    name: str = "warp_vec"
