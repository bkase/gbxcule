"""Warp-based backends (CPU + CUDA)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gbxcule.core.state_io import PyBoyState

import os

import numpy as np

from gbxcule.backends.common import (
    DEFAULT_ACTION_CODEC_ID,
    ArraySpec,
    CpuState,
    Device,
    NDArrayBool,
    NDArrayF32,
    NDArrayI32,
    Stage,
    action_codec_spec,
    as_i32_actions,
    empty_obs,
    flags_from_f,
    resolve_action_codec,
)
from gbxcule.core.abi import (
    DOWNSAMPLE_H,
    DOWNSAMPLE_W,
    MEM_SIZE,
    OBS_DIM_DEFAULT,
    SCREEN_H,
    SCREEN_W,
    SERIAL_MAX,
)
from gbxcule.core.action_codec import action_codec_kernel_id
from gbxcule.core.cartridge import (
    BOOTROM_SIZE,
    CART_STATE_BANK_MODE,
    CART_STATE_BOOTROM_ENABLED,
    CART_STATE_MBC_KIND,
    CART_STATE_RAM_BANK,
    CART_STATE_RAM_ENABLE,
    CART_STATE_ROM_BANK_HI,
    CART_STATE_ROM_BANK_LO,
    CART_STATE_RTC_DAYS_HIGH,
    CART_STATE_RTC_DAYS_LOW,
    CART_STATE_RTC_HOURS,
    CART_STATE_RTC_LAST_CYCLE,
    CART_STATE_RTC_LATCH,
    CART_STATE_RTC_LATCHED_DAYS_HIGH,
    CART_STATE_RTC_LATCHED_DAYS_LOW,
    CART_STATE_RTC_LATCHED_HOURS,
    CART_STATE_RTC_LATCHED_MINUTES,
    CART_STATE_RTC_LATCHED_SECONDS,
    CART_STATE_RTC_MINUTES,
    CART_STATE_RTC_SECONDS,
    CART_STATE_RTC_SELECT,
    CART_STATE_STRIDE,
    MBC_KIND_MBC1,
    MBC_KIND_MBC3,
    MBC_KIND_ROM_ONLY,
    CartType,
    parse_cartridge_header,
)
from gbxcule.kernels.cpu_step import (
    get_cpu_step_kernel,
    get_warp,
    warmup_warp_cpu,
    warmup_warp_cuda,
)
from gbxcule.kernels.ppu_render_downsampled import get_ppu_render_downsampled_kernel
from gbxcule.kernels.ppu_step import get_ppu_render_bg_kernel

BOOTROM_PATH = Path("bench/roms/bootrom_fast_dmg.bin")
ROM_LIMIT = 0x8000


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
        action_codec: str = DEFAULT_ACTION_CODEC_ID,
        stage: Stage = "emulate_only",
        render_bg: bool = False,
        render_pixels: bool = False,
    ) -> None:
        """Initialize the warp_vec backend."""
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        if self._obs_dim != OBS_DIM_DEFAULT:
            raise ValueError(
                f"obs_dim {self._obs_dim} unsupported; expected {OBS_DIM_DEFAULT}"
            )
        self._rom_path = rom_path
        self._initialized = False
        self._frames_per_step = frames_per_step
        self._release_after_frames = release_after_frames
        self._base_seed = base_seed
        self.device = device
        self._device_name = device_name
        self._sync_after_step = True
        self._stage: Stage = stage
        self._render_bg = render_bg
        self._render_pixels = render_pixels
        self._ppu_render_kernel = None
        self._ppu_render_downsampled_kernel = None
        self._action_codec = resolve_action_codec(action_codec)
        self.action_codec = action_codec_spec(action_codec)
        self.num_actions = self._action_codec.num_actions
        self._action_codec_kernel_id = action_codec_kernel_id(action_codec)

        self._wp = get_warp()
        self._device = self._wp.get_device(self._device_name)
        self._warmup()
        self._kernel = get_cpu_step_kernel(stage=self._stage, obs_dim=self._obs_dim)

        self._mem = None
        self._rom = None
        self._bootrom = None
        self._cart_ram = None
        self._cart_state = None
        self._rom_bytes = None
        self._bootrom_bytes = None
        self._rom_bank_count = 0
        self._rom_bank_mask = -1
        self._ram_bank_count = 0
        self._ram_byte_length = 0
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
        self._trap_flag = None
        self._trap_pc = None
        self._trap_opcode = None
        self._trap_kind = None
        self._actions = None
        self._joyp_select = None
        self._serial_buf = None
        self._serial_len = None
        self._serial_overflow = None
        self._ime = None
        self._ime_delay = None
        self._halted = None
        self._div_counter = None
        self._timer_prev_in = None
        self._tima_reload_pending = None
        self._tima_reload_delay = None
        self._ppu_scanline_cycle = None
        self._ppu_ly = None
        self._ppu_window_line = None
        self._ppu_stat_prev = None
        self._bg_lcdc_latch_env0 = None
        self._bg_scx_latch_env0 = None
        self._bg_scy_latch_env0 = None
        self._bg_bgp_latch_env0 = None
        self._win_wx_latch_env0 = None
        self._win_wy_latch_env0 = None
        self._win_line_latch_env0 = None
        self._obj_obp0_latch_env0 = None
        self._obj_obp1_latch_env0 = None
        self._frame_bg_shade_env0 = None
        self._pix = None
        self._pix_torch = None
        self._reward = None
        self._obs = None

        self.action_spec = ArraySpec(
            shape=(num_envs,),
            dtype="int32",
            meaning=f"action index [0, {self.num_actions})",
        )
        self.obs_spec = ArraySpec(
            shape=(num_envs, obs_dim),
            dtype="float32",
            meaning="Observation vector per environment",
        )

    def _warmup(self) -> None:
        if self.device == "cpu":
            warmup_warp_cpu(stage=self._stage, obs_dim=self._obs_dim)
        else:
            warmup_warp_cuda(
                self._device_name,
                stage=self._stage,
                obs_dim=self._obs_dim,
            )

    def _synchronize(self) -> None:
        if self._sync_after_step:
            self._wp.synchronize()

    def _load_rom_bytes(self) -> bytes:
        rom_path = Path(self._rom_path)
        if not rom_path.exists():
            raise FileNotFoundError(f"ROM file not found: {rom_path}")
        rom_bytes = rom_path.read_bytes()
        # Banked cartridges can exceed 64KB; header parsing in reset() validates size.
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
        spec = parse_cartridge_header(rom_bytes)
        if len(rom_bytes) < spec.rom_byte_length:
            raise ValueError(
                f"ROM smaller than header advertises: "
                f"{len(rom_bytes)} < {spec.rom_byte_length}"
            )
        if len(rom_bytes) > spec.rom_byte_length:
            rom_bytes = rom_bytes[: spec.rom_byte_length]

        self._rom_bytes = rom_bytes
        self._bootrom_bytes = bootrom_bytes
        self._rom_bank_count = spec.rom_bank_count
        self._rom_bank_mask = (
            spec.rom_bank_mask if spec.rom_bank_mask is not None else -1
        )
        self._ram_bank_count = spec.ram_bank_count
        self._ram_byte_length = spec.ram_byte_length

        rom_np = np.frombuffer(rom_bytes, dtype=np.uint8)
        self._rom = self._wp.array(rom_np, dtype=self._wp.uint8, device=self._device)
        bootrom_np = np.frombuffer(bootrom_bytes, dtype=np.uint8)
        self._bootrom = self._wp.array(
            bootrom_np, dtype=self._wp.uint8, device=self._device
        )

        ram_alloc_bytes = self._ram_byte_length
        if ram_alloc_bytes <= 0:
            ram_alloc_bytes = 1
        cart_ram_np = np.zeros(self.num_envs * ram_alloc_bytes, dtype=np.uint8)
        self._cart_ram = self._wp.array(
            cart_ram_np, dtype=self._wp.uint8, device=self._device
        )

        cart_state_np = np.zeros(self.num_envs * CART_STATE_STRIDE, dtype=np.int32)
        mbc_kind = MBC_KIND_ROM_ONLY
        if spec.cart_type in {
            CartType.MBC1,
            CartType.MBC1_RAM,
            CartType.MBC1_RAM_BATTERY,
        }:
            mbc_kind = MBC_KIND_MBC1
        elif spec.cart_type in {
            CartType.MBC3,
            CartType.MBC3_RAM,
            CartType.MBC3_RAM_BATTERY,
            CartType.MBC3_TIMER_BATTERY,
            CartType.MBC3_TIMER_RAM_BATTERY,
        }:
            mbc_kind = MBC_KIND_MBC3
        for env_idx in range(self.num_envs):
            base = env_idx * CART_STATE_STRIDE
            cart_state_np[base + CART_STATE_MBC_KIND] = mbc_kind
            cart_state_np[base + CART_STATE_RAM_ENABLE] = 0
            cart_state_np[base + CART_STATE_ROM_BANK_LO] = 1
            cart_state_np[base + CART_STATE_ROM_BANK_HI] = 0
            cart_state_np[base + CART_STATE_RAM_BANK] = 0
            cart_state_np[base + CART_STATE_BANK_MODE] = 0
            cart_state_np[base + CART_STATE_BOOTROM_ENABLED] = 1
            cart_state_np[base + CART_STATE_RTC_SELECT] = 0
            cart_state_np[base + CART_STATE_RTC_LATCH] = 0
            cart_state_np[base + CART_STATE_RTC_SECONDS] = 0
            cart_state_np[base + CART_STATE_RTC_MINUTES] = 0
            cart_state_np[base + CART_STATE_RTC_HOURS] = 0
            cart_state_np[base + CART_STATE_RTC_DAYS_LOW] = 0
            cart_state_np[base + CART_STATE_RTC_DAYS_HIGH] = 0
            cart_state_np[base + CART_STATE_RTC_LAST_CYCLE] = 0
            cart_state_np[base + CART_STATE_RTC_LATCHED_SECONDS] = 0
            cart_state_np[base + CART_STATE_RTC_LATCHED_MINUTES] = 0
            cart_state_np[base + CART_STATE_RTC_LATCHED_HOURS] = 0
            cart_state_np[base + CART_STATE_RTC_LATCHED_DAYS_LOW] = 0
            cart_state_np[base + CART_STATE_RTC_LATCHED_DAYS_HIGH] = 0
        self._cart_state = self._wp.array(
            cart_state_np, dtype=self._wp.int32, device=self._device
        )

        mem_np = np.zeros(self.num_envs * MEM_SIZE, dtype=np.uint8)
        for env_idx in range(self.num_envs):
            base = env_idx * MEM_SIZE
            mem_np[base : base + BOOTROM_SIZE] = bootrom_np

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
        self._trap_flag = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._trap_pc = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._trap_opcode = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._trap_kind = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._actions = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        joyp_init = np.full(self.num_envs, 0x30, dtype=np.uint8)
        self._joyp_select = self._wp.array(
            joyp_init, dtype=self._wp.uint8, device=self._device
        )
        self._serial_buf = self._wp.zeros(
            self.num_envs * SERIAL_MAX, dtype=self._wp.uint8, device=self._device
        )
        self._serial_len = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._serial_overflow = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._ime = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._ime_delay = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._halted = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._div_counter = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._timer_prev_in = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._tima_reload_pending = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._tima_reload_delay = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._ppu_scanline_cycle = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._ppu_ly = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._ppu_window_line = self._wp.zeros(
            self.num_envs, dtype=self._wp.int32, device=self._device
        )
        self._ppu_stat_prev = self._wp.zeros(
            self.num_envs, dtype=self._wp.uint8, device=self._device
        )
        self._bg_lcdc_latch_env0 = self._wp.zeros(
            SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        self._bg_scx_latch_env0 = self._wp.zeros(
            SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        self._bg_scy_latch_env0 = self._wp.zeros(
            SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        self._bg_bgp_latch_env0 = self._wp.zeros(
            SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        self._win_wx_latch_env0 = self._wp.zeros(
            SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        self._win_wy_latch_env0 = self._wp.zeros(
            SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        self._win_line_latch_env0 = self._wp.zeros(
            SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        self._obj_obp0_latch_env0 = self._wp.zeros(
            SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        self._obj_obp1_latch_env0 = self._wp.zeros(
            SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        self._frame_bg_shade_env0 = self._wp.zeros(
            SCREEN_W * SCREEN_H, dtype=self._wp.uint8, device=self._device
        )
        if self._render_pixels:
            self._pix = self._wp.zeros(
                self.num_envs * DOWNSAMPLE_W * DOWNSAMPLE_H,
                dtype=self._wp.uint8,
                device=self._device,
            )
            self._pix_torch = None
        else:
            self._pix = None
            self._pix_torch = None
        self._reward = self._wp.zeros(
            self.num_envs, dtype=self._wp.float32, device=self._device
        )
        self._obs = self._wp.zeros(
            self.num_envs * self._obs_dim, dtype=self._wp.float32, device=self._device
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

        if (
            self._mem is None
            or self._rom is None
            or self._bootrom is None
            or self._cart_ram is None
            or self._cart_state is None
            or self._actions is None
        ):
            raise RuntimeError("Backend not initialized. Call reset() first.")

        if hasattr(self._actions, "assign"):
            self._actions.assign(actions)
        else:
            try:
                host_actions = self._wp.array(
                    actions, dtype=self._wp.int32, device="cpu"
                )
                self._wp.copy(self._actions, host_actions)
            except Exception:
                self._actions = self._wp.array(
                    actions, dtype=self._wp.int32, device=self._device
                )

        self._launch_step(self._actions)

        obs = empty_obs(self.num_envs, self._obs_dim)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=np.bool_)
        trunc = np.zeros((self.num_envs,), dtype=np.bool_)

        return obs, reward, done, trunc, {}

    def _launch_step(self, actions: Any) -> None:
        if (
            self._mem is None
            or self._rom is None
            or self._bootrom is None
            or self._cart_ram is None
            or self._cart_state is None
        ):
            raise RuntimeError("Backend not initialized. Call reset() first.")
        self._wp.launch(
            self._kernel,
            dim=self.num_envs,
            inputs=[
                self._mem,
                self._rom,
                self._bootrom,
                self._cart_ram,
                self._cart_state,
                int(self._rom_bank_count),
                int(self._rom_bank_mask),
                int(self._ram_bank_count),
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
                self._trap_flag,
                self._trap_pc,
                self._trap_opcode,
                self._trap_kind,
                actions,
                self._joyp_select,
                self._serial_buf,
                self._serial_len,
                self._serial_overflow,
                self._ime,
                self._ime_delay,
                self._halted,
                self._div_counter,
                self._timer_prev_in,
                self._tima_reload_pending,
                self._tima_reload_delay,
                self._ppu_scanline_cycle,
                self._ppu_ly,
                self._ppu_window_line,
                self._ppu_stat_prev,
                self._bg_lcdc_latch_env0,
                self._bg_scx_latch_env0,
                self._bg_scy_latch_env0,
                self._bg_bgp_latch_env0,
                self._win_wx_latch_env0,
                self._win_wy_latch_env0,
                self._win_line_latch_env0,
                self._obj_obp0_latch_env0,
                self._obj_obp1_latch_env0,
                int(self._action_codec_kernel_id),
                self._reward,
                self._obs,
                int(self._frames_per_step),
                int(self._release_after_frames),
            ],
            device=self._device,
        )
        if self._render_bg:
            if self._ppu_render_kernel is None:
                self._ppu_render_kernel = get_ppu_render_bg_kernel()
            self._wp.launch(
                self._ppu_render_kernel,
                dim=SCREEN_W * SCREEN_H,
                inputs=[
                    self._mem,
                    self._bg_lcdc_latch_env0,
                    self._bg_scx_latch_env0,
                    self._bg_scy_latch_env0,
                    self._bg_bgp_latch_env0,
                    self._win_wx_latch_env0,
                    self._win_wy_latch_env0,
                    self._win_line_latch_env0,
                    self._obj_obp0_latch_env0,
                    self._obj_obp1_latch_env0,
                    self._frame_bg_shade_env0,
                ],
                device=self._device,
            )
        if self._render_pixels:
            if self._pix is None:
                raise RuntimeError("Pixel buffer not initialized. Call reset() first.")
            if self._ppu_render_downsampled_kernel is None:
                self._ppu_render_downsampled_kernel = (
                    get_ppu_render_downsampled_kernel()
                )
            self._wp.launch(
                self._ppu_render_downsampled_kernel,
                dim=self.num_envs * DOWNSAMPLE_W * DOWNSAMPLE_H,
                inputs=[
                    self._mem,
                    self._pix,
                ],
                device=self._device,
            )
        self._synchronize()

    def get_pc_snapshot(self) -> NDArrayI32:
        """Return a snapshot of PC values for all environments."""
        if self._pc is None or not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        self._wp.synchronize()
        return np.array(self._pc.numpy(), copy=True)

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
        if hi <= ROM_LIMIT:
            return self._read_rom_slice(lo, hi)
        if lo >= ROM_LIMIT:
            return mem_np[base + lo : base + hi].tobytes()
        rom_part = self._read_rom_slice(lo, ROM_LIMIT)
        mem_part = mem_np[base + ROM_LIMIT : base + hi].tobytes()
        return rom_part + mem_part

    def _read_rom_slice(self, lo: int, hi: int) -> bytes:
        if self._rom_bytes is None or self._bootrom_bytes is None:
            raise RuntimeError("ROM not initialized. Call reset() first.")
        if lo < 0 or hi < 0 or lo > ROM_LIMIT or hi > ROM_LIMIT or lo > hi:
            raise ValueError(f"Invalid ROM range: lo={lo} hi={hi}")
        out = bytearray(hi - lo)
        for idx, addr in enumerate(range(lo, hi)):
            if addr < BOOTROM_SIZE:
                out[idx] = self._bootrom_bytes[addr]
            else:
                out[idx] = (
                    self._rom_bytes[addr] if addr < len(self._rom_bytes) else 0xFF
                )
        return bytes(out)

    @staticmethod
    def _cart_state_to_dict(state: np.ndarray) -> dict[str, int]:
        if state.shape[0] < CART_STATE_STRIDE:
            raise ValueError(
                f"cart_state length {state.shape[0]} < {CART_STATE_STRIDE}"
            )
        return {
            "mbc_kind": int(state[CART_STATE_MBC_KIND]),
            "ram_enable": int(state[CART_STATE_RAM_ENABLE]),
            "rom_bank_lo": int(state[CART_STATE_ROM_BANK_LO]),
            "rom_bank_hi": int(state[CART_STATE_ROM_BANK_HI]),
            "ram_bank": int(state[CART_STATE_RAM_BANK]),
            "bank_mode": int(state[CART_STATE_BANK_MODE]),
            "bootrom_enabled": int(state[CART_STATE_BOOTROM_ENABLED]),
            "rtc_select": int(state[CART_STATE_RTC_SELECT]),
            "rtc_latch": int(state[CART_STATE_RTC_LATCH]),
            "rtc_seconds": int(state[CART_STATE_RTC_SECONDS]),
            "rtc_minutes": int(state[CART_STATE_RTC_MINUTES]),
            "rtc_hours": int(state[CART_STATE_RTC_HOURS]),
            "rtc_days_low": int(state[CART_STATE_RTC_DAYS_LOW]),
            "rtc_days_high": int(state[CART_STATE_RTC_DAYS_HIGH]),
            "rtc_last_cycle": int(state[CART_STATE_RTC_LAST_CYCLE]),
            "rtc_latched_seconds": int(state[CART_STATE_RTC_LATCHED_SECONDS]),
            "rtc_latched_minutes": int(state[CART_STATE_RTC_LATCHED_MINUTES]),
            "rtc_latched_hours": int(state[CART_STATE_RTC_LATCHED_HOURS]),
            "rtc_latched_days_low": int(state[CART_STATE_RTC_LATCHED_DAYS_LOW]),
            "rtc_latched_days_high": int(state[CART_STATE_RTC_LATCHED_DAYS_HIGH]),
        }

    def read_frame_bg_shade_env0(self) -> bytes:
        """Read env0 BG shade framebuffer (CPU backend only)."""
        if self._frame_bg_shade_env0 is None or not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        self._wp.synchronize()
        return self._frame_bg_shade_env0.numpy().tobytes()

    def pixels_wp(self):
        """Return the downsampled pixel buffer (Warp array)."""
        if self._pix is None or not self._initialized:
            raise RuntimeError("Pixel buffer not initialized. Call reset() first.")
        return self._pix

    def pixels_torch(self):
        """Return a torch view of the downsampled pixel buffer."""
        if self._pix is None or not self._initialized:
            raise RuntimeError("Pixel buffer not initialized. Call reset() first.")
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("Torch not available for pixels_torch().")
        if self._pix_torch is None:
            pix_t = self._wp.to_torch(self._pix)
            self._pix_torch = pix_t.view(self.num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W)
        return self._pix_torch

    def read_serial(self, env_idx: int) -> bytes:
        """Read captured serial bytes for a specific env (CPU only)."""
        if self._serial_buf is None or self._serial_len is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        lengths = self._serial_len.numpy()
        length = int(lengths[env_idx])
        if length <= 0:
            return b""
        length = min(length, SERIAL_MAX)
        base = env_idx * SERIAL_MAX
        buf_np = self._serial_buf.numpy()
        return buf_np[base : base + length].tobytes()

    def drain_serial(self, env_idx: int) -> bytes:
        """Read and clear captured serial bytes for a specific env."""
        data = self.read_serial(env_idx)
        if self._serial_len is None or self._serial_overflow is None:
            return data
        lengths = self._serial_len.numpy()
        overflow = self._serial_overflow.numpy()
        lengths[env_idx] = 0
        overflow[env_idx] = 0
        if hasattr(self._serial_len, "assign"):
            self._serial_len.assign(lengths)
        if hasattr(self._serial_overflow, "assign"):
            self._serial_overflow.assign(overflow)
        return data

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

        # All register arrays are guaranteed non-None after initialization
        assert self._sp is not None
        assert self._a is not None
        assert self._b is not None
        assert self._c is not None
        assert self._d is not None
        assert self._e is not None
        assert self._h is not None
        assert self._l is not None
        assert self._f is not None
        assert self._instr_count is not None
        assert self._cycle_count is not None
        assert self._trap_flag is not None
        assert self._trap_pc is not None
        assert self._trap_opcode is not None
        assert self._trap_kind is not None

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
        trap_flag = int(self._trap_flag.numpy()[env_idx])
        trap_pc = int(self._trap_pc.numpy()[env_idx]) & 0xFFFF
        trap_opcode = int(self._trap_opcode.numpy()[env_idx]) & 0xFF
        trap_kind = int(self._trap_kind.numpy()[env_idx])

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
            trap_flag=trap_flag,
            trap_pc=trap_pc,
            trap_opcode=trap_opcode,
            trap_kind=trap_kind,
        )

    def get_cart_state(self, env_idx: int) -> dict[str, int]:
        """Get cart/MBC state snapshot for a specific environment."""
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if not self._initialized or self._cart_state is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if self.device == "cuda":
            self._wp.synchronize()
        cart_state_np = self._cart_state.numpy()
        base = env_idx * CART_STATE_STRIDE
        state = np.array(cart_state_np[base : base + CART_STATE_STRIDE], copy=True)
        return self._cart_state_to_dict(state)

    def save_state(self, env_idx: int = 0) -> PyBoyState:
        """Save the current state of an environment as a PyBoyState.

        Args:
            env_idx: Environment index to save.

        Returns:
            PyBoyState object containing the complete emulator state.
        """
        from gbxcule.core.state_io import state_from_warp_backend

        return state_from_warp_backend(self, env_idx)

    def save_state_file(self, path: str, env_idx: int = 0) -> None:
        """Save the current state of an environment to a PyBoy v9 state file.

        Args:
            path: Path to write the .state file.
            env_idx: Environment index to save.
        """
        from gbxcule.core.state_io import save_pyboy_state, state_from_warp_backend

        state = state_from_warp_backend(self, env_idx)
        save_pyboy_state(state, path)

    def load_state(self, state: PyBoyState, env_idx: int = 0) -> None:
        """Load a PyBoyState into an environment.

        Args:
            state: The state to load.
            env_idx: Environment index to modify.
        """
        from gbxcule.core.state_io import apply_state_to_warp_backend

        apply_state_to_warp_backend(state, self, env_idx)

    def load_state_file(self, path: str, env_idx: int = 0) -> None:
        """Load a PyBoy v9 state file into an environment.

        Args:
            path: Path to the .state file.
            env_idx: Environment index to modify.
        """
        from gbxcule.core.state_io import apply_state_to_warp_backend, load_pyboy_state

        expected_cart_ram_size = 0
        if self._rom_bytes is not None:
            spec = parse_cartridge_header(self._rom_bytes)
            expected_cart_ram_size = int(spec.ram_byte_length)
        state = load_pyboy_state(path, expected_cart_ram_size=expected_cart_ram_size)
        apply_state_to_warp_backend(state, self, env_idx)

    def close(self) -> None:
        """Close the backend and release resources."""
        self._initialized = False
        self._mem = None
        self._rom = None
        self._bootrom = None
        self._cart_ram = None
        self._cart_state = None
        self._rom_bytes = None
        self._bootrom_bytes = None
        self._rom_bank_count = 0
        self._rom_bank_mask = -1
        self._ram_bank_count = 0
        self._ram_byte_length = 0
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
        self._trap_flag = None
        self._trap_pc = None
        self._trap_opcode = None
        self._trap_kind = None
        self._actions = None
        self._joyp_select = None
        self._serial_buf = None
        self._serial_len = None
        self._serial_overflow = None
        self._ime = None
        self._ime_delay = None
        self._halted = None
        self._div_counter = None
        self._timer_prev_in = None
        self._tima_reload_pending = None
        self._tima_reload_delay = None
        self._ppu_scanline_cycle = None
        self._ppu_ly = None
        self._ppu_window_line = None
        self._ppu_stat_prev = None
        self._bg_lcdc_latch_env0 = None
        self._bg_scx_latch_env0 = None
        self._bg_scy_latch_env0 = None
        self._bg_bgp_latch_env0 = None
        self._win_wx_latch_env0 = None
        self._win_wy_latch_env0 = None
        self._win_line_latch_env0 = None
        self._obj_obp0_latch_env0 = None
        self._obj_obp1_latch_env0 = None
        self._frame_bg_shade_env0 = None
        self._pix = None
        self._pix_torch = None
        self._ppu_render_downsampled_kernel = None
        self._reward = None
        self._obs = None


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
        action_codec: str = DEFAULT_ACTION_CODEC_ID,
        stage: Stage = "emulate_only",
        render_bg: bool = False,
        render_pixels: bool = False,
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
            action_codec=action_codec,
            stage=stage,
            render_bg=render_bg,
            render_pixels=render_pixels,
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
        action_codec: str = DEFAULT_ACTION_CODEC_ID,
        stage: Stage = "emulate_only",
        render_bg: bool = False,
        render_pixels: bool = False,
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
            action_codec=action_codec,
            stage=stage,
            render_bg=render_bg,
            render_pixels=render_pixels,
        )
        self._sync_after_step = False
        self._mem_readback = None
        self._mem_readback_capacity = 0
        self._frame_readback = None
        self._frame_readback_capacity = 0
        self._serial_readback = None
        self._serial_readback_capacity = 0
        self._cart_state_readback = None
        self._cart_state_readback_capacity = 0

    def step_torch(self, actions):  # type: ignore[no-untyped-def]
        """Step all envs using torch CUDA actions (no host staging).

        This is a CUDA-only fast path intended for RL wrappers.
        """
        if self.device != "cuda":
            raise RuntimeError("step_torch() is only supported on CUDA backends.")
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        import importlib

        try:
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise RuntimeError("Torch not available for step_torch().") from exc

        tensor_type = getattr(torch, "Tensor", None)
        if tensor_type is None or not isinstance(actions, tensor_type):
            raise TypeError("actions must be a torch.Tensor")
        if actions.device.type != "cuda":
            raise ValueError("actions must be a CUDA tensor")
        if actions.ndim != 1 or int(actions.shape[0]) != self.num_envs:
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if actions.dtype is not torch.int32:
            raise ValueError("actions must have dtype torch.int32 (no implicit cast)")

        if os.environ.get("GBXCULE_VALIDATE_ACTIONS") == "1":
            invalid = (actions < 0) | (actions >= self.num_actions)
            if torch.any(invalid).item():
                raise ValueError("actions contain out-of-range values")

        actions_wp = self._wp.from_torch(actions)
        stream = self._wp.stream_from_torch(torch.cuda.current_stream())
        with self._wp.ScopedStream(stream):
            self._launch_step(actions_wp)

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        if self._mem is None or not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if lo < 0 or hi < 0 or lo > MEM_SIZE or hi > MEM_SIZE or lo > hi:
            raise ValueError(f"Invalid memory range: lo={lo} hi={hi}")
        if hi <= ROM_LIMIT:
            return self._read_rom_slice(lo, hi)
        if lo >= ROM_LIMIT:
            return self._read_mem_device_range(env_idx, lo, hi)
        rom_part = self._read_rom_slice(lo, ROM_LIMIT)
        mem_part = self._read_mem_device_range(env_idx, ROM_LIMIT, hi)
        return rom_part + mem_part

    def _read_mem_device_range(self, env_idx: int, lo: int, hi: int) -> bytes:
        count = hi - lo
        if count <= 0:
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

    def read_frame_bg_shade_env0(self) -> bytes:
        if self._frame_bg_shade_env0 is None or not self._initialized:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        count = SCREEN_W * SCREEN_H
        if self._frame_readback is None or self._frame_readback_capacity < count:
            self._frame_readback = self._wp.empty(
                count, dtype=self._wp.uint8, device="cpu", pinned=True
            )
            self._frame_readback_capacity = count
        self._wp.copy(
            self._frame_readback,
            self._frame_bg_shade_env0,
            dest_offset=0,
            src_offset=0,
            count=count,
        )
        self._wp.synchronize()
        return self._frame_readback.numpy()[:count].tobytes()

    def read_serial(self, env_idx: int) -> bytes:
        if self._serial_buf is None or self._serial_len is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        self._wp.synchronize()
        lengths = self._serial_len.numpy()
        length = int(lengths[env_idx])
        if length <= 0:
            return b""
        length = min(length, SERIAL_MAX)
        if self._serial_readback is None or self._serial_readback_capacity < length:
            self._serial_readback = self._wp.empty(
                length, dtype=self._wp.uint8, device="cpu", pinned=True
            )
            self._serial_readback_capacity = length
        base = env_idx * SERIAL_MAX
        self._wp.copy(
            self._serial_readback,
            self._serial_buf,
            dest_offset=0,
            src_offset=base,
            count=length,
        )
        self._wp.synchronize()
        return self._serial_readback.numpy()[:length].tobytes()

    def get_cart_state(self, env_idx: int) -> dict[str, int]:
        """Get cart/MBC state snapshot for a specific environment."""
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError(f"env_idx {env_idx} out of range [0, {self.num_envs})")
        if not self._initialized or self._cart_state is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        count = CART_STATE_STRIDE
        if (
            self._cart_state_readback is None
            or self._cart_state_readback_capacity < count
        ):
            self._cart_state_readback = self._wp.empty(
                count, dtype=self._wp.int32, device="cpu", pinned=True
            )
            self._cart_state_readback_capacity = count
        base = env_idx * CART_STATE_STRIDE
        self._wp.copy(
            self._cart_state_readback,
            self._cart_state,
            dest_offset=0,
            src_offset=base,
            count=count,
        )
        self._wp.synchronize()
        state = np.array(self._cart_state_readback.numpy()[:count], copy=True)
        return self._cart_state_to_dict(state)

    def write_memory(self, env_idx: int, addr: int, data: bytes) -> None:
        raise NotImplementedError(
            "CUDA write_memory is not implemented yet (debug-only for CPU)."
        )

    def close(self) -> None:
        super().close()
        self._mem_readback = None
        self._mem_readback_capacity = 0
        self._frame_readback = None
        self._frame_readback_capacity = 0
        self._serial_readback = None
        self._serial_readback_capacity = 0
        self._cart_state_readback = None
        self._cart_state_readback_capacity = 0


class WarpVecBackend(WarpVecCpuBackend):
    """Alias for warp_vec backend name."""

    name: str = "warp_vec"
