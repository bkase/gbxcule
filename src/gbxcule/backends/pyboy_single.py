"""Single-env PyBoy backend (trusted reference).

This backend wraps PyBoy to provide a VecBackend-conforming interface
for a single environment. It is used as the correctness oracle for
verification and as a baseline for benchmarks.
"""

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
    get_pyboy_class,
    resolve_action_codec,
)
from gbxcule.core.action_schedule import split_press_release_ticks, validate_schedule
from gbxcule.core.obs import build_obs_v3_from_state


class PyBoySingleBackend:
    """Single-environment PyBoy backend conforming to VecBackend protocol.

    This backend runs PyBoy headless and provides deterministic step semantics
    with explicit button press/release timing.

    Attributes:
        name: Backend identifier ("pyboy_single").
        device: Device type ("cpu").
        num_envs: Number of environments (always 1).
        action_spec: Action array specification.
        obs_spec: Observation array specification.
    """

    name: str = "pyboy_single"
    device: Device = "cpu"
    num_envs: int = 1
    _BOOTROM_PATH = Path("bench/roms/bootrom_fast_dmg.bin")

    def __init__(
        self,
        rom_path: str,
        *,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        obs_dim: int = 32,
        action_codec: str = DEFAULT_ACTION_CODEC_ID,
        log_level: str = "ERROR",
        render_frames: bool = False,
    ) -> None:
        """Initialize the backend.

        Args:
            rom_path: Path to the ROM file.
            frames_per_step: Frames to advance per step call.
            release_after_frames: Frames after which to release button.
            obs_dim: Observation vector dimension.
        action_codec: Action codec id (e.g., "pokemonred_puffer_v1").
            log_level: PyBoy log level (e.g., "ERROR", "WARNING").
            render_frames: Whether to render frames for screen reads.
        """
        self._rom_path = rom_path
        validate_schedule(frames_per_step, release_after_frames)
        self._frames_per_step = frames_per_step
        self._release_after_frames = release_after_frames
        self._obs_dim = obs_dim
        self._log_level = log_level
        self._render_frames = render_frames
        self._action_codec = resolve_action_codec(action_codec)
        self.action_codec = action_codec_spec(action_codec)
        self.num_actions = self._action_codec.num_actions

        # Specs
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

        # PyBoy instance (created on reset)
        self._pyboy: Any = None

    def _make_pyboy(self) -> Any:
        """Create a fresh PyBoy instance."""
        if not self._BOOTROM_PATH.exists():
            raise FileNotFoundError(
                f"Boot ROM not found: {self._BOOTROM_PATH}. "
                "Expected repo-local fast boot ROM."
            )
        PyBoy = get_pyboy_class()
        pyboy = PyBoy(
            self._rom_path,
            window="null",
            sound_emulated=False,
            bootrom=str(self._BOOTROM_PATH),
        )
        pyboy.set_emulation_speed(0)  # No speed limit
        return pyboy

    def reset(self, seed: int | None = None) -> tuple[NDArrayF32, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional random seed (not used by emulator, but recorded).

        Returns:
            Tuple of (observations, info_dict).
        """
        # Close previous instance if any
        if self._pyboy is not None:
            self._pyboy.stop(save=False)

        # Create fresh emulator
        self._pyboy = self._make_pyboy()

        # Build observation
        obs = self._build_obs()

        info: dict[str, Any] = {"seed": seed}
        return obs, info

    def step(
        self, actions: NDArrayI32
    ) -> tuple[NDArrayF32, NDArrayF32, NDArrayBool, NDArrayBool, dict[str, Any]]:
        """Step the environment forward.

        Args:
            actions: Action array, shape (1,), dtype int32.

        Returns:
            Tuple of (obs, reward, done, trunc, info).
        """
        if self._pyboy is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        # Validate and cast actions
        actions = as_i32_actions(actions, self.num_envs)
        action = int(actions[0])

        # Get button name (raises ValueError if out of range)
        button = self._action_codec.to_pyboy_button(action)

        # Apply action with explicit press/release timing
        if button is not None:
            # Press button
            self._pyboy.button_press(button)

            # Tick for release_after_frames
            pressed_ticks, remaining_ticks = split_press_release_ticks(
                self._frames_per_step, self._release_after_frames
            )
            for _ in range(pressed_ticks):
                self._pyboy.tick(render=self._render_frames)

            # Release button
            self._pyboy.button_release(button)

            # Tick remaining frames
            for _ in range(remaining_ticks):
                self._pyboy.tick(render=self._render_frames)
        else:
            # Noop: just tick all frames
            for _ in range(self._frames_per_step):
                self._pyboy.tick(render=self._render_frames)

        # Build outputs
        obs = self._build_obs()
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.array([False], dtype=np.bool_)
        trunc = np.array([False], dtype=np.bool_)
        info: dict[str, Any] = {}

        return obs, reward, done, trunc, info

    def get_cpu_state(self, env_idx: int) -> CpuState:
        """Get CPU register state.

        Args:
            env_idx: Environment index (must be 0).

        Returns:
            CpuState dictionary with register values and flags.

        Raises:
            ValueError: If env_idx != 0.
            RuntimeError: If backend not initialized.
        """
        if env_idx != 0:
            raise ValueError(f"env_idx must be 0 for single-env backend, got {env_idx}")
        if self._pyboy is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        reg = self._pyboy.register_file

        # Read registers
        pc = int(reg.PC)
        sp = int(reg.SP)
        a = int(reg.A)
        f = int(reg.F)
        b = int(reg.B)
        c = int(reg.C)
        d = int(reg.D)
        e = int(reg.E)
        hl = int(reg.HL)

        # Derive H and L from HL
        h = (hl >> 8) & 0xFF
        l = hl & 0xFF  # noqa: E741 - canonical register name

        # Derive flags from F
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
            instr_count=None,  # Not available from PyBoy
            cycle_count=None,  # Not available from PyBoy
        )

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        """Read a slice of Game Boy memory.

        Args:
            env_idx: Environment index (must be 0).
            lo: Lower address (inclusive).
            hi: Upper address (exclusive).

        Returns:
            Bytes for the requested memory slice.
        """
        if env_idx != 0:
            raise ValueError(f"env_idx must be 0 for single-env backend, got {env_idx}")
        if self._pyboy is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if lo < 0 or hi > 0x10000 or lo >= hi:
            raise ValueError(f"Invalid memory range: {lo:#06x}:{hi:#06x}")

        return bytes(self._pyboy.memory[lo:hi])

    def close(self) -> None:
        """Clean up resources."""
        if self._pyboy is not None:
            self._pyboy.stop(save=False)
            self._pyboy = None

    def _build_obs(self) -> NDArrayF32:
        """Build observation vector from current state.

        Returns normalized register values in the first slots,
        with remaining slots as zeros.
        """
        obs = empty_obs(self.num_envs, self._obs_dim)

        if self._pyboy is not None:
            reg = self._pyboy.register_file
            hl = int(reg.HL)
            wram = self._pyboy.memory[0xC000:0xC010]
            obs_vec = build_obs_v3_from_state(
                pc=int(reg.PC),
                sp=int(reg.SP),
                a=int(reg.A),
                f=int(reg.F),
                b=int(reg.B),
                c=int(reg.C),
                d=int(reg.D),
                e=int(reg.E),
                h=(hl >> 8) & 0xFF,
                l_reg=hl & 0xFF,
                wram16=wram,
                obs_dim=self._obs_dim,
            )
            obs[0, : obs_vec.shape[0]] = obs_vec

        return obs
