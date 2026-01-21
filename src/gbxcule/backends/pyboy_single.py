"""Single-env PyBoy backend (trusted reference).

This backend wraps PyBoy to provide a VecBackend-conforming interface
for a single environment. It is used as the correctness oracle for
verification and as a baseline for benchmarks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gbxcule.backends.common import (
    ArraySpec,
    CpuState,
    Device,
    NDArrayBool,
    NDArrayF32,
    NDArrayI32,
    action_to_button,
    as_i32_actions,
    empty_obs,
    flags_from_f,
)


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

    def __init__(
        self,
        rom_path: str,
        *,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        obs_dim: int = 32,
        log_level: str = "ERROR",
    ) -> None:
        """Initialize the backend.

        Args:
            rom_path: Path to the ROM file.
            frames_per_step: Frames to advance per step call.
            release_after_frames: Frames after which to release button.
            obs_dim: Observation vector dimension.
            log_level: PyBoy log level (e.g., "ERROR", "WARNING").
        """
        self._rom_path = rom_path
        self._frames_per_step = frames_per_step
        self._release_after_frames = release_after_frames
        self._obs_dim = obs_dim
        self._log_level = log_level

        # Specs
        self.action_spec = ArraySpec(
            shape=(self.num_envs,),
            dtype="int32",
            meaning="action index [0, 9)",
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
        from pyboy import PyBoy

        pyboy = PyBoy(
            self._rom_path,
            window="null",
            sound_emulated=False,
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
        button = action_to_button(action)

        # Apply action with explicit press/release timing
        if button is not None:
            # Press button
            self._pyboy.button_press(button)

            # Tick for release_after_frames
            for _ in range(self._release_after_frames):
                self._pyboy.tick(render=False)

            # Release button
            self._pyboy.button_release(button)

            # Tick remaining frames
            remaining = self._frames_per_step - self._release_after_frames
            for _ in range(remaining):
                self._pyboy.tick(render=False)
        else:
            # Noop: just tick all frames
            for _ in range(self._frames_per_step):
                self._pyboy.tick(render=False)

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

            # Normalize registers to [0, 1] range
            # PC and SP are 16-bit, others are 8-bit
            obs[0, 0] = reg.PC / 65535.0
            obs[0, 1] = reg.SP / 65535.0
            obs[0, 2] = reg.A / 255.0
            obs[0, 3] = reg.F / 255.0
            obs[0, 4] = reg.B / 255.0
            obs[0, 5] = reg.C / 255.0
            obs[0, 6] = reg.D / 255.0
            obs[0, 7] = reg.E / 255.0
            obs[0, 8] = (reg.HL >> 8) / 255.0  # H
            obs[0, 9] = (reg.HL & 0xFF) / 255.0  # L
            # Remaining slots are zeros

        return obs
