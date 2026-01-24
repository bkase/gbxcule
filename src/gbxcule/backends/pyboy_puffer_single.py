"""Single-env PyBoy backend using puffer-style input timing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gbxcule.backends.common import (
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
from gbxcule.core.action_schedule import (
    run_puffer_press_release_schedule,
    validate_schedule,
)

_BOOTROM_PATH = Path("bench/roms/bootrom_fast_dmg.bin")

_WINDOW_EVENT_MAP: dict[str, tuple[Any, Any]] | None = None


def _get_window_event_map() -> dict[str, tuple[Any, Any]]:
    global _WINDOW_EVENT_MAP
    if _WINDOW_EVENT_MAP is None:
        from pyboy.utils import WindowEvent

        _WINDOW_EVENT_MAP = {
            "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
            "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
            "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
            "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
            "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
            "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
            "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
            "select": (
                WindowEvent.PRESS_BUTTON_SELECT,
                WindowEvent.RELEASE_BUTTON_SELECT,
            ),
        }
    return _WINDOW_EVENT_MAP


class PyBoyPufferSingleBackend:
    """Single-environment PyBoy backend using puffer-style timing."""

    name: str = "pyboy_puffer_single"
    device: Device = "cpu"
    num_envs: int = 1

    def __init__(
        self,
        rom_path: str,
        *,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        obs_dim: int = 32,
        action_codec: str = "pokemonred_puffer_v0",
        log_level: str = "ERROR",
        render_frames: bool = False,
    ) -> None:
        self._rom_path = rom_path
        validate_schedule(frames_per_step, release_after_frames)
        if frames_per_step < 1:
            raise ValueError("frames_per_step must be >= 1")
        self._frames_per_step = frames_per_step
        self._release_after_frames = release_after_frames
        self._obs_dim = obs_dim
        self._log_level = log_level
        self._render_frames = render_frames
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

        self._pyboy: Any = None

    def _make_pyboy(self) -> Any:
        if not _BOOTROM_PATH.exists():
            raise FileNotFoundError(
                f"Boot ROM not found: {_BOOTROM_PATH}. "
                "Expected repo-local fast boot ROM."
            )
        PyBoy = get_pyboy_class()
        pyboy = PyBoy(
            self._rom_path,
            window="null",
            sound_emulated=False,
            bootrom=str(_BOOTROM_PATH),
        )
        pyboy.set_emulation_speed(0)
        return pyboy

    def reset(self, seed: int | None = None) -> tuple[NDArrayF32, dict[str, Any]]:
        if self._pyboy is not None:
            self._pyboy.stop(save=False)
        self._pyboy = self._make_pyboy()

        obs = self._build_obs()
        info: dict[str, Any] = {"seed": seed}
        return obs, info

    def step(
        self, actions: NDArrayI32
    ) -> tuple[NDArrayF32, NDArrayF32, NDArrayBool, NDArrayBool, dict[str, Any]]:
        if self._pyboy is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        actions = as_i32_actions(actions, self.num_envs)
        action = int(actions[0])
        button = self._action_codec.to_pyboy_button(action)

        if button is None:
            self._tick_frames()
        else:
            press_event, release_event = _get_window_event_map()[button]
            run_puffer_press_release_schedule(
                send_input=self._pyboy.send_input,
                tick=self._pyboy.tick,
                press_event=press_event,
                release_event=release_event,
                frames_per_step=self._frames_per_step,
                release_after_frames=self._release_after_frames,
            )

        obs = self._build_obs()
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.array([False], dtype=np.bool_)
        trunc = np.array([False], dtype=np.bool_)
        info: dict[str, Any] = {}
        return obs, reward, done, trunc, info

    def get_cpu_state(self, env_idx: int) -> CpuState:
        if env_idx != 0:
            raise ValueError(f"env_idx must be 0 for single-env backend, got {env_idx}")
        if self._pyboy is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")

        reg = self._pyboy.register_file
        pc = int(reg.PC)
        sp = int(reg.SP)
        a = int(reg.A)
        f = int(reg.F)
        b = int(reg.B)
        c = int(reg.C)
        d = int(reg.D)
        e = int(reg.E)
        hl = int(reg.HL)
        h = (hl >> 8) & 0xFF
        l = hl & 0xFF  # noqa: E741 - canonical register name
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
            instr_count=None,
            cycle_count=None,
        )

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        if env_idx != 0:
            raise ValueError(f"env_idx must be 0 for single-env backend, got {env_idx}")
        if self._pyboy is None:
            raise RuntimeError("Backend not initialized. Call reset() first.")
        if lo < 0 or hi > 0x10000 or lo >= hi:
            raise ValueError(f"Invalid memory range: {lo:#06x}:{hi:#06x}")
        return bytes(self._pyboy.memory[lo:hi])

    def close(self) -> None:
        if self._pyboy is not None:
            self._pyboy.stop(save=False)
            self._pyboy = None

    def _tick_frames(self) -> None:
        if self._frames_per_step > 1:
            self._pyboy.tick(self._frames_per_step - 1, render=self._render_frames)
        self._pyboy.tick(1, render=self._render_frames)

    def _build_obs(self) -> NDArrayF32:
        obs = empty_obs(self.num_envs, self._obs_dim)
        if self._pyboy is None:
            return obs

        reg = self._pyboy.register_file
        obs[0, 0] = reg.PC / 65535.0
        obs[0, 1] = reg.SP / 65535.0
        obs[0, 2] = reg.A / 255.0
        obs[0, 3] = reg.F / 255.0
        obs[0, 4] = reg.B / 255.0
        obs[0, 5] = reg.C / 255.0
        obs[0, 6] = reg.D / 255.0
        obs[0, 7] = reg.E / 255.0
        obs[0, 8] = (reg.HL >> 8) / 255.0
        obs[0, 9] = (reg.HL & 0xFF) / 255.0
        return obs
