"""Gymnasium-compatible PyBoy environment for micro-ROM baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from gbxcule.backends.common import get_pyboy_class, resolve_action_codec
from gbxcule.core.action_schedule import (
    run_puffer_press_release_schedule,
    validate_schedule,
)

_BOOTROM_PATH = Path("bench/roms/bootrom_fast_dmg.bin")


_WINDOW_EVENT_MAP: dict[str, tuple[Any, Any]] | None = None


def _get_window_event_map() -> dict[str, tuple[Any, Any]]:
    """Return mapping from PyBoy button names to WindowEvent press/release."""
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


class PyBoyGymEnv(gym.Env):
    """Minimal PyBoy Gymnasium environment for micro-ROMs."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        rom_path: str,
        *,
        frames_per_step: int = 24,
        release_after_frames: int = 8,
        obs_dim: int = 32,
        action_codec: str = "pokemonred_puffer_v1",
        log_level: str = "ERROR",
    ) -> None:
        if not Path(rom_path).exists():
            raise FileNotFoundError(f"ROM file not found: {rom_path}")
        if not _BOOTROM_PATH.exists():
            raise FileNotFoundError(
                f"Boot ROM not found: {_BOOTROM_PATH}. "
                "Expected repo-local fast boot ROM."
            )
        validate_schedule(frames_per_step, release_after_frames)
        if frames_per_step < 1:
            raise ValueError("frames_per_step must be >= 1")

        self._rom_path = rom_path
        self._frames_per_step = frames_per_step
        self._release_after_frames = release_after_frames
        self._obs_dim = obs_dim
        self._log_level = log_level
        self._action_codec = resolve_action_codec(action_codec)

        self.action_space = gym.spaces.Discrete(self._action_codec.num_actions)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._pyboy: Any = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._pyboy is not None:
            self._pyboy.stop(save=False)

        PyBoy = get_pyboy_class()
        self._pyboy = PyBoy(
            self._rom_path,
            window="null",
            sound_emulated=False,
            bootrom=str(_BOOTROM_PATH),
        )
        self._pyboy.set_emulation_speed(0)

        obs = self._build_obs()
        info: dict[str, Any] = {"seed": seed}
        return obs, info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._pyboy is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action = int(action.item()) if isinstance(action, np.ndarray) else int(action)

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
        reward = 0.0
        done = False
        truncated = False
        info: dict[str, Any] = {}
        return obs, reward, done, truncated, info

    def close(self) -> None:
        if self._pyboy is not None:
            self._pyboy.stop(save=False)
            self._pyboy = None

    def _tick_frames(self) -> None:
        if self._frames_per_step > 1:
            self._pyboy.tick(self._frames_per_step - 1, render=False)
        self._pyboy.tick(1, render=False)

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros((self._obs_dim,), dtype=np.float32)
        if self._pyboy is None:
            return obs

        reg = self._pyboy.register_file
        obs[0] = reg.PC / 65535.0
        obs[1] = reg.SP / 65535.0
        obs[2] = reg.A / 255.0
        obs[3] = reg.F / 255.0
        obs[4] = reg.B / 255.0
        obs[5] = reg.C / 255.0
        obs[6] = reg.D / 255.0
        obs[7] = reg.E / 255.0
        obs[8] = (reg.HL >> 8) / 255.0
        obs[9] = (reg.HL & 0xFF) / 255.0
        return obs
