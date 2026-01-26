#!/usr/bin/env python3
"""Capture a Pallet Town house goal trace using PyBoy and save actions/state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyboy import PyBoy
from pyboy.utils import WindowEvent

ROM_PATH = Path("red.gb")
OUT_DIR = Path("states/rl_m5_goal_pallet_house")
OUT_STATE = Path("states/pokemonred_pallet_house.state")
OUT_ACTIONS = Path("states/rl_m5_actions_pallet_house.jsonl")


def main() -> None:
    if not ROM_PATH.exists():
        raise FileNotFoundError(f"ROM not found: {ROM_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pyboy: Any = PyBoy(str(ROM_PATH), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    actions: list[list[int]] = []

    def tick(n: int) -> None:
        for _ in range(n):
            pyboy.tick(render=True)

    def press(
        event_press, event_release, hold_frames: int = 8, wait_frames: int = 40
    ) -> None:
        pyboy.send_input(event_press)
        tick(hold_frames)
        pyboy.send_input(event_release)
        tick(wait_frames)

    def record_action(
        action_idx: int,
        event_press,
        event_release,
        hold_frames: int = 8,
        wait_frames: int = 16,
    ) -> None:
        actions.append([action_idx])
        pyboy.send_input(event_press)
        tick(hold_frames)
        pyboy.send_input(event_release)
        tick(wait_frames)

    # Boot to intro/title
    tick(650)

    # Press START to begin
    press(WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START, 8, 120)

    # Advance intro to name screen
    for _ in range(30):
        press(WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A, 8, 40)

    # Choose default player name (START)
    press(WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START, 8, 80)

    # Advance to rival name screen
    for _ in range(10):
        press(WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A, 8, 40)

    # Choose default rival name (START)
    press(WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START, 8, 80)

    # Advance story to room
    for _ in range(120):
        press(WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A, 8, 40)

    # Clear remaining dialog quickly using B
    for _ in range(200):
        press(WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B, 4, 4)

    # Save start state for RL replay (after dialog cleared)
    pyboy.screen.image.save(OUT_DIR / "pyboy_start_debug.png")
    with OUT_STATE.open("wb") as f:
        pyboy.save_state(f)

    # Now record movement actions while inside house
    # Use codec indices: A=0, B=1, START=2, UP=3, DOWN=4, LEFT=5, RIGHT=6
    # Move right and down a few tiles inside the house
    for _ in range(3):
        record_action(6, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)
    for _ in range(2):
        record_action(4, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)

    # Save screenshot for goal verification
    pyboy.screen.image.save(OUT_DIR / "pyboy_goal_debug.png")

    # Save actions trace
    with OUT_ACTIONS.open("w", encoding="utf-8") as f:
        for step in actions:
            f.write(json.dumps(step) + "\n")

    pyboy.stop(save=False)
    print(f"Saved state to {OUT_STATE}")
    print(f"Saved actions to {OUT_ACTIONS}")


if __name__ == "__main__":
    main()
