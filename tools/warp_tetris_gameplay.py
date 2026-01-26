#!/usr/bin/env python3
"""Start from existing tetris_start.state and navigate to actual gameplay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyboy import PyBoy

from gbxcule.backends.warp_vec import WarpVecCpuBackend

# Action indices (pokemonred_puffer_v0 codec)
A = 0
B = 1
START = 2
UP = 3
DOWN = 4
LEFT = 5
RIGHT = 6


def main() -> None:
    rom_path = "tetris.gb"
    input_state = Path("states/tetris_start.state")
    out_state = Path("states/tetris_warp_gameplay.state")
    out_screenshot = Path("states/tetris_warp_gameplay.png")
    out_meta = Path("states/tetris_warp_gameplay.meta.json")

    if not Path(rom_path).exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not input_state.exists():
        raise FileNotFoundError(f"Input state not found: {input_state}")

    out_state.parent.mkdir(parents=True, exist_ok=True)

    # Create Warp backend
    backend = WarpVecCpuBackend(
        rom_path=rom_path,
        num_envs=1,
        frames_per_step=24,
        release_after_frames=8,
        render_bg=True,
    )

    obs, info = backend.reset(seed=42)
    print("Backend initialized")

    # Load existing state (at menu screen)
    backend.load_state_file(str(input_state), env_idx=0)
    print(f"Loaded state from {input_state}")

    total_steps = 0

    def step(action: int = 0) -> None:
        """Execute one step (24 frames) with the given action."""
        nonlocal total_steps
        actions = np.array([action], dtype=np.int32)
        backend.step(actions)
        total_steps += 1

    def step_n(n: int, action: int = 0) -> None:
        """Execute n steps with the given action."""
        for _ in range(n):
            step(action)

    # From the menu screen (1PLAYER selected), press START to select game type
    print("Navigating menus...")

    # Press START to select 1 PLAYER
    step(START)
    step_n(3)  # Wait for game type screen (A-TYPE / B-TYPE)

    # Press START to select A-TYPE
    step(START)
    step_n(3)  # Wait for level/music selection

    # Press START to start with default level/music
    step(START)
    step_n(5)  # Wait for game to initialize

    print(f"After menu navigation: {total_steps} steps")

    # Wait for blocks to start falling - step more frames
    print("Waiting for gameplay...")
    step_n(10)

    # Save the Warp state
    backend.save_state_file(str(out_state), env_idx=0)
    print(f"Warp state saved to {out_state}")

    # Now load into PyBoy to verify and take a proper screenshot
    print("Verifying with PyBoy...")
    pyboy = PyBoy(rom_path, window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    with out_state.open("rb") as f:
        pyboy.load_state(f)

    # Tick to render
    pyboy.tick(render=True)

    # Take screenshot
    pyboy.screen.image.save(out_screenshot)
    print(f"Screenshot saved to {out_screenshot}")

    pyboy.stop(save=False)

    total_frames = total_steps * 24

    # Save metadata
    meta = {
        "rom": rom_path,
        "input_state": str(input_state),
        "total_steps": total_steps,
        "total_frames": total_frames,
        "frames_per_step": 24,
        "screenshot": str(out_screenshot),
        "state": str(out_state),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Metadata saved to {out_meta}")

    print(f"\nDone! Captured state after {total_steps} steps ({total_frames} frames).")


if __name__ == "__main__":
    main()
