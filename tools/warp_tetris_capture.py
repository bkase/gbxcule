#!/usr/bin/env python3
"""Capture Tetris gameplay state using the Warp Runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from gbxcule.backends.warp_vec import WarpVecCpuBackend
from gbxcule.core.abi import SCREEN_H, SCREEN_W

# Action indices (pokemonred_puffer_v1 codec)
NOOP = 0
A = 1
B = 2
START = 3
UP = 4
DOWN = 5
LEFT = 6
RIGHT = 7


def main() -> None:
    rom_path = "tetris.gb"
    out_state = Path("states/tetris_warp_gameplay.state")
    out_screenshot = Path("states/tetris_warp_gameplay.png")
    out_meta = Path("states/tetris_warp_gameplay.meta.json")

    if not Path(rom_path).exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")

    out_state.parent.mkdir(parents=True, exist_ok=True)

    # Create Warp backend with proper press/release timing
    # Each step = 24 frames, button held for 8 frames then released
    backend = WarpVecCpuBackend(
        rom_path=rom_path,
        num_envs=1,
        frames_per_step=24,
        release_after_frames=8,
        render_bg=True,
    )

    obs, info = backend.reset(seed=42)
    print("Backend initialized")

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

    def take_screenshot() -> Image.Image:
        """Take a screenshot from the Warp backend."""
        frame_bytes = backend.read_frame_bg_shade_env0()
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            (SCREEN_H, SCREEN_W)
        )
        # Convert grayscale shades (0-3) to actual grayscale (0-255)
        # GB uses 0=white, 3=black
        frame_scaled = 255 - (frame_np * 85)  # 0->255, 1->170, 2->85, 3->0
        return Image.fromarray(frame_scaled, mode="L")

    # Boot sequence - wait for LCD to turn on (~120 frames = 5 steps)
    print("Boot sequence...")
    step_n(5)

    # Press START to enter menu
    print("Pressing START to enter menu...")
    step(START)
    step_n(3)  # Wait for menu to settle (~72 frames)

    # Press START again to start game (selects 1 player mode)
    print("Pressing START to start game...")
    step(START)
    step_n(2)  # Wait for game to initialize (~48 frames)

    # Wait for gameplay - look for the first piece to appear
    print("Waiting for gameplay to start...")
    prev_screenshot = take_screenshot()
    fall_step = None

    for _i in range(30):  # Max steps to wait
        step()
        current_screenshot = take_screenshot()
        # Check if screen changed (piece falling)
        prev_arr = np.array(prev_screenshot)
        curr_arr = np.array(current_screenshot)
        diff = np.sum(prev_arr != curr_arr)
        if diff > 100:  # Significant screen change
            fall_step = total_steps
            print(f"Detected screen change at step {total_steps} (diff={diff} pixels)")
            break
        prev_screenshot = current_screenshot

    # Step a few more to ensure piece is clearly visible
    step_n(5)

    # Take final screenshot
    screenshot = take_screenshot()
    screenshot.save(out_screenshot)
    print(f"Screenshot saved to {out_screenshot}")

    # Save state
    backend.save_state_file(str(out_state), env_idx=0)
    print(f"State saved to {out_state}")

    total_frames = total_steps * 24

    # Save metadata
    meta = {
        "rom": rom_path,
        "total_steps": total_steps,
        "total_frames": total_frames,
        "fall_step": fall_step,
        "frames_per_step": 24,
        "screenshot": str(out_screenshot),
        "state": str(out_state),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Metadata saved to {out_meta}")

    print(f"\nDone! Captured state after {total_steps} steps ({total_frames} frames).")


if __name__ == "__main__":
    main()
