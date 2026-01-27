#!/usr/bin/env python3
"""Play Tetris with random actions using Warp CPU backend and capture screenshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from gbxcule.backends.warp_vec import WarpVecCpuBackend
from gbxcule.core.abi import SCREEN_H, SCREEN_W

# Action indices (pokemonred_puffer_v1 codec)
# 0=NOOP, 1=A, 2=B, 3=START, 4=UP, 5=DOWN, 6=LEFT, 7=RIGHT
NUM_ACTIONS = 8


def take_screenshot(backend: WarpVecCpuBackend) -> Image.Image:
    """Take a screenshot from the Warp backend's background renderer."""
    frame_bytes = backend.read_frame_bg_shade_env0()
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((SCREEN_H, SCREEN_W))
    # Convert grayscale shades (0-3) to actual grayscale (0-255)
    # GB uses 0=white, 3=black
    frame_scaled = (255 - (frame_np * 85)).astype(np.uint8)
    return Image.fromarray(frame_scaled, mode="L")


def main() -> None:
    rom_path = "tetris.gb"
    state_path = Path("states/tetris_gameplay.state")
    out_dir = Path("tetris_screenshots")

    if not Path(rom_path).exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Create Warp backend with background rendering enabled
    backend = WarpVecCpuBackend(
        rom_path=rom_path,
        num_envs=1,
        frames_per_step=24,
        release_after_frames=8,
        render_bg=True,
    )

    obs, info = backend.reset(seed=42)
    print("Warp backend initialized")

    # Load the gameplay state
    backend.load_state_file(str(state_path), env_idx=0)
    print(f"Loaded state from {state_path}")

    # Random number generator
    rng = np.random.default_rng(seed=12345)

    total_steps = 0
    num_batches = 128
    steps_per_batch = 5

    print(f"Running {num_batches} batches of {steps_per_batch} random actions each...")
    print(f"Total steps: {num_batches * steps_per_batch}")

    for batch in range(num_batches):
        # Execute 5 random actions
        for _ in range(steps_per_batch):
            action = rng.integers(0, NUM_ACTIONS)
            actions = np.array([action], dtype=np.int32)
            backend.step(actions)
            total_steps += 1

        # Take screenshot using Warp's background renderer
        screenshot = take_screenshot(backend)
        screenshot_path = out_dir / f"screenshot_{batch:03d}.png"
        screenshot.save(screenshot_path)

        if (batch + 1) % 16 == 0:
            print(f"  Batch {batch + 1}/{num_batches} complete ({total_steps} steps)")

    print(f"\nDone! {num_batches} screenshots saved to {out_dir}/")
    print(f"Total steps executed: {total_steps}")


if __name__ == "__main__":
    main()
