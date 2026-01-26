#!/usr/bin/env python3
"""Verify the gameplay state works with Warp Runtime and create final checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyboy import PyBoy

from gbxcule.backends.warp_vec import WarpVecCpuBackend

# Action indices
A = 0
B = 1
START = 2
UP = 3
DOWN = 4
LEFT = 5
RIGHT = 6


def main() -> None:
    rom_path = "tetris.gb"
    input_state = Path("states/tetris_gameplay.state")
    out_state = Path("states/tetris_warp_checkpoint.state")
    out_screenshot = Path("states/tetris_warp_checkpoint.png")
    out_meta = Path("states/tetris_warp_checkpoint.meta.json")

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
    print("Warp backend initialized")

    # Load the PyBoy gameplay state
    backend.load_state_file(str(input_state), env_idx=0)
    print(f"Loaded state from {input_state}")

    # Step a few times to let the piece fall more
    print("Stepping in Warp...")
    for i in range(5):
        actions = np.array([0], dtype=np.int32)  # No input
        backend.step(actions)
        print(f"  Step {i + 1}")

    # Save Warp state
    backend.save_state_file(str(out_state), env_idx=0)
    print(f"Warp state saved to {out_state}")

    # Verify by loading into PyBoy and taking screenshot
    print("Verifying state with PyBoy...")
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

    # Save metadata
    meta = {
        "rom": rom_path,
        "input_state": str(input_state),
        "warp_steps": 5,
        "frames_per_step": 24,
        "total_additional_frames": 5 * 24,
        "screenshot": str(out_screenshot),
        "state": str(out_state),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Metadata saved to {out_meta}")

    print("\nDone! Warp gameplay checkpoint created.")


if __name__ == "__main__":
    main()
