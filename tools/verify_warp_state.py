#!/usr/bin/env python3
"""Verify Warp state by loading into PyBoy and taking a screenshot."""

from __future__ import annotations

from pathlib import Path

from pyboy import PyBoy


def main() -> None:
    rom_path = "tetris.gb"
    state_path = Path("states/tetris_warp_gameplay.state")
    out_screenshot = Path("states/tetris_warp_verify.png")

    if not Path(rom_path).exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")

    # Load into PyBoy
    pyboy = PyBoy(rom_path, window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    with state_path.open("rb") as f:
        pyboy.load_state(f)

    # Tick once to render
    pyboy.tick(render=True)

    # Take screenshot
    out_screenshot.parent.mkdir(parents=True, exist_ok=True)
    pyboy.screen.image.save(out_screenshot)
    print(f"Screenshot saved to {out_screenshot}")

    pyboy.stop(save=False)


if __name__ == "__main__":
    main()
