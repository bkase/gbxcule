#!/usr/bin/env python3
"""Create Tetris gameplay state using PyBoy directly."""

from __future__ import annotations

import json
from pathlib import Path

from pyboy import PyBoy


def main() -> None:
    rom_path = "tetris.gb"
    input_state = Path("states/tetris_start.state")
    out_state = Path("states/tetris_gameplay.state")
    out_screenshot = Path("states/tetris_gameplay.png")
    out_meta = Path("states/tetris_gameplay.meta.json")

    if not Path(rom_path).exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not input_state.exists():
        raise FileNotFoundError(f"Input state not found: {input_state}")

    out_state.parent.mkdir(parents=True, exist_ok=True)

    pyboy = PyBoy(rom_path, window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    # Load existing state (at menu screen - 1PLAYER/2PLAYER selection)
    with input_state.open("rb") as f:
        pyboy.load_state(f)
    print(f"Loaded state from {input_state}")

    ticks = 0

    def tick(n: int = 1) -> None:
        nonlocal ticks
        for _ in range(n):
            pyboy.tick(render=True)
            ticks += 1

    def press_button(button: str, hold_frames: int = 8, wait_frames: int = 60) -> None:
        pyboy.button_press(button)
        tick(hold_frames)
        pyboy.button_release(button)
        tick(wait_frames)

    # Take initial screenshot to see where we are
    pyboy.screen.image.save("states/tetris_debug_1.png")
    print("Debug screenshot 1 saved")

    # From menu screen: Press START to select 1 PLAYER
    print("Selecting 1 PLAYER...")
    press_button("start", hold_frames=8, wait_frames=60)
    pyboy.screen.image.save("states/tetris_debug_2.png")
    print("Debug screenshot 2 saved (after 1 PLAYER)")

    # Now at game type selection (A-TYPE / B-TYPE)
    # Press START to select A-TYPE
    print("Selecting A-TYPE...")
    press_button("start", hold_frames=8, wait_frames=60)
    pyboy.screen.image.save("states/tetris_debug_3.png")
    print("Debug screenshot 3 saved (after A-TYPE)")

    # Now at level/music selection
    # Press START to start game
    print("Starting game...")
    press_button("start", hold_frames=8, wait_frames=120)
    pyboy.screen.image.save("states/tetris_debug_4.png")
    print("Debug screenshot 4 saved (after START game)")

    # Wait for gameplay to begin and first piece to appear
    print("Waiting for blocks to fall...")
    for i in range(300):
        tick(1)
        if i % 50 == 0:
            pyboy.screen.image.save(f"states/tetris_debug_wait_{i}.png")
            print(f"Debug screenshot at tick {ticks}")

    # Take final screenshot
    pyboy.screen.image.save(out_screenshot)
    print(f"Final screenshot saved to {out_screenshot}")

    # Save state
    with out_state.open("wb") as f:
        pyboy.save_state(f)
    print(f"State saved to {out_state}")

    # Save metadata
    meta = {
        "rom": rom_path,
        "input_state": str(input_state),
        "total_ticks": ticks,
        "screenshot": str(out_screenshot),
        "state": str(out_state),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Metadata saved to {out_meta}")

    pyboy.stop(save=False)
    print(f"\nDone! Captured state after {ticks} ticks.")


if __name__ == "__main__":
    main()
