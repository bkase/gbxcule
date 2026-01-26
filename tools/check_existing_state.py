#!/usr/bin/env python3
"""Check existing tetris_start.state by loading into PyBoy."""

from pathlib import Path

from pyboy import PyBoy


def main() -> None:
    rom_path = "tetris.gb"
    state_path = Path("states/tetris_start.state")

    if not state_path.exists():
        print(f"State not found: {state_path}")
        return

    pyboy = PyBoy(rom_path, window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    with state_path.open("rb") as f:
        pyboy.load_state(f)

    # Tick a few times
    for _ in range(10):
        pyboy.tick(render=True)

    pyboy.screen.image.save("states/tetris_start_check.png")
    print("Saved states/tetris_start_check.png")

    pyboy.stop(save=False)


if __name__ == "__main__":
    main()
