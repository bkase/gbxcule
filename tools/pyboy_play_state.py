#!/usr/bin/env python3
"""Play a ROM in PyBoy starting from a specific state with keyboard controls."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyboy import PyBoy
from pyboy.utils import WindowEvent

try:
    import sdl2
    from sdl2.ext import get_events
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "SDL2 is required for interactive play. Install pysdl2 and pysdl2-dll."
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", default=None, help="Optional .state file to load")
    parser.add_argument(
        "--save-state",
        default="states/pyboy_quicksave.state",
        help="Path for quick-save (F5)",
    )
    parser.add_argument(
        "--load-state",
        default=None,
        help="Path for quick-load (F9). Defaults to --save-state.",
    )
    parser.add_argument("--scale", type=int, default=3, help="Window scale factor")
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Emulation speed (1 = real-time)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rom_path = Path(args.rom)
    state_path = Path(args.state) if args.state else None
    save_state = Path(args.save_state)
    load_state = Path(args.load_state) if args.load_state else save_state

    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if state_path is not None and not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")

    save_state.parent.mkdir(parents=True, exist_ok=True)

    pyboy = PyBoy(str(rom_path), window="SDL2", scale=int(args.scale), no_input=True)
    pyboy.set_emulation_speed(int(args.speed))

    if state_path is not None:
        with state_path.open("rb") as f:
            pyboy.load_state(f)

    print(
        "Controls: arrows=move, Z=A, X=B, Enter=Start, RightShift=Select, "
        "F5=save, F9=load, Esc=quit"
    )
    print(f"Quick-save: {save_state}")
    print(f"Quick-load: {load_state}")

    key_down = {
        sdl2.SDLK_UP: WindowEvent.PRESS_ARROW_UP,
        sdl2.SDLK_DOWN: WindowEvent.PRESS_ARROW_DOWN,
        sdl2.SDLK_LEFT: WindowEvent.PRESS_ARROW_LEFT,
        sdl2.SDLK_RIGHT: WindowEvent.PRESS_ARROW_RIGHT,
        sdl2.SDLK_z: WindowEvent.PRESS_BUTTON_A,
        sdl2.SDLK_x: WindowEvent.PRESS_BUTTON_B,
        sdl2.SDLK_RETURN: WindowEvent.PRESS_BUTTON_START,
        sdl2.SDLK_RSHIFT: WindowEvent.PRESS_BUTTON_SELECT,
    }
    key_up = {
        sdl2.SDLK_UP: WindowEvent.RELEASE_ARROW_UP,
        sdl2.SDLK_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
        sdl2.SDLK_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
        sdl2.SDLK_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
        sdl2.SDLK_z: WindowEvent.RELEASE_BUTTON_A,
        sdl2.SDLK_x: WindowEvent.RELEASE_BUTTON_B,
        sdl2.SDLK_RETURN: WindowEvent.RELEASE_BUTTON_START,
        sdl2.SDLK_RSHIFT: WindowEvent.RELEASE_BUTTON_SELECT,
    }

    running = True
    while running:
        for event in get_events():
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
            if event.type == sdl2.SDL_KEYDOWN:
                key = event.key.keysym.sym
                if key == sdl2.SDLK_ESCAPE:
                    running = False
                    break
                if key == sdl2.SDLK_F5:
                    with save_state.open("wb") as f:
                        pyboy.save_state(f)
                    print(f"Saved state to {save_state}")
                    continue
                if key == sdl2.SDLK_F9:
                    if load_state.exists():
                        with load_state.open("rb") as f:
                            pyboy.load_state(f)
                        print(f"Loaded state from {load_state}")
                    else:
                        print(f"Load failed, file not found: {load_state}")
                    continue
                mapped = key_down.get(key)
                if mapped is not None:
                    pyboy.send_input(mapped)
            elif event.type == sdl2.SDL_KEYUP:
                key = event.key.keysym.sym
                mapped = key_up.get(key)
                if mapped is not None:
                    pyboy.send_input(mapped)

        if not pyboy.tick():
            break

    pyboy.stop(save=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
