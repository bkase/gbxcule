#!/usr/bin/env python3
"""Capture a Tetris PyBoy state right as the first piece starts falling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pyboy import PyBoy
from pyboy.plugins.game_wrapper_tetris import GameWrapperTetris


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a Tetris .state at the first falling piece."
    )
    parser.add_argument(
        "--rom",
        type=Path,
        default=Path("tetris.gb"),
        help="Path to Tetris ROM.",
    )
    parser.add_argument(
        "--bootrom",
        type=Path,
        default=Path("bench/roms/bootrom_fast_dmg.bin"),
        help="Path to boot ROM (fast DMG).",
    )
    parser.add_argument(
        "--no-bootrom",
        action="store_true",
        help="Disable boot ROM (use PyBoy default init state).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("states/tetris_start.state"),
        help="Output .state path.",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=Path("states/tetris_start.png"),
        help="Screenshot path captured at the saved state.",
    )
    parser.add_argument(
        "--screenshot-pre",
        type=Path,
        default=Path("states/tetris_start_pre.png"),
        help="Screenshot path captured just after game start.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("states/tetris_start.meta.json"),
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--boot-frames",
        type=int,
        default=120,
        help="Frames to advance before first Start press.",
    )
    parser.add_argument(
        "--menu-frames",
        type=int,
        default=60,
        help="Frames to advance after first Start (menu settle).",
    )
    parser.add_argument(
        "--post-start-frames",
        type=int,
        default=30,
        help="Frames to advance after second Start (game init).",
    )
    parser.add_argument(
        "--press-frames",
        type=int,
        default=5,
        help="Frames to hold the Start button.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Max frames to wait for the first falling movement.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.rom.exists():
        raise FileNotFoundError(f"ROM not found: {args.rom}")
    if args.no_bootrom:
        bootrom_path: Path | None = None
    else:
        bootrom_path = args.bootrom
        if not bootrom_path.exists():
            raise FileNotFoundError(f"Boot ROM not found: {bootrom_path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)

    def _make_pyboy(bootrom: Path | None) -> PyBoy:
        if bootrom is None:
            pyboy = PyBoy(
                str(args.rom),
                window="null",
                sound_emulated=False,
            )
        else:
            pyboy = PyBoy(
                str(args.rom),
                window="null",
                sound_emulated=False,
                bootrom=str(bootrom),
            )
        pyboy.set_emulation_speed(0)
        return pyboy

    pyboy = _make_pyboy(bootrom_path)
    wrapper = pyboy.game_wrapper
    if not isinstance(wrapper, GameWrapperTetris):
        raise RuntimeError("PyBoy did not load the Tetris game wrapper.")

    ticks = 0
    bootrom_used = bootrom_path is not None

    def _tick(frames: int) -> None:
        nonlocal ticks
        for _ in range(frames):
            pyboy.tick(render=True)
            ticks += 1

    # Boot sequence
    _tick(args.boot_frames)
    if bootrom_path is not None and (pyboy.memory[0xFF40] & 0x80) == 0:
        # LCD stayed off after boot; restart without boot ROM to allow rendering.
        pyboy.stop(save=False)
        bootrom_path = None
        bootrom_used = False
        pyboy = _make_pyboy(None)
        wrapper = pyboy.game_wrapper
        if not isinstance(wrapper, GameWrapperTetris):
            raise RuntimeError("PyBoy did not load the Tetris game wrapper.")
        ticks = 0
        _tick(args.boot_frames)

    # Enter menu
    pyboy.button_press("start")
    _tick(args.press_frames)
    pyboy.button_release("start")
    _tick(args.menu_frames)

    # Start game
    pyboy.button_press("start")
    _tick(args.press_frames)
    pyboy.button_release("start")
    _tick(args.post_start_frames)

    def _screen_is_blank() -> bool:
        frame = np.asarray(pyboy.screen.image, dtype=np.uint8)
        return frame.size == 0 or np.all(frame == frame.flat[0])

    # Ensure LCD is on before capturing.
    for _ in range(args.max_frames):
        if (pyboy.memory[0xFF40] & 0x80) != 0 and not _screen_is_blank():
            break
        _tick(1)

    # Screenshot right after game start (before first fall).
    args.screenshot_pre.parent.mkdir(parents=True, exist_ok=True)
    pyboy.screen.image.save(args.screenshot_pre)

    prev_frame = np.asarray(pyboy.screen.image, dtype=np.uint8)
    fall_tick: int | None = None
    for _ in range(args.max_frames):
        _tick(1)
        if (pyboy.memory[0xFF40] & 0x80) == 0 or _screen_is_blank():
            prev_frame = np.asarray(pyboy.screen.image, dtype=np.uint8)
            continue
        frame = np.asarray(pyboy.screen.image, dtype=np.uint8)
        if not np.array_equal(frame, prev_frame):
            fall_tick = ticks
            break
        prev_frame = frame

    # Screenshot and save state at the first detected falling step
    # (or fallback to current frame).
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    pyboy.screen.image.save(args.screenshot)
    with args.out.open("wb") as f:
        pyboy.save_state(f)

    meta = {
        "rom": str(args.rom),
        "bootrom": str(args.bootrom),
        "ticks_total": ticks,
        "fall_tick": fall_tick,
        "boot_frames": args.boot_frames,
        "menu_frames": args.menu_frames,
        "post_start_frames": args.post_start_frames,
        "press_frames": args.press_frames,
        "max_frames": args.max_frames,
        "screenshot": str(args.screenshot),
        "screenshot_pre": str(args.screenshot_pre),
        "bootrom_used": bootrom_used,
    }
    args.meta.write_text(json.dumps(meta, indent=2) + "\n")

    pyboy.stop(save=False)

    print(f"Saved state to {args.out} after {ticks} frames (fall={fall_tick}).")


if __name__ == "__main__":
    main()
