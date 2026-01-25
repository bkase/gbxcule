#!/usr/bin/env python3
"""Verify Warp vs PyBoy parity from a saved Tetris state with random actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from pyboy import PyBoy

from gbxcule.backends.common import flags_from_f, resolve_action_codec
from gbxcule.backends.warp_vec import WarpVecCpuBackend
from gbxcule.core.action_schedule import split_press_release_ticks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare WarpVec CPU backend vs PyBoy from a saved state."
    )
    parser.add_argument(
        "--rom",
        type=Path,
        default=Path("tetris.gb"),
        help="Path to Tetris ROM.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("states/tetris_start.state"),
        help="Path to PyBoy .state file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of random action steps to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for action sampling.",
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=24,
        help="Frames per step for both backends.",
    )
    parser.add_argument(
        "--release-after-frames",
        type=int,
        default=8,
        help="Frames to hold a button before release.",
    )
    parser.add_argument(
        "--actions-out",
        type=Path,
        default=Path("states/tetris_actions.jsonl"),
        help="Where to write sampled actions (jsonl).",
    )
    return parser.parse_args()


FlagsDict = dict[str, int]
CpuStateDict = dict[str, int | FlagsDict]


def _pyboy_step(
    pyboy: Any,
    button: str | None,
    frames_per_step: int,
    release_after_frames: int,
) -> None:
    if button is None:
        pyboy.tick(frames_per_step, False)
        return

    pressed_ticks, remaining_ticks = split_press_release_ticks(
        frames_per_step, release_after_frames
    )
    pyboy.button_press(button)
    for _ in range(pressed_ticks):
        pyboy.tick(1, False)
    pyboy.button_release(button)
    for _ in range(remaining_ticks):
        pyboy.tick(1, False)


def _pyboy_cpu_state(pyboy: Any) -> CpuStateDict:
    reg = pyboy.register_file
    pc = int(reg.PC) & 0xFFFF
    sp = int(reg.SP) & 0xFFFF
    a = int(reg.A) & 0xFF
    f = int(reg.F) & 0xF0
    b = int(reg.B) & 0xFF
    c = int(reg.C) & 0xFF
    d = int(reg.D) & 0xFF
    e = int(reg.E) & 0xFF
    hl = int(reg.HL) & 0xFFFF
    h = (hl >> 8) & 0xFF
    l = hl & 0xFF  # noqa: E741 - canonical register name
    flags = cast(FlagsDict, dict(flags_from_f(f)))
    return {
        "pc": pc,
        "sp": sp,
        "a": a,
        "f": f,
        "b": b,
        "c": c,
        "d": d,
        "e": e,
        "h": h,
        "l": l,
        "flags": flags,
    }


def _warp_cpu_state(warp: WarpVecCpuBackend) -> CpuStateDict:
    state = warp.get_cpu_state(0)
    flags = cast(FlagsDict, dict(state["flags"]))
    return {
        "pc": int(state["pc"]),
        "sp": int(state["sp"]),
        "a": int(state["a"]),
        "f": int(state["f"]),
        "b": int(state["b"]),
        "c": int(state["c"]),
        "d": int(state["d"]),
        "e": int(state["e"]),
        "h": int(state["h"]),
        "l": int(state["l"]),
        "flags": flags,
    }


def _diff_states(ref: CpuStateDict, dut: CpuStateDict) -> dict[str, object] | None:
    diff: dict[str, object] = {}
    for key in ("pc", "sp", "a", "f", "b", "c", "d", "e", "h", "l"):
        if ref[key] != dut[key]:
            diff[key] = {"ref": ref[key], "dut": dut[key]}
    ref_flags = cast(FlagsDict, ref["flags"])
    dut_flags = cast(FlagsDict, dut["flags"])
    flag_diff: dict[str, object] = {}
    for flag in ("z", "n", "h", "c"):
        if ref_flags[flag] != dut_flags[flag]:
            flag_diff[flag] = {"ref": ref_flags[flag], "dut": dut_flags[flag]}
    if flag_diff:
        diff["flags"] = flag_diff
    return diff or None


def main() -> None:
    args = _parse_args()
    if not args.rom.exists():
        raise FileNotFoundError(f"ROM not found: {args.rom}")
    if not args.state.exists():
        raise FileNotFoundError(f"State not found: {args.state}")

    codec = resolve_action_codec()
    rng = np.random.default_rng(args.seed)
    actions = rng.integers(0, codec.num_actions, size=args.steps, dtype=np.int32)

    args.actions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.actions_out.open("w") as f:
        for action in actions:
            f.write(json.dumps({"action": int(action)}) + "\n")

    warp = WarpVecCpuBackend(
        str(args.rom),
        num_envs=1,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        stage="emulate_only",
    )
    warp.reset()
    warp.load_state_file(str(args.state), env_idx=0)

    pyboy = PyBoy(str(args.rom), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)
    with args.state.open("rb") as f:
        pyboy.load_state(f)

    mismatch = None
    try:
        for step, action in enumerate(actions, start=1):
            button = codec.to_pyboy_button(int(action))
            _pyboy_step(
                pyboy,
                button,
                args.frames_per_step,
                args.release_after_frames,
            )
            warp.step(np.array([action], dtype=np.int32))

            ref_state = _pyboy_cpu_state(pyboy)
            dut_state = _warp_cpu_state(warp)
            diff = _diff_states(ref_state, dut_state)
            if diff:
                mismatch = {"step": step, "action": int(action), "diff": diff}
                break
    finally:
        pyboy.stop(save=False)
        warp.close()

    if mismatch:
        mismatch_path = Path("states/tetris_parity_mismatch.json")
        mismatch_path.write_text(json.dumps(mismatch, indent=2) + "\n")
        raise SystemExit(
            f"Mismatch at step {mismatch['step']} (action={mismatch['action']}), "
            f"details in {mismatch_path}"
        )

    print(f"Parity OK for {args.steps} steps (seed={args.seed}).")


if __name__ == "__main__":
    main()
