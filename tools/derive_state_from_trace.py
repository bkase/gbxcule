#!/usr/bin/env python3
"""Derive a new .state file by replaying an action trace from a start state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--start-state", required=True, help="Path to start .state")
    parser.add_argument("--actions", required=True, help="Path to actions.jsonl")
    parser.add_argument("--output-state", required=True, help="Output .state path")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--action-codec", default=None)
    parser.add_argument("--output-png", default=None, help="Optional PNG preview path")
    return parser.parse_args()


def _load_actions_jsonl(path: Path) -> list[list[int]]:
    actions: list[list[int]] = []
    for line_num, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc
        if isinstance(payload, int):
            actions.append([int(payload)])
        elif isinstance(payload, list):
            actions.append([int(x) for x in payload])
        else:
            raise ValueError(f"Unsupported payload at line {line_num}: {payload!r}")
    if not actions:
        raise ValueError(f"No actions found in {path}")
    return actions


def _infer_num_envs(actions: list[list[int]]) -> int:
    sizes = {len(step) for step in actions}
    if len(sizes) != 1:
        raise ValueError(f"Inconsistent action widths: {sorted(sizes)}")
    return next(iter(sizes))


def _write_png(frame: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    palette = np.array([255, 170, 85, 0], dtype=np.uint8)
    img = Image.fromarray(palette[frame], mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> int:
    args = _parse_args()
    rom_path = Path(args.rom)
    start_state = Path(args.start_state)
    actions_path = Path(args.actions)
    output_state = Path(args.output_state)
    output_png = Path(args.output_png) if args.output_png else None

    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not start_state.exists():
        raise FileNotFoundError(f"Start state not found: {start_state}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions not found: {actions_path}")

    actions = _load_actions_jsonl(actions_path)
    num_envs = _infer_num_envs(actions)

    from gbxcule.backends.warp_vec import WarpVecCpuBackend

    backend_kwargs: dict[str, Any] = {}
    if args.action_codec is not None:
        backend_kwargs["action_codec"] = args.action_codec

    backend = WarpVecCpuBackend(
        str(rom_path),
        num_envs=num_envs,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        obs_dim=32,
        render_pixels=True,
        force_lcdc_on_render=True,
        **backend_kwargs,
    )
    try:
        backend.reset(seed=0)
        backend.load_state_file(str(start_state), env_idx=0)
        for step_actions in actions:
            act = np.asarray(step_actions, dtype=np.int32)
            backend.step(act)
        backend.save_state_file(str(output_state), env_idx=0)
        if output_png is not None:
            backend.render_pixels_snapshot()
            pix = backend.pixels_wp().numpy()
            frame = pix.reshape(num_envs, 72, 80)[0].copy()
            _write_png(frame, output_png)
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
