#!/usr/bin/env python3
"""Capture a pixels-only goal template from a saved .state file."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W
from gbxcule.rl.goal_template import (
    GOAL_TEMPLATE_SCHEMA_VERSION,
    GoalTemplateMeta,
    compute_sha256,
    save_goal_template,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--goal-state", required=True, help="Path to goal .state file")
    parser.add_argument(
        "--start-state",
        default=None,
        help="Optional start .state file (for start_scaled.png)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--stack-k", type=int, default=1)
    parser.add_argument("--action-codec", default=None)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--k-consecutive", type=int, default=4)
    parser.add_argument("--dist-metric", default="l1_mean_norm")
    parser.add_argument("--pipeline-version", type=int, default=1)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def _render_from_state(
    rom_path: Path,
    state_path: Path,
    *,
    frames_per_step: int,
    release_after_frames: int,
    action_codec: str | None,
) -> tuple[np.ndarray, str]:
    from gbxcule.backends.warp_vec import WarpVecCpuBackend

    backend_kwargs: dict[str, Any] = {}
    if action_codec is not None:
        backend_kwargs["action_codec"] = action_codec

    backend = WarpVecCpuBackend(
        str(rom_path),
        num_envs=1,
        frames_per_step=frames_per_step,
        release_after_frames=release_after_frames,
        obs_dim=32,
        render_pixels=True,
        force_lcdc_on_render=True,
        **backend_kwargs,
    )
    try:
        backend.reset(seed=0)
        backend.load_state_file(str(state_path), env_idx=0)
        backend.render_pixels_snapshot()
        pix = backend.pixels_wp().numpy()
        frame = pix.reshape(1, DOWNSAMPLE_H, DOWNSAMPLE_W)[0].copy()
        return frame, backend.action_codec.id
    finally:
        backend.close()


def _write_scaled_png(frame: np.ndarray, path: Path, *, scale: int) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    palette = np.array([255, 170, 85, 0], dtype=np.uint8)
    img = Image.fromarray(palette[frame], mode="L")
    if scale > 1:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            resample=Image.NEAREST,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _try_preview(paths: Iterable[Path]) -> None:
    previewers = [
        ("chafa", lambda p: ["chafa", str(p)]),
        ("kitty", lambda p: ["kitty", "+kitten", "icat", str(p)]),
        ("imgcat", lambda p: ["imgcat", str(p)]),
    ]
    available = None
    for name, cmd in previewers:
        if shutil.which(name):
            available = cmd
            break
    if available is None:
        for path in paths:
            print(f"PNG: {path}")
        return
    for path in paths:
        subprocess.run(available(path), check=False)


def main() -> int:
    args = _parse_args()
    rom_path = Path(args.rom)
    goal_state = Path(args.goal_state)
    start_state = Path(args.start_state) if args.start_state else None
    output_dir = Path(args.output_dir)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not goal_state.exists():
        raise FileNotFoundError(f"Goal state not found: {goal_state}")
    if start_state is not None and not start_state.exists():
        raise FileNotFoundError(f"Start state not found: {start_state}")

    goal_frame, action_codec_id = _render_from_state(
        rom_path,
        goal_state,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        action_codec=args.action_codec,
    )
    goal_scaled = output_dir / "goal_template_scaled.png"
    _write_scaled_png(goal_frame, goal_scaled, scale=int(args.scale))

    meta = GoalTemplateMeta(
        schema_version=GOAL_TEMPLATE_SCHEMA_VERSION,
        created_at=GoalTemplateMeta.now_iso(),
        rom_path=str(rom_path),
        rom_sha256=compute_sha256(rom_path),
        state_path=str(goal_state),
        state_sha256=compute_sha256(goal_state),
        actions_path="state_only",
        actions_sha256="",
        action_codec_id=action_codec_id,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        downsample_h=DOWNSAMPLE_H,
        downsample_w=DOWNSAMPLE_W,
        stack_k=args.stack_k,
        shade_levels=4,
        dist_metric=args.dist_metric,
        tau=float(args.tau),
        k_consecutive=int(args.k_consecutive),
        pipeline_version=int(args.pipeline_version),
    )

    save_goal_template(output_dir, goal_frame, meta, force=args.force)

    paths = [goal_scaled]
    if start_state is not None:
        start_frame, _ = _render_from_state(
            rom_path,
            start_state,
            frames_per_step=args.frames_per_step,
            release_after_frames=args.release_after_frames,
            action_codec=args.action_codec,
        )
        start_scaled = output_dir / "start_scaled.png"
        _write_scaled_png(start_frame, start_scaled, scale=int(args.scale))
        paths.insert(0, start_scaled)

    if not args.no_preview:
        _try_preview(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
