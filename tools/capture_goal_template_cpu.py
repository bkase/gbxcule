#!/usr/bin/env python3
"""Capture a pixels-only goal template using the CPU Warp backend."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W
from gbxcule.rl.goal_template import (
    GOAL_TEMPLATE_SCHEMA_VERSION,
    GoalTemplateMeta,
    compute_sha256,
    load_actions_trace_jsonl,
    save_goal_template,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", required=True, help="Path to .state file")
    parser.add_argument("--actions", required=True, help="Path to actions.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--stack-k", type=int, default=4)
    parser.add_argument("--action-codec", default=None)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--k-consecutive", type=int, default=2)
    parser.add_argument("--dist-metric", default="l1_mean_norm")
    parser.add_argument("--pipeline-version", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug-png", default=None)
    return parser.parse_args()


def _infer_num_envs(actions: list[list[int]]) -> int:
    sizes = {len(step) for step in actions}
    if len(sizes) != 1:
        raise ValueError(f"Inconsistent action widths: {sorted(sizes)}")
    num_envs = next(iter(sizes))
    if num_envs < 1:
        raise ValueError("actions trace must include at least one env")
    return num_envs


def _write_debug_png(template: np.ndarray, output: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    frame = template[-1] if template.ndim == 3 else template
    palette = np.array([255, 170, 85, 0], dtype=np.uint8)
    img = Image.fromarray(palette[frame], mode="L")
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> int:
    args = _parse_args()
    rom_path = Path(args.rom)
    state_path = Path(args.state)
    actions_path = Path(args.actions)
    output_dir = Path(args.output_dir)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions not found: {actions_path}")

    actions = load_actions_trace_jsonl(actions_path)
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
        backend.load_state_file(str(state_path), env_idx=0)
        for step_actions in actions:
            act = np.asarray(step_actions, dtype=np.int32)
            backend.step(act)
        backend.render_pixels_snapshot()
        pix = backend.pixels_wp().numpy()
        template = pix.reshape(num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W)[0].copy()
        action_codec_id = backend.action_codec.id
    finally:
        backend.close()

    meta = GoalTemplateMeta(
        schema_version=GOAL_TEMPLATE_SCHEMA_VERSION,
        created_at=GoalTemplateMeta.now_iso(),
        rom_path=str(rom_path),
        rom_sha256=compute_sha256(rom_path),
        state_path=str(state_path),
        state_sha256=compute_sha256(state_path),
        actions_path=str(actions_path),
        actions_sha256=compute_sha256(actions_path),
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

    save_goal_template(output_dir, template, meta, force=args.force)
    if args.debug_png:
        _write_debug_png(template, Path(args.debug_png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
