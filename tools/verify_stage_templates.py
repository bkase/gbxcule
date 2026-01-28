#!/usr/bin/env python3
"""Verify stage goal templates against manifest and basic distance gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W
from gbxcule.rl.goal_template import GoalTemplateMeta, load_goal_template


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="states/rl_stages.json")
    parser.add_argument("--rom", default="red.gb")
    parser.add_argument("--include-pending", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--start-margin", type=float, default=0.01)
    parser.add_argument("--approx-steps", type=int, default=3)
    parser.add_argument("--no-approx", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _render_with_noop_steps(
    rom_path: Path,
    state_path: Path,
    *,
    frames_per_step: int,
    release_after_frames: int,
    action_codec: str | None,
    steps: int,
) -> np.ndarray:
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
        if steps > 0:
            actions = np.zeros((1,), dtype=np.int32)
            for _ in range(steps):
                backend.step(actions)
        backend.render_pixels_snapshot()
        pix = backend.pixels_wp().numpy()
        frame = pix.reshape(1, DOWNSAMPLE_H, DOWNSAMPLE_W)[0].copy()
        return frame
    finally:
        backend.close()


def _dist_l1_mean(frame: np.ndarray, goal: np.ndarray) -> float:
    if frame.dtype != np.uint8 or goal.dtype != np.uint8:
        raise ValueError("frame and goal must be uint8")
    if goal.ndim == 2 and frame.ndim == 2:
        sel_frame = frame
        sel_goal = goal
    elif goal.ndim == 2 and frame.ndim == 3:
        sel_frame = frame[-1]
        sel_goal = goal
    elif goal.ndim == 3 and frame.ndim == 3:
        sel_frame = frame
        sel_goal = goal
    else:
        raise ValueError(f"Unsupported shapes frame={frame.shape} goal={goal.shape}")
    diff = np.abs(sel_frame.astype(np.float32) - sel_goal.astype(np.float32))
    return float(diff.mean() / 3.0)


def _meta_to_dict(meta: GoalTemplateMeta) -> dict[str, Any]:
    data = asdict(meta)
    data["tau"] = float(meta.tau)
    return data


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    rom_path = Path(args.rom)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")

    manifest = _load_manifest(manifest_path)
    abi = manifest.get("abi", {})
    stages = manifest.get("stages", [])
    results: list[dict[str, Any]] = []
    ok = True

    for stage in stages:
        status = stage.get("status", "ready")
        if status == "pending" and not args.include_pending:
            continue
        stage_id = stage.get("id")
        goal_dir = Path(stage["goal_dir"])
        start_state = Path(stage["start_state"])
        goal_state = Path(stage["goal_source_state"])
        entry: dict[str, Any] = {"id": stage_id, "ok": True, "checks": {}}

        missing = []
        for path in [goal_dir, start_state, goal_state]:
            if not path.exists():
                missing.append(str(path))
        if missing:
            entry["ok"] = False
            entry["checks"]["missing"] = missing
            results.append(entry)
            ok = False
            continue

        template, meta = load_goal_template(goal_dir)
        meta_dict = _meta_to_dict(meta)
        entry["meta"] = meta_dict

        def _abi_mismatch(key: str, expected: Any, actual: Any) -> None:
            entry["checks"].setdefault("abi_mismatch", []).append(
                {"key": key, "expected": expected, "actual": actual}
            )

        if abi:
            for key, expected in abi.items():
                actual = meta_dict.get(key)
                if actual != expected:
                    _abi_mismatch(key, expected, actual)

        goal_frame, codec_id = _render_from_state(
            rom_path,
            goal_state,
            frames_per_step=meta.frames_per_step,
            release_after_frames=meta.release_after_frames,
            action_codec=meta.action_codec_id,
        )
        if codec_id != meta.action_codec_id:
            entry["checks"].setdefault("codec_mismatch", []).append(
                {"expected": meta.action_codec_id, "actual": codec_id}
            )

        goal_dist = _dist_l1_mean(goal_frame, template)
        entry["checks"]["goal_dist"] = goal_dist
        if goal_dist > float(args.epsilon):
            entry["checks"]["goal_dist_ok"] = False
            entry["ok"] = False
        else:
            entry["checks"]["goal_dist_ok"] = True

        start_frame, _ = _render_from_state(
            rom_path,
            start_state,
            frames_per_step=meta.frames_per_step,
            release_after_frames=meta.release_after_frames,
            action_codec=meta.action_codec_id,
        )
        start_dist = _dist_l1_mean(start_frame, template)
        entry["checks"]["start_dist"] = start_dist
        min_start = float(meta.tau) + float(args.start_margin)
        entry["checks"]["start_min"] = min_start
        if start_dist <= min_start:
            entry["checks"]["start_dist_ok"] = False
            entry["ok"] = False
        else:
            entry["checks"]["start_dist_ok"] = True

        if not args.no_approx and int(args.approx_steps) > 0:
            approx_ok = True
            approx_samples: list[float] = []
            for step_idx in range(int(args.approx_steps)):
                frame = _render_with_noop_steps(
                    rom_path,
                    goal_state,
                    frames_per_step=meta.frames_per_step,
                    release_after_frames=meta.release_after_frames,
                    action_codec=meta.action_codec_id,
                    steps=step_idx + 1,
                )
                dist = _dist_l1_mean(frame, template)
                approx_samples.append(dist)
                if dist > float(meta.tau):
                    approx_ok = False
            entry["checks"]["approx_dists"] = approx_samples
            entry["checks"]["approx_ok"] = approx_ok
            if not approx_ok:
                entry["ok"] = False

        results.append(entry)
        if not entry["ok"]:
            ok = False

    if args.json:
        print(json.dumps({"ok": ok, "results": results}, indent=2))
    else:
        for entry in results:
            status = "ok" if entry["ok"] else "FAIL"
            print(f"{entry['id']}: {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
