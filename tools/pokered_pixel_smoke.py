#!/usr/bin/env python3
"""Pixel smoke test for Pokemon Red (env0 BG shades, 24 frames/step)."""

from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend
from gbxcule.core.abi import SCREEN_H, SCREEN_W

DEFAULT_CONFIG = Path("configs/m0_pokered_smoke.json")
ACTION_MIN = 0
ACTION_MAX = 6


def _blake2b_hex(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _hash_file(path: Path) -> str:
    return _blake2b_hex(path.read_bytes())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_actions_file(path: Path) -> list[int]:
    actions: list[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            actions.append(int(line))
            continue
        if isinstance(payload, int):
            actions.append(payload)
        elif isinstance(payload, dict) and "action" in payload:
            actions.append(int(payload["action"]))
        else:
            raise ValueError(f"Unrecognized action line in {path}: {line}")
    return actions


def _validate_actions(actions: Iterable[int]) -> list[int]:
    out: list[int] = []
    for action in actions:
        if action < ACTION_MIN or action > ACTION_MAX:
            raise ValueError(f"Action {action} out of range [{ACTION_MIN}, {ACTION_MAX}]")
        out.append(int(action))
    if not out:
        raise ValueError("No actions provided")
    return out


def _write_shade_png(shades: np.ndarray, path: Path) -> None:
    palette = np.array(
        [
            [255, 255, 255, 255],
            [170, 170, 170, 255],
            [85, 85, 85, 255],
            [0, 0, 0, 255],
        ],
        dtype=np.uint8,
    )
    rgba = palette[shades]
    Image.fromarray(rgba, mode="RGBA").save(path)


def _resolve_backend(name: str, **kwargs):
    if name == "warp_vec_cpu":
        return WarpVecCpuBackend(**kwargs)
    if name == "warp_vec_cuda":
        try:
            import warp as wp
        except Exception as exc:  # pragma: no cover - optional CUDA path
            raise RuntimeError("Warp is required for CUDA backend") from exc
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("CUDA backend requested but no CUDA device found")
        return WarpVecCudaBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pokemon Red pixel smoke (env0 BG shades)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--backend",
        choices=["warp_vec_cpu", "warp_vec_cuda"],
        default="warp_vec_cpu",
        help="Warp backend to use",
    )
    parser.add_argument("--rom", type=Path, default=None)
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--frames-per-step", type=int, default=None)
    parser.add_argument("--release-after-frames", type=int, default=None)
    parser.add_argument("--actions-file", type=Path, default=None)
    parser.add_argument("--actions-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_json(args.config)

    rom_path = args.rom or Path(cfg.get("rom", "red.gb"))
    state_path = args.state or Path(cfg.get("state", ""))
    action_codec = cfg.get("action_codec", "pokemonred_puffer_v0")
    frames_per_step = args.frames_per_step or int(cfg.get("frames_per_step", 24))
    release_after_frames = args.release_after_frames or int(
        cfg.get("release_after_frames", 8)
    )
    steps = args.steps if args.steps is not None else cfg.get("steps")
    save_every = args.save_every if args.save_every is not None else cfg.get(
        "save_every", 1
    )

    if frames_per_step != 24:
        raise ValueError(
            f"frames_per_step must be 24 for m0, got {frames_per_step}"
        )

    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")

    if args.actions_file:
        actions = _load_actions_file(args.actions_file)
    elif args.actions_seed is not None:
        if steps is None:
            steps = int(cfg.get("steps", 12))
        rng = np.random.default_rng(seed=args.actions_seed)
        actions = rng.integers(ACTION_MIN, ACTION_MAX + 1, size=int(steps)).tolist()
    else:
        actions = list(cfg.get("actions", []))

    actions = _validate_actions(actions)

    if steps is None:
        steps = len(actions)
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if len(actions) < steps:
        raise ValueError(
            f"Not enough actions ({len(actions)}) for steps={steps}; provide more"
        )
    if len(actions) > steps:
        actions = actions[:steps]

    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("bench/runs/m0_smoke") / stamp
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = _resolve_backend(
        args.backend,
        rom_path=str(rom_path),
        num_envs=1,
        frames_per_step=frames_per_step,
        release_after_frames=release_after_frames,
        render_bg=True,
        action_codec=action_codec,
    )

    backend.reset(seed=0)
    backend.load_state_file(str(state_path), env_idx=0)

    meta = {
        "backend": args.backend,
        "rom": str(rom_path),
        "rom_hash": _hash_file(rom_path),
        "state": str(state_path),
        "state_hash": _hash_file(state_path),
        "action_codec": action_codec,
        "frames_per_step": frames_per_step,
        "release_after_frames": release_after_frames,
        "steps": steps,
        "save_every": int(save_every),
        "actions_source": (
            str(args.actions_file)
            if args.actions_file
            else ("seed" if args.actions_seed is not None else "config")
        ),
        "actions_seed": args.actions_seed,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    actions_path = output_dir / "actions.jsonl"
    frames_path = output_dir / "frames.jsonl"

    with actions_path.open("w", encoding="utf-8") as actions_f, frames_path.open(
        "w", encoding="utf-8"
    ) as frames_f:
        for step_idx, action in enumerate(actions):
            actions_f.write(json.dumps({"step": step_idx, "action": action}) + "\n")
            backend.step(np.array([action], dtype=np.int32))
            frame_bytes = backend.read_frame_bg_shade_env0()
            frame_hash = _blake2b_hex(frame_bytes)
            frames_f.write(
                json.dumps(
                    {
                        "step": step_idx,
                        "action": action,
                        "frame_hash": frame_hash,
                    }
                )
                + "\n"
            )

            if save_every and step_idx % int(save_every) == 0:
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                    SCREEN_H, SCREEN_W
                )
                _write_shade_png(frame, output_dir / f"frame_{step_idx:04d}.png")

    backend.close()
    print(f"Wrote artifacts to {output_dir}")


if __name__ == "__main__":
    main()
