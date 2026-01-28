#!/usr/bin/env python3
"""Dump a single-env policy rollout to MP4 via FFmpeg."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W


def _require_torch():
    import importlib

    return importlib.import_module("torch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", required=True, help="Path to .state file")
    parser.add_argument("--goal-dir", required=True, help="Goal template directory")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--stack-k", type=int, default=None)
    parser.add_argument("--action-codec", default=None)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument("--output-dir", default="bench/runs/rl_m5_rollout")
    parser.add_argument("--output", default="rollout.mp4")
    return parser.parse_args()


def _write_frame_png(frame: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("PIL is required to write frames") from exc
    palette = np.array([255, 170, 85, 0], dtype=np.uint8)
    img = Image.fromarray(palette[frame], mode="L")
    img.save(path)


def _encode_mp4(frames_dir: Path, fps: int, output: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode MP4")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = _parse_args()
    torch = _require_torch()
    if not torch.cuda.is_available():
        print(json.dumps({"skipped": "torch CUDA not available"}))
        return 0

    rom_path = Path(args.rom)
    state_path = Path(args.state)
    goal_dir = Path(args.goal_dir)
    ckpt_path = Path(args.checkpoint)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not goal_dir.exists():
        raise FileNotFoundError(f"Goal dir not found: {goal_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from gbxcule.rl.goal_template import load_goal_template
    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv

    _, meta = load_goal_template(goal_dir)
    stack_k = args.stack_k if args.stack_k is not None else int(meta.stack_k)
    action_codec = args.action_codec or meta.action_codec_id

    env = PokeredPixelsGoalEnv(
        str(rom_path),
        state_path=str(state_path),
        goal_dir=str(goal_dir),
        num_envs=1,
        frames_per_step=int(args.frames_per_step),
        release_after_frames=int(args.release_after_frames),
        stack_k=int(stack_k),
        action_codec=action_codec,
        info_mode="full",
    )
    model = PixelActorCriticCNN(
        num_actions=env.backend.num_actions, in_frames=env.stack_k
    )
    model.to(device="cuda")
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = output_dir / args.output
    log_path = output_dir / "rollout.jsonl"
    meta_path = output_dir / "meta.json"

    meta_payload = {
        "rom": str(rom_path),
        "state": str(state_path),
        "goal_dir": str(goal_dir),
        "checkpoint": str(ckpt_path),
        "frames_per_step": int(args.frames_per_step),
        "release_after_frames": int(args.release_after_frames),
        "stack_k": int(stack_k),
        "action_codec": action_codec,
        "fps": int(args.fps),
        "steps": int(args.steps),
        "greedy": bool(args.greedy),
        "seed": int(args.seed),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")

    obs = env.reset(seed=int(args.seed))
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        log_path.open("w", encoding="utf-8") as log_f,
    ):
        frames_dir = Path(tmpdir)
        for step_idx in range(int(args.steps)):
            logits, values = model(obs)
            if args.greedy:
                actions_i64 = torch.argmax(logits, dim=-1)
            else:
                actions_i64 = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                ).squeeze(1)
            actions = actions_i64.to(torch.int32)
            obs, reward, done, trunc, info = env.step(actions)
            pix = env.backend.pixels_wp().numpy()
            frame = pix.reshape(1, DOWNSAMPLE_H, DOWNSAMPLE_W)[0].copy()
            _write_frame_png(frame, frames_dir / f"frame_{step_idx:06d}.png")
            dist_val = None
            if isinstance(info, dict) and "dist" in info:
                dist_any: Any = info["dist"]
                if isinstance(dist_any, (int, float)):
                    dist_val = float(dist_any)
                else:
                    try:
                        dist_val = float(dist_any[0].item())
                    except Exception:
                        dist_val = None
            log_f.write(
                json.dumps(
                    {
                        "step": step_idx,
                        "action": int(actions_i64[0].item()),
                        "reward": float(reward[0].item()),
                        "value": float(values[0].item()),
                        "dist": dist_val,
                        "done": bool(done[0].item()),
                        "trunc": bool(trunc[0].item()),
                    }
                )
                + "\n"
            )
            if args.stop_on_done and bool((done | trunc)[0].item()):
                break

        _encode_mp4(frames_dir, int(args.fps), out_mp4)

    env.close()
    print(f"Wrote MP4: {out_mp4}")
    print(f"Wrote log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
