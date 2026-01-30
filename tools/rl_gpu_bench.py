#!/usr/bin/env python3
"""Benchmark GPU RL loop components with JSON output."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _require_torch():
    import importlib

    return importlib.import_module("torch")


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="red.gb", help="Path to ROM")
    parser.add_argument(
        "--state", default="states/rl_stage1_exit_oak/start.state", help="State path"
    )
    parser.add_argument(
        "--goal-dir", default="states/rl_stage1_exit_oak", help="Goal template dir"
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    return parser.parse_args()


def _bench_loop(env, model, obs, actions, iters: int):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.no_grad():
            logits, values = model(obs)
            actions_i64 = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            ).squeeze(1)
            actions.copy_(actions_i64.to(torch.int32))
        env.backend.step_torch(actions)
        env.backend.render_pixels_snapshot_torch()
        obs = env.backend.pixels_torch().unsqueeze(1)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms


def _bench_component(fn, iters: int):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main() -> int:
    args = _parse_args()
    if not _cuda_available():
        print(json.dumps({"skipped": "CUDA not available"}))
        return 0

    torch = _require_torch()
    if not torch.cuda.is_available():
        print(json.dumps({"skipped": "torch CUDA not available"}))
        return 0

    rom_path = Path(args.rom)
    state_path = Path(args.state)
    goal_dir = Path(args.goal_dir)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not goal_dir.exists():
        raise FileNotFoundError(f"Goal dir not found: {goal_dir}")

    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv

    env = PokeredPixelsGoalEnv(
        str(rom_path),
        state_path=str(state_path),
        goal_dir=str(goal_dir),
        num_envs=args.num_envs,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        max_steps=500,
    )
    env.reset(seed=1234)

    model = PixelActorCriticCNN(num_actions=env.backend.num_actions, in_frames=1).to(
        "cuda"
    )

    env.backend.render_pixels_snapshot_torch()
    obs = env.backend.pixels_torch().unsqueeze(1)

    actions = torch.zeros((args.num_envs,), dtype=torch.int32, device="cuda")

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            logits, values = model(obs)
            actions_i64 = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            ).squeeze(1)
            actions.copy_(actions_i64.to(torch.int32))
        env.backend.step_torch(actions)
        env.backend.render_pixels_snapshot_torch()
        obs = env.backend.pixels_torch().unsqueeze(1)
    torch.cuda.synchronize()

    iters = int(args.iters)

    model_ms = _bench_component(
        lambda: model(obs)[0],
        iters,
    )

    sample_ms = _bench_component(
        lambda: torch.multinomial(torch.softmax(model(obs)[0], dim=-1), 1),
        iters,
    )

    step_ms = _bench_component(lambda: env.backend.step_torch(actions), iters)
    render_ms = _bench_component(
        lambda: env.backend.render_pixels_snapshot_torch(), iters
    )
    pixels_ms = _bench_component(lambda: env.backend.pixels_torch().unsqueeze(1), iters)

    full_ms = _bench_loop(env, model, obs, actions, iters)
    sps = float(args.num_envs * iters) / (full_ms / 1000.0)

    payload = {
        "num_envs": int(args.num_envs),
        "frames_per_step": int(args.frames_per_step),
        "release_after_frames": int(args.release_after_frames),
        "iters": iters,
        "warmup": int(args.warmup),
        "model_ms": float(model_ms),
        "action_sample_ms": float(sample_ms),
        "env_step_ms": float(step_ms),
        "render_ms": float(render_ms),
        "pixels_ms": float(pixels_ms),
        "full_step_ms": float(full_ms),
        "sps": float(sps),
        "timestamp_s": time.time(),
    }

    output = json.dumps(payload)
    print(output)

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
