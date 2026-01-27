"""Benchmark CUDA pixel env throughput and allocation stability."""

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
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", default=None, help="Path to .state file")
    parser.add_argument("--goal-dir", default=None, help="Goal template directory")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--stack-k", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--bench-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force-resets", action="store_true")
    parser.add_argument("--skip-reset-if-empty", action="store_true")
    return parser.parse_args()


def _make_env(args):  # type: ignore[no-untyped-def]
    if args.goal_dir is None:
        from gbxcule.rl.pokered_pixels_env import PokeredPixelsEnv

        return PokeredPixelsEnv(
            args.rom,
            num_envs=args.num_envs,
            frames_per_step=args.frames_per_step,
            release_after_frames=args.release_after_frames,
            stack_k=args.stack_k,
        )

    if args.state is None:
        raise ValueError("--state is required when using --goal-dir")
    from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv

    return PokeredPixelsGoalEnv(
        args.rom,
        state_path=args.state,
        goal_dir=args.goal_dir,
        num_envs=args.num_envs,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        stack_k=args.stack_k,
        max_steps=args.max_steps if not args.force_resets else 1,
        skip_reset_if_empty=args.skip_reset_if_empty,
    )


def _run() -> dict:
    args = _parse_args()
    if not _cuda_available():
        return {"skipped": "CUDA not available"}
    torch = _require_torch()
    if not torch.cuda.is_available():
        return {"skipped": "torch CUDA not available"}

    rom_path = Path(args.rom)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if args.state is not None and not Path(args.state).exists():
        raise FileNotFoundError(f"State not found: {args.state}")
    if args.goal_dir is not None and not Path(args.goal_dir).exists():
        raise FileNotFoundError(f"Goal dir not found: {args.goal_dir}")

    env = _make_env(args)
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        obs = env.reset(seed=args.seed)
        actions = torch.zeros((env.num_envs,), device="cuda", dtype=torch.int32)
        for _ in range(args.warmup_steps):
            step_out = env.step(actions)
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
        torch.cuda.synchronize()

        mem_before = torch.cuda.max_memory_allocated()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(args.bench_steps):
            step_out = env.step(actions)
            obs = step_out[0] if isinstance(step_out, tuple) else step_out
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        mem_after = torch.cuda.max_memory_allocated()
        env_steps_per_sec = (
            float(args.bench_steps * env.num_envs) / (elapsed_ms / 1000.0)
        )

        return {
            "num_envs": int(env.num_envs),
            "stack_k": int(args.stack_k),
            "frames_per_step": int(args.frames_per_step),
            "release_after_frames": int(args.release_after_frames),
            "max_steps": int(args.max_steps),
            "warmup_steps": int(args.warmup_steps),
            "bench_steps": int(args.bench_steps),
            "env_steps_per_sec": env_steps_per_sec,
            "ms_per_step": float(elapsed_ms / max(1, args.bench_steps)),
            "max_mem_alloc_before": int(mem_before),
            "max_mem_alloc_after": int(mem_after),
            "max_mem_alloc_delta": int(mem_after - mem_before),
            "force_resets": bool(args.force_resets),
            "skip_reset_if_empty": bool(args.skip_reset_if_empty),
            "goal_dir": args.goal_dir is not None,
        }
    finally:
        env.close()


def main() -> int:
    payload = _run()
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
