#!/usr/bin/env python3
"""Profile a single PPO update and report CUDA memcpy events."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from gbxcule.rl.async_ppo_engine import AsyncPPOEngine, AsyncPPOEngineConfig


def _require_torch():  # type: ignore[no-untyped-def]
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
    parser.add_argument("--obs-format", choices=("u8", "packed2"), default="u8")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps-per-rollout", type=int, default=4)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if HtoD/DtoH memcpy events are detected",
    )
    return parser.parse_args()


def _has_assets(rom_path: Path, state_path: Path, goal_dir: Path) -> None:
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not goal_dir.exists():
        raise FileNotFoundError(f"Goal dir not found: {goal_dir}")


def main() -> int:
    args = _parse_args()
    if not _cuda_available():
        return 0

    torch = _require_torch()
    if not torch.cuda.is_available():
        return 0

    rom_path = Path(args.rom)
    state_path = Path(args.state)
    goal_dir = Path(args.goal_dir)
    _has_assets(rom_path, state_path, goal_dir)

    config = AsyncPPOEngineConfig(
        rom_path=str(rom_path),
        state_path=str(state_path),
        goal_dir=str(goal_dir),
        device="cuda",
        obs_format=args.obs_format,
        num_envs=args.num_envs,
        steps_per_rollout=args.steps_per_rollout,
        updates=args.updates,
        minibatch_size=args.minibatch_size,
    )
    engine = AsyncPPOEngine(config)
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            engine.run(updates=args.updates)

        memcpy_events = []
        for event in prof.key_averages():
            key = event.key
            if "memcpy" in key.lower():
                memcpy_events.append(key)

        if memcpy_events:
            print("Memcpy events detected:")
            for key in memcpy_events:
                print(f"  {key}")
            if args.strict:
                for key in memcpy_events:
                    lowered = key.lower()
                    if "htod" in lowered or "dtoh" in lowered:
                        return 1
        else:
            print("No memcpy events detected.")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
