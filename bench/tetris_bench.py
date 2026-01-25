#!/usr/bin/env python3
"""Tetris-specific benchmark: PyBoy baseline vs WarpVec CUDA.

This script benchmarks Tetris specifically, loading saved gameplay state where
possible (WarpVec) or starting fresh (PyBoy puffer vec).

Usage:
    uv run python bench/tetris_bench.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def run_pyboy_baseline(
    rom_path: str,
    env_counts: list[int],
    steps: int,
    warmup_steps: int,
    frames_per_step: int,
    release_after_frames: int,
    output_dir: Path,
    puffer_vec_backend: str = "puffer_mp_sync",
) -> list[dict[str, Any]]:
    """Run PyBoy puffer vec baseline benchmarks."""
    from gbxcule.backends.pyboy_puffer_vec import PyBoyPufferVecBackend

    results = []
    for num_envs in env_counts:
        print(f"  PyBoy puffer vec: {num_envs} envs...", end=" ", flush=True)

        backend = PyBoyPufferVecBackend(
            rom_path,
            num_envs=num_envs,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
            vec_backend=puffer_vec_backend,
        )

        # Reset
        backend.reset(seed=42)

        # Generate random actions
        rng = np.random.default_rng(1234)
        actions = rng.integers(
            0, backend.num_actions, size=(steps + warmup_steps, num_envs)
        ).astype(np.int32)

        # Warmup
        for i in range(warmup_steps):
            backend.step(actions[i])

        # Benchmark
        t0 = time.perf_counter()
        for i in range(warmup_steps, steps + warmup_steps):
            backend.step(actions[i])
        elapsed = time.perf_counter() - t0

        backend.close()

        steps_per_sec = steps / elapsed
        envs_x_steps = num_envs * steps / elapsed
        print(f"{steps_per_sec:.1f} steps/s, {envs_x_steps:.1f} env*steps/s")

        result = {
            "backend": "pyboy_puffer_vec",
            "num_envs": num_envs,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": steps_per_sec,
            "env_steps_per_sec": envs_x_steps,
            "frames_per_step": frames_per_step,
            "release_after_frames": release_after_frames,
        }
        results.append(result)

    return results


def run_warp_cuda_dut(
    rom_path: str,
    state_path: str | None,
    env_counts: list[int],
    steps: int,
    warmup_steps: int,
    frames_per_step: int,
    release_after_frames: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run WarpVec CUDA benchmarks with optional state loading."""
    from gbxcule.backends.warp_vec import WarpVecCudaBackend

    results = []
    for num_envs in env_counts:
        print(f"  WarpVec CUDA: {num_envs} envs...", end=" ", flush=True)

        backend = WarpVecCudaBackend(
            rom_path,
            num_envs=num_envs,
            frames_per_step=frames_per_step,
            release_after_frames=release_after_frames,
        )

        # Reset
        backend.reset(seed=42)

        # Load state into all environments if provided
        if state_path:
            for env_idx in range(num_envs):
                backend.load_state_file(state_path, env_idx=env_idx)

        # Generate random actions
        rng = np.random.default_rng(1234)
        actions = rng.integers(
            0, backend.num_actions, size=(steps + warmup_steps, num_envs)
        ).astype(np.int32)

        # Warmup
        for i in range(warmup_steps):
            backend.step(actions[i])
        # Sync to ensure warmup completes before timing
        backend._wp.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for i in range(warmup_steps, steps + warmup_steps):
            backend.step(actions[i])
        # Sync to ensure all GPU work completes before stopping timer
        backend._wp.synchronize()
        elapsed = time.perf_counter() - t0

        backend.close()

        steps_per_sec = steps / elapsed
        envs_x_steps = num_envs * steps / elapsed
        print(f"{steps_per_sec:.1f} steps/s, {envs_x_steps:.1f} env*steps/s")

        result = {
            "backend": "warp_vec_cuda",
            "num_envs": num_envs,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": steps_per_sec,
            "env_steps_per_sec": envs_x_steps,
            "frames_per_step": frames_per_step,
            "release_after_frames": release_after_frames,
            "state_loaded": state_path is not None,
        }
        results.append(result)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tetris benchmark: PyBoy vs WarpVec CUDA"
    )
    parser.add_argument(
        "--rom", type=str, default="tetris.gb", help="Path to Tetris ROM"
    )
    parser.add_argument(
        "--state",
        type=str,
        default="states/tetris_gameplay.state",
        help="Initial state file for WarpVec",
    )
    parser.add_argument(
        "--baseline-env-counts",
        type=str,
        default="1,8,64,128",
        help="Comma-separated env counts for PyBoy baseline",
    )
    parser.add_argument(
        "--dut-env-counts",
        type=str,
        default="1,8,64,512,2048,8192,16384",
        help="Comma-separated env counts for WarpVec CUDA",
    )
    parser.add_argument("--steps", type=int, default=200, help="Steps per benchmark")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps")
    parser.add_argument(
        "--frames-per-step", type=int, default=24, help="Frames per step"
    )
    parser.add_argument(
        "--release-after-frames",
        type=int,
        default=8,
        help="Release button after N frames",
    )
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument(
        "--puffer-vec-backend",
        type=str,
        default="puffer_mp_sync",
        help="PufferLib vec backend",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip PyBoy baseline"
    )
    parser.add_argument("--skip-dut", action="store_true", help="Skip WarpVec CUDA")
    args = parser.parse_args(argv)

    rom_path = Path(args.rom)
    if not rom_path.exists():
        print(f"Error: ROM not found: {rom_path}", file=sys.stderr)
        return 1

    state_path = args.state
    if state_path and not Path(state_path).exists():
        print(
            f"Warning: State file not found: {state_path}. WarpVec will start fresh.",
            file=sys.stderr,
        )
        state_path = None

    baseline_env_counts = [int(x) for x in args.baseline_env_counts.split(",")]
    dut_env_counts = [int(x) for x in args.dut_env_counts.split(",")]

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("bench/runs/reports") / f"{timestamp}_tetris"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Tetris Benchmark")
    print("================")
    print(f"ROM: {rom_path}")
    print(f"State: {state_path or '(none)'}")
    print(f"Steps: {args.steps} (warmup: {args.warmup_steps})")
    print(f"Frames per step: {args.frames_per_step}")
    print(f"Output: {output_dir}")
    print()

    all_results: list[dict[str, Any]] = []

    # Run PyBoy baseline
    if not args.skip_baseline:
        print("Running PyBoy puffer vec baseline...")
        baseline_results = run_pyboy_baseline(
            str(rom_path),
            baseline_env_counts,
            args.steps,
            args.warmup_steps,
            args.frames_per_step,
            args.release_after_frames,
            output_dir,
            args.puffer_vec_backend,
        )
        all_results.extend(baseline_results)
        print()

    # Run WarpVec CUDA
    if not args.skip_dut:
        print("Running WarpVec CUDA...")
        dut_results = run_warp_cuda_dut(
            str(rom_path),
            state_path,
            dut_env_counts,
            args.steps,
            args.warmup_steps,
            args.frames_per_step,
            args.release_after_frames,
            output_dir,
        )
        all_results.extend(dut_results)
        print()

    # Save results
    results_file = output_dir / "tetris_bench_results.json"
    with results_file.open("w") as f:
        json.dump(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "rom": str(rom_path),
                "state": state_path,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {results_file}")

    # Print summary
    print("\nSummary:")
    print("-" * 70)
    print(f"{'Backend':<20} {'Envs':>8} {'Steps/s':>12} {'Env*Steps/s':>15}")
    print("-" * 70)
    for r in all_results:
        backend = r["backend"]
        envs = r["num_envs"]
        sps = r["steps_per_sec"]
        esps = r["env_steps_per_sec"]
        print(f"{backend:<20} {envs:>8} {sps:>12.1f} {esps:>15.1f}")
    print("-" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
