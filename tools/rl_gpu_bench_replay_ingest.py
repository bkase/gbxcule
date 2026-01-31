#!/usr/bin/env python3
"""Profile CUDA replay ingestion for Dreamer v3 (M6)."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from gbxcule.rl.dreamer_v3.cuda_guard import find_host_memcpy_events, find_memcpy_events
from gbxcule.rl.dreamer_v3.ingest_cuda import ReplayIngestorCUDA
from gbxcule.rl.dreamer_v3.replay_commit import ReplayCommitManager
from gbxcule.rl.dreamer_v3.replay_cuda import ReplayRingCUDA


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
    parser.add_argument("--rom", default="bench/roms/out/BG_STATIC.gb")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--commit-stride", type=int, default=8)
    parser.add_argument("--safety-margin", type=int, default=64)
    parser.add_argument("--profile", action="store_true", help="Enable profiler")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if HtoD/DtoH memcpy events are detected",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not _cuda_available():
        return 0

    torch = _require_torch()
    if not torch.cuda.is_available():
        return 0

    rom_path = Path(args.rom)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")

    from gbxcule.backends.warp_vec import WarpVecCudaBackend

    backend = WarpVecCudaBackend(
        str(rom_path),
        num_envs=args.num_envs,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels_packed=True,
    )
    ring = ReplayRingCUDA(
        capacity=max(args.steps + 2, args.commit_stride * 2),
        num_envs=args.num_envs,
        device="cuda",
    )
    commit = ReplayCommitManager(
        commit_stride=args.commit_stride,
        safety_margin=args.safety_margin,
        device="cuda",
    )
    ingestor = ReplayIngestorCUDA(ring, commit)
    action = torch.zeros((args.num_envs,), dtype=torch.int32, device="cuda")
    reward = torch.zeros((args.num_envs,), dtype=torch.float32, device="cuda")
    terminated = torch.zeros((args.num_envs,), dtype=torch.bool, device="cuda")
    truncated = torch.zeros((args.num_envs,), dtype=torch.bool, device="cuda")

    def _render(slot):  # type: ignore[no-untyped-def]
        backend.render_pixels_snapshot_packed_to_torch(slot, 0)

    def _loop() -> None:
        for _ in range(args.steps):
            ingestor.commit_action(action)
            backend.step_torch(action)
            ingestor.set_next_obs(
                _render,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

    try:
        backend.reset(seed=0)
        ingestor.start(_render)
        start = time.perf_counter()
        if args.profile:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                _loop()
            elapsed = time.perf_counter() - start
            memcpy_events = find_memcpy_events(prof)
            host_events = find_host_memcpy_events(memcpy_events)
            if memcpy_events:
                print("Memcpy events detected:")
                for key in memcpy_events:
                    print(f"  {key}")
            else:
                print("No memcpy events detected.")
            if host_events and args.strict:
                return 1
        else:
            _loop()
            elapsed = time.perf_counter() - start
        sps = args.num_envs * args.steps / elapsed if elapsed > 0 else 0.0
        print(f"ingest_sps={sps:.2f}")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
