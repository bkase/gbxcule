#!/usr/bin/env python3
"""Profile matching harness exactly - just step(), no render, no model."""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*cuda capability.*")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gbxcule.backends.warp_vec import WarpVecCudaBackend

ROM = "red.gb"
NUM_ENVS = 8192
ITERS = 20


def main():
    print(f"Profiling HARNESS-MATCH: {NUM_ENVS} envs, {ITERS} iters")

    backend = WarpVecCudaBackend(
        ROM,
        num_envs=NUM_ENVS,
        frames_per_step=24,
        release_after_frames=8,
        stage="emulate_only",
        obs_dim=32,
    )
    backend.reset(seed=1234)
    print("Backend ready")

    actions = np.zeros(NUM_ENVS, dtype=np.int32)

    # Warmup
    for _ in range(5):
        backend.step(actions)

    import warp as wp

    wp.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(ITERS):
        backend.step(actions)
    wp.synchronize()
    elapsed = time.perf_counter() - t0

    sps = NUM_ENVS * ITERS / elapsed
    print(f"Time: {elapsed:.3f}s")
    print(f"SPS: {sps:,.0f}")
    print(f"Per-step: {elapsed / ITERS * 1000:.1f}ms")

    backend.close()


if __name__ == "__main__":
    main()
