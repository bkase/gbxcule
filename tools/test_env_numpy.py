#!/usr/bin/env python3
"""Test env stepping with numpy actions (like benchmark harness)."""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*cuda capability.*")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gbxcule.backends.warp_vec import WarpVecCudaBackend

ROM = "red.gb"
NUM_ENVS = 4096
NUM_STEPS = 128


def main():
    print(f"Testing env-only (numpy): {NUM_ENVS} envs, {NUM_STEPS} steps")

    backend = WarpVecCudaBackend(
        ROM,
        num_envs=NUM_ENVS,
        frames_per_step=24,
        release_after_frames=8,
        obs_dim=32,
    )
    backend.reset(seed=1234)
    print("Backend ready")

    # Pre-create random actions
    actions = np.random.randint(
        0, backend.num_actions, size=(NUM_ENVS,), dtype=np.int32
    )

    print(f"Starting {NUM_STEPS} steps...")

    import warp as wp

    wp.synchronize()
    start = time.perf_counter()

    for step in range(NUM_STEPS):
        backend.step(actions)
        if step % 32 == 0:
            print(f"  Step {step}...")

    wp.synchronize()
    elapsed = time.perf_counter() - start

    total_steps = NUM_ENVS * NUM_STEPS
    sps = total_steps / elapsed
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Total SPS: {sps:,.0f}")
    print(f"Per-step time: {elapsed / NUM_STEPS * 1000:.1f}ms")

    backend.close()


if __name__ == "__main__":
    main()
