#!/usr/bin/env python3
"""Test just env stepping without model to isolate bottleneck."""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv

ROM = "red.gb"
NUM_ENVS = 4096
NUM_STEPS = 128


def main():
    print(f"Testing env-only: {NUM_ENVS} envs, {NUM_STEPS} steps")

    env = PokeredPixelsGoalEnv(
        ROM,
        state_path="states/rl_stage1_exit_oak/start.state",
        goal_dir="states/rl_stage1_exit_oak",
        num_envs=NUM_ENVS,
        frames_per_step=24,
        release_after_frames=8,
        max_steps=500,
    )
    env.reset(seed=1234)
    torch.cuda.synchronize()
    print("Environment ready")

    # Pre-create random actions
    actions = torch.randint(
        0, env.backend.num_actions, (NUM_ENVS,), dtype=torch.int32, device="cuda"
    )

    print(f"Starting {NUM_STEPS} steps...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    for step in range(NUM_STEPS):
        env.backend.step_torch(actions)
        env.backend.render_pixels_snapshot_torch()
        _ = env.backend.pixels_torch()
        if step % 32 == 0:
            print(f"  Step {step}...")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_steps = NUM_ENVS * NUM_STEPS
    sps = total_steps / elapsed
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Total SPS: {sps:,.0f}")
    print(f"Per-step time: {elapsed / NUM_STEPS * 1000:.1f}ms")

    env.close()


if __name__ == "__main__":
    main()
