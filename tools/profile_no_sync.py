#!/usr/bin/env python3
"""Profile without per-step sync to match harness behavior."""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gbxcule.rl.models import PixelActorCriticCNN
from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv

ROM = "red.gb"
NUM_ENVS = 8192
ITERS = 20


def main():
    print(f"Profiling WITHOUT per-step sync: {NUM_ENVS} envs, {ITERS} iters")

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

    model = PixelActorCriticCNN(num_actions=env.backend.num_actions, in_frames=1)
    model.to("cuda")

    env.backend.render_pixels_snapshot_torch()
    obs = env.backend.pixels_torch().unsqueeze(1)

    # Warmup
    actions = torch.randint(
        0, env.backend.num_actions, (NUM_ENVS,), dtype=torch.int32, device="cuda"
    )
    for _ in range(5):
        with torch.no_grad():
            logits, values = model(obs)
        env.backend.step_torch(actions)
        env.backend.render_pixels_snapshot_torch()
        obs = env.backend.pixels_torch().unsqueeze(1)

    # Benchmark - NO per-step sync (like harness)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(ITERS):
        with torch.no_grad():
            logits, values = model(obs)
            actions_i64 = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            ).squeeze(1)
        actions = actions_i64.to(torch.int32)
        env.backend.step_torch(actions)
        env.backend.render_pixels_snapshot_torch()
        obs = env.backend.pixels_torch().unsqueeze(1)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sps = NUM_ENVS * ITERS / elapsed
    print(f"Time: {elapsed:.3f}s")
    print(f"SPS: {sps:,.0f}")
    print(f"Per-step: {elapsed / ITERS * 1000:.1f}ms")

    env.close()


if __name__ == "__main__":
    main()
