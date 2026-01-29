#!/usr/bin/env python3
"""Profile GPU training loop components to find bottleneck."""

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
NUM_ENVS = 512
WARMUP = 5
ITERS = 20


def sync():
    """Force CUDA sync for accurate timing."""
    torch.cuda.synchronize()


def main():
    print(f"Profiling GPU training with {NUM_ENVS} envs...")
    print()

    # Create env
    print("1. Creating environment...")
    t0 = time.perf_counter()
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
    sync()
    print(f"   Env creation: {time.perf_counter() - t0:.3f}s")

    # Create model
    print("2. Creating model...")
    t0 = time.perf_counter()
    model = PixelActorCriticCNN(num_actions=env.backend.num_actions, in_frames=1)
    model.to("cuda")
    sync()
    print(f"   Model creation: {time.perf_counter() - t0:.3f}s")

    # Get initial observation
    print("3. Getting initial obs...")
    t0 = time.perf_counter()
    env.backend.render_pixels_snapshot_torch()
    obs = env.backend.pixels_torch().unsqueeze(1)  # [N, 1, 72, 80]
    sync()
    print(f"   Initial obs: {time.perf_counter() - t0:.3f}s")
    print(f"   Obs shape: {obs.shape}, device: {obs.device}, dtype: {obs.dtype}")
    print()

    # Warmup
    print(f"Warming up ({WARMUP} iters)...")
    actions = torch.randint(
        0, env.backend.num_actions, (NUM_ENVS,), dtype=torch.int32, device="cuda"
    )
    for _ in range(WARMUP):
        with torch.no_grad():
            logits, values = model(obs)
        env.backend.step_torch(actions)
        env.backend.render_pixels_snapshot_torch()
        obs = env.backend.pixels_torch().unsqueeze(1)
    sync()
    print()

    # Benchmark model forward
    print(f"4. Benchmarking model forward ({ITERS} iters)...")
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        with torch.no_grad():
            logits, values = model(obs)
    sync()
    model_time = time.perf_counter() - t0
    model_per_iter = model_time / ITERS * 1000
    print(f"   Model forward: {model_time:.3f}s total, {model_per_iter:.2f}ms/iter")

    # Benchmark action sampling
    print(f"5. Benchmarking action sampling ({ITERS} iters)...")
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        with torch.no_grad():
            logits, values = model(obs)
            actions_i64 = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            ).squeeze(1)
            actions = actions_i64.to(torch.int32)
    sync()
    sample_time = time.perf_counter() - t0
    sample_per_iter = sample_time / ITERS * 1000
    print(f"   Action sampling: {sample_time:.3f}s total, {sample_per_iter:.2f}ms/iter")

    # Benchmark env step
    print(f"6. Benchmarking env step ({ITERS} iters)...")
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        env.backend.step_torch(actions)
    sync()
    step_time = time.perf_counter() - t0
    step_per_iter = step_time / ITERS * 1000
    print(f"   Env step: {step_time:.3f}s total, {step_per_iter:.2f}ms/iter")

    # Benchmark render
    print(f"7. Benchmarking render ({ITERS} iters)...")
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        env.backend.render_pixels_snapshot_torch()
    sync()
    render_time = time.perf_counter() - t0
    render_per_iter = render_time / ITERS * 1000
    print(f"   Render: {render_time:.3f}s total, {render_per_iter:.2f}ms/iter")

    # Benchmark pixels_torch
    print(f"8. Benchmarking pixels_torch ({ITERS} iters)...")
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        obs = env.backend.pixels_torch().unsqueeze(1)
    sync()
    pixels_time = time.perf_counter() - t0
    pixels_per_iter = pixels_time / ITERS * 1000
    print(f"   pixels_torch: {pixels_time:.3f}s total, {pixels_per_iter:.2f}ms/iter")

    # Benchmark full step (like in training loop)
    print(f"9. Benchmarking full step ({ITERS} iters)...")
    sync()
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
    sync()
    full_time = time.perf_counter() - t0
    full_per_iter = full_time / ITERS * 1000
    sps = NUM_ENVS * ITERS / full_time
    print(f"   Full step: {full_time:.3f}s total, {full_per_iter:.2f}ms/iter")
    print(f"   Steps/sec: {sps:,.0f}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model forward:      {model_per_iter:6.2f}ms/iter")
    print(f"Action sampling:    {sample_per_iter:6.2f}ms/iter")
    print(f"Env step:           {step_per_iter:6.2f}ms/iter")
    print(f"Render:             {render_per_iter:6.2f}ms/iter")
    print(f"pixels_torch:       {pixels_per_iter:6.2f}ms/iter")
    print(f"Full step:          {full_per_iter:6.2f}ms/iter")
    print()
    overhead = full_per_iter - (
        sample_per_iter + step_per_iter + render_per_iter + pixels_per_iter
    )
    expected = sample_per_iter + step_per_iter + render_per_iter + pixels_per_iter
    print(f"Expected (sum):     {expected:.2f}ms")
    print(f"Overhead:           {overhead:.2f}ms")
    print()
    print(f"Effective SPS: {sps:,.0f} (target: 50k+)")

    env.close()


if __name__ == "__main__":
    main()
