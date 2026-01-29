#!/usr/bin/env python3
"""Profile with model to measure its overhead."""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gbxcule.backends.warp_vec import WarpVecCudaBackend
from gbxcule.rl.models import PixelActorCriticCNN

ROM = "red.gb"
NUM_ENVS = 8192
ITERS = 20


def main():
    print(f"Profiling WITH MODEL: {NUM_ENVS} envs, {ITERS} iters")

    backend = WarpVecCudaBackend(
        ROM,
        num_envs=NUM_ENVS,
        frames_per_step=24,
        release_after_frames=8,
        stage="emulate_only",
        obs_dim=32,
        render_pixels=True,
    )
    backend.reset(seed=1234)
    print("Backend ready")

    model = PixelActorCriticCNN(num_actions=backend.num_actions, in_frames=1)
    model.to("cuda")
    print("Model ready")

    # Get initial obs
    backend.render_pixels_snapshot_torch()
    obs = backend.pixels_torch().unsqueeze(1)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            logits, values = model(obs)
            actions_i64 = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            ).squeeze(1)
        actions = actions_i64.to(torch.int32)
        backend.step_torch(actions)
        backend.render_pixels_snapshot_torch()
        obs = backend.pixels_torch().unsqueeze(1)

    torch.cuda.synchronize()

    # Benchmark - sync only at end
    t0 = time.perf_counter()
    for _ in range(ITERS):
        with torch.no_grad():
            logits, values = model(obs)
            actions_i64 = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            ).squeeze(1)
        actions = actions_i64.to(torch.int32)
        backend.step_torch(actions)
        backend.render_pixels_snapshot_torch()
        obs = backend.pixels_torch().unsqueeze(1)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sps = NUM_ENVS * ITERS / elapsed
    print(f"Time: {elapsed:.3f}s")
    print(f"SPS: {sps:,.0f}")
    print(f"Per-step: {elapsed / ITERS * 1000:.1f}ms")

    backend.close()


if __name__ == "__main__":
    main()
