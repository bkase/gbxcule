#!/usr/bin/env python3
"""Evaluate a trained policy and generate trace for video.

Usage:
  uv run python tools/eval_policy_cpu.py --checkpoint <ckpt> --output-trace <trace.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--output-trace", required=True, help="Output trace JSON path")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps to run")
    parser.add_argument(
        "--deterministic", action="store_true", help="Use greedy actions"
    )
    args = parser.parse_args()

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.backends.warp_vec import WarpVecCpuBackend
    from gbxcule.core.state_io import apply_state_to_warp_backend, load_pyboy_state
    from gbxcule.rl.goal_template import load_goal_template
    from gbxcule.rl.models import PixelActorCriticCNN

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    # Create backend
    print("Creating CPU backend...")
    backend = WarpVecCpuBackend(
        cfg["rom"],
        num_envs=1,
        frames_per_step=cfg["frames_per_step"],
        release_after_frames=cfg["release_after_frames"],
        render_pixels=True,
    )

    # Load goal template
    print(f"Loading goal template from {cfg['goal_dir']}...")
    template, meta = load_goal_template(
        Path(cfg["goal_dir"]),
        action_codec_id=backend.action_codec.id,
        frames_per_step=cfg["frames_per_step"],
        release_after_frames=cfg["release_after_frames"],
        stack_k=1,
        dist_metric=None,
        pipeline_version=None,
    )
    goal_np = np.array(template, dtype=np.float32)
    if goal_np.ndim == 3:
        goal_np = goal_np.squeeze(0)

    # Load initial state
    print(f"Loading initial state from {cfg['state']}...")
    initial_state = load_pyboy_state(cfg["state"], expected_cart_ram_size=32768)

    # Load model
    model = PixelActorCriticCNN(num_actions=backend.num_actions, in_frames=1)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Reset
    backend.reset()
    apply_state_to_warp_backend(initial_state, backend, env_idx=0)

    # Get obs
    def get_obs():
        backend.render_pixels_snapshot()
        pixels_flat = backend.pixels_wp().numpy()
        return pixels_flat.reshape(1, 72, 80)

    def compute_dist(obs_np):
        diff = np.abs(obs_np[0].astype(np.float32) - goal_np)
        return diff.mean() / 3.0

    # Run episode
    obs_np = get_obs()
    actions = []
    tau = cfg.get("tau", 0.05)
    dist = compute_dist(obs_np)  # Initial distance

    print(f"Running evaluation (max {args.max_steps} steps)...")
    for step in range(args.max_steps):
        obs_t = torch.from_numpy(obs_np).unsqueeze(1)  # [1, 1, 72, 80]

        with torch.no_grad():
            logits, _ = model(obs_t)
            if args.deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1).squeeze(1)

        action_np = action.numpy().astype(np.int32)
        actions.append(int(action_np[0]))

        backend.step(action_np)
        obs_np = get_obs()
        dist = compute_dist(obs_np)

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}: dist={dist:.4f}")

        if dist < tau:
            print(f"GOAL reached at step {step + 1}! (dist={dist:.4f})")
            break

    backend.close()

    # Save trace
    trace_data = {
        "start_state": cfg["state"],
        "goal_dir": cfg["goal_dir"],
        "steps": len(actions),
        "final_dist": float(dist),
        "actions": actions,
        "checkpoint": args.checkpoint,
        "deterministic": args.deterministic,
    }

    output_path = Path(args.output_trace)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace_data, f)

    print(f"Trace saved to: {output_path}")
    print(f"  Steps: {len(actions)}")
    print(f"  Final distance: {dist:.4f}")
    print(f"  Goal reached: {dist < tau}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
