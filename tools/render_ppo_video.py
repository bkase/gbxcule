#!/usr/bin/env python3
"""Render a video of the PPO parcel model playing Pokemon Red."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--rom", default="red.gb", help="Path to ROM")
    parser.add_argument(
        "--state",
        default="states/rl_stage1_exit_oak_start.state",
        help="Starting state",
    )
    parser.add_argument("--steps", type=int, default=2048, help="Number of steps")
    parser.add_argument("--output", default="ppo_rollout.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax actions")
    args = parser.parse_args()

    import torch

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.rl.dual_lobe_model import DualLobeActorCritic
    from gbxcule.rl.pokered_packed_parcel_env import (
        EVENTS_LENGTH,
        PokeredPackedParcelEnv,
        SENSES_DIM,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cuda")

    print("Creating environment (single env for video)...")
    env = PokeredPackedParcelEnv(
        args.rom,
        state_path=args.state,
        num_envs=1,
        max_steps=args.steps + 100,
        info_mode="full",
    )

    print("Creating model...")
    model = DualLobeActorCritic(
        num_actions=env.num_actions,
        senses_dim=SENSES_DIM,
        events_dim=EVENTS_LENGTH,
    ).to("cuda")
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Rolling out {args.steps} steps...")
    obs = env.reset()
    pixels = obs["pixels"]
    senses = obs["senses"]
    events = obs["events"]

    frames = []
    rewards = []
    total_reward = 0.0

    # Get unpacked pixels for video
    from gbxcule.rl.packed_pixels import get_unpack_lut

    lut = get_unpack_lut(device="cuda", dtype=torch.uint8)

    for step in range(args.steps):
        with torch.no_grad():
            logits, values = model(pixels, senses, events)

            if args.deterministic:
                actions = logits.argmax(dim=-1).to(torch.int32)
            else:
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(1).to(torch.int32)

        # Unpack pixels for frame
        unpacked = lut[pixels.to(torch.int64)].reshape(1, 1, 72, 80)
        frame = unpacked[0, 0].cpu().numpy()
        # Scale to 0-255 grayscale
        frame = (frame.astype(np.float32) / 3.0 * 255).astype(np.uint8)
        frames.append(frame)

        # Step
        next_obs, reward, terminated, truncated, info = env.step(actions)
        done = terminated | truncated

        r = reward[0].item()
        rewards.append(r)
        total_reward += r

        if step % 200 == 0:
            map_id = info.get("map_id", torch.tensor([0]))[0].item()
            x = info.get("x", torch.tensor([0]))[0].item()
            y = info.get("y", torch.tensor([0]))[0].item()
            print(
                f"  Step {step}: map={int(map_id)}, pos=({int(x)},{int(y)}), "
                f"reward={r:.4f}, total={total_reward:.2f}"
            )

        if done.any():
            print(f"  Episode done at step {step}")
            env.reset_mask(done)

        pixels = next_obs["pixels"]
        senses = next_obs["senses"]
        events = next_obs["events"]

    env.close()

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Mean reward per step: {total_reward / args.steps:.4f}")

    # Write video
    print(f"\nWriting video to {args.output}...")
    try:
        import cv2

        # Upscale frames for visibility (4x)
        scale = 4
        h, w = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.output, fourcc, args.fps, (w * scale, h * scale), False)

        for frame in frames:
            upscaled = cv2.resize(
                frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
            )
            out.write(upscaled)

        out.release()
        print(f"Video saved: {args.output}")

    except ImportError:
        print("cv2 not available, saving as numpy array instead")
        np.savez(args.output.replace(".mp4", ".npz"), frames=np.array(frames))
        print(f"Frames saved: {args.output.replace('.mp4', '.npz')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
