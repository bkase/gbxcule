#!/usr/bin/env python3
"""Fast GPU PPO training using reduced frames_per_step for better throughput.

Uses frames_per_step=4 instead of 24 for ~6x speedup.
This means actions are held for ~66ms instead of ~400ms.

Usage:
  uv run python tools/rl_train_gpu_fast.py --rom red.gb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch


@dataclass(frozen=True)
class TrainConfig:
    rom: str
    frames_per_step: int
    release_after_frames: int
    num_envs: int
    steps_per_rollout: int
    updates: int
    lr: float
    gamma: float
    gae_lambda: float
    clip: float
    value_coef: float
    entropy_coef: float
    seed: int
    output_dir: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="red.gb", help="Path to ROM")
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=4,
        help="Frames per step (lower = faster but harder)",
    )
    parser.add_argument("--release-after-frames", type=int, default=2)
    parser.add_argument(
        "--num-envs", type=int, default=4096, help="Parallel environments"
    )
    parser.add_argument("--steps-per-rollout", type=int, default=128)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.backends.warp_vec import WarpVecCudaBackend
    from gbxcule.core.reset_cache import ResetCache
    from gbxcule.rl.goal_template import load_goal_template
    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses
    from gbxcule.rl.rollout import RolloutBuffer

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("bench/runs/rl_gpu_fast") / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    ckpt_path = output_dir / "checkpoint.pt"

    cfg = TrainConfig(
        rom=str(Path(args.rom)),
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        num_envs=args.num_envs,
        steps_per_rollout=args.steps_per_rollout,
        updates=args.updates,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        output_dir=str(output_dir),
    )

    print(f"Config: num_envs={cfg.num_envs}, steps_per_rollout={cfg.steps_per_rollout}")
    print(f"Frames per step: {cfg.frames_per_step} (fast mode)")
    print(f"Output: {output_dir}")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    print(f"Creating GPU backend with {cfg.num_envs} envs...")

    backend = WarpVecCudaBackend(
        cfg.rom,
        num_envs=cfg.num_envs,
        frames_per_step=cfg.frames_per_step,
        release_after_frames=cfg.release_after_frames,
        obs_dim=32,
        render_pixels=True,
    )

    print("Initializing backend...")
    backend.reset(seed=cfg.seed)

    # Load start state and create reset cache
    backend.load_state_file("states/rl_stage1_exit_oak/start.state", env_idx=0)
    reset_cache = ResetCache.from_backend(backend, env_idx=0)
    print("Reset cache created")

    # Apply start state to all envs
    all_mask = torch.ones(cfg.num_envs, dtype=torch.uint8, device="cuda")
    reset_cache.apply_mask_torch(all_mask)
    print("All envs initialized")

    # Load goal template (skip validation - just need the pixels)
    template, meta = load_goal_template(
        Path("states/rl_stage1_exit_oak"),
        action_codec_id=backend.action_codec.id,
        frames_per_step=None,
        release_after_frames=None,
        stack_k=1,
        dist_metric=None,
        pipeline_version=None,
    )
    goal_np = template
    if goal_np.ndim == 3:
        goal_np = goal_np.squeeze(0)
    goal = torch.tensor(goal_np, device="cuda", dtype=torch.uint8).unsqueeze(
        0
    )  # [1, 72, 80]
    print(f"Goal template shape: {goal.shape}")

    model = PixelActorCriticCNN(num_actions=backend.num_actions, in_frames=1)
    model.to(device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rollout = RolloutBuffer(
        steps=cfg.steps_per_rollout,
        num_envs=cfg.num_envs,
        stack_k=1,
        device="cuda",
    )

    # Get initial obs
    print("Getting initial observation...")
    backend.render_pixels_snapshot_torch()
    obs = backend.pixels_torch().unsqueeze(1)  # [N, 1, 72, 80]
    print(f"Initial obs shape: {obs.shape}")

    run_id = output_dir.name or uuid4().hex[:8]
    train_start = time.time()
    print("Starting training loop...")

    # Episode tracking
    episode_steps = torch.zeros(cfg.num_envs, dtype=torch.int32, device="cuda")
    prev_dist = torch.ones(cfg.num_envs, dtype=torch.float32, device="cuda")
    total_goals = 0
    total_episodes = 0

    tau = 0.05
    step_cost = -0.01
    alpha = 1.0
    goal_bonus = 10.0
    max_steps = 3000  # More steps needed with lower frames_per_step

    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(
            json.dumps(
                {
                    "meta": {
                        "run_id": run_id,
                        "stage": "exit_oak",
                        "num_envs": cfg.num_envs,
                        "seed": cfg.seed,
                        "frames_per_step": cfg.frames_per_step,
                    },
                    "config": asdict(cfg),
                }
            )
            + "\n"
        )

        for update_idx in range(cfg.updates):
            rollout.reset()
            update_start = time.time()
            update_goals = 0
            update_episodes = 0

            for _step in range(cfg.steps_per_rollout):
                with torch.no_grad():
                    logits, values = model(obs)
                    actions_i64 = torch.multinomial(
                        torch.softmax(logits, dim=-1), num_samples=1
                    ).squeeze(1)
                    logprobs = logprob_from_logits(logits, actions_i64)

                actions = actions_i64.to(torch.int32)

                # Step env
                backend.step_torch(actions)
                backend.render_pixels_snapshot_torch()
                next_obs = backend.pixels_torch().unsqueeze(1)
                episode_steps += 1

                # Compute distance to goal
                diff = torch.abs(next_obs.float() - goal.float())
                curr_dist = diff.mean(dim=(1, 2, 3)) / 3.0

                # Check done (goal reached)
                done = curr_dist < tau

                # Check truncation (max steps)
                trunc = episode_steps >= max_steps

                # Compute reward
                reward = torch.full((cfg.num_envs,), step_cost, device="cuda")
                reward += alpha * (prev_dist - curr_dist)
                reward[done] += goal_bonus

                reset_mask = done | trunc
                update_goals += int(done.sum().item())
                update_episodes += int(reset_mask.sum().item())

                # Store in rollout buffer
                rollout.add(
                    obs,
                    actions,
                    reward,
                    reset_mask,
                    values.detach(),
                    logprobs.detach(),
                )

                # Handle resets
                if reset_mask.any():
                    reset_cache.apply_mask_torch(reset_mask.to(torch.uint8))
                    episode_steps[reset_mask] = 0

                    # Re-render after reset
                    backend.render_pixels_snapshot_torch()
                    next_obs = backend.pixels_torch().unsqueeze(1)
                    curr_dist = (
                        torch.abs(next_obs.float() - goal.float()).mean(dim=(1, 2, 3))
                        / 3.0
                    )

                prev_dist = curr_dist
                obs = next_obs

            # Compute advantages
            with torch.no_grad():
                _, last_value = model(obs)

            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.dones,
                last_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            # PPO update
            batch = rollout.as_batch(flatten_obs=True)
            logits, values = model(batch["obs_u8"])
            losses = ppo_losses(
                logits,
                batch["actions"],
                batch["logprobs"],
                returns.reshape(-1),
                advantages.reshape(-1),
                values,
                clip=cfg.clip,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
            )

            optimizer.zero_grad()
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_goals += update_goals
            total_episodes += update_episodes

            env_steps = (update_idx + 1) * cfg.num_envs * cfg.steps_per_rollout
            update_time = time.time() - update_start
            sps = cfg.num_envs * cfg.steps_per_rollout / update_time
            sr = update_goals / max(1, update_episodes)

            record = {
                "run_id": run_id,
                "update": update_idx,
                "env_steps": env_steps,
                "wall_time_s": time.time() - train_start,
                "loss_total": float(losses["loss_total"].item()),
                "entropy": float(losses["entropy"].item()),
                "sps": int(sps),
                "goals": update_goals,
                "episodes": update_episodes,
                "success_rate": sr,
                "total_goals": total_goals,
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

            if (update_idx + 1) % 10 == 0 or update_idx == 0:
                print(
                    f"Update {update_idx + 1}/{cfg.updates} | "
                    f"Steps: {env_steps:,} | "
                    f"SPS: {sps:,.0f} | "
                    f"Goals: {update_goals}/{update_episodes} ({100 * sr:.0f}%) | "
                    f"Loss: {losses['loss_total']:.4f}"
                )

            # Save checkpoint
            if (update_idx + 1) % 50 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "update": update_idx + 1,
                        "total_goals": total_goals,
                    },
                    ckpt_path,
                )

    # Final checkpoint
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "update": cfg.updates,
            "total_goals": total_goals,
        },
        ckpt_path,
    )

    backend.close()

    sr = total_goals / max(1, total_episodes)
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total updates: {cfg.updates}")
    print(f"Total env steps: {cfg.updates * cfg.num_envs * cfg.steps_per_rollout:,}")
    print(f"Total goals: {total_goals} ({100 * sr:.1f}% success rate)")
    print(f"Checkpoint: {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
