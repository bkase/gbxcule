#!/usr/bin/env python3
"""PPO training loop using CPU backend (for systems with GPU compatibility issues).

Usage:
  uv run python tools/rl_train_cpu.py --rom red.gb --state <state> --goal-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch


@dataclass(frozen=True)
class TrainConfig:
    rom: str
    state: str
    goal_dir: str
    frames_per_step: int
    release_after_frames: int
    num_envs: int
    steps_per_rollout: int
    max_steps: int
    updates: int
    lr: float
    gamma: float
    gae_lambda: float
    clip: float
    value_coef: float
    entropy_coef: float
    tau: float
    step_cost: float
    alpha: float
    goal_bonus: float
    seed: int
    output_dir: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", required=True, help="Path to .state file")
    parser.add_argument("--goal-dir", required=True, help="Goal template directory")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps-per-rollout", type=int, default=256)
    parser.add_argument(
        "--max-steps", type=int, default=512, help="Max steps per episode"
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--tau", type=float, default=0.05, help="Goal distance threshold"
    )
    parser.add_argument("--step-cost", type=float, default=-0.01)
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Distance shaping coefficient"
    )
    parser.add_argument("--goal-bonus", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.backends.warp_vec import WarpVecCpuBackend
    from gbxcule.core.state_io import apply_state_to_warp_backend, load_pyboy_state
    from gbxcule.rl.goal_template import compute_sha256, load_goal_template
    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("bench/runs/rl_cpu") / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    ckpt_path = output_dir / "checkpoint.pt"

    cfg = TrainConfig(
        rom=str(Path(args.rom)),
        state=str(Path(args.state)),
        goal_dir=str(Path(args.goal_dir)),
        frames_per_step=int(args.frames_per_step),
        release_after_frames=int(args.release_after_frames),
        num_envs=args.num_envs,
        steps_per_rollout=args.steps_per_rollout,
        max_steps=args.max_steps,
        updates=args.updates,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        tau=args.tau,
        step_cost=args.step_cost,
        alpha=args.alpha,
        goal_bonus=args.goal_bonus,
        seed=args.seed,
        output_dir=str(output_dir),
    )

    print(f"Config: {cfg}")
    print(f"Output: {output_dir}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create CPU backend
    print(f"Creating CPU backend with {cfg.num_envs} envs...")
    backend = WarpVecCpuBackend(
        cfg.rom,
        num_envs=cfg.num_envs,
        frames_per_step=cfg.frames_per_step,
        release_after_frames=cfg.release_after_frames,
        render_pixels=True,
    )

    # Load goal template
    print(f"Loading goal template from {cfg.goal_dir}...")
    template, meta = load_goal_template(
        Path(cfg.goal_dir),
        action_codec_id=backend.action_codec.id,
        frames_per_step=cfg.frames_per_step,
        release_after_frames=cfg.release_after_frames,
        stack_k=1,
        dist_metric=None,
        pipeline_version=None,
    )
    goal_np = np.array(template, dtype=np.float32)  # [72, 80]
    if goal_np.ndim == 3:
        goal_np = goal_np.squeeze(0)  # Remove stack dim if present

    # Load initial state
    print(f"Loading initial state from {cfg.state}...")
    initial_state = load_pyboy_state(cfg.state, expected_cart_ram_size=32768)

    num_actions = backend.num_actions
    print(f"Action space: {num_actions} actions")
    print(f"Max steps per episode: {cfg.max_steps}")
    print(f"Steps per rollout: {cfg.steps_per_rollout}")

    # Model (on CPU)
    model = PixelActorCriticCNN(num_actions=num_actions, in_frames=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    start_update = 0
    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = int(ckpt.get("update", 0))

    # Rollout storage
    obs_buffer = torch.zeros(
        (cfg.steps_per_rollout, cfg.num_envs, 1, 72, 80), dtype=torch.uint8
    )
    action_buffer = torch.zeros(
        (cfg.steps_per_rollout, cfg.num_envs), dtype=torch.int32
    )
    reward_buffer = torch.zeros(
        (cfg.steps_per_rollout, cfg.num_envs), dtype=torch.float32
    )
    done_buffer = torch.zeros((cfg.steps_per_rollout, cfg.num_envs), dtype=torch.bool)
    value_buffer = torch.zeros(
        (cfg.steps_per_rollout, cfg.num_envs), dtype=torch.float32
    )
    logprob_buffer = torch.zeros(
        (cfg.steps_per_rollout, cfg.num_envs), dtype=torch.float32
    )

    # Episode tracking
    episode_steps = np.zeros(cfg.num_envs, dtype=np.int32)
    prev_dist = np.ones(cfg.num_envs, dtype=np.float32)

    def reset_all():
        """Reset all envs to initial state."""
        nonlocal episode_steps, prev_dist
        backend.reset()
        for env_idx in range(cfg.num_envs):
            apply_state_to_warp_backend(initial_state, backend, env_idx=env_idx)
        episode_steps[:] = 0
        prev_dist[:] = 1.0

    def get_obs():
        """Get current observation."""
        backend.render_pixels_snapshot()
        pixels_flat = backend.pixels_wp().numpy()
        return pixels_flat.reshape(cfg.num_envs, 72, 80)

    def compute_dist(obs_np):
        """Compute L1 distance to goal (normalized to [0, 1])."""
        # obs_np: [num_envs, 72, 80], goal_np: [72, 80]
        diff = np.abs(obs_np.astype(np.float32) - goal_np)
        return diff.mean(axis=(1, 2)) / 3.0  # Normalize by max shade value

    def compute_reward_shaped(prev_d, curr_d, done):
        """Compute shaped reward."""
        # Step cost + distance shaping + goal bonus
        r = np.full(cfg.num_envs, cfg.step_cost, dtype=np.float32)
        r += cfg.alpha * (prev_d - curr_d)  # Reward for getting closer
        r[done] += cfg.goal_bonus
        return r

    # Initial reset
    reset_all()
    obs_np = get_obs()
    prev_dist = compute_dist(obs_np)

    run_id = output_dir.name or uuid4().hex[:8]
    train_start = time.time()

    # Stats
    total_goals = 0
    total_episodes = 0
    best_success_rate = 0.0

    with log_path.open("a", encoding="utf-8") as log_f:
        if start_update == 0:
            log_f.write(
                json.dumps(
                    {
                        "meta": {
                            "run_id": run_id,
                            "backend": "cpu",
                            "rom_path": cfg.rom,
                            "rom_sha256": compute_sha256(Path(cfg.rom)),
                            "state_path": cfg.state,
                            "state_sha256": compute_sha256(Path(cfg.state)),
                            "goal_dir": cfg.goal_dir,
                            "num_envs": cfg.num_envs,
                            "num_actions": num_actions,
                            "seed": cfg.seed,
                        },
                        "config": asdict(cfg),
                    }
                )
                + "\n"
            )

        for update_idx in range(start_update, cfg.updates):
            update_start = time.time()
            update_goals = 0
            update_episodes = 0

            # Collect rollout
            for step in range(cfg.steps_per_rollout):
                # Get torch obs
                obs_t = torch.from_numpy(obs_np).unsqueeze(1)  # [N, 1, 72, 80]

                # Model forward
                with torch.no_grad():
                    logits, values = model(obs_t)
                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, num_samples=1).squeeze(1)
                    logprobs = logprob_from_logits(logits, actions)

                # Store
                obs_buffer[step] = obs_t
                action_buffer[step] = actions
                value_buffer[step] = values
                logprob_buffer[step] = logprobs

                # Step env
                actions_np = actions.numpy().astype(np.int32)
                backend.step(actions_np)
                episode_steps += 1

                # Get new obs
                obs_np = get_obs()
                curr_dist = compute_dist(obs_np)

                # Check done (goal reached)
                done = curr_dist < cfg.tau

                # Check truncation (max steps)
                trunc = episode_steps >= cfg.max_steps
                reset_mask = done | trunc

                # Compute reward
                reward = compute_reward_shaped(prev_dist, curr_dist, done)
                reward_buffer[step] = torch.from_numpy(reward)
                done_buffer[step] = torch.from_numpy(reset_mask)

                # Stats
                update_goals += done.sum()
                update_episodes += reset_mask.sum()

                # Reset envs that are done
                for env_idx in np.where(reset_mask)[0]:
                    apply_state_to_warp_backend(initial_state, backend, env_idx=env_idx)
                    episode_steps[env_idx] = 0
                    # Re-render after reset
                    backend.render_pixels_snapshot()
                    pixels_flat = backend.pixels_wp().numpy()
                    obs_np = pixels_flat.reshape(cfg.num_envs, 72, 80)
                    curr_dist = compute_dist(obs_np)

                prev_dist = curr_dist

            # Compute GAE
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_np).unsqueeze(1)
                _, last_value = model(obs_t)

            advantages, returns = compute_gae(
                reward_buffer,
                value_buffer,
                done_buffer,
                last_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            # PPO update
            batch_obs = obs_buffer.reshape(-1, 1, 72, 80)
            batch_actions = action_buffer.reshape(-1)
            batch_logprobs = logprob_buffer.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_advantages = advantages.reshape(-1)

            logits, values = model(batch_obs)
            losses = ppo_losses(
                logits,
                batch_actions,
                batch_logprobs,
                batch_returns,
                batch_advantages,
                values,
                clip=cfg.clip,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
            )

            optimizer.zero_grad()
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # Stats
            total_goals += update_goals
            total_episodes += update_episodes
            env_steps = (update_idx + 1) * cfg.num_envs * cfg.steps_per_rollout
            update_time = time.time() - update_start
            sps = cfg.num_envs * cfg.steps_per_rollout / update_time

            success_rate = update_goals / max(1, update_episodes)
            best_success_rate = max(best_success_rate, success_rate)

            record = {
                "run_id": run_id,
                "update": update_idx,
                "env_steps": env_steps,
                "wall_time_s": time.time() - train_start,
                "loss_total": float(losses["loss_total"].item()),
                "loss_policy": float(losses["loss_policy"].item()),
                "loss_value": float(losses["loss_value"].item()),
                "entropy": float(losses["entropy"].item()),
                "approx_kl": float(losses["approx_kl"].item()),
                "sps": int(sps),
                "update_goals": int(update_goals),
                "update_episodes": int(update_episodes),
                "success_rate": float(success_rate),
                "total_goals": int(total_goals),
                "total_episodes": int(total_episodes),
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

            sr_pct = 100 * success_rate
            print(
                f"Update {update_idx + 1}/{cfg.updates} | "
                f"Steps: {env_steps:,} | "
                f"SPS: {sps:.0f} | "
                f"Goals: {update_goals}/{update_episodes} ({sr_pct:.0f}%) | "
                f"Loss: {losses['loss_total'].item():.4f} | "
                f"Entropy: {losses['entropy'].item():.2f}"
            )

            # Save checkpoint
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "update": update_idx + 1,
                },
                ckpt_path,
            )

    backend.close()

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total updates: {cfg.updates}")
    print(f"Total env steps: {cfg.updates * cfg.num_envs * cfg.steps_per_rollout:,}")
    print(f"Total goals: {total_goals}")
    print(f"Total episodes: {total_episodes}")
    print(f"Best success rate: {100 * best_success_rate:.1f}%")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Log: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
