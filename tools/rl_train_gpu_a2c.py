#!/usr/bin/env python3
"""A2C training using Warp GPU backend with many parallel environments.

The GPU backend runs Warp's CUDA kernels which are fast even on sm_121.
PyTorch model runs on CPU (to avoid PyTorch CUDA compatibility issues),
with data transferred via numpy arrays.

Usage:
  uv run python tools/rl_train_gpu_a2c.py --rom red.gb --state <state> --goal-dir <dir>
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
    n_steps: int  # Steps between updates (A2C typically uses small values like 5-20)
    max_steps: int
    updates: int
    lr: float
    gamma: float
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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1024,
        help="Parallel environments (GPU can handle many)",
    )
    parser.add_argument("--n-steps", type=int, default=20, help="Steps between updates")
    parser.add_argument(
        "--max-steps", type=int, default=512, help="Max steps per episode"
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
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

    from gbxcule.backends.warp_vec import WarpVecCudaBackend
    from gbxcule.core.reset_cache import ResetCache
    from gbxcule.core.state_io import apply_state_to_warp_backend, load_pyboy_state
    from gbxcule.rl.goal_template import compute_sha256, load_goal_template
    from gbxcule.rl.models import PixelActorCriticCNN

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("bench/runs/rl_gpu_a2c") / time.strftime("%Y%m%d_%H%M%S")
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
        n_steps=args.n_steps,
        max_steps=args.max_steps,
        updates=args.updates,
        lr=args.lr,
        gamma=args.gamma,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        tau=args.tau,
        step_cost=args.step_cost,
        alpha=args.alpha,
        goal_bonus=args.goal_bonus,
        seed=args.seed,
        output_dir=str(output_dir),
    )

    print(f"Config: num_envs={cfg.num_envs}, n_steps={cfg.n_steps}")
    print(f"Output: {output_dir}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create GPU backend (Warp CUDA - fast on all GPUs)
    print(f"Creating GPU backend with {cfg.num_envs} envs...")
    backend = WarpVecCudaBackend(
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
    goal_np = np.array(template, dtype=np.float32)
    if goal_np.ndim == 3:
        goal_np = goal_np.squeeze(0)

    # Load initial state
    print(f"Loading initial state from {cfg.state}...")
    initial_state = load_pyboy_state(cfg.state, expected_cart_ram_size=32768)

    num_actions = backend.num_actions
    print(f"Action space: {num_actions} actions")
    print(f"Max steps per episode: {cfg.max_steps}")
    print(f"Steps between updates: {cfg.n_steps}")

    # Model (on CPU - PyTorch CUDA has issues on sm_121)
    model = PixelActorCriticCNN(num_actions=num_actions, in_frames=1)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr, alpha=0.99, eps=1e-5)

    start_update = 0
    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = int(ckpt.get("update", 0))

    # Episode tracking (numpy for GPU<->CPU transfer)
    episode_steps = np.zeros(cfg.num_envs, dtype=np.int32)
    prev_dist = np.ones(cfg.num_envs, dtype=np.float32)
    reset_cache = None  # Will be initialized after first reset

    def reset_all():
        """Reset all envs to initial state."""
        nonlocal episode_steps, prev_dist, reset_cache
        import warp as wp

        backend.reset()
        # Apply state only to env 0, then create reset cache
        apply_state_to_warp_backend(initial_state, backend, env_idx=0)
        # Create reset cache from env 0
        reset_cache = ResetCache.from_backend(backend, env_idx=0)
        # Apply to all envs using warp mask (no torch)
        all_mask_np = np.ones(cfg.num_envs, dtype=np.uint8)
        all_mask_wp = wp.array(all_mask_np, dtype=wp.uint8, device="cuda")
        reset_cache._launch_masked_copies(all_mask_wp)
        wp.synchronize()
        episode_steps[:] = 0
        prev_dist[:] = 1.0

    def get_obs():
        """Get current observation as numpy array."""
        backend.render_pixels_snapshot()
        pixels_flat = backend.pixels_wp().numpy()
        return pixels_flat.reshape(cfg.num_envs, 72, 80)

    def compute_dist(obs_np):
        """Compute L1 distance to goal (normalized to [0, 1])."""
        diff = np.abs(obs_np.astype(np.float32) - goal_np)
        return diff.mean(axis=(1, 2)) / 3.0

    def compute_reward_shaped(prev_d, curr_d, done):
        """Compute shaped reward."""
        r = np.full(cfg.num_envs, cfg.step_cost, dtype=np.float32)
        r += cfg.alpha * (prev_d - curr_d)
        r[done] += cfg.goal_bonus
        return r

    # Initial reset
    print("Resetting all environments...")
    reset_all()
    obs_np = get_obs()
    prev_dist = compute_dist(obs_np)
    print("Ready to train!")

    run_id = output_dir.name or uuid4().hex[:8]
    train_start = time.time()

    # Stats
    total_goals = 0
    total_episodes = 0
    best_success_rate = 0.0

    # Rollout storage
    obs_buffer = np.zeros((cfg.n_steps, cfg.num_envs, 1, 72, 80), dtype=np.uint8)
    action_buffer = np.zeros((cfg.n_steps, cfg.num_envs), dtype=np.int64)
    reward_buffer = np.zeros((cfg.n_steps, cfg.num_envs), dtype=np.float32)
    done_buffer = np.zeros((cfg.n_steps, cfg.num_envs), dtype=bool)
    value_buffer = np.zeros((cfg.n_steps, cfg.num_envs), dtype=np.float32)

    with log_path.open("a", encoding="utf-8") as log_f:
        if start_update == 0:
            log_f.write(
                json.dumps(
                    {
                        "meta": {
                            "run_id": run_id,
                            "backend": "gpu",
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

            # Collect n_steps of experience
            for step in range(cfg.n_steps):
                # Get torch obs for model
                obs_t = torch.from_numpy(obs_np).unsqueeze(1)  # [N, 1, 72, 80]

                # Model forward (CPU)
                with torch.no_grad():
                    logits, values = model(obs_t)
                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, num_samples=1).squeeze(1)

                # Store in buffers
                obs_buffer[step] = obs_np[:, np.newaxis, :, :]  # Add channel dim
                action_buffer[step] = actions.numpy()
                value_buffer[step] = values.numpy()

                # Step GPU backend with numpy actions
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
                reward_buffer[step] = reward
                done_buffer[step] = reset_mask

                # Stats
                update_goals += done.sum()
                update_episodes += reset_mask.sum()

                # Reset envs that are done using reset_cache (warp mask, no torch)
                if reset_mask.any():
                    import warp as wp

                    assert reset_cache is not None
                    mask_wp = wp.array(
                        reset_mask.astype(np.uint8), dtype=wp.uint8, device="cuda"
                    )
                    reset_cache._launch_masked_copies(mask_wp)
                    episode_steps[reset_mask] = 0
                    # Re-render after reset
                    obs_np = get_obs()
                    curr_dist = compute_dist(obs_np)

                prev_dist = curr_dist

            # Compute returns and advantages (A2C style - no GAE)
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_np).unsqueeze(1)
                _, last_value = model(obs_t)
                last_value_np = last_value.numpy()

            returns = np.zeros((cfg.n_steps, cfg.num_envs), dtype=np.float32)
            R = last_value_np * (~done_buffer[-1]).astype(np.float32)
            for t in range(cfg.n_steps - 1, -1, -1):
                R = reward_buffer[t] + cfg.gamma * R * (~done_buffer[t]).astype(
                    np.float32
                )
                returns[t] = R

            advantages = returns - value_buffer

            # A2C update
            obs_batch = torch.from_numpy(obs_buffer.reshape(-1, 1, 72, 80))
            actions_batch = torch.from_numpy(action_buffer.reshape(-1)).long()
            returns_batch = torch.from_numpy(returns.reshape(-1))
            advantages_batch = torch.from_numpy(advantages.reshape(-1))

            # Forward pass
            logits, values = model(obs_batch)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Policy loss
            action_log_probs = log_probs.gather(1, actions_batch.unsqueeze(1)).squeeze(
                1
            )
            policy_loss = -(action_log_probs * advantages_batch.detach()).mean()

            # Value loss
            value_loss = ((returns_batch - values) ** 2).mean()

            # Entropy bonus
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            entropy_loss = -cfg.entropy_coef * entropy

            # Total loss
            loss = policy_loss + cfg.value_coef * value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            # Stats
            total_goals += update_goals
            total_episodes += update_episodes
            env_steps = (update_idx + 1) * cfg.num_envs * cfg.n_steps
            update_time = time.time() - update_start
            sps = cfg.num_envs * cfg.n_steps / update_time

            success_rate = (
                update_goals / max(1, update_episodes) if update_episodes > 0 else 0.0
            )
            best_success_rate = max(best_success_rate, success_rate)

            record = {
                "run_id": run_id,
                "update": update_idx,
                "env_steps": env_steps,
                "wall_time_s": time.time() - train_start,
                "loss_total": float(loss.item()),
                "loss_policy": float(policy_loss.item()),
                "loss_value": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "sps": int(sps),
                "update_goals": int(update_goals),
                "update_episodes": int(update_episodes),
                "success_rate": float(success_rate),
                "total_goals": int(total_goals),
                "total_episodes": int(total_episodes),
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

            if (update_idx + 1) % 10 == 0 or update_idx == 0:
                sr_pct = 100 * success_rate
                print(
                    f"Update {update_idx + 1}/{cfg.updates} | "
                    f"Steps: {env_steps:,} | "
                    f"SPS: {sps:,.0f} | "
                    f"Goals: {update_goals}/{update_episodes} ({sr_pct:.0f}%) | "
                    f"Loss: {loss.item():.4f} | "
                    f"Entropy: {entropy.item():.2f}"
                )

            # Save checkpoint every 100 updates
            if (update_idx + 1) % 100 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "update": update_idx + 1,
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
        },
        ckpt_path,
    )

    backend.close()

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total updates: {cfg.updates}")
    print(f"Total env steps: {cfg.updates * cfg.num_envs * cfg.n_steps:,}")
    print(f"Total goals: {total_goals}")
    print(f"Total episodes: {total_episodes}")
    print(f"Best success rate: {100 * best_success_rate:.1f}%")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Log: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
