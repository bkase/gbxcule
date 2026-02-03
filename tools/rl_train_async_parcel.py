#!/usr/bin/env python3
"""Async PPO training for Oak's Parcel with interaction curiosity.

Features:
- Double-buffered async rollouts for better GPU utilization
- Interaction exploration: rewards pressing A at new (x,y) locations
- Hash-based exploration ("fresh snow") with curiosity reset
- Frame stacking (4 frames) for short-term memory
- Higher entropy (0.03) to prevent policy collapse

Usage:
  uv run python tools/rl_train_async_parcel.py --rom red.gb \
    --state states/rl_oak_parcel/start.state --num-envs 8192
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


def _git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()[:8]
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", True


@dataclass(frozen=True)
class Config:
    """Training configuration."""
    rom: str
    state: str
    num_envs: int
    frames_per_step: int
    release_after_frames: int
    action_codec: str
    max_steps: int
    # Reward config
    snow_bonus: float
    get_parcel_bonus: float
    deliver_bonus: float
    dialogue_bonus: float
    item_pickup_bonus: float
    interaction_curiosity_bonus: float  # NEW: bonus for A at new location
    curiosity_reset_on_parcel: bool
    # PPO hyperparameters
    gamma: float
    gae_lambda: float
    clip: float
    value_coef: float
    entropy_coef: float
    lr: float
    grad_clip: float
    ppo_epochs: int
    minibatch_size: int
    # Training config
    seed: int
    total_env_steps: int
    steps_per_rollout: int
    checkpoint_every: int
    output_tag: str
    run_root: str
    # Model config
    num_frames: int
    base_channels: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--action-codec", default="pokemonred_puffer_v1")
    parser.add_argument("--max-steps", type=int, default=4096)
    # Reward
    parser.add_argument("--snow-bonus", type=float, default=0.01)
    parser.add_argument("--get-parcel-bonus", type=float, default=5.0)
    parser.add_argument("--deliver-bonus", type=float, default=10.0)
    parser.add_argument("--dialogue-bonus", type=float, default=0.01)
    parser.add_argument("--item-pickup-bonus", type=float, default=0.05)
    parser.add_argument("--interaction-curiosity-bonus", type=float, default=0.015)
    parser.add_argument("--no-curiosity-reset", action="store_true")
    # PPO - higher entropy to prevent collapse
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.03)  # Higher!
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    # Training
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--total-env-steps", type=int, default=10_000_000)
    parser.add_argument("--steps-per-rollout", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--output-tag", default="async_parcel")
    parser.add_argument("--run-root", default="bench/runs/rl")
    parser.add_argument("--resume", default=None)
    # Model
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    cfg = Config(
        rom=str(args.rom),
        state=str(args.state),
        num_envs=int(args.num_envs),
        frames_per_step=int(args.frames_per_step),
        release_after_frames=int(args.release_after_frames),
        action_codec=str(args.action_codec),
        max_steps=int(args.max_steps),
        snow_bonus=float(args.snow_bonus),
        get_parcel_bonus=float(args.get_parcel_bonus),
        deliver_bonus=float(args.deliver_bonus),
        dialogue_bonus=float(args.dialogue_bonus),
        item_pickup_bonus=float(args.item_pickup_bonus),
        interaction_curiosity_bonus=float(args.interaction_curiosity_bonus),
        curiosity_reset_on_parcel=not bool(args.no_curiosity_reset),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip=float(args.clip),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        lr=float(args.lr),
        grad_clip=float(args.grad_clip),
        ppo_epochs=int(args.ppo_epochs),
        minibatch_size=int(args.minibatch_size),
        seed=int(args.seed),
        total_env_steps=int(args.total_env_steps),
        steps_per_rollout=int(args.steps_per_rollout),
        checkpoint_every=int(args.checkpoint_every),
        output_tag=str(args.output_tag),
        run_root=str(args.run_root),
        num_frames=int(args.num_frames),
        base_channels=int(args.base_channels),
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    dtype = torch.bfloat16
    print(f"Using dtype: {dtype}")

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.rl.experiment import Experiment
    from gbxcule.rl.frame_stack_env import FrameStackEnv
    from gbxcule.rl.nature_cnn import NatureCNN
    from gbxcule.rl.pokered_packed_parcel_env import PokeredPackedParcelEnv
    from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses

    # Create environment
    print(f"Creating environment with {cfg.num_envs} parallel envs...")
    base_env = PokeredPackedParcelEnv(
        cfg.rom,
        state_path=cfg.state,
        num_envs=cfg.num_envs,
        frames_per_step=cfg.frames_per_step,
        release_after_frames=cfg.release_after_frames,
        action_codec=cfg.action_codec,
        max_steps=cfg.max_steps,
        snow_bonus=cfg.snow_bonus,
        get_parcel_bonus=cfg.get_parcel_bonus,
        deliver_bonus=cfg.deliver_bonus,
        dialogue_bonus=cfg.dialogue_bonus,
        item_pickup_bonus=cfg.item_pickup_bonus,
        curiosity_reset_on_parcel=cfg.curiosity_reset_on_parcel,
        info_mode="stats",
    )
    env = FrameStackEnv(base_env, num_frames=cfg.num_frames)
    print(f"Environment created: {cfg.num_envs} envs, {env.num_actions} actions")
    print(f"Frame stacking: {cfg.num_frames} frames")

    # Interaction curiosity tracking: hash table for (x, y) where A was pressed
    # Using epoch trick similar to snow exploration
    INTERACTION_HASH_SIZE = 16384
    HASH_PRIME_X = 73856093
    HASH_PRIME_Y = 19349663
    interaction_table = torch.zeros(
        (cfg.num_envs, INTERACTION_HASH_SIZE), dtype=torch.int16, device="cuda"
    )
    interaction_epoch = torch.ones((cfg.num_envs,), dtype=torch.int16, device="cuda")
    A_BUTTON_ACTION = 0  # Action index for A button

    def compute_interaction_curiosity(
        actions: torch.Tensor, x: torch.Tensor, y: torch.Tensor, facing: torch.Tensor
    ) -> torch.Tensor:
        """Compute curiosity bonus for pressing A at new facing tile locations.

        When you press A in Pokemon, you interact with the tile you're FACING,
        not the tile you're standing on. This computes the facing tile based on
        the player's direction and rewards exploring new interaction targets.
        """
        # Only reward if action is A
        is_a_press = actions == A_BUTTON_ACTION

        # Compute facing tile offset based on direction
        # facing: 0=down, 4=up, 8=left, 12=right
        dx = torch.where(facing == 8, -1, torch.where(facing == 12, 1, 0))
        dy = torch.where(facing == 0, 1, torch.where(facing == 4, -1, 0))
        facing_x = x.int() + dx
        facing_y = y.int() + dy

        # Hash the facing tile position (what we're interacting with)
        hash_idx = ((facing_x * HASH_PRIME_X) ^ (facing_y * HASH_PRIME_Y)) % INTERACTION_HASH_SIZE

        # Check if this facing tile is new for interaction
        stored_epoch = interaction_table.gather(1, hash_idx.unsqueeze(1)).squeeze(1)
        is_novel = stored_epoch != interaction_epoch

        # Mark as visited (only if A was pressed)
        update_mask = is_a_press & is_novel
        if update_mask.any():
            interaction_table.scatter_(
                1,
                hash_idx[update_mask].unsqueeze(1),
                interaction_epoch[update_mask].unsqueeze(1)
            )

        # Reward: bonus only if A pressed at novel facing tile
        return (is_a_press & is_novel).float() * cfg.interaction_curiosity_bonus

    def reset_interaction_curiosity(mask: torch.Tensor) -> None:
        """Reset interaction curiosity for masked environments (epoch trick)."""
        nonlocal interaction_epoch
        if not mask.any():
            return
        new_epoch = interaction_epoch + 1
        interaction_epoch = torch.where(mask, new_epoch.to(torch.int16), interaction_epoch)
        # Handle overflow
        overflowed = mask & (interaction_epoch == 0)
        if overflowed.any():
            interaction_table[overflowed] = 0
            interaction_epoch[overflowed] = 1

    # Create model
    print(f"Creating NatureCNN (in_frames={cfg.num_frames})...")
    model = NatureCNN(
        num_actions=env.num_actions,
        in_frames=cfg.num_frames,
    )
    model = model.to("cuda", dtype=dtype)
    print(f"Using entropy_coef={cfg.entropy_coef} (higher to prevent collapse)")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Resume checkpoint
    resume_env_steps = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        resume_env_steps = ckpt.get("env_steps", 0)

    # Setup experiment
    git_commit, git_dirty = _git_info()
    rom_path = Path(cfg.rom)

    experiment = Experiment(
        algo="async_ppo",
        rom_id=rom_path.stem,
        tag=cfg.output_tag,
        run_root=cfg.run_root,
        meta={"git_commit": git_commit, "git_dirty": git_dirty},
        config=asdict(cfg),
    )

    # Allocate rollout buffers (uint8 for pixels to save memory)
    print(f"Allocating rollout buffers ({cfg.steps_per_rollout} steps)...")
    rollout_pixels = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs, cfg.num_frames, 72, 80),
        dtype=torch.uint8,
        device="cuda",
    )
    rollout_actions = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs), dtype=torch.int64, device="cuda"
    )
    rollout_rewards = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs), dtype=torch.float32, device="cuda"
    )
    rollout_dones = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs), dtype=torch.bool, device="cuda"
    )
    rollout_values = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs), dtype=torch.float32, device="cuda"
    )
    rollout_logprobs = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs), dtype=torch.float32, device="cuda"
    )

    # RAM addresses for position and facing
    MAP_ID_ADDR = 0xD35E
    PLAYER_X_ADDR = 0xD362
    PLAYER_Y_ADDR = 0xD361
    PLAYER_FACING_ADDR = 0xC109  # 0=down, 4=up, 8=left, 12=right

    # Initialize
    print("Resetting environment...")
    obs = env.reset(seed=cfg.seed)
    pixels = obs["pixels"]

    iterations = max(
        1, (cfg.total_env_steps + cfg.num_envs * cfg.steps_per_rollout - 1)
        // (cfg.num_envs * cfg.steps_per_rollout)
    )
    env_steps = resume_env_steps
    train_steps = 0
    start = time.time()

    # Stats
    parcels_picked = 0
    parcels_delivered = 0
    total_interaction_bonus = 0.0

    print(f"\nStarting training for {iterations} iterations...")
    print(f"  total_env_steps={cfg.total_env_steps}")
    print(f"  max_steps={cfg.max_steps} (episode truncation)")
    print(f"  gamma={cfg.gamma}, gae_lambda={cfg.gae_lambda}")
    print(f"  entropy_coef={cfg.entropy_coef}")
    print(f"  interaction_curiosity_bonus={cfg.interaction_curiosity_bonus}")
    print(f"  curiosity_reset={cfg.curiosity_reset_on_parcel}")
    print(f"  checkpoint_every={cfg.checkpoint_every}")

    try:
        for iter_idx in range(iterations):
            iter_start = time.time()

            # === PHASE 1: COLLECT ROLLOUT ===
            rollout_start = time.time()
            model.eval()
            iter_interaction_bonus = 0.0
            with torch.no_grad():
                for t in range(cfg.steps_per_rollout):
                    # Forward pass in bfloat16
                    pixels_norm = pixels.to(dtype) / 3.0
                    logits, values = model(pixels_norm)

                    # Convert to float32 for sampling
                    logits = logits.float()
                    values = values.float()

                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, num_samples=1).squeeze(1)
                    logprobs = logprob_from_logits(logits, actions)

                    # Store
                    rollout_pixels[t].copy_(pixels)
                    rollout_actions[t].copy_(actions)
                    rollout_values[t].copy_(values)
                    rollout_logprobs[t].copy_(logprobs)

                    # Step
                    next_obs, reward, terminated, truncated, info = env.step(
                        actions.to(torch.int32)
                    )
                    done = terminated | truncated

                    # Compute interaction curiosity bonus (for facing tile, not standing tile)
                    mem = base_env.backend.memory_torch()
                    x = mem[:, PLAYER_X_ADDR]
                    y = mem[:, PLAYER_Y_ADDR]
                    facing = mem[:, PLAYER_FACING_ADDR]
                    interaction_bonus = compute_interaction_curiosity(actions, x, y, facing)
                    reward = reward + interaction_bonus
                    iter_interaction_bonus += interaction_bonus.sum().item()

                    rollout_rewards[t].copy_(reward)
                    rollout_dones[t].copy_(done)

                    # Track stats
                    if "got_parcel" in info:
                        parcels_picked += int(info["got_parcel"].sum().item())
                    if "delivered" in info:
                        parcels_delivered += int(info["delivered"].sum().item())

                    # Handle resets
                    if done.any():
                        env.reset_mask(done)
                        # Reset interaction curiosity for done envs
                        reset_interaction_curiosity(done)

                    pixels = next_obs["pixels"]

                # Bootstrap value
                pixels_norm = pixels.to(dtype) / 3.0
                _, last_value = model(pixels_norm)
                last_value = last_value.float()

            env_steps += cfg.num_envs * cfg.steps_per_rollout
            total_interaction_bonus += iter_interaction_bonus
            rollout_time = time.time() - rollout_start

            # === PHASE 2: COMPUTE GAE ===
            gae_start = time.time()
            advantages, returns = compute_gae(
                rollout_rewards,
                rollout_values,
                rollout_dones,
                last_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )
            gae_time = time.time() - gae_start

            # === PHASE 3: PPO UPDATE ===
            ppo_start = time.time()
            model.train()

            batch_size = cfg.steps_per_rollout * cfg.num_envs
            flat_pixels = rollout_pixels.reshape(batch_size, cfg.num_frames, 72, 80)
            flat_actions = rollout_actions.reshape(batch_size)
            flat_old_logprobs = rollout_logprobs.reshape(batch_size)
            flat_returns = returns.reshape(batch_size)
            flat_advantages = advantages.reshape(batch_size)

            # Normalize advantages
            adv_mean = flat_advantages.mean()
            adv_std = flat_advantages.std(unbiased=False)
            flat_advantages_norm = (flat_advantages - adv_mean) / (adv_std + 1e-8)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_clipfrac = 0.0
            num_updates = 0

            for _ in range(cfg.ppo_epochs):
                perm = torch.randperm(batch_size, device="cuda")

                for mb_start in range(0, batch_size, cfg.minibatch_size):
                    mb_end = min(mb_start + cfg.minibatch_size, batch_size)
                    mb_idx = perm[mb_start:mb_end]

                    mb_pixels = flat_pixels[mb_idx].to(dtype) / 3.0
                    mb_actions = flat_actions[mb_idx]
                    mb_old_logprobs = flat_old_logprobs[mb_idx]
                    mb_returns = flat_returns[mb_idx]
                    mb_advantages = flat_advantages_norm[mb_idx]

                    # Forward in bfloat16
                    logits, values = model(mb_pixels)
                    logits = logits.float()
                    values = values.float()

                    # Compute losses
                    losses = ppo_losses(
                        logits,
                        mb_actions,
                        mb_old_logprobs,
                        mb_returns,
                        mb_advantages,
                        values,
                        clip=cfg.clip,
                        value_coef=cfg.value_coef,
                        entropy_coef=cfg.entropy_coef,
                        normalize_adv=False,
                    )

                    # Backward
                    optimizer.zero_grad()
                    losses["loss_total"].backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_clip
                        )
                    optimizer.step()

                    total_policy_loss += losses["loss_policy"].item()
                    total_value_loss += losses["loss_value"].item()
                    total_entropy += losses["entropy"].item()
                    total_clipfrac += losses["clipfrac"].item()
                    num_updates += 1
                    train_steps += 1

            ppo_time = time.time() - ppo_start

            # Logging - comprehensive metrics like Dreamer
            iter_time = time.time() - iter_start
            wall_time = time.time() - start
            env_steps_iter = cfg.num_envs * cfg.steps_per_rollout
            sps = env_steps_iter / iter_time if iter_time > 0 else 0.0
            train_sps = num_updates / iter_time if iter_time > 0 else 0.0

            avg_policy_loss = total_policy_loss / max(1, num_updates)
            avg_value_loss = total_value_loss / max(1, num_updates)
            avg_entropy = total_entropy / max(1, num_updates)
            avg_clipfrac = total_clipfrac / max(1, num_updates)

            # Compute additional statistics
            reward_flat = rollout_rewards.flatten()
            value_flat = rollout_values.flatten()
            adv_flat = advantages.flatten()

            metrics = {
                # Core progress
                "iteration": iter_idx + 1,
                "env_steps": env_steps,
                "train_steps": train_steps,
                "opt_steps": train_steps,
                "updates": num_updates,
                "wall_time_s": wall_time,
                # Throughput
                "sps": sps,
                "train_sps": train_sps,
                "iter_time_s": iter_time,
                "env_steps_iter": env_steps_iter,
                # Phase timing (for profiling)
                "timing/rollout_s": rollout_time,
                "timing/gae_s": gae_time,
                "timing/ppo_s": ppo_time,
                "timing/rollout_pct": 100.0 * rollout_time / iter_time if iter_time > 0 else 0,
                "timing/gae_pct": 100.0 * gae_time / iter_time if iter_time > 0 else 0,
                "timing/ppo_pct": 100.0 * ppo_time / iter_time if iter_time > 0 else 0,
                # Losses
                "Loss/policy": avg_policy_loss,
                "Loss/value": avg_value_loss,
                "Loss/total": avg_policy_loss + cfg.value_coef * avg_value_loss - cfg.entropy_coef * avg_entropy,
                # Policy stats
                "entropy": avg_entropy,
                "clipfrac": avg_clipfrac,
                # Reward stats
                "reward/mean": reward_flat.mean().item(),
                "reward/std": reward_flat.std().item(),
                "reward/min": reward_flat.min().item(),
                "reward/max": reward_flat.max().item(),
                "reward/sum": reward_flat.sum().item(),
                # Value stats
                "value/mean": value_flat.mean().item(),
                "value/std": value_flat.std().item(),
                "value/min": value_flat.min().item(),
                "value/max": value_flat.max().item(),
                # Advantage stats
                "advantage/mean": adv_flat.mean().item(),
                "advantage/std": adv_flat.std().item(),
                # Return stats
                "return/mean": flat_returns.mean().item(),
                "return/std": flat_returns.std().item(),
                # Task-specific
                "parcels_picked": parcels_picked,
                "parcels_delivered": parcels_delivered,
                "interaction_bonus_iter": iter_interaction_bonus,
                "interaction_bonus_total": total_interaction_bonus,
                # Config (for reference in analysis)
                "config/num_envs": cfg.num_envs,
                "config/steps_per_rollout": cfg.steps_per_rollout,
                "config/gamma": cfg.gamma,
                "config/gae_lambda": cfg.gae_lambda,
                "config/entropy_coef": cfg.entropy_coef,
                "config/lr": cfg.lr,
                "config/interaction_curiosity_bonus": cfg.interaction_curiosity_bonus,
                "config/snow_bonus": cfg.snow_bonus,
            }
            experiment.log_metrics(metrics)

            print(
                f"Iter {iter_idx + 1}/{iterations} | "
                f"env_steps={env_steps:,} | "
                f"sps={sps:,.0f} | "
                f"reward={rollout_rewards.mean().item():.4f} | "
                f"entropy={avg_entropy:.4f} | "
                f"int_bonus={iter_interaction_bonus:.1f} | "
                f"picked={parcels_picked} | delivered={parcels_delivered}",
                flush=True
            )

            # Checkpoint
            if (iter_idx + 1) % cfg.checkpoint_every == 0:
                ckpt_path = experiment.run_dir / "checkpoints" / "checkpoint.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)

                # Get state dict
                if hasattr(model, "_orig_mod"):
                    model_state = model._orig_mod.state_dict()
                else:
                    model_state = model.state_dict()

                torch.save(
                    {
                        "model": model_state,
                        "optimizer": optimizer.state_dict(),
                        "env_steps": env_steps,
                        "train_steps": train_steps,
                        "config": asdict(cfg),
                    },
                    ckpt_path,
                )
                print(f"  Checkpoint saved to {ckpt_path}", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        env.close()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Run dir: {experiment.run_dir}")
    print(f"  Total env steps: {env_steps:,}")
    print(f"  Total train steps: {train_steps:,}")
    print(f"  Parcels picked: {parcels_picked}")
    print(f"  Parcels delivered: {parcels_delivered}")
    print(f"  Total interaction bonus: {total_interaction_bonus:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
