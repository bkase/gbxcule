#!/usr/bin/env python3
"""PPO training with ImpalaResNet for Oak's Parcel (Pokemon Red).

Optimized for NVIDIA GB10 (Blackwell Spark):
- ImpalaResNet with frame stacking (4 frames) for short-term memory
- torch.compile with reduce-overhead mode
- bfloat16 for native Blackwell performance
- High throughput via parallel environments

Usage:
  uv run python tools/rl_train_impala.py --rom red.gb \
    --state states/rl_oak_parcel_start.state --num-envs 16384
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
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--action-codec", default="pokemonred_puffer_v1")
    parser.add_argument("--max-steps", type=int, default=2048)
    # Reward
    parser.add_argument("--snow-bonus", type=float, default=0.01)
    parser.add_argument("--get-parcel-bonus", type=float, default=5.0)
    parser.add_argument("--deliver-bonus", type=float, default=10.0)
    parser.add_argument("--dialogue-bonus", type=float, default=0.01)
    parser.add_argument("--item-pickup-bonus", type=float, default=0.05)
    parser.add_argument("--no-curiosity-reset", action="store_true")
    # PPO
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    # Training
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--total-env-steps", type=int, default=100_000_000)
    parser.add_argument("--steps-per-rollout", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-tag", default="impala_parcel")
    parser.add_argument("--run-root", default="bench/runs/rl")
    parser.add_argument("--resume", default=None)
    # Model
    parser.add_argument("--num-frames", type=int, default=4, help="Frame stack size")
    parser.add_argument("--base-channels", type=int, default=64, help="Base channels")
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

    # Use bfloat16 for native Blackwell performance
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

    # Create model (NatureCNN is faster than ImpalaResNet)
    print(f"Creating NatureCNN (in_frames={cfg.num_frames})...")
    model = NatureCNN(
        num_actions=env.num_actions,
        in_frames=cfg.num_frames,
    )
    model = model.to("cuda", dtype=dtype)

    # Skip torch.compile for now - causes long JIT overhead
    # model = torch.compile(model, mode="reduce-overhead")
    print("Using eager mode (torch.compile disabled)")

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
        algo="impala_ppo",
        rom_id=rom_path.stem,
        tag=cfg.output_tag,
        run_root=cfg.run_root,
        meta={"git_commit": git_commit, "git_dirty": git_dirty},
        config=asdict(cfg),
    )

    # Allocate rollout buffers (uint8 for pixels to save memory)
    print(f"Allocating rollout buffers ({cfg.steps_per_rollout} steps)...")
    # Store pixels as uint8 (0-3 for 2bpp), convert to float on-the-fly
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

    print(f"\nStarting training for {iterations} iterations...")
    print(f"  total_env_steps={cfg.total_env_steps}")
    print(f"  gamma={cfg.gamma}, gae_lambda={cfg.gae_lambda}")
    print(f"  curiosity_reset={cfg.curiosity_reset_on_parcel}")

    try:
        for iter_idx in range(iterations):
            iter_start = time.time()

            # === PHASE 1: COLLECT ROLLOUT ===
            model.eval()
            with torch.no_grad():
                for t in range(cfg.steps_per_rollout):
                    # Forward pass in bfloat16 (normalize uint8 0-3 to 0-1)
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

                    pixels = next_obs["pixels"]

                # Bootstrap value
                pixels_norm = pixels.to(dtype) / 3.0
                _, last_value = model(pixels_norm)
                last_value = last_value.float()

            env_steps += cfg.num_envs * cfg.steps_per_rollout

            # === PHASE 2: COMPUTE GAE ===
            advantages, returns = compute_gae(
                rollout_rewards,
                rollout_values,
                rollout_dones,
                last_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            # === PHASE 3: PPO UPDATE ===
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

            # Logging
            iter_time = time.time() - iter_start
            wall_time = time.time() - start
            sps = (cfg.num_envs * cfg.steps_per_rollout) / iter_time

            avg_policy_loss = total_policy_loss / max(1, num_updates)
            avg_value_loss = total_value_loss / max(1, num_updates)
            avg_entropy = total_entropy / max(1, num_updates)
            avg_clipfrac = total_clipfrac / max(1, num_updates)

            metrics = {
                "env_steps": env_steps,
                "train_steps": train_steps,
                "wall_time_s": wall_time,
                "sps": sps,
                "iter_time_s": iter_time,
                "Loss/policy": avg_policy_loss,
                "Loss/value": avg_value_loss,
                "entropy": avg_entropy,
                "clipfrac": avg_clipfrac,
                "reward_mean": rollout_rewards.mean().item(),
                "reward_std": rollout_rewards.std().item(),
                "value_mean": rollout_values.mean().item(),
                "adv_mean": advantages.mean().item(),
                "parcels_picked": parcels_picked,
                "parcels_delivered": parcels_delivered,
            }
            experiment.log_metrics(metrics)

            print(
                f"Iter {iter_idx + 1}/{iterations} | "
                f"env_steps={env_steps:,} | "
                f"sps={sps:,.0f} | "
                f"reward={rollout_rewards.mean().item():.4f} | "
                f"entropy={avg_entropy:.4f} | "
                f"picked={parcels_picked} | delivered={parcels_delivered}"
            )

            # Checkpoint
            if (iter_idx + 1) % cfg.checkpoint_every == 0:
                ckpt_path = experiment.run_dir / "checkpoints" / "checkpoint.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)

                # Get state dict (handle compiled model)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
