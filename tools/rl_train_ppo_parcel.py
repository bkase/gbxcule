#!/usr/bin/env python3
"""Sync PPO training for Oak's Parcel task (Pokemon Red).

This implements the PPO-based approach for the Oak's Parcel quest with:
- Hash-based exploration ("fresh snow") with curiosity reset on key events
- Dual-lobe architecture (pixels + senses + events)
- Long-horizon tuning (gamma=0.999, GAE lambda=0.98)
- Support for up to 16k parallel GameBoy environments

Usage:
  uv run python tools/rl_train_ppo_parcel.py --rom red.gb \
    --state states/pokemonred_bulbasaur_roundtrip2.state --num-envs 16384
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _require_torch():
    import importlib

    return importlib.import_module("torch")


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def _git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", True


def _system_meta() -> dict[str, Any]:
    torch = _require_torch()
    warp_version = None
    try:
        import warp as wp

        warp_version = getattr(wp, "__version__", None)
    except Exception:
        warp_version = None
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "warp_version": warp_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
    }


@dataclass(frozen=True)
class PPOParcelConfig:
    """Configuration for PPO parcel training."""

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
    gamma: float  # Long horizon: 0.999
    gae_lambda: float  # High lambda: 0.98
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
    steps_per_rollout: int  # Long rollouts for credit assignment
    checkpoint_every: int
    output_tag: str
    run_root: str
    # Model config
    cnn_channels: tuple[int, ...]
    mlp_hidden: int
    fusion_hidden: int

    def validate(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if self.steps_per_rollout < 1:
            raise ValueError("steps_per_rollout must be >= 1")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if not (0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be >= 1")
        if self.minibatch_size < 1:
            raise ValueError("minibatch_size must be >= 1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to Pokemon Red ROM")
    parser.add_argument("--state", required=True, help="Path to starting state file")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16384,
        help="Number of parallel envs (default: 16384)",
    )
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--action-codec", default="pokemonred_puffer_v1")
    parser.add_argument(
        "--max-steps", type=int, default=2048, help="Max steps per episode"
    )
    # Reward config
    parser.add_argument("--snow-bonus", type=float, default=0.01)
    parser.add_argument("--get-parcel-bonus", type=float, default=5.0)
    parser.add_argument("--deliver-bonus", type=float, default=10.0)
    parser.add_argument(
        "--dialogue-bonus",
        type=float,
        default=0.01,
        help="Bonus for entering dialogue/menu",
    )
    parser.add_argument(
        "--item-pickup-bonus", type=float, default=0.05, help="Bonus per item picked up"
    )
    parser.add_argument(
        "--no-curiosity-reset",
        action="store_true",
        help="Disable curiosity reset on parcel pickup",
    )
    # PPO hyperparameters - tuned for long horizon
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Discount factor (default: 0.999 for long horizon)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.98,
        help="GAE lambda (default: 0.98 for long horizon)",
    )
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument(
        "--value-coef", type=float, default=0.5, help="Value loss coefficient"
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient"
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument(
        "--grad-clip", type=float, default=0.5, help="Gradient clipping max norm"
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=4, help="PPO epochs per update"
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=4096,
        help="Minibatch size for PPO updates",
    )
    # Training config
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=100_000_000,
        help="Total environment steps",
    )
    parser.add_argument(
        "--steps-per-rollout",
        type=int,
        default=128,
        help="Steps per rollout (recommend 128-256)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=10, help="Checkpoint every N iterations"
    )
    parser.add_argument("--output-tag", default="ppo_parcel")
    parser.add_argument("--run-root", default="bench/runs/rl")
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint.pt to resume from"
    )
    # Model config
    parser.add_argument(
        "--cnn-channels",
        type=str,
        default="32,64,64",
        help="CNN channel sizes (comma-separated)",
    )
    parser.add_argument(
        "--mlp-hidden", type=int, default=128, help="MLP hidden size for senses/events"
    )
    parser.add_argument(
        "--fusion-hidden", type=int, default=512, help="Fusion layer hidden size"
    )
    return parser.parse_args()


def _resolve_iterations(cfg: PPOParcelConfig) -> int:
    steps_per_iter = int(cfg.num_envs * cfg.steps_per_rollout)
    return max(1, int((cfg.total_env_steps + steps_per_iter - 1) // steps_per_iter))


def _get_state_dict(model):
    """Get state dict from model, handling torch.compile wrapped models."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def _save_checkpoint(path, payload) -> None:
    torch = _require_torch()
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> int:
    args = _parse_args()
    cnn_channels = tuple(int(x) for x in args.cnn_channels.split(","))

    cfg = PPOParcelConfig(
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
        cnn_channels=cnn_channels,
        mlp_hidden=int(args.mlp_hidden),
        fusion_hidden=int(args.fusion_hidden),
    )
    cfg.validate()

    torch = _require_torch()
    if not _cuda_available() or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PPO parcel training.")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.rl.dual_lobe_model import DualLobeActorCritic
    from gbxcule.rl.experiment import Experiment
    from gbxcule.rl.goal_template import compute_sha256
    from gbxcule.rl.pokered_packed_parcel_env import (
        EVENTS_LENGTH,
        SENSES_DIM,
        PokeredPackedParcelEnv,
    )
    from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses

    print(f"Creating environment with {cfg.num_envs} parallel envs...")
    env = PokeredPackedParcelEnv(
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
    print(f"Environment created: {env.num_envs} envs, {env.num_actions} actions")

    # Create model
    print("Creating DualLobeActorCritic model...")
    model = DualLobeActorCritic(
        num_actions=env.num_actions,
        senses_dim=SENSES_DIM,
        events_dim=EVENTS_LENGTH,
        cnn_channels=cfg.cnn_channels,
        mlp_hidden=cfg.mlp_hidden,
        fusion_hidden=cfg.fusion_hidden,
    ).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Resume from checkpoint if provided
    resume_env_steps = 0
    ckpt = None
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        resume_env_steps = int(ckpt.get("env_steps", 0) or 0)
        print(f"  Resumed: env_steps={resume_env_steps}")

    # Setup experiment tracking
    rom_path = Path(cfg.rom)
    state_path = Path(cfg.state)
    rom_sha = compute_sha256(rom_path)
    state_sha = compute_sha256(state_path)
    git_commit, git_dirty = _git_info()

    meta = {
        "rom": {"rom_path": str(rom_path), "rom_sha256": rom_sha},
        "state": {"state_path": str(state_path), "state_sha256": state_sha},
        "env": {
            "num_envs": cfg.num_envs,
            "frames_per_step": cfg.frames_per_step,
            "release_after_frames": cfg.release_after_frames,
            "max_steps": cfg.max_steps,
            "snow_bonus": cfg.snow_bonus,
            "get_parcel_bonus": cfg.get_parcel_bonus,
            "deliver_bonus": cfg.deliver_bonus,
            "dialogue_bonus": cfg.dialogue_bonus,
            "item_pickup_bonus": cfg.item_pickup_bonus,
            "curiosity_reset_on_parcel": cfg.curiosity_reset_on_parcel,
        },
        "ppo": {
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "clip": cfg.clip,
            "value_coef": cfg.value_coef,
            "entropy_coef": cfg.entropy_coef,
            "lr": cfg.lr,
            "ppo_epochs": cfg.ppo_epochs,
            "minibatch_size": cfg.minibatch_size,
            "steps_per_rollout": cfg.steps_per_rollout,
        },
        "model": {
            "cnn_channels": cfg.cnn_channels,
            "mlp_hidden": cfg.mlp_hidden,
            "fusion_hidden": cfg.fusion_hidden,
        },
        "code": {"git_commit": git_commit, "git_dirty": git_dirty},
        "system": _system_meta(),
    }

    experiment = Experiment(
        algo="ppo_parcel",
        rom_id=rom_path.stem,
        tag=cfg.output_tag,
        run_root=cfg.run_root,
        meta=meta,
        config=asdict(cfg),
    )

    # Allocate rollout buffers
    print(f"Allocating rollout buffers ({cfg.steps_per_rollout} steps)...")
    rollout_pixels = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs, 1, 72, 20),
        dtype=torch.uint8,
        device="cuda",
    )
    rollout_senses = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs, SENSES_DIM),
        dtype=torch.float32,
        device="cuda",
    )
    rollout_events = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs, EVENTS_LENGTH),
        dtype=torch.uint8,
        device="cuda",
    )
    rollout_actions = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs),
        dtype=torch.int32,
        device="cuda",
    )
    rollout_rewards = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs),
        dtype=torch.float32,
        device="cuda",
    )
    rollout_dones = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs),
        dtype=torch.bool,
        device="cuda",
    )
    rollout_values = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs),
        dtype=torch.float32,
        device="cuda",
    )
    rollout_logprobs = torch.empty(
        (cfg.steps_per_rollout, cfg.num_envs),
        dtype=torch.float32,
        device="cuda",
    )

    # Initialize environment
    print("Resetting environment...")
    obs = env.reset_torch(seed=cfg.seed)
    pixels = obs["pixels"]
    senses = obs["senses"]
    events = obs["events"]

    iterations = _resolve_iterations(cfg)
    env_steps = resume_env_steps
    train_steps = 0
    start = time.time()
    repro = " ".join([json.dumps(arg) if " " in arg else arg for arg in sys.argv])

    # Stats tracking
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
                    # Get action from policy
                    logits, values = model(pixels, senses, events)
                    probs = torch.softmax(logits, dim=-1)
                    actions_i64 = torch.multinomial(probs, num_samples=1).squeeze(1)
                    logprobs = logprob_from_logits(logits, actions_i64)
                    actions = actions_i64.to(torch.int32)

                    # Store in rollout buffer
                    rollout_pixels[t].copy_(pixels)
                    rollout_senses[t].copy_(senses)
                    rollout_events[t].copy_(events)
                    rollout_actions[t].copy_(actions)
                    rollout_values[t].copy_(values)
                    rollout_logprobs[t].copy_(logprobs)

                    # Step environment
                    next_obs, reward, terminated, truncated, info = env.step_torch(
                        actions
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

                    # Update observations
                    pixels = next_obs["pixels"]
                    senses = next_obs["senses"]
                    events = next_obs["events"]

                # Get bootstrap value
                _, last_value = model(pixels, senses, events)

            env_steps += cfg.num_envs * cfg.steps_per_rollout

            # === PHASE 2: COMPUTE ADVANTAGES ===
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

            # Flatten rollout data
            batch_size = cfg.steps_per_rollout * cfg.num_envs
            flat_pixels = rollout_pixels.reshape(batch_size, 1, 72, 20)
            flat_senses = rollout_senses.reshape(batch_size, SENSES_DIM)
            flat_events = rollout_events.reshape(batch_size, EVENTS_LENGTH)
            flat_actions = rollout_actions.reshape(batch_size)
            flat_old_logprobs = rollout_logprobs.reshape(batch_size)
            flat_returns = returns.reshape(batch_size)
            flat_advantages = advantages.reshape(batch_size)

            # Normalize advantages
            adv_mean = flat_advantages.mean()
            adv_std = flat_advantages.std(unbiased=False)
            flat_advantages_norm = (flat_advantages - adv_mean) / (adv_std + 1e-8)

            # PPO epochs
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_clipfrac = 0.0
            num_updates = 0

            for _ in range(cfg.ppo_epochs):
                # Shuffle and create minibatches
                perm = torch.randperm(batch_size, device="cuda")

                for mb_start in range(0, batch_size, cfg.minibatch_size):
                    mb_end = min(mb_start + cfg.minibatch_size, batch_size)
                    mb_idx = perm[mb_start:mb_end]

                    # Get minibatch data
                    mb_pixels = flat_pixels[mb_idx]
                    mb_senses = flat_senses[mb_idx]
                    mb_events = flat_events[mb_idx]
                    mb_actions = flat_actions[mb_idx]
                    mb_old_logprobs = flat_old_logprobs[mb_idx]
                    mb_returns = flat_returns[mb_idx]
                    mb_advantages = flat_advantages_norm[mb_idx]

                    # Forward pass
                    logits, values = model(mb_pixels, mb_senses, mb_events)

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
                        normalize_adv=False,  # Already normalized
                    )

                    # Backward pass
                    optimizer.zero_grad()
                    losses["loss_total"].backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_clip
                        )
                    optimizer.step()

                    # Track stats
                    total_policy_loss += losses["loss_policy"].item()
                    total_value_loss += losses["loss_value"].item()
                    total_entropy += losses["entropy"].item()
                    total_clipfrac += losses["clipfrac"].item()
                    num_updates += 1
                    train_steps += 1

            # Average stats
            avg_policy_loss = (
                total_policy_loss / num_updates if num_updates > 0 else 0.0
            )
            avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0.0
            avg_entropy = total_entropy / num_updates if num_updates > 0 else 0.0
            avg_clipfrac = total_clipfrac / num_updates if num_updates > 0 else 0.0

            # === LOGGING ===
            iter_time = time.time() - iter_start
            wall_time = time.time() - start
            sps = (cfg.num_envs * cfg.steps_per_rollout) / iter_time

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

            # Print progress
            if (iter_idx + 1) % 10 == 0 or iter_idx == 0:
                print(
                    f"Iter {iter_idx + 1}/{iterations} | "
                    f"env_steps={env_steps:,} | "
                    f"sps={sps:,.0f} | "
                    f"reward={rollout_rewards.mean().item():.4f} | "
                    f"entropy={avg_entropy:.4f} | "
                    f"picked={parcels_picked} | "
                    f"delivered={parcels_delivered}"
                )

            # === CHECKPOINT ===
            if cfg.checkpoint_every > 0 and (iter_idx + 1) % cfg.checkpoint_every == 0:
                payload = {
                    "model": _get_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "env_steps": env_steps,
                    "train_steps": train_steps,
                    "parcels_picked": parcels_picked,
                    "parcels_delivered": parcels_delivered,
                }
                ckpt_name = f"ckpt_{iter_idx + 1}.pt"
                experiment.save_checkpoint(ckpt_name, payload)

    except Exception as exc:
        experiment.write_failure_bundle(
            kind="ppo_parcel_train",
            error=exc,
            extra={"env_steps": env_steps, "train_steps": train_steps},
            repro=repro,
        )
        raise
    finally:
        env.close()

    # Save final checkpoint
    payload = {
        "model": _get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
        "env_steps": env_steps,
        "train_steps": train_steps,
        "parcels_picked": parcels_picked,
        "parcels_delivered": parcels_delivered,
    }
    experiment.save_checkpoint("checkpoint.pt", payload)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Run dir: {experiment.run_dir}")
    print(f"  Total env steps: {env_steps:,}")
    print(f"  Total train steps: {train_steps:,}")
    print(f"  Parcels picked: {parcels_picked}")
    print(f"  Parcels delivered: {parcels_delivered}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
