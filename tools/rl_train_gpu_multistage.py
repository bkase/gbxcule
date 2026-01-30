#!/usr/bin/env python3
"""Multi-stage PPO training on GPU - trains a single policy across all stages.

Uses curriculum learning: envs progress through stages sequentially.
When an env completes a stage, it advances to the next stage.

Usage:
  uv run python tools/rl_train_gpu_multistage.py --rom red.gb
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

# Suppress PyTorch sm_121 warning - it still works
warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch


@dataclass
class StageConfig:
    name: str
    state_path: str
    goal_dir: str
    max_steps: int


# Define all stages
STAGES = [
    StageConfig(
        "exit_oak",
        "states/rl_stage1_exit_oak/start.state",
        "states/rl_stage1_exit_oak",
        500,
    ),
    StageConfig(
        "return_home",
        "states/rl_stage2_return_home/start.state",
        "states/rl_stage2_return_home",
        1500,
    ),
    StageConfig(
        "enter_house",
        "states/rl_stage3_enter_house/start.state",
        "states/rl_stage3_enter_house",
        100,
    ),
    StageConfig(
        "time_to_go",
        "states/rl_stage5_time_to_go/start.state",
        "states/rl_stage5_time_to_go",
        4000,
    ),
]


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
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument(
        "--num-envs", type=int, default=8192, help="Parallel environments"
    )
    parser.add_argument("--steps-per-rollout", type=int, default=128)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv
    from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses
    from gbxcule.rl.rollout import RolloutBuffer
    from gbxcule.rl.stage_goal_distance import compute_stage_goal_distance

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("bench/runs/rl_gpu_multistage") / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    ckpt_path = output_dir / "checkpoint.pt"

    cfg = TrainConfig(
        rom=str(Path(args.rom)),
        frames_per_step=int(args.frames_per_step),
        release_after_frames=int(args.release_after_frames),
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
    print(f"Stages: {[s.name for s in STAGES]}")
    print(f"Output: {output_dir}")

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Create one env per stage for now - we'll use stage 0 as the main env
    # and switch states when envs complete their current stage
    print(f"Creating GPU env with {cfg.num_envs} envs...")

    # Start all envs at stage 0
    env = PokeredPixelsGoalEnv(
        cfg.rom,
        state_path=STAGES[0].state_path,
        goal_dir=STAGES[0].goal_dir,
        num_envs=cfg.num_envs,
        frames_per_step=cfg.frames_per_step,
        release_after_frames=cfg.release_after_frames,
        max_steps=max(s.max_steps for s in STAGES),  # Use max for flexibility
    )

    # Load goal templates for all stages
    from gbxcule.core.reset_cache import ResetCache
    from gbxcule.rl.goal_template import load_goal_template

    # Initialize backend by calling reset first
    print("Initializing backend...")
    env.reset(seed=cfg.seed)

    stage_goals = []
    stage_caches = []

    for i, stage in enumerate(STAGES):
        print(f"Loading stage {i}: {stage.name}")
        template, meta = load_goal_template(
            Path(stage.goal_dir),
            action_codec_id=env.backend.action_codec.id,
            frames_per_step=cfg.frames_per_step,
            release_after_frames=cfg.release_after_frames,
            stack_k=1,
            dist_metric=None,
            pipeline_version=None,
        )
        goal_t = torch.tensor(template, device="cuda", dtype=torch.uint8)
        if goal_t.ndim == 2:
            goal_t = goal_t.unsqueeze(0)  # [1, 72, 80]
        stage_goals.append(goal_t)
    stage_goals_f = torch.stack(stage_goals).to(dtype=torch.float32)

    # Create reset caches after setting up env with each stage's state
    print("Creating reset caches for each stage...")
    for i, stage in enumerate(STAGES):
        # Load state file into env 0, then create reset cache from it
        env.backend.load_state_file(stage.state_path, env_idx=0)
        cache = ResetCache.from_backend(env.backend, env_idx=0)
        stage_caches.append(cache)
        print(f"  Stage {i} ({stage.name}): cache created")

    # Initialize all envs to stage 0
    print("Initializing env stages...")
    env_stages = torch.zeros(cfg.num_envs, dtype=torch.int64, device="cuda")
    all_mask = torch.ones(cfg.num_envs, dtype=torch.uint8, device="cuda")
    stage_caches[0].apply_mask_torch(all_mask)
    print("All envs initialized to stage 0")

    # Model on CUDA (slower but avoids data transfer overhead)
    model = PixelActorCriticCNN(
        num_actions=env.backend.num_actions,
        in_frames=1,  # Single frame, not stacked
    )
    model.to(device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rollout = RolloutBuffer(
        steps=cfg.steps_per_rollout,
        num_envs=cfg.num_envs,
        stack_k=1,
        device="cuda",
    )

    start_update = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = int(ckpt.get("update", 0))
        if "env_stages" in ckpt:
            env_stages = ckpt["env_stages"].to("cuda")

    # Get initial obs
    print("Getting initial observation...")
    env.backend.render_pixels_snapshot_torch()
    obs = env.backend.pixels_torch().unsqueeze(1)  # [N, 1, 72, 80]
    print(f"Initial obs shape: {obs.shape}, device: {obs.device}")

    run_id = output_dir.name or uuid4().hex[:8]
    train_start = time.time()
    print("Starting training loop...")

    # Per-stage stats
    stage_goals_count = [0] * len(STAGES)
    stage_episodes_count = [0] * len(STAGES)

    # Episode tracking
    episode_steps = torch.zeros(cfg.num_envs, dtype=torch.int32, device="cuda")
    prev_dist = torch.ones(cfg.num_envs, dtype=torch.float32, device="cuda")

    max_steps_tensor = torch.tensor(
        [stage.max_steps for stage in STAGES], device="cuda"
    )
    tau = 0.05
    step_cost = -0.01
    alpha = 1.0
    goal_bonus = 10.0

    with log_path.open("a", encoding="utf-8") as log_f:
        if start_update == 0:
            log_f.write(
                json.dumps(
                    {
                        "meta": {
                            "run_id": run_id,
                            "stages": [s.name for s in STAGES],
                            "num_envs": cfg.num_envs,
                            "seed": cfg.seed,
                        },
                        "config": asdict(cfg),
                    }
                )
                + "\n"
            )

        for update_idx in range(start_update, cfg.updates):
            rollout.reset()
            update_start = time.time()
            update_stage_goals = torch.zeros(
                (len(STAGES),), device="cuda", dtype=torch.int64
            )
            update_stage_episodes = torch.zeros(
                (len(STAGES),), device="cuda", dtype=torch.int64
            )

            for _step in range(cfg.steps_per_rollout):
                with torch.no_grad():
                    logits, values = model(obs)
                    actions_i64 = torch.multinomial(
                        torch.softmax(logits, dim=-1), num_samples=1
                    ).squeeze(1)
                    logprobs = logprob_from_logits(logits, actions_i64)

                actions = actions_i64.to(torch.int32)

                # Step env
                env.backend.step_torch(actions)
                env.backend.render_pixels_snapshot_torch()
                next_obs = env.backend.pixels_torch().unsqueeze(1)
                episode_steps += 1

                # Compute distances for current stage goals
                curr_dist = compute_stage_goal_distance(
                    next_obs, stage_goals_f, env_stages
                )

                # Check done (goal reached)
                done = curr_dist < tau

                # Check truncation (max steps per stage)
                env_max_steps = max_steps_tensor[env_stages]
                trunc = episode_steps >= env_max_steps

                # Compute reward
                reward = torch.full((cfg.num_envs,), step_cost, device="cuda")
                reward += alpha * (prev_dist - curr_dist)
                reward[done] += goal_bonus

                reset_mask = done | trunc

                # Store in rollout buffer
                rollout.add(
                    obs,
                    actions,
                    reward,
                    reset_mask,
                    values.detach(),
                    logprobs.detach(),
                )

                # Track stats and handle stage transitions
                for stage_idx in range(len(STAGES)):
                    stage_mask = env_stages == stage_idx
                    update_stage_goals[stage_idx] += (done & stage_mask).sum()
                    update_stage_episodes[stage_idx] += (reset_mask & stage_mask).sum()

                # Handle resets and stage progression (no per-step sync)
                advancing = done & (env_stages < len(STAGES) - 1)
                env_stages = torch.where(advancing, env_stages + 1, env_stages)

                failing = trunc | (done & (env_stages == len(STAGES) - 1))
                env_stages = torch.where(
                    failing, torch.zeros_like(env_stages), env_stages
                )

                for stage_idx in range(len(STAGES)):
                    stage_reset_mask = reset_mask & (env_stages == stage_idx)
                    stage_caches[stage_idx].apply_mask_torch(
                        stage_reset_mask.to(torch.uint8)
                    )

                episode_steps = torch.where(
                    reset_mask, torch.zeros_like(episode_steps), episode_steps
                )

                # Re-render after reset (always, to avoid host sync)
                env.backend.render_pixels_snapshot_torch()
                next_obs = env.backend.pixels_torch().unsqueeze(1)
                curr_dist = compute_stage_goal_distance(
                    next_obs, stage_goals_f, env_stages
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

            # Stats
            update_stage_goals_list = update_stage_goals.detach().cpu().tolist()
            update_stage_episodes_list = update_stage_episodes.detach().cpu().tolist()
            for i in range(len(STAGES)):
                stage_goals_count[i] += update_stage_goals_list[i]
                stage_episodes_count[i] += update_stage_episodes_list[i]

            env_steps = (update_idx + 1) * cfg.num_envs * cfg.steps_per_rollout
            update_time = time.time() - update_start
            sps = cfg.num_envs * cfg.steps_per_rollout / update_time

            # Stage distribution
            stage_dist = [(env_stages == i).sum().item() for i in range(len(STAGES))]

            record = {
                "run_id": run_id,
                "update": update_idx,
                "env_steps": env_steps,
                "wall_time_s": time.time() - train_start,
                "loss_total": float(losses["loss_total"].item()),
                "entropy": float(losses["entropy"].item()),
                "sps": int(sps),
                "stage_goals": update_stage_goals_list,
                "stage_episodes": update_stage_episodes_list,
                "stage_dist": stage_dist,
                "total_stage_goals": stage_goals_count.copy(),
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

            if (update_idx + 1) % 10 == 0 or update_idx == 0:
                sr_str = " | ".join(
                    [
                        f"S{i}:{update_stage_goals[i]}/{update_stage_episodes[i]}"
                        for i in range(len(STAGES))
                        if update_stage_episodes[i] > 0
                    ]
                )
                print(
                    f"Update {update_idx + 1}/{cfg.updates} | "
                    f"Steps: {env_steps:,} | "
                    f"SPS: {sps:,.0f} | "
                    f"{sr_str} | "
                    f"Dist: {stage_dist}"
                )

            # Save checkpoint
            if (update_idx + 1) % 50 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "update": update_idx + 1,
                        "env_stages": env_stages.cpu(),
                        "stage_goals": stage_goals_count,
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
            "env_stages": env_stages.cpu(),
            "stage_goals": stage_goals_count,
        },
        ckpt_path,
    )

    env.close()

    print()
    print("=" * 60)
    print("MULTI-STAGE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total updates: {cfg.updates}")
    print(f"Total env steps: {cfg.updates * cfg.num_envs * cfg.steps_per_rollout:,}")
    for i, stage in enumerate(STAGES):
        sr = stage_goals_count[i] / max(1, stage_episodes_count[i])
        goals = stage_goals_count[i]
        print(f"Stage {i} ({stage.name}): {goals} goals, {100 * sr:.1f}% success")
    print(f"Checkpoint: {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
