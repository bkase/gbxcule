#!/usr/bin/env python3
"""Simple training loop profile - minimal syncs for realistic timing."""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*cuda capability.*")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gbxcule.rl.models import PixelActorCriticCNN
from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv
from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses
from gbxcule.rl.rollout import RolloutBuffer

ROM = "red.gb"
NUM_ENVS = 4096  # Higher for better GPU utilization
STEPS_PER_ROLLOUT = 128
NUM_UPDATES = 5


def main():
    print(
        f"Simple training profile: {NUM_ENVS} envs, {STEPS_PER_ROLLOUT} steps/rollout"
    )
    print()

    # Create env
    env = PokeredPixelsGoalEnv(
        ROM,
        state_path="states/rl_stage1_exit_oak/start.state",
        goal_dir="states/rl_stage1_exit_oak",
        num_envs=NUM_ENVS,
        frames_per_step=24,
        release_after_frames=8,
        max_steps=500,
    )
    env.reset(seed=1234)
    torch.cuda.synchronize()
    print("Environment ready")

    # Create model
    model = PixelActorCriticCNN(num_actions=env.backend.num_actions, in_frames=1)
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    print("Model ready")

    # Create rollout buffer
    rollout = RolloutBuffer(
        steps=STEPS_PER_ROLLOUT,
        num_envs=NUM_ENVS,
        stack_k=1,
        device="cuda",
    )

    # Get initial observation
    env.backend.render_pixels_snapshot_torch()
    obs = env.backend.pixels_torch().unsqueeze(1)
    print(f"Obs shape: {obs.shape}")
    print()

    # Training loop
    for update_idx in range(NUM_UPDATES):
        rollout.reset()
        torch.cuda.synchronize()
        update_start = time.perf_counter()

        # --- Rollout phase ---
        rollout_start = time.perf_counter()
        for _step in range(STEPS_PER_ROLLOUT):
            with torch.no_grad():
                logits, values = model(obs)
                actions_i64 = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                ).squeeze(1)
                logprobs = logprob_from_logits(logits, actions_i64)

            actions = actions_i64.to(torch.int32)
            env.backend.step_torch(actions)
            env.backend.render_pixels_snapshot_torch()
            next_obs = env.backend.pixels_torch().unsqueeze(1)

            # Dummy reward/done
            reward = torch.zeros(NUM_ENVS, device="cuda")
            done = torch.zeros(NUM_ENVS, dtype=torch.bool, device="cuda")

            rollout.add(obs, actions, reward, done, values.detach(), logprobs.detach())
            obs = next_obs

        torch.cuda.synchronize()
        rollout_time = time.perf_counter() - rollout_start

        # --- PPO update phase ---
        ppo_start = time.perf_counter()

        with torch.no_grad():
            _, last_value = model(obs)

        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            last_value,
            gamma=0.99,
            gae_lambda=0.95,
        )

        batch = rollout.as_batch(flatten_obs=True)
        logits, values = model(batch["obs_u8"])
        losses = ppo_losses(
            logits,
            batch["actions"],
            batch["logprobs"],
            returns.reshape(-1),
            advantages.reshape(-1),
            values,
            clip=0.1,
            value_coef=0.5,
            entropy_coef=0.01,
        )

        optimizer.zero_grad()
        losses["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        torch.cuda.synchronize()
        ppo_time = time.perf_counter() - ppo_start
        total_time = time.perf_counter() - update_start

        env_steps = NUM_ENVS * STEPS_PER_ROLLOUT
        sps = env_steps / total_time
        print(
            f"Update {update_idx + 1}/{NUM_UPDATES}: "
            f"rollout={rollout_time:.2f}s, ppo={ppo_time:.2f}s, "
            f"total={total_time:.2f}s, SPS={sps:,.0f}"
        )

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
