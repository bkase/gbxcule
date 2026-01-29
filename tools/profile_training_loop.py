#!/usr/bin/env python3
"""Profile the full training loop with high env count to find bottlenecks."""

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
NUM_ENVS = 4096  # Higher env count for better GPU utilization
STEPS_PER_ROLLOUT = 128
NUM_UPDATES = 3


def sync():
    torch.cuda.synchronize()


def main():
    print(f"Profiling: {NUM_ENVS} envs, {STEPS_PER_ROLLOUT} steps/rollout...")
    print()

    # Create env
    print("Creating environment...")
    t0 = time.perf_counter()
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
    sync()
    print(f"  Env creation: {time.perf_counter() - t0:.3f}s")

    # Create model
    print("Creating model...")
    t0 = time.perf_counter()
    model = PixelActorCriticCNN(num_actions=env.backend.num_actions, in_frames=1)
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    sync()
    print(f"  Model creation: {time.perf_counter() - t0:.3f}s")

    # Create rollout buffer
    rollout = RolloutBuffer(
        steps=STEPS_PER_ROLLOUT,
        num_envs=NUM_ENVS,
        stack_k=1,
        device="cuda",
    )

    # Get initial observation
    env.backend.render_pixels_snapshot_torch()
    obs = env.backend.pixels_torch().unsqueeze(1)  # [N, 1, 72, 80]
    print(f"  Obs shape: {obs.shape}")
    print()

    # Random actions for testing
    actions = torch.randint(
        0, env.backend.num_actions, (NUM_ENVS,), dtype=torch.int32, device="cuda"
    )

    # Profile training updates
    for update_idx in range(NUM_UPDATES):
        print(f"Update {update_idx + 1}/{NUM_UPDATES}")
        rollout.reset()
        update_start = time.perf_counter()

        # Rollout phase
        t_rollout_start = time.perf_counter()
        t_model = 0.0
        t_step = 0.0
        t_render = 0.0
        t_buffer = 0.0

        for _step in range(STEPS_PER_ROLLOUT):
            # Model forward
            sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, values = model(obs)
                actions_i64 = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                ).squeeze(1)
                logprobs = logprob_from_logits(logits, actions_i64)
            sync()
            t_model += time.perf_counter() - t0

            actions = actions_i64.to(torch.int32)

            # Env step
            t0 = time.perf_counter()
            env.backend.step_torch(actions)
            sync()
            t_step += time.perf_counter() - t0

            # Render
            t0 = time.perf_counter()
            env.backend.render_pixels_snapshot_torch()
            next_obs = env.backend.pixels_torch().unsqueeze(1)
            sync()
            t_render += time.perf_counter() - t0

            # Dummy reward/done for profiling
            reward = torch.zeros(NUM_ENVS, device="cuda")
            done = torch.zeros(NUM_ENVS, dtype=torch.bool, device="cuda")

            # Store in buffer
            t0 = time.perf_counter()
            rollout.add(
                obs,
                actions,
                reward,
                done,
                values.detach(),
                logprobs.detach(),
            )
            sync()
            t_buffer += time.perf_counter() - t0

            obs = next_obs

        t_rollout = time.perf_counter() - t_rollout_start

        # PPO update phase
        t_ppo_start = time.perf_counter()

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

        sync()
        t_ppo = time.perf_counter() - t_ppo_start
        t_total = time.perf_counter() - update_start

        # Stats
        env_steps = NUM_ENVS * STEPS_PER_ROLLOUT
        sps = env_steps / t_total

        print("  Rollout breakdown:")
        ms = 1000 / STEPS_PER_ROLLOUT
        print(f"    Model forward:  {t_model:6.3f}s ({t_model * ms:6.2f}ms/step)")
        print(f"    Env step:       {t_step:6.3f}s ({t_step * ms:6.2f}ms/step)")
        print(f"    Render:         {t_render:6.3f}s ({t_render * ms:6.2f}ms/step)")
        print(f"    Buffer add:     {t_buffer:6.3f}s ({t_buffer * ms:6.2f}ms/step)")
        print(f"  Total rollout:    {t_rollout:6.3f}s")
        print(f"  PPO update:       {t_ppo:6.3f}s")
        print(f"  Total update:     {t_total:6.3f}s")
        print(f"  SPS:              {sps:,.0f}")
        print()

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
