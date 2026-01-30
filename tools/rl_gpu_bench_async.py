#!/usr/bin/env python3
"""Async PPO benchmark with double-buffered rollouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="red.gb", help="Path to ROM")
    parser.add_argument(
        "--state", default="states/rl_stage1_exit_oak/start.state", help="State path"
    )
    parser.add_argument(
        "--goal-dir", default="states/rl_stage1_exit_oak", help="Goal template dir"
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--steps-per-rollout", type=int, default=32)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not _cuda_available():
        print(json.dumps({"skipped": "CUDA not available"}))
        return 0

    torch = _require_torch()
    if not torch.cuda.is_available():
        print(json.dumps({"skipped": "torch CUDA not available"}))
        return 0

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gbxcule.backends.warp_vec import WarpVecCudaBackend
    from gbxcule.core.reset_cache import ResetCache
    from gbxcule.rl.async_ppo import AsyncPPOBufferManager
    from gbxcule.rl.goal_template import load_goal_template
    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_update_minibatch
    from gbxcule.rl.rollout import RolloutBuffer

    rom_path = Path(args.rom)
    state_path = Path(args.state)
    goal_dir = Path(args.goal_dir)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not goal_dir.exists():
        raise FileNotFoundError(f"Goal dir not found: {goal_dir}")

    backend = WarpVecCudaBackend(
        str(rom_path),
        num_envs=args.num_envs,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        obs_dim=32,
        render_pixels=True,
    )
    backend.reset(seed=args.seed)

    backend.load_state_file(str(state_path), env_idx=0)
    reset_cache = ResetCache.from_backend(backend, env_idx=0)
    all_mask = torch.ones(args.num_envs, dtype=torch.uint8, device="cuda")
    reset_cache.apply_mask_torch(all_mask)

    template, _ = load_goal_template(
        Path(args.goal_dir),
        action_codec_id=backend.action_codec.id,
        frames_per_step=None,
        release_after_frames=None,
        stack_k=1,
        dist_metric=None,
        pipeline_version=None,
    )
    goal_np = template.squeeze(0) if template.ndim == 3 else template
    goal = torch.tensor(goal_np, device="cuda", dtype=torch.uint8).unsqueeze(0)
    goal_f = goal.to(dtype=torch.float32)

    actor_model = PixelActorCriticCNN(num_actions=backend.num_actions, in_frames=1).to(
        "cuda"
    )
    learner_model = PixelActorCriticCNN(
        num_actions=backend.num_actions, in_frames=1
    ).to("cuda")
    optimizer = torch.optim.Adam(learner_model.parameters(), lr=args.lr)

    rollout_buffers = [
        RolloutBuffer(
            steps=args.steps_per_rollout,
            num_envs=args.num_envs,
            stack_k=1,
            device="cuda",
        )
        for _ in range(2)
    ]

    manager = AsyncPPOBufferManager(num_buffers=2)
    actor_stream = torch.cuda.Stream()
    learner_stream = torch.cuda.Stream()

    # Warm start obs and state
    with torch.cuda.stream(actor_stream):
        backend.render_pixels_snapshot_torch()
        obs = backend.pixels_torch().unsqueeze(1)
        episode_steps = torch.zeros(args.num_envs, dtype=torch.int32, device="cuda")
        prev_dist = torch.ones(args.num_envs, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    tau = 0.05
    step_cost = -0.01
    alpha = 1.0
    goal_bonus = 10.0
    max_steps = 3000

    def _sync_actor_weights():
        for p_actor, p_learner in zip(
            actor_model.parameters(), learner_model.parameters(), strict=True
        ):
            p_actor.data.copy_(p_learner.data)

    start = time.perf_counter()
    for update_idx in range(args.updates):
        buf_idx = update_idx % 2
        prev_idx = (update_idx - 1) % 2

        with torch.cuda.stream(learner_stream):
            if update_idx > 0:
                manager.wait_ready(prev_idx, learner_stream)
                rollout = rollout_buffers[prev_idx]
                with torch.no_grad():
                    _, last_value = learner_model(obs)
                advantages, returns = compute_gae(
                    rollout.rewards,
                    rollout.values,
                    rollout.dones,
                    last_value,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
                batch = rollout.as_batch(flatten_obs=True)
                ppo_update_minibatch(
                    model=learner_model,
                    optimizer=optimizer,
                    obs=batch["obs_u8"],
                    actions=batch["actions"],
                    old_logprobs=batch["logprobs"],
                    returns=returns.reshape(-1),
                    advantages=advantages.reshape(-1),
                    clip=args.clip,
                    value_coef=args.value_coef,
                    entropy_coef=args.entropy_coef,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    grad_clip=args.grad_clip,
                )
                _sync_actor_weights()
                manager.mark_free(prev_idx, learner_stream)

        with torch.cuda.stream(actor_stream):
            manager.wait_free(buf_idx, actor_stream)
            rollout = rollout_buffers[buf_idx]
            rollout.reset()
            for _ in range(args.steps_per_rollout):
                with torch.no_grad():
                    logits, values = actor_model(obs)
                    actions_i64 = torch.multinomial(
                        torch.softmax(logits, dim=-1), num_samples=1
                    ).squeeze(1)
                    logprobs = logprob_from_logits(logits, actions_i64)

                actions = actions_i64.to(torch.int32)
                backend.step_torch(actions)
                backend.render_pixels_snapshot_torch()
                next_obs = backend.pixels_torch().unsqueeze(1)
                episode_steps = episode_steps + 1

                diff = torch.abs(next_obs.float() - goal_f)
                curr_dist = diff.mean(dim=(1, 2, 3)) / 3.0
                done = curr_dist < tau
                trunc = episode_steps >= max_steps
                reward = torch.full((args.num_envs,), step_cost, device="cuda")
                reward += alpha * (prev_dist - curr_dist)
                reward[done] += goal_bonus
                reset_mask = done | trunc

                rollout.add(
                    obs,
                    actions,
                    reward,
                    reset_mask,
                    values.detach(),
                    logprobs.detach(),
                )

                reset_cache.apply_mask_torch(reset_mask.to(torch.uint8))
                episode_steps = torch.where(
                    reset_mask, torch.zeros_like(episode_steps), episode_steps
                )
                backend.render_pixels_snapshot_torch()
                next_obs = backend.pixels_torch().unsqueeze(1)
                curr_dist = (
                    torch.abs(next_obs.float() - goal_f).mean(dim=(1, 2, 3)) / 3.0
                )

                prev_dist = curr_dist
                obs = next_obs

            manager.mark_ready(buf_idx, actor_stream, policy_version=update_idx)

    with torch.cuda.stream(learner_stream):
        last_idx = (args.updates - 1) % 2
        manager.wait_ready(last_idx, learner_stream)
        rollout = rollout_buffers[last_idx]
        with torch.no_grad():
            _, last_value = learner_model(obs)
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            last_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        batch = rollout.as_batch(flatten_obs=True)
        ppo_update_minibatch(
            model=learner_model,
            optimizer=optimizer,
            obs=batch["obs_u8"],
            actions=batch["actions"],
            old_logprobs=batch["logprobs"],
            returns=returns.reshape(-1),
            advantages=advantages.reshape(-1),
            clip=args.clip,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            grad_clip=args.grad_clip,
        )
        _sync_actor_weights()
        manager.mark_free(last_idx, learner_stream)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    env_steps = args.num_envs * args.steps_per_rollout * args.updates
    sps = env_steps / elapsed

    payload = {
        "num_envs": int(args.num_envs),
        "frames_per_step": int(args.frames_per_step),
        "release_after_frames": int(args.release_after_frames),
        "steps_per_rollout": int(args.steps_per_rollout),
        "updates": int(args.updates),
        "ppo_epochs": int(args.ppo_epochs),
        "minibatch_size": int(args.minibatch_size),
        "env_steps": int(env_steps),
        "elapsed_s": float(elapsed),
        "sps": float(sps),
    }

    output = json.dumps(payload)
    print(output)
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")

    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
