"""Greedy evaluation for PPO checkpoint (pixels-only, CUDA).

Usage:
  uv run python tools/rl_m5_eval.py --rom <rom> --state <state> \
    --goal-dir <dir> --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", required=True, help="Path to .state file")
    parser.add_argument("--goal-dir", required=True, help="Goal template directory")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=4)
    return parser.parse_args()


def _eval_greedy(env, model, episodes: int):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    successes = 0
    steps_to_goal: list[int] = []
    returns: list[float] = []
    for _ in range(episodes):
        obs = env.reset()
        done = torch.zeros((env.num_envs,), device="cuda", dtype=torch.bool)
        ep_return = torch.zeros((env.num_envs,), device="cuda", dtype=torch.float32)
        ep_steps = torch.zeros((env.num_envs,), device="cuda", dtype=torch.int32)
        while True:
            logits, _ = model(obs)
            actions = torch.argmax(logits, dim=-1).to(torch.int32)
            obs, reward, done, trunc, _ = env.step(actions)
            ep_return.add_(reward)
            ep_steps.add_(1)
            if torch.any(done | trunc):
                break
        mask = done | trunc
        successes += int(done[mask].sum().item())
        steps_to_goal.extend(ep_steps[mask].tolist())
        returns.extend(ep_return[mask].tolist())
    success_rate = successes / max(1, len(steps_to_goal))
    median_steps = (
        int(torch.tensor(steps_to_goal).median().item()) if steps_to_goal else 0
    )
    mean_return = float(torch.tensor(returns).mean().item()) if returns else 0.0
    return success_rate, median_steps, mean_return


def main() -> int:
    args = _parse_args()
    if not _cuda_available():
        print(json.dumps({"skipped": "CUDA not available"}))
        return 0
    torch = _require_torch()
    if not torch.cuda.is_available():
        print(json.dumps({"skipped": "torch CUDA not available"}))
        return 0

    rom_path = Path(args.rom)
    state_path = Path(args.state)
    goal_dir = Path(args.goal_dir)
    ckpt_path = Path(args.checkpoint)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not goal_dir.exists():
        raise FileNotFoundError(f"Goal dir not found: {goal_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv

    env = PokeredPixelsGoalEnv(
        str(rom_path),
        state_path=str(state_path),
        goal_dir=str(goal_dir),
        num_envs=args.num_envs,
    )
    model = PixelActorCriticCNN(
        num_actions=env.backend.num_actions, in_frames=env.stack_k
    )
    model.to(device="cuda")

    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model"])

    success_rate, median_steps, mean_return = _eval_greedy(env, model, args.episodes)
    print(
        json.dumps(
            {
                "success_rate": success_rate,
                "median_steps_to_goal": median_steps,
                "mean_return": mean_return,
            }
        )
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
