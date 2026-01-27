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
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-codec", default=None)
    parser.add_argument("--dump-trajectory", default=None)
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

    from gbxcule.rl.eval import run_greedy_eval
    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv

    env = PokeredPixelsGoalEnv(
        str(rom_path),
        state_path=str(state_path),
        goal_dir=str(goal_dir),
        num_envs=args.num_envs,
        frames_per_step=int(args.frames_per_step),
        release_after_frames=int(args.release_after_frames),
        action_codec=args.action_codec,
    )
    model = PixelActorCriticCNN(
        num_actions=env.backend.num_actions, in_frames=env.stack_k
    )
    model.to(device="cuda")

    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model"])

    summary = run_greedy_eval(
        env,
        model,
        episodes=int(args.episodes),
        seed=int(args.seed),
        trajectory_path=(
            Path(args.dump_trajectory) if args.dump_trajectory is not None else None
        ),
    )
    print(
        json.dumps(
            {
                "episodes": summary.episodes,
                "successes": summary.successes,
                "success_rate": summary.success_rate,
                "median_steps_to_goal": summary.median_steps_to_goal,
                "mean_return": summary.mean_return,
                "steps_p50_success": summary.steps_p50_success,
                "return_mean_success": summary.return_mean_success,
                "return_mean_fail": summary.return_mean_fail,
                "dist_at_end_p50": summary.dist_at_end_p50,
            }
        )
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
