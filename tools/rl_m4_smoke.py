"""Determinism smoke for M4 wrapper (reward/done/trunc + autoreset).

Usage:
  uv run python tools/rl_m4_smoke.py --rom <rom> --state <state> --goal-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch as torch_typing


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


def _hash_pixels_u64(pix, torch):  # type: ignore[no-untyped-def]
    flat = pix.reshape(pix.shape[0], -1).to(torch.int64)
    prime = torch.tensor(1315423911, device=flat.device, dtype=torch.int64)
    return (flat * prime).sum(dim=1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", required=True, help="Path to .state file")
    parser.add_argument("--goal-dir", required=True, help="Goal template directory")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--stack-k", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _run_trace(
    *,
    rom: str,
    state: str,
    goal_dir: str,
    num_envs: int,
    steps: int,
    frames_per_step: int,
    release_after_frames: int,
    stack_k: int | None,
    max_steps: int,
    seed: int,
):
    torch = _require_torch()
    from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv

    env = PokeredPixelsGoalEnv(
        rom,
        state_path=state,
        goal_dir=goal_dir,
        num_envs=num_envs,
        frames_per_step=frames_per_step,
        release_after_frames=release_after_frames,
        stack_k=stack_k,
        max_steps=max_steps,
    )
    try:
        env.reset(seed=seed)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        seq = []
        for step_idx in range(steps):
            actions = torch.randint(
                0,
                env.backend.num_actions,
                (num_envs,),
                device="cuda",
                dtype=torch.int32,
                generator=gen,
            )
            _, reward, done, trunc, info = env.step(actions)
            pix_hash = _hash_pixels_u64(env.pixels, torch).tolist()
            reset_mask = info.get("reset_mask")
            if reset_mask is None or not hasattr(reset_mask, "detach"):
                raise RuntimeError("reset_mask missing from env info")
            reset_mask_t = cast("torch_typing.Tensor", reset_mask)
            seq.append(
                {
                    "step": step_idx,
                    "pix_hash": pix_hash,
                    "reward": reward.detach().cpu().tolist(),
                    "done": done.detach().cpu().tolist(),
                    "trunc": trunc.detach().cpu().tolist(),
                    "reset": reset_mask_t.detach().cpu().tolist(),
                }
            )
        return seq
    finally:
        env.close()


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
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not goal_dir.exists():
        raise FileNotFoundError(f"Goal dir not found: {goal_dir}")

    seq_a = _run_trace(
        rom=str(rom_path),
        state=str(state_path),
        goal_dir=str(goal_dir),
        num_envs=args.num_envs,
        steps=args.steps,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        stack_k=args.stack_k,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    seq_b = _run_trace(
        rom=str(rom_path),
        state=str(state_path),
        goal_dir=str(goal_dir),
        num_envs=args.num_envs,
        steps=args.steps,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        stack_k=args.stack_k,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    if seq_a != seq_b:
        raise SystemExit("Determinism check failed: sequences differ")
    print(json.dumps({"ok": True, "steps": args.steps, "num_envs": args.num_envs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
