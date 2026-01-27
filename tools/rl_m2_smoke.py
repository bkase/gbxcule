"""Smoke test for RL M2 pixels wrapper (determinism + stream correctness).

Usage:
  uv run python tools/rl_m2_smoke.py --rom bench/roms/out/BG_SCROLL_SIGNED.gb
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


def _hash_pixels_u64(pix, torch):  # type: ignore[no-untyped-def]
    flat = pix.reshape(pix.shape[0], -1).to(torch.int64)
    prime = torch.tensor(1315423911, device=flat.device, dtype=torch.int64)
    return (flat * prime).sum(dim=1)


def _run_trace(
    rom_path: str,
    *,
    num_envs: int,
    steps: int,
    frames_per_step: int,
    release_after_frames: int,
    seed: int,
) -> list[list[int]]:
    torch = _require_torch()
    from gbxcule.rl.pokered_pixels_env import PokeredPixelsEnv

    env = PokeredPixelsEnv(
        rom_path,
        num_envs=num_envs,
        frames_per_step=frames_per_step,
        release_after_frames=release_after_frames,
    )
    try:
        env.reset(seed=seed)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        hashes: list[list[int]] = []
        for step_idx in range(steps):
            actions = torch.randint(
                0,
                env.backend.num_actions,
                (num_envs,),
                device="cuda",
                dtype=torch.int32,
                generator=gen,
            )
            env.step(actions)
            h = _hash_pixels_u64(env.pixels, torch).tolist()
            hashes.append(h)
            print(
                json.dumps(
                    {
                        "step": step_idx,
                        "hashes": h,
                    }
                )
            )
        return hashes
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rom",
        default="bench/roms/out/BG_SCROLL_SIGNED.gb",
        help="Path to ROM (default: BG_SCROLL_SIGNED.gb).",
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--frames-per-step", type=int, default=1)
    parser.add_argument("--release-after-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not _cuda_available():
        print(json.dumps({"skipped": "CUDA not available"}))
        return 0

    torch = _require_torch()
    if not torch.cuda.is_available():
        print(json.dumps({"skipped": "torch CUDA not available"}))
        return 0

    rom_path = str(Path(args.rom))
    if not Path(rom_path).exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")

    hashes_a = _run_trace(
        rom_path,
        num_envs=args.num_envs,
        steps=args.steps,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        seed=args.seed,
    )
    hashes_b = _run_trace(
        rom_path,
        num_envs=args.num_envs,
        steps=args.steps,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        seed=args.seed,
    )
    if hashes_a != hashes_b:
        raise SystemExit("Determinism check failed: hash sequences differ")
    print(json.dumps({"ok": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
