#!/usr/bin/env python3
"""Async PPO benchmark using the experiment harness."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from gbxcule.rl.async_ppo_engine import AsyncPPOEngine, AsyncPPOEngineConfig
from gbxcule.rl.experiment import Experiment


def _require_torch():  # type: ignore[no-untyped-def]
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
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory root for experiment runs (default: bench/runs/rl)",
    )
    parser.add_argument("--tag", default="bench", help="Run tag for the output dir")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info() -> tuple[str | None, bool]:
    commit = None
    dirty = False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            commit = result.stdout.strip() or None
    except Exception:
        commit = None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = bool(status.stdout.strip())
    except Exception:
        dirty = False
    return commit, dirty


def _system_info(torch) -> dict[str, Any]:
    warp_version = None
    try:
        import warp as wp

        warp_version = getattr(wp, "__version__", None)
    except Exception:
        warp_version = None
    gpu_name = None
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return {
        "platform": platform.system(),
        "python": platform.python_version(),
        "torch_version": str(torch.__version__),
        "warp_version": warp_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }


def _build_meta(
    *,
    rom_path: Path,
    state_path: Path,
    num_envs: int,
    frames_per_step: int,
    release_after_frames: int,
    stack_k: int,
    obs_format: str,
    action_codec_id: str,
    algo_name: str,
    algo_version: str,
    torch,
) -> dict[str, Any]:
    git_commit, git_dirty = _git_info()
    return {
        "rom": {
            "rom_path": str(rom_path),
            "rom_sha256": _sha256(rom_path),
        },
        "state": {
            "state_path": str(state_path),
            "state_sha256": _sha256(state_path),
        },
        "env": {
            "num_envs": int(num_envs),
            "frames_per_step": int(frames_per_step),
            "release_after_frames": int(release_after_frames),
            "stack_k": int(stack_k),
        },
        "pipeline": {
            "obs_format": obs_format,
            "action_codec_id": action_codec_id,
        },
        "algo": {
            "algo_name": algo_name,
            "algo_version": algo_version,
        },
        "code": {
            "git_commit": git_commit,
            "git_dirty": bool(git_dirty),
        },
        "system": _system_info(torch),
    }


def main() -> int:
    args = _parse_args()
    if not _cuda_available():
        return 0

    torch = _require_torch()
    if not torch.cuda.is_available():
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

    config = AsyncPPOEngineConfig(
        rom_path=str(rom_path),
        state_path=str(state_path),
        goal_dir=str(goal_dir),
        device="cuda",
        num_envs=args.num_envs,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        steps_per_rollout=args.steps_per_rollout,
        updates=args.updates,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )

    engine = AsyncPPOEngine(config)
    try:
        meta = _build_meta(
            rom_path=rom_path,
            state_path=state_path,
            num_envs=args.num_envs,
            frames_per_step=args.frames_per_step,
            release_after_frames=args.release_after_frames,
            stack_k=1,
            obs_format="u8",
            action_codec_id=engine.backend.action_codec.id,
            algo_name="ppo",
            algo_version="async_ppo_engine",
            torch=torch,
        )
        output_root = (
            Path(args.output_dir) if args.output_dir else Path("bench/runs/rl")
        )
        exp = Experiment(
            algo="ppo",
            rom_id=rom_path.stem,
            tag=args.tag,
            run_root=output_root,
            meta=meta,
            config=config,
        )
        engine.experiment = exp
        engine.run(updates=args.updates)
    finally:
        engine.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
