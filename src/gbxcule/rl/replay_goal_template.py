"""Replay a trace and validate goal detection (exit 0/1)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gbxcule.rl.goal_match import compute_dist_l1, compute_done, update_consecutive
from gbxcule.rl.goal_template import load_actions_trace_jsonl, load_goal_template
from gbxcule.rl.pokered_pixels_env import PokeredPixelsEnv


def _require_torch() -> Any:
    import importlib

    return importlib.import_module("torch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", required=True, help="Path to .state file")
    parser.add_argument("--actions", required=True, help="Path to actions.jsonl")
    parser.add_argument("--goal-dir", required=True, help="Goal template directory")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--stack-k", type=int, default=None)
    parser.add_argument("--action-codec", default=None)
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--k-consecutive", type=int, default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibrate-output", default=None)
    return parser.parse_args()


def _infer_num_envs(actions: list[list[int]]) -> int:
    sizes = {len(step) for step in actions}
    if len(sizes) != 1:
        raise ValueError(f"Inconsistent action widths: {sorted(sizes)}")
    num_envs = next(iter(sizes))
    if num_envs < 1:
        raise ValueError("actions trace must include at least one env")
    return num_envs


def _write_failure_receipt(
    output_dir: Path,
    *,
    dist_curve: list[list[float]],
    last_frames: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dist_path = output_dir / "dist_curve.json"
    dist_path.write_text(json.dumps(dist_curve, indent=2), encoding="utf-8")
    np.save(output_dir / "last_frames.npy", last_frames)
    try:
        from PIL import Image
    except Exception:
        return
    palette = np.array([255, 170, 85, 0], dtype=np.uint8)
    frame = last_frames[0]
    img = Image.fromarray(palette[frame], mode="L")
    img.save(output_dir / "last_frame_env0.png")


def main() -> int:
    args = _parse_args()
    rom_path = Path(args.rom)
    state_path = Path(args.state)
    actions_path = Path(args.actions)
    goal_dir = Path(args.goal_dir)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions not found: {actions_path}")

    template, meta = load_goal_template(
        goal_dir,
        action_codec_id=args.action_codec,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        stack_k=args.stack_k,
        dist_metric=None,
        pipeline_version=None,
    )
    action_codec_id = args.action_codec or meta.action_codec_id
    if action_codec_id != meta.action_codec_id:
        raise ValueError(
            f"action_codec_id {action_codec_id} != template {meta.action_codec_id}"
        )

    actions = load_actions_trace_jsonl(actions_path)
    num_envs = _infer_num_envs(actions)

    torch = _require_torch()
    template_t = torch.tensor(template, device="cuda", dtype=torch.uint8)
    tau = float(args.tau) if args.tau is not None else float(meta.tau)
    k_consecutive = (
        int(args.k_consecutive)
        if args.k_consecutive is not None
        else int(meta.k_consecutive)
    )

    env = PokeredPixelsEnv(
        str(rom_path),
        state_path=str(state_path),
        num_envs=num_envs,
        frames_per_step=args.frames_per_step,
        release_after_frames=args.release_after_frames,
        stack_k=args.stack_k or meta.stack_k,
        action_codec=action_codec_id,
    )
    log_file = None
    dist_curve: list[list[float]] = []
    dist_env0: list[float] = []
    done_any = False
    record_dist = args.log_jsonl is not None or args.output_dir is not None
    try:
        env.reset()
        consec = torch.zeros((num_envs,), device="cuda", dtype=torch.int32)
        if args.log_jsonl:
            log_path = Path(args.log_jsonl)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
        for step_idx, step_actions in enumerate(actions):
            act = torch.tensor(step_actions, device="cuda", dtype=torch.int32)
            env.step(act)
            frame = env.obs if template_t.ndim == 3 else env.pixels
            dist = compute_dist_l1(frame, template_t)
            consec = update_consecutive(consec, dist, tau=tau)
            done = compute_done(consec, k_consecutive=k_consecutive)
            done_any = bool(torch.any(done).item())
            dist_cpu = dist.detach().cpu().tolist() if record_dist else None
            if dist_cpu is not None:
                dist_curve.append(dist_cpu)
            dist_env0.append(float(dist[0].item()))
            if log_file is not None:
                log_file.write(
                    json.dumps(
                        {
                            "step": step_idx,
                            "actions": step_actions,
                            "dist": dist_cpu,
                            "done_any": done_any,
                        }
                    )
                    + "\n"
                )
            if done_any and not args.calibrate:
                break
        if args.calibrate:
            dist_arr = np.array(dist_env0, dtype=np.float32)
            tail = dist_arr[-10:] if dist_arr.size >= 10 else dist_arr
            hist_bins = np.linspace(0.0, 1.0, 21)
            hist, edges = np.histogram(dist_arr, bins=hist_bins)
            summary = {
                "steps": int(dist_arr.size),
                "min": float(dist_arr.min()) if dist_arr.size else None,
                "max": float(dist_arr.max()) if dist_arr.size else None,
                "mean": float(dist_arr.mean()) if dist_arr.size else None,
                "tail_min": float(tail.min()) if tail.size else None,
                "recommended_tau": float(tail.min() + 0.01) if tail.size else None,
                "histogram": {
                    "bins": edges.tolist(),
                    "counts": hist.tolist(),
                },
            }
            if args.calibrate_output:
                out_path = Path(args.calibrate_output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            else:
                print(json.dumps(summary, indent=2))
            return 0
        if done_any:
            return 0
        if args.output_dir is not None:
            last_frames = env.pixels.detach().cpu().numpy()
            _write_failure_receipt(
                Path(args.output_dir),
                dist_curve=dist_curve,
                last_frames=last_frames,
            )
        return 1
    finally:
        if log_file is not None:
            log_file.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
