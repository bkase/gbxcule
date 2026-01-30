#!/usr/bin/env python3
"""Run an async PPO benchmark sweep and emit results + plots."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from gbxcule.rl.async_ppo_engine import AsyncPPOEngine, AsyncPPOEngineConfig

PIPELINES = {
    "u8_baseline": {
        "obs_format": "u8",
        "direct_write": False,
        "render_mode": "manual",
    },
    "packed2_directwrite": {
        "obs_format": "packed2",
        "direct_write": True,
        "render_mode": "manual",
    },
}


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


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="red.gb", help="Path to ROM")
    parser.add_argument(
        "--state", default="states/rl_stage1_exit_oak/start.state", help="State path"
    )
    parser.add_argument(
        "--goal-dir", default="states/rl_stage1_exit_oak", help="Goal template dir"
    )
    parser.add_argument(
        "--output-dir",
        default="bench/sweeps/async_ppo",
        help="Output directory root",
    )
    parser.add_argument(
        "--num-envs",
        default="1024,2048,4096,8192,16384",
        help="Comma-separated env counts",
    )
    parser.add_argument(
        "--obs-format",
        dest="obs_formats",
        default="u8_baseline,packed2_directwrite",
        help="Comma-separated pipeline labels",
    )
    parser.add_argument(
        "--obs-formats",
        dest="obs_formats",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--modes",
        default="async",
        help="Comma-separated modes (async,sync)",
    )
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--steps-per-rollout", type=int, default=64)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _ensure_assets(rom_path: Path, state_path: Path, goal_dir: Path) -> None:
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"State not found: {state_path}")
    if not goal_dir.exists():
        raise FileNotFoundError(f"Goal dir not found: {goal_dir}")


def _format_num(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _write_summary(records: list[dict[str, Any]], summary_path: Path) -> None:
    rows: list[str] = []
    header = "| envs | SPS u8 async | SPS packed2 async | speedup | overlap_eff |"
    rows.append(header)
    rows.append("| --- | --- | --- | --- | --- |")

    async_records = [rec for rec in records if rec["mode"] == "async"]
    envs = sorted({rec["num_envs"] for rec in async_records})
    for env in envs:
        u8 = next(
            (
                rec
                for rec in async_records
                if rec["num_envs"] == env and rec["obs_format"] == "u8_baseline"
            ),
            None,
        )
        packed = next(
            (
                rec
                for rec in async_records
                if rec["num_envs"] == env and rec["obs_format"] == "packed2_directwrite"
            ),
            None,
        )
        u8_sps = u8["sps"] if u8 else None
        packed_sps = packed["sps"] if packed else None
        speedup = None
        if u8_sps and packed_sps:
            speedup = packed_sps / u8_sps
        overlap_eff = packed["overlap_efficiency"] if packed else None
        rows.append(
            "| "
            + " | ".join(
                [
                    str(env),
                    _format_num(u8_sps),
                    _format_num(packed_sps),
                    _format_num(speedup),
                    _format_num(overlap_eff),
                ]
            )
            + " |"
        )

    narrative = "No async records to summarize."
    packed_sps_values = [
        rec for rec in async_records if rec["obs_format"] == "packed2_directwrite"
    ]
    if packed_sps_values:
        best = max(packed_sps_values, key=lambda rec: rec["sps"])
        narrative = (
            "Packed2 async peak SPS is "
            f"{best['sps']:.2f} at envs={best['num_envs']}. "
            "Throughput typically increases until the GPU saturates."
        )

    summary_path.write_text(
        "\n".join(
            [
                "# Async PPO Sweep Summary",
                "",
                *rows,
                "",
                narrative,
            ]
        ),
        encoding="utf-8",
    )


def _plot_results(records: list[dict[str, Any]], plots_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    async_records = [rec for rec in records if rec["mode"] == "async"]
    envs = sorted({rec["num_envs"] for rec in async_records})

    plt.figure(figsize=(7, 4))
    for label in ("u8_baseline", "packed2_directwrite"):
        points = [rec for rec in async_records if rec["obs_format"] == label]
        if not points:
            continue
        sps_map = {rec["num_envs"]: rec["sps"] for rec in points}
        envs_for_plot = [env for env in envs if env in sps_map]
        sps = [sps_map[env] for env in envs_for_plot]
        if envs_for_plot:
            plt.plot(envs_for_plot, sps, marker="o", label=label)
    plt.xlabel("Num envs")
    plt.ylabel("SPS")
    plt.title("SPS vs envs (async)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "sps_vs_envs.png")
    plt.close()

    speedups = []
    speedup_envs = []
    for env in envs:
        u8 = next(
            (
                rec
                for rec in async_records
                if rec["num_envs"] == env and rec["obs_format"] == "u8_baseline"
            ),
            None,
        )
        packed = next(
            (
                rec
                for rec in async_records
                if rec["num_envs"] == env and rec["obs_format"] == "packed2_directwrite"
            ),
            None,
        )
        if not u8 or not packed:
            continue
        speedup_envs.append(env)
        speedups.append(packed["sps"] / u8["sps"])

    if speedups:
        plt.figure(figsize=(7, 4))
        plt.plot(speedup_envs, speedups, marker="o")
        plt.xlabel("Num envs")
        plt.ylabel("Packed2 speedup")
        plt.title("Packed2 speedup vs envs (async)")
        plt.tight_layout()
        plt.savefig(plots_dir / "packed_speedup.png")
        plt.close()

    plt.figure(figsize=(7, 4))
    for label in ("u8_baseline", "packed2_directwrite"):
        points = [rec for rec in async_records if rec["obs_format"] == label]
        if not points:
            continue
        eff_map = {rec["num_envs"]: rec["overlap_efficiency"] for rec in points}
        envs_for_plot = [env for env in envs if env in eff_map]
        eff = [eff_map[env] for env in envs_for_plot]
        if envs_for_plot:
            plt.plot(envs_for_plot, eff, marker="o", label=label)
    plt.xlabel("Num envs")
    plt.ylabel("Overlap efficiency")
    plt.title("Overlap efficiency vs envs (async)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "overlap_efficiency.png")
    plt.close()


def _run_one(
    *,
    torch,
    mode: str,
    pipeline_label: str,
    config: AsyncPPOEngineConfig,
) -> dict[str, Any]:
    engine = AsyncPPOEngine(config)
    try:
        if mode == "sync":
            stream = torch.cuda.current_stream()
            engine.actor_stream = stream
            engine.learner_stream = stream
        torch.cuda.reset_peak_memory_stats()
        metrics = engine.run(updates=config.updates)
        peak_mem = torch.cuda.max_memory_allocated()
    finally:
        engine.close()
    return {
        **metrics,
        "gpu_mem_bytes": int(peak_mem),
        "mode": mode,
        "obs_format": pipeline_label,
        "obs_format_raw": config.obs_format,
        "direct_write": bool(PIPELINES[pipeline_label]["direct_write"]),
        "render_mode": PIPELINES[pipeline_label]["render_mode"],
        "num_envs": int(config.num_envs),
        "frames_per_step": int(config.frames_per_step),
        "release_after_frames": int(config.release_after_frames),
        "steps_per_rollout": int(config.steps_per_rollout),
        "updates": int(config.updates),
        "ppo_epochs": int(config.ppo_epochs),
        "minibatch_size": int(config.minibatch_size),
    }


def main() -> int:
    args = _parse_args()
    if not _cuda_available():
        return 0

    torch = _require_torch()
    if not torch.cuda.is_available():
        return 0

    num_envs_list = _parse_int_list(args.num_envs)
    pipeline_labels = _parse_str_list(args.obs_formats)
    modes = _parse_str_list(args.modes)

    for label in pipeline_labels:
        if label not in PIPELINES:
            raise ValueError(f"Unknown pipeline label: {label}")
    for mode in modes:
        if mode not in ("async", "sync"):
            raise ValueError(f"Unknown mode: {mode}")

    rom_path = Path(args.rom)
    state_path = Path(args.state)
    goal_dir = Path(args.goal_dir)
    _ensure_assets(rom_path, state_path, goal_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    records: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for mode in modes:
            for num_envs in num_envs_list:
                for pipeline_label in pipeline_labels:
                    config = AsyncPPOEngineConfig(
                        rom_path=str(rom_path),
                        state_path=str(state_path),
                        goal_dir=str(goal_dir),
                        device="cuda",
                        obs_format=PIPELINES[pipeline_label]["obs_format"],
                        num_envs=num_envs,
                        frames_per_step=args.frames_per_step,
                        release_after_frames=args.release_after_frames,
                        steps_per_rollout=args.steps_per_rollout,
                        updates=args.updates,
                        ppo_epochs=args.ppo_epochs,
                        minibatch_size=args.minibatch_size,
                        seed=args.seed,
                    )
                    started = time.perf_counter()
                    record = _run_one(
                        torch=torch,
                        mode=mode,
                        pipeline_label=pipeline_label,
                        config=config,
                    )
                    record["wall_clock_s"] = float(time.perf_counter() - started)
                    records.append(record)
                    handle.write(json.dumps(record) + "\n")

    _write_summary(records, output_dir / "summary.md")
    _plot_results(records, plots_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
