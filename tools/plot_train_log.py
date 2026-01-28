#!/usr/bin/env python3
"""Plot training metrics from train_log.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Path to train_log.jsonl")
    parser.add_argument("--out", default="train.png", help="Output PNG path")
    parser.add_argument(
        "--smoothing", type=int, default=1, help="Moving average window"
    )
    return parser.parse_args()


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and "meta" in payload:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out: list[float] = []
    acc = 0.0
    for idx, val in enumerate(values):
        acc += val
        if idx >= window:
            acc -= values[idx - window]
            out.append(acc / float(window))
        else:
            out.append(acc / float(idx + 1))
    return out


def main() -> int:
    args = _parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"log not found: {log_path}")

    records = _load_records(log_path)
    if not records:
        raise SystemExit("No records found")

    steps = [
        int(r.get("env_steps", r.get("opt_steps", idx)))
        for idx, r in enumerate(records)
    ]
    reward = _smooth(
        [float(r.get("reward_mean", 0.0)) for r in records], args.smoothing
    )
    done_rate = _smooth(
        [float(r.get("done_rate", 0.0)) for r in records], args.smoothing
    )
    trunc_rate = _smooth(
        [float(r.get("trunc_rate", 0.0)) for r in records], args.smoothing
    )
    dist_p50 = _smooth([float(r.get("dist_p50", 0.0)) for r in records], args.smoothing)
    entropy = _smooth(
        [float(r.get("entropy_mean", r.get("entropy", 0.0))) for r in records],
        args.smoothing,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting. Install it and retry."
        ) from exc

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    ax_reward, ax_done, ax_dist, ax_entropy = axes.flatten()

    ax_reward.plot(steps, reward, color="tab:blue")
    ax_reward.set_title("reward_mean")
    ax_reward.grid(True, linestyle="--", alpha=0.3)

    ax_done.plot(steps, done_rate, label="done_rate", color="tab:green")
    ax_done.plot(steps, trunc_rate, label="trunc_rate", color="tab:orange")
    ax_done.set_title("done/trunc rate")
    ax_done.legend()
    ax_done.grid(True, linestyle="--", alpha=0.3)

    ax_dist.plot(steps, dist_p50, color="tab:purple")
    ax_dist.set_title("dist_p50")
    ax_dist.grid(True, linestyle="--", alpha=0.3)

    ax_entropy.plot(steps, entropy, color="tab:red")
    ax_entropy.set_title("entropy_mean")
    ax_entropy.grid(True, linestyle="--", alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel("env_steps" if "env_steps" in records[0] else "opt_steps")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
