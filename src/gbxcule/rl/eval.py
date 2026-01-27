"""Greedy eval utilities for pixels-only RL envs (autoreset)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl. Install with `uv sync --group rl`."
        ) from exc


@dataclass
class EvalSummary:
    episodes: int
    successes: int
    success_rate: float
    median_steps_to_goal: int | None
    mean_return: float


def _lower_median_int(values: Iterable[int]) -> int | None:
    data = sorted(values)
    if not data:
        return None
    idx = (len(data) - 1) // 2
    return int(data[idx])


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def run_greedy_eval(  # type: ignore[no-untyped-def]
    env,
    model,
    *,
    episodes: int,
    trajectory_path: Path | None = None,
    max_steps: int | None = None,
) -> EvalSummary:
    """Run greedy evaluation on an autoreset env and return summary metrics."""
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    torch = _require_torch()

    num_envs = int(env.num_envs)
    if trajectory_path is not None and num_envs != 1:
        raise ValueError("trajectory_path is only supported for num_envs == 1")

    obs = env.reset()
    device = obs.device

    ep_returns = torch.zeros((num_envs,), device=device, dtype=torch.float32)
    ep_steps = torch.zeros((num_envs,), device=device, dtype=torch.int32)

    successes = 0
    returns: list[float] = []
    steps_success: list[int] = []
    episodes_collected = 0

    trajectory: list[dict[str, Any]] = []
    episode_idx = 0

    model.eval()
    with torch.inference_mode():
        while episodes_collected < episodes:
            logits, _ = model(obs)
            actions = torch.argmax(logits, dim=-1).to(torch.int32)
            obs, reward, done, trunc, info = env.step(actions)

            ep_returns.add_(reward)
            ep_steps.add_(1)

            terminated = done | trunc

            if num_envs == 1 and trajectory_path is not None:
                dist = None
                if isinstance(info, dict) and "dist" in info:
                    dist = float(info["dist"][0].item())
                trajectory.append(
                    {
                        "episode_idx": episode_idx,
                        "t": int(ep_steps[0].item()),
                        "action": int(actions[0].item()),
                        "reward": float(reward[0].item()),
                        "done": bool(done[0].item()),
                        "trunc": bool(trunc[0].item()),
                        "dist": dist,
                    }
                )

            if torch.any(terminated):
                idxs = torch.nonzero(terminated, as_tuple=False).flatten()
                for idx in idxs.tolist():
                    episodes_collected += 1
                    if bool(done[idx].item()):
                        successes += 1
                        steps_success.append(int(ep_steps[idx].item()))
                    returns.append(float(ep_returns[idx].item()))
                    ep_returns[idx] = 0.0
                    ep_steps[idx] = 0
                    if num_envs == 1 and trajectory_path is not None:
                        episode_idx += 1
                    if episodes_collected >= episodes:
                        break

            if (
                max_steps is not None
                and episodes_collected < episodes
                and int(ep_steps.max().item()) >= int(max_steps)
            ):
                raise RuntimeError(
                    f"max_steps {max_steps} reached without episode termination"
                )

    if trajectory_path is not None:
        _write_jsonl(trajectory_path, trajectory)

    mean_return = float(sum(returns) / max(1, len(returns)))
    success_rate = float(successes / episodes)
    median_steps = _lower_median_int(steps_success)

    return EvalSummary(
        episodes=episodes,
        successes=successes,
        success_rate=success_rate,
        median_steps_to_goal=median_steps,
        mean_return=mean_return,
    )
