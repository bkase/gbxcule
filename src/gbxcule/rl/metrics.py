"""Metrics utilities for RL training logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_torch():
    import importlib

    return importlib.import_module("torch")


def compute_entropy_from_logits(logits):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def compute_action_hist(actions, num_actions: int):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    return torch.bincount(actions, minlength=num_actions)


def compute_dist_percentiles(dist):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    qs = torch.tensor([0.1, 0.5, 0.9], device=dist.device, dtype=dist.dtype)
    return torch.quantile(dist, qs)


def compute_value_stats(values):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    mean = values.mean()
    var = torch.clamp(values.var(unbiased=False), min=0.0)
    return mean, torch.sqrt(var)


@dataclass
class MetricsAccumulator:
    num_envs: int
    num_actions: int
    device: str = "cuda"

    def __post_init__(self) -> None:
        torch = _require_torch()
        self._torch = torch
        self.reset()

    def reset(self) -> None:
        torch = self._torch
        self.env_steps = 0
        self.reward_sum = torch.tensor(0.0, device=self.device)
        self.done_sum = torch.tensor(0.0, device=self.device)
        self.trunc_sum = torch.tensor(0.0, device=self.device)
        self.reset_sum = torch.tensor(0.0, device=self.device)
        self.entropy_sum = torch.tensor(0.0, device=self.device)
        self.value_sum = torch.tensor(0.0, device=self.device)
        self.value_sq_sum = torch.tensor(0.0, device=self.device)
        self.action_hist = torch.zeros(
            (self.num_actions,), device=self.device, dtype=torch.int64
        )
        self._last_dist = None

    def update(
        self,
        *,
        reward,
        done,
        trunc,
        reset_mask,
        dist,
        actions,
        logits,
        values,
    ) -> None:  # type: ignore[no-untyped-def]
        torch = self._torch
        steps = int(reward.numel())
        self.env_steps += steps
        self.reward_sum += reward.sum()
        self.done_sum += done.to(dtype=torch.float32).sum()
        self.trunc_sum += trunc.to(dtype=torch.float32).sum()
        self.reset_sum += reset_mask.to(dtype=torch.float32).sum()
        entropy = compute_entropy_from_logits(logits)
        self.entropy_sum += entropy.sum()
        self.value_sum += values.sum()
        self.value_sq_sum += (values * values).sum()
        self.action_hist += compute_action_hist(actions, self.num_actions)
        self._last_dist = dist

    def as_record(self) -> dict[str, Any]:
        torch = self._torch
        if self.env_steps <= 0:
            raise RuntimeError("No steps accumulated")
        env_steps = float(self.env_steps)
        reward_mean = (self.reward_sum / env_steps).item()
        done_rate = (self.done_sum / env_steps).item()
        trunc_rate = (self.trunc_sum / env_steps).item()
        reset_rate = (self.reset_sum / env_steps).item()
        entropy_mean = (self.entropy_sum / env_steps).item()
        value_mean = (self.value_sum / env_steps).item()
        value_var = torch.clamp(
            (self.value_sq_sum / env_steps) - (self.value_sum / env_steps) ** 2,
            min=0.0,
        )
        value_std = torch.sqrt(value_var).item()
        if self._last_dist is None:
            raise RuntimeError("Missing last dist for percentiles")
        p10, p50, p90 = compute_dist_percentiles(self._last_dist)
        return {
            "reward_mean": float(reward_mean),
            "done_rate": float(done_rate),
            "trunc_rate": float(trunc_rate),
            "reset_rate": float(reset_rate),
            "dist_p10": float(p10.item()),
            "dist_p50": float(p50.item()),
            "dist_p90": float(p90.item()),
            "entropy_mean": float(entropy_mean),
            "value_mean": float(value_mean),
            "value_std": float(value_std),
            "action_hist": self.action_hist.detach().cpu().tolist(),
        }
