"""Lambda-return utilities for Dreamer v3 behavior learning (M5)."""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


def lambda_returns(
    rewards,  # type: ignore[no-untyped-def]
    values,  # type: ignore[no-untyped-def]
    continues_gamma,  # type: ignore[no-untyped-def]
    lmbda: float = 0.95,
):
    """Compute Dreamer-style lambda returns (sheeprl parity).

    Args:
    - rewards: time-major [T, B, ...]
    - values: time-major [T, B, ...]
    - continues_gamma: time-major [T, B, ...] already multiplied by gamma
    - lmbda: lambda parameter in [0, 1]
    """
    torch = _require_torch()
    if not 0.0 <= lmbda <= 1.0:
        raise ValueError("lmbda must be in [0, 1]")
    if rewards.shape[0] != values.shape[0]:
        raise ValueError("rewards and values must share time dimension")
    if rewards.shape[0] != continues_gamma.shape[0]:
        raise ValueError("continues_gamma must match rewards time dimension")
    if rewards.shape[0] < 1:
        raise ValueError("rewards must be non-empty")

    vals = [values[-1:]]
    interm = rewards + continues_gamma * values * (1 - lmbda)
    for t in reversed(range(len(continues_gamma))):
        vals.append(interm[t : t + 1] + continues_gamma[t : t + 1] * lmbda * vals[-1])
    return torch.cat(list(reversed(vals))[:-1], dim=0)
