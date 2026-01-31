"""Target network utilities for Dreamer v3 behavior learning (M5)."""

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


def update_target(source, target, tau: float) -> None:  # type: ignore[no-untyped-def]
    """EMA update for target network."""
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    for src_param, tgt_param in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)


def maybe_update_target(
    source,
    target,
    tau: float,
    *,
    step: int,
    update_freq: int,
) -> bool:  # type: ignore[no-untyped-def]
    """Update target network on a fixed cadence.

    Returns True if an update was applied.
    """
    if update_freq < 1:
        raise ValueError("update_freq must be >= 1")
    if step % update_freq != 0:
        return False
    if step == 0:
        tau = 1.0
    update_target(source, target, tau)
    return True
