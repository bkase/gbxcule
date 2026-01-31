"""Replay ratio scheduling utilities."""

from __future__ import annotations

import warnings
from typing import Any


class Ratio:
    """Hafner-style ratio scheduler (gradient steps per policy step)."""

    def __init__(self, ratio: float, pretrain_steps: int = 0) -> None:
        if pretrain_steps < 0:
            raise ValueError("pretrain_steps must be >= 0")
        if ratio < 0:
            raise ValueError("ratio must be >= 0")
        self._pretrain_steps = int(pretrain_steps)
        self._ratio = float(ratio)
        self._prev: int | None = None

    def __call__(self, step: int) -> int:
        if self._ratio == 0:
            return 0
        if step < 0:
            raise ValueError("step must be >= 0")
        if self._prev is None:
            self._prev = int(step)
            repeats = int(step * self._ratio)
            if self._pretrain_steps > 0:
                if step < self._pretrain_steps:
                    warnings.warn(
                        "pretrain_steps exceeds current steps; reducing to "
                        "current step",
                        stacklevel=2,
                    )
                    self._pretrain_steps = int(step)
                repeats = int(self._pretrain_steps * self._ratio)
            return repeats
        repeats = int((step - self._prev) * self._ratio)
        if self._ratio > 0:
            self._prev += int(repeats / self._ratio)
        return repeats

    def state_dict(self) -> dict[str, Any]:
        return {
            "_ratio": self._ratio,
            "_prev": self._prev,
            "_pretrain_steps": self._pretrain_steps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> Ratio:
        self._ratio = float(state_dict["_ratio"])
        self._prev = state_dict.get("_prev")
        self._pretrain_steps = int(state_dict["_pretrain_steps"])
        return self


__all__ = ["Ratio"]
