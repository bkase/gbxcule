"""Schema contract tests for Dreamer v3."""

from __future__ import annotations

from typing import Any

import pytest

from gbxcule.rl.dreamer_v3 import schema


def _require_torch() -> Any:
    try:
        import torch
    except Exception:
        pytest.skip("torch required")
    return torch


def test_time_major_passes() -> None:
    torch = _require_torch()
    tensor = torch.zeros(2, 3, 4)
    schema.assert_time_major(tensor, "tensor")


def test_packed2_obs_contract() -> None:
    torch = _require_torch()
    obs = torch.zeros(2, 3, 1, 72, 20, dtype=torch.uint8)
    schema.assert_packed2_obs(obs, "obs")


def test_packed2_obs_rejects_shape() -> None:
    torch = _require_torch()
    obs = torch.zeros(2, 3, 1, 70, 20, dtype=torch.uint8)
    with pytest.raises(ValueError):
        schema.assert_packed2_obs(obs, "obs")
