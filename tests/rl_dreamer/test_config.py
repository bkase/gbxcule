"""Config validation tests for Dreamer v3."""

from __future__ import annotations

from typing import Any

import pytest

from gbxcule.rl.dreamer_v3.config import (
    DreamerV3Config,
    PrecisionPolicy,
    validate_config,
)


def _require_torch() -> Any:
    try:
        import torch
    except Exception:
        pytest.skip("torch required")
    return torch


def test_default_config_valid() -> None:
    cfg = DreamerV3Config()
    errors = validate_config(cfg)
    assert errors == []


def test_invalid_bins_rejected() -> None:
    cfg = DreamerV3Config()
    cfg.world_model.reward_model.bins = 1
    cfg.critic.bins = 0
    errors = validate_config(cfg)
    assert any("reward_model.bins" in err for err in errors)
    assert any("critic.bins" in err for err in errors)


def test_invalid_percentiles_rejected() -> None:
    cfg = DreamerV3Config()
    cfg.actor.moments.percentile.low = 0.9
    cfg.actor.moments.percentile.high = 0.1
    errors = validate_config(cfg)
    assert any("percentile.low" in err for err in errors)
    assert any("percentile.high" in err for err in errors)


def test_precision_dtype_validation() -> None:
    cfg = DreamerV3Config(precision=PrecisionPolicy(model_dtype="float32"))
    errors = validate_config(cfg)
    assert errors == []
    cfg.precision.model_dtype = "float16"
    errors = validate_config(cfg)
    assert any("model_dtype" in err for err in errors)


def test_precision_accepts_torch_dtype() -> None:
    torch = _require_torch()
    cfg = DreamerV3Config(precision=PrecisionPolicy(model_dtype=torch.float32))
    errors = validate_config(cfg)
    assert errors == []
