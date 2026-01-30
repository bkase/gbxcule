from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.return_ema import ReturnEMA  # noqa: E402
from tests.rl_dreamer.fixtures import load_fixture  # noqa: E402


def test_return_ema_cases() -> None:
    payload = load_fixture("return_ema_cases.json")
    config = payload["config"]
    ema = ReturnEMA(
        decay=config["decay"],
        percentiles=tuple(config["percentiles"]),
        max_value=config["max_value"],
    )
    for case in payload["updates"]:
        values = torch.tensor(case["values"], dtype=torch.float32)
        offset, invscale = ema.update(values)
        assert offset.item() == pytest.approx(case["low"], abs=1e-5)
        assert ema.high is not None
        assert ema.high.item() == pytest.approx(case["high"], abs=1e-5)
        assert invscale.item() == pytest.approx(case["invscale"], abs=1e-5)


def test_return_ema_constant_input() -> None:
    ema = ReturnEMA(decay=0.99, percentiles=(0.05, 0.95), max_value=1.0)
    values = torch.full((10,), 5.0, dtype=torch.float32)
    offset, invscale = ema.update(values)
    assert torch.isfinite(offset).item()
    assert torch.isfinite(invscale).item()
    assert invscale.item() >= 1.0


def test_return_ema_outlier_stability() -> None:
    ema = ReturnEMA(decay=0.99, percentiles=(0.05, 0.95), max_value=1.0)
    values = torch.tensor([0.0, 0.0, 0.0, 1000.0], dtype=torch.float32)
    offset, invscale = ema.update(values)
    assert torch.isfinite(offset).item()
    assert torch.isfinite(invscale).item()
