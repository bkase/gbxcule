"""Return normalization helpers for Dreamer v3 (M2)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


class ReturnEMA:
    """Percentile-based return normalizer (sheeprl Moments parity)."""

    def __init__(
        self,
        decay: float = 0.99,
        percentiles: tuple[float, float] = (0.05, 0.95),
        max_value: float = 1.0,
    ) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        low, high = percentiles
        if not 0.0 < low < 1.0 or not 0.0 < high < 1.0 or low >= high:
            raise ValueError("percentiles must satisfy 0 < low < high < 1")
        if max_value <= 0:
            raise ValueError("max_value must be > 0")
        self._decay = float(decay)
        self._percentile_low = float(low)
        self._percentile_high = float(high)
        self._max = float(max_value)
        self._low = None
        self._high = None

    @property
    def low(self):  # type: ignore[no-untyped-def]
        return self._low

    @property
    def high(self):  # type: ignore[no-untyped-def]
        return self._high

    def update(self, values):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        x = values
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.float().detach().flatten()
        low = torch.quantile(x, self._percentile_low)
        high = torch.quantile(x, self._percentile_high)
        if self._low is None:
            self._low = torch.zeros((), dtype=torch.float32, device=x.device)
            self._high = torch.zeros((), dtype=torch.float32, device=x.device)
        assert self._low is not None
        assert self._high is not None
        self._low = self._decay * self._low + (1 - self._decay) * low
        self._high = self._decay * self._high + (1 - self._decay) * high
        invscale = torch.max(
            torch.tensor(1.0 / self._max, device=x.device), self._high - self._low
        )
        return self._low.detach(), invscale.detach()

    def normalize(self, values):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        offset, invscale = self.update(values)
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values, device=offset.device)
        return (values - offset) / invscale


def update_moments_sequence(
    updates: Iterable[Iterable[float]],
    *,
    decay: float = 0.99,
    percentiles: tuple[float, float] = (0.05, 0.95),
    max_value: float = 1.0,
):  # type: ignore[no-untyped-def]
    """Helper for tests: run ReturnEMA update over a sequence of batches."""
    torch = _require_torch()
    moments = ReturnEMA(decay=decay, percentiles=percentiles, max_value=max_value)
    results = []
    for batch in updates:
        values = torch.tensor(list(batch), dtype=torch.float32)
        low, invscale = moments.update(values)
        assert moments.high is not None
        results.append((low.item(), moments.high.item(), invscale.item()))
    return results
