"""Distribution helpers for Dreamer v3 (M2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gbxcule.rl.dreamer_v3.math import symexp, symlog, twohot


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


class SymlogDistribution:
    def __init__(
        self,
        mode,  # type: ignore[no-untyped-def]
        dims: int,
        *,
        dist: str = "mse",
        agg: str = "sum",
        tol: float = 1e-8,
    ):
        torch = _require_torch()
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]
        if not isinstance(self._mode, torch.Tensor):
            raise TypeError("mode must be torch.Tensor")

    @property
    def mode(self):  # type: ignore[no-untyped-def]
        return symexp(self._mode)

    @property
    def mean(self):  # type: ignore[no-untyped-def]
        return symexp(self._mode)

    def log_prob(self, value):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        if self._mode.shape != value.shape:
            raise ValueError("value shape mismatch")
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class MSEDistribution:
    def __init__(self, mode, dims: int, *, agg: str = "sum"):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self._batch_shape = mode.shape[: len(mode.shape) - dims]
        self._event_shape = mode.shape[len(mode.shape) - dims :]
        if not isinstance(self._mode, torch.Tensor):
            raise TypeError("mode must be torch.Tensor")

    @property
    def mode(self):  # type: ignore[no-untyped-def]
        return self._mode

    @property
    def mean(self):  # type: ignore[no-untyped-def]
        return self._mode

    def log_prob(self, value):  # type: ignore[no-untyped-def]
        if self._mode.shape != value.shape:
            raise ValueError("value shape mismatch")
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class TwoHotEncodingDistribution:
    def __init__(
        self,
        logits,  # type: ignore[no-untyped-def]
        bins=None,
        *,
        dims: int = 0,
        low: float = -20.0,
        high: float = 20.0,
        transfwd: Callable[[Any], Any] = symlog,
        transbwd: Callable[[Any], Any] = symexp,
    ) -> None:
        torch = _require_torch()
        import torch.nn.functional as F

        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be torch.Tensor")
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        if bins is None:
            bins = torch.linspace(
                low, high, logits.shape[-1], device=logits.device, dtype=logits.dtype
            )
        else:
            bins = torch.as_tensor(bins, dtype=logits.dtype, device=logits.device)
        self.bins = bins
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self._batch_shape = logits.shape[: len(logits.shape) - dims]
        self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)

    @property
    def mean(self):  # type: ignore[no-untyped-def]
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    @property
    def mode(self):  # type: ignore[no-untyped-def]
        return self.mean

    def log_prob(self, value):  # type: ignore[no-untyped-def]
        torch = _require_torch()
        value = self.transfwd(value)
        target = twohot(value, self.bins)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        prod = target * log_pred
        if self.dims:
            return prod.sum(dim=self.dims)
        return prod


class SymlogTwoHot(TwoHotEncodingDistribution):
    pass


class BernoulliSafeMode(_require_torch().distributions.Bernoulli):  # type: ignore[misc]
    @property
    def mode(self):  # type: ignore[no-untyped-def]
        return (self.probs > 0.5).to(self.probs)
