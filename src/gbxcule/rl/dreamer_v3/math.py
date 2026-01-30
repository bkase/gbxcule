"""Math primitives for Dreamer v3 (M2)."""

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


def _as_float32(x):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if not torch.is_floating_point(x) or x.dtype is torch.float64:
        x = x.to(dtype=torch.float32)
    return x


def symlog(x):  # type: ignore[no-untyped-def]
    """Symmetric log: sign(x) * log(1 + |x|)."""
    torch = _require_torch()
    x = _as_float32(x)
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):  # type: ignore[no-untyped-def]
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    torch = _require_torch()
    x = _as_float32(x)
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def _validate_bins(bins):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    if not isinstance(bins, torch.Tensor):
        bins = torch.as_tensor(bins, dtype=torch.float32)
    if bins.ndim != 1:
        raise ValueError("bins must be 1D")
    if bins.numel() < 2:
        raise ValueError("bins must have at least 2 entries")
    return bins.to(dtype=torch.float32)


def twohot(y, bins):  # type: ignore[no-untyped-def]
    """Two-hot weights in symlog space (sheeprl parity)."""
    torch = _require_torch()
    import torch.nn.functional as F

    y = _as_float32(y)
    squeeze_added = False
    if y.ndim == 0:
        y = y.reshape(1, 1)
        squeeze_added = True
    elif y.shape[-1] != 1:
        y = y.unsqueeze(-1)
        squeeze_added = True
    bins = _validate_bins(bins).to(device=y.device)

    below = (bins <= y).to(torch.int32).sum(dim=-1, keepdim=True) - 1
    above = below + 1

    max_idx = torch.full_like(above, bins.numel() - 1)
    above = torch.minimum(above, max_idx)
    below = torch.maximum(below, torch.zeros_like(below))

    equal = below == above
    dist_to_below = torch.where(equal, torch.ones_like(y), torch.abs(bins[below] - y))
    dist_to_above = torch.where(equal, torch.ones_like(y), torch.abs(bins[above] - y))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total

    below_oh = F.one_hot(below.to(torch.int64), bins.numel())
    above_oh = F.one_hot(above.to(torch.int64), bins.numel())
    target = (
        below_oh * weight_below[..., None] + above_oh * weight_above[..., None]
    ).squeeze(-2)
    if squeeze_added and target.ndim >= 2:
        target = target.squeeze(-2)
    return target


def twohot_to_value(weights, bins, *, keepdim: bool = True):  # type: ignore[no-untyped-def]
    """Decode two-hot weights back to values in symlog space."""
    weights = _as_float32(weights)
    bins = _validate_bins(bins).to(device=weights.device)
    value = (weights * bins).sum(dim=-1, keepdim=keepdim)
    return value
