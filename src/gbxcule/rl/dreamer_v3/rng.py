"""Explicit RNG utilities for Dreamer v3 (M0)."""

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


def require_generator(gen: Any) -> Any:
    torch = _require_torch()
    if gen is None or not isinstance(gen, torch.Generator):
        raise ValueError("generator must be a torch.Generator")
    return gen


def _generator_device(gen: Any):  # type: ignore[no-untyped-def]
    return getattr(gen, "device", None)


def make_generator(seed: int, device=None):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    gen = torch.Generator() if device is None else torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def fork_generator(gen):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    gen = require_generator(gen)
    device = _generator_device(gen)
    seed = torch.randint(0, 2**63 - 1, (1,), generator=gen).item()
    return make_generator(int(seed), device=device)


def randn(shape, *, generator, device=None, dtype=None):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    gen = require_generator(generator)
    if device is None:
        device = _generator_device(gen)
    return torch.randn(shape, generator=gen, device=device, dtype=dtype)


def rand(shape, *, generator, device=None, dtype=None):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    gen = require_generator(generator)
    if device is None:
        device = _generator_device(gen)
    return torch.rand(shape, generator=gen, device=device, dtype=dtype)


def randint(low, high, shape, *, generator, device=None, dtype=None):  # type: ignore[no-untyped-def]
    torch = _require_torch()
    gen = require_generator(generator)
    if device is None:
        device = _generator_device(gen)
    return torch.randint(low, high, shape, generator=gen, device=device, dtype=dtype)
