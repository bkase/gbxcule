"""RNG contract tests for Dreamer v3."""

from __future__ import annotations

from typing import Any

import pytest

from gbxcule.rl.dreamer_v3 import rng


def _require_torch() -> Any:
    try:
        import torch
    except Exception:
        pytest.skip("torch required")
    return torch


def test_generator_required() -> None:
    _require_torch()
    with pytest.raises(ValueError):
        rng.require_generator(None)


def test_make_generator_determinism() -> None:
    torch = _require_torch()
    gen_a = rng.make_generator(123)
    gen_b = rng.make_generator(123)
    sample_a = rng.rand((4,), generator=gen_a)
    sample_b = rng.rand((4,), generator=gen_b)
    assert torch.allclose(sample_a, sample_b)


def test_fork_generator_determinism() -> None:
    torch = _require_torch()
    base = rng.make_generator(999)
    child_a = rng.fork_generator(base)
    child_b = rng.fork_generator(base)
    sample_a = rng.rand((4,), generator=child_a)
    sample_b = rng.rand((4,), generator=child_b)
    assert not torch.allclose(sample_a, sample_b)
