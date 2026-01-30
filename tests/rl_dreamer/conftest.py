"""Pytest config for Dreamer v3 tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _enable_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    try:
        import torch
    except Exception:
        return
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        return
