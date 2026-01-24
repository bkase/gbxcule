"""Smoke tests for warp_vec_cuda backend (CUDA only)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from gbxcule.backends.warp_vec import WarpVecCudaBackend

from .conftest import ROM_PATH, require_rom


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def test_warp_vec_cuda_smoke() -> None:
    """CUDA backend can reset, step, and report CPU state."""
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    require_rom(ROM_PATH)
    backend = WarpVecCudaBackend(str(ROM_PATH), num_envs=1, obs_dim=32)
    try:
        backend.reset()
        actions = np.zeros((1,), dtype=np.int32)
        for _ in range(2):
            backend.step(actions)
        state = backend.get_cpu_state(0)
        assert state["instr_count"] is not None and state["instr_count"] > 0
        assert state["cycle_count"] is not None and state["cycle_count"] > 0
    finally:
        backend.close()
