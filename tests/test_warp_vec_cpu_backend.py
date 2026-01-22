"""Tests for warp_vec_cpu backend."""

from __future__ import annotations

import numpy as np
import pytest

from gbxcule.backends.warp_vec import WarpVecCpuBackend

from .conftest import ROM_PATH, BackendComplianceTests


class TestWarpVecCpuCompliance(BackendComplianceTests):
    """Compliance tests for WarpVecCpuBackend."""

    expected_name = "warp_vec_cpu"
    expected_num_envs = 4
    obs_dim = 32

    @pytest.fixture
    def backend(self) -> WarpVecCpuBackend:
        """Create a backend instance for testing."""
        if not ROM_PATH.exists():
            pytest.skip(f"Test ROM not found: {ROM_PATH}")
        return WarpVecCpuBackend(str(ROM_PATH), num_envs=4, obs_dim=32)


class TestWarpVecCpuSpecific:
    """Tests specific to WarpVecCpuBackend."""

    @pytest.fixture
    def backend(self) -> WarpVecCpuBackend:
        """Create a backend instance for testing."""
        if not ROM_PATH.exists():
            pytest.skip(f"Test ROM not found: {ROM_PATH}")
        return WarpVecCpuBackend(str(ROM_PATH), num_envs=2, obs_dim=32)

    def test_step_increments_counter(self, backend: WarpVecCpuBackend) -> None:
        """step() advances instruction and cycle counters."""
        backend.reset()
        actions = np.zeros((2,), dtype=np.int32)
        state0 = backend.get_cpu_state(0)
        backend.step(actions)
        state1 = backend.get_cpu_state(0)
        assert state1["instr_count"] > state0["instr_count"]
        assert state1["cycle_count"] > state0["cycle_count"]
        assert state1["pc"] >= 0x0100

    def test_multi_env_counters_progress(self, backend: WarpVecCpuBackend) -> None:
        """All env counters advance each step."""
        backend.reset()
        actions = np.zeros((2,), dtype=np.int32)
        backend.step(actions)
        state0 = backend.get_cpu_state(0)
        state1 = backend.get_cpu_state(1)
        assert state0["instr_count"] > 0
        assert state1["instr_count"] > 0
        assert state0["instr_count"] == state1["instr_count"]

    def test_determinism_across_instances(self) -> None:
        """Identical runs produce identical state progression."""
        if not ROM_PATH.exists():
            pytest.skip(f"Test ROM not found: {ROM_PATH}")
        backend_a = WarpVecCpuBackend(str(ROM_PATH), num_envs=1, obs_dim=32)
        backend_b = WarpVecCpuBackend(str(ROM_PATH), num_envs=1, obs_dim=32)
        backend_a.reset(seed=123)
        backend_b.reset(seed=123)
        actions = np.zeros((1,), dtype=np.int32)
        for _ in range(3):
            backend_a.step(actions)
            backend_b.step(actions)
        assert backend_a.get_cpu_state(0) == backend_b.get_cpu_state(0)
