"""Tests for pyboy_single backend.

Tests cover:
- Backend initialization (headless)
- reset() returns correct obs shape/dtype
- step() returns correct shapes/dtypes
- get_cpu_state() returns canonical keys and flags
- Invalid action raises ValueError
- close() stops emulator cleanly
"""

from __future__ import annotations

import pytest

from gbxcule.backends.pyboy_single import PyBoySingleBackend

from .conftest import ROM_PATH, BackendComplianceTests


class TestPyBoySingleCompliance(BackendComplianceTests):
    """Compliance tests for PyBoySingleBackend."""

    expected_name = "pyboy_single"
    expected_num_envs = 1
    obs_dim = 32

    @pytest.fixture
    def backend(self) -> PyBoySingleBackend:
        """Create a backend instance for testing."""
        if not ROM_PATH.exists():
            pytest.skip(f"Test ROM not found: {ROM_PATH}")
        return PyBoySingleBackend(str(ROM_PATH), obs_dim=32)


class TestPyBoySingleSpecific:
    """Tests specific to PyBoySingleBackend (not in compliance suite)."""

    @pytest.fixture
    def backend(self) -> PyBoySingleBackend:
        """Create a backend instance for testing."""
        if not ROM_PATH.exists():
            pytest.skip(f"Test ROM not found: {ROM_PATH}")
        return PyBoySingleBackend(str(ROM_PATH), obs_dim=32)

    def test_reset_can_be_called_multiple_times(
        self, backend: PyBoySingleBackend
    ) -> None:
        """reset() can be called multiple times."""
        backend.reset()
        obs, _ = backend.reset()
        assert obs.shape == (1, 32)

    def test_get_cpu_state_invalid_env_idx(self, backend: PyBoySingleBackend) -> None:
        """get_cpu_state() raises ValueError for invalid env_idx."""
        backend.reset()
        with pytest.raises(ValueError, match="env_idx must be 0"):
            backend.get_cpu_state(1)

    def test_get_cpu_state_without_reset_raises(
        self, backend: PyBoySingleBackend
    ) -> None:
        """get_cpu_state() without reset raises RuntimeError."""
        with pytest.raises(RuntimeError, match="not initialized"):
            backend.get_cpu_state(0)

    def test_get_cpu_state_counters_are_none(self, backend: PyBoySingleBackend) -> None:
        """PyBoy doesn't provide counters, so they should be None."""
        backend.reset()
        state = backend.get_cpu_state(0)
        assert state["instr_count"] is None
        assert state["cycle_count"] is None

    def test_close_stops_emulator(self, backend: PyBoySingleBackend) -> None:
        """close() stops the emulator."""
        backend.reset()
        backend.close()
        assert backend._pyboy is None

    def test_reset_after_close(self, backend: PyBoySingleBackend) -> None:
        """reset() works after close()."""
        backend.reset()
        backend.close()
        obs, _ = backend.reset()
        assert obs.shape == (1, 32)
