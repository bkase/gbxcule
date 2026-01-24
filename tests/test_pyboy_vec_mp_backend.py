"""Tests for pyboy_vec_mp backend reliability.

Tests cover:
- Backend initialization with multiple workers
- reset() and step() work without hangs
- close() joins workers cleanly
- get_cpu_state() returns canonical keys
- Deterministic seeding produces stable derived seeds
"""

from __future__ import annotations

import numpy as np
import pytest

from gbxcule.backends.pyboy_vec_mp import PyBoyMpConfig, PyBoyVecMpBackend

from .conftest import ROM_PATH, BackendComplianceTests, require_rom


class TestConfig:
    """Tests for PyBoyMpConfig validation."""

    def test_valid_config(self) -> None:
        """Valid configuration is accepted."""
        require_rom(ROM_PATH)
        config = PyBoyMpConfig(
            num_envs=4,
            num_workers=2,
            frames_per_step=24,
            release_after_frames=8,
            rom_path=str(ROM_PATH),
        )
        assert config.num_envs == 4
        assert config.num_workers == 2

    def test_invalid_num_envs(self) -> None:
        """num_envs < 1 raises ValueError."""
        require_rom(ROM_PATH)
        with pytest.raises(ValueError, match="num_envs must be >= 1"):
            PyBoyMpConfig(
                num_envs=0,
                num_workers=1,
                frames_per_step=24,
                release_after_frames=8,
                rom_path=str(ROM_PATH),
            )

    def test_invalid_num_workers(self) -> None:
        """num_workers < 1 raises ValueError."""
        require_rom(ROM_PATH)
        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            PyBoyMpConfig(
                num_envs=4,
                num_workers=0,
                frames_per_step=24,
                release_after_frames=8,
                rom_path=str(ROM_PATH),
            )

    def test_workers_exceed_envs(self) -> None:
        """num_workers > num_envs raises ValueError."""
        require_rom(ROM_PATH)
        with pytest.raises(ValueError, match="cannot exceed"):
            PyBoyMpConfig(
                num_envs=2,
                num_workers=4,
                frames_per_step=24,
                release_after_frames=8,
                rom_path=str(ROM_PATH),
            )

    def test_invalid_release_frames(self) -> None:
        """release_after_frames > frames_per_step raises ValueError."""
        require_rom(ROM_PATH)
        with pytest.raises(ValueError, match="cannot exceed"):
            PyBoyMpConfig(
                num_envs=4,
                num_workers=2,
                frames_per_step=10,
                release_after_frames=20,
                rom_path=str(ROM_PATH),
            )


class TestPyBoyVecMpCompliance(BackendComplianceTests):
    """Compliance tests for PyBoyVecMpBackend."""

    expected_name = "pyboy_vec_mp"
    expected_num_envs = 4
    obs_dim = 32

    @pytest.fixture
    def backend(self) -> PyBoyVecMpBackend:
        """Create a backend instance for testing."""
        require_rom(ROM_PATH)
        be = PyBoyVecMpBackend(str(ROM_PATH), num_envs=4, num_workers=2, obs_dim=32)
        yield be
        be.close()


class TestPyBoyVecMpSpecific:
    """Tests specific to PyBoyVecMpBackend (not in compliance suite)."""

    @pytest.fixture
    def backend(self) -> PyBoyVecMpBackend:
        """Create a backend instance for testing."""
        require_rom(ROM_PATH)
        return PyBoyVecMpBackend(str(ROM_PATH), num_envs=4, num_workers=2, obs_dim=32)

    def test_reset_info_contains_rom_sha256(self, backend: PyBoyVecMpBackend) -> None:
        """reset() info contains rom_sha256."""
        try:
            _, info = backend.reset(seed=123)
            assert "rom_sha256" in info
        finally:
            backend.close()

    def test_step_small_count_no_hang(self, backend: PyBoyVecMpBackend) -> None:
        """step() runs 8 times without hanging."""
        try:
            backend.reset(seed=123)
            actions = np.zeros((4,), dtype=np.int32)

            for _ in range(8):
                obs, reward, done, trunc, info = backend.step(actions)
                assert obs.shape == (4, 32)
                assert reward.shape == (4,)
                assert done.shape == (4,)
                assert trunc.shape == (4,)
        finally:
            backend.close()

    def test_step_with_various_actions(self, backend: PyBoyVecMpBackend) -> None:
        """step() handles different action values correctly."""
        try:
            backend.reset(seed=123)

            for _ in range(4):
                max_action = max(0, backend.num_actions - 1)
                actions = np.array(
                    [0, min(1, max_action), min(2, max_action), max_action],
                    dtype=np.int32,
                )
                obs, _, _, _, _ = backend.step(actions)
                assert obs.shape == (4, 32)
        finally:
            backend.close()

    def test_get_cpu_state_all_envs(self, backend: PyBoyVecMpBackend) -> None:
        """get_cpu_state() works for all environment indices."""
        try:
            backend.reset(seed=123)

            for env_idx in range(4):
                state = backend.get_cpu_state(env_idx)
                assert "pc" in state
                assert "flags" in state
        finally:
            backend.close()

    def test_get_cpu_state_invalid_idx(self, backend: PyBoyVecMpBackend) -> None:
        """get_cpu_state() raises for invalid env_idx."""
        try:
            backend.reset(seed=123)
            with pytest.raises(ValueError, match="out of range"):
                backend.get_cpu_state(10)
        finally:
            backend.close()

    def test_close_after_reset_and_steps(self, backend: PyBoyVecMpBackend) -> None:
        """close() joins workers cleanly after work."""
        backend.reset(seed=123)
        actions = np.zeros((4,), dtype=np.int32)

        for _ in range(4):
            backend.step(actions)

        backend.close()
        assert not backend._initialized

    def test_close_idempotent(self, backend: PyBoyVecMpBackend) -> None:
        """close() can be called multiple times."""
        backend.reset(seed=123)
        backend.close()
        backend.close()

    def test_no_zombie_workers(self, backend: PyBoyVecMpBackend) -> None:
        """Workers are properly terminated after close."""
        backend.reset(seed=123)
        workers = backend._workers.copy()

        backend.close()

        for proc in workers:
            assert not proc.is_alive()


class TestDeterministicSeeding:
    """Tests for deterministic seed derivation."""

    @pytest.fixture
    def backend(self) -> PyBoyVecMpBackend:
        """Create a backend instance for testing."""
        require_rom(ROM_PATH)
        return PyBoyVecMpBackend(str(ROM_PATH), num_envs=4, num_workers=2, obs_dim=32)

    def test_derived_seeds_are_stable(self, backend: PyBoyVecMpBackend) -> None:
        """Same seed produces same derived seeds."""
        try:
            _, info1 = backend.reset(seed=42)
            seeds1 = info1.get("derived_seeds")

            _, info2 = backend.reset(seed=42)
            seeds2 = info2.get("derived_seeds")

            assert seeds1 is not None
            assert seeds2 is not None
            assert seeds1 == seeds2
        finally:
            backend.close()

    def test_different_seeds_produce_different_derived(
        self, backend: PyBoyVecMpBackend
    ) -> None:
        """Different base seeds produce different derived seeds."""
        try:
            _, info1 = backend.reset(seed=42)
            seeds1 = info1.get("derived_seeds")

            _, info2 = backend.reset(seed=99)
            seeds2 = info2.get("derived_seeds")

            assert seeds1 != seeds2
        finally:
            backend.close()

    def test_derived_seeds_are_unique_per_env(self, backend: PyBoyVecMpBackend) -> None:
        """Each environment gets a unique derived seed."""
        try:
            _, info = backend.reset(seed=42)
            seeds = info.get("derived_seeds")

            assert seeds is not None
            assert len(seeds) == 4
            assert len(set(seeds)) == 4
        finally:
            backend.close()
