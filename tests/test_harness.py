"""Tests for harness action generation, SPS math, and artifact schema.

Tests cover:
- Deterministic action generation
- noop and seeded_random generators
- Action generator metadata
- SPS formula correctness (with fake backend)
- Artifact schema validation
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

# Add bench directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))

from harness import (  # noqa: E402
    ACTION_GEN_VERSION,
    generate_actions,
    get_action_gen_metadata,
    run_benchmark,
    write_artifact,
    write_scaling_artifact,
)

from gbxcule.backends.common import ArraySpec, CpuState, Device

# ---------------------------------------------------------------------------
# Fake Backend for testing
# ---------------------------------------------------------------------------


class FakeVecBackend:
    """Fake backend for testing harness without PyBoy dependency."""

    name: str = "fake_backend"
    device: Device = "cpu"

    def __init__(self, num_envs: int = 4, obs_dim: int = 32) -> None:
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        self._step_count = 0
        self._initialized = False

        self.action_spec = ArraySpec(
            shape=(num_envs,),
            dtype="int32",
            meaning="Action index per environment",
        )
        self.obs_spec = ArraySpec(
            shape=(num_envs, obs_dim),
            dtype="float32",
            meaning="Observation vector per environment",
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._initialized = True
        self._step_count = 0
        obs = np.zeros((self.num_envs, self._obs_dim), dtype=np.float32)
        return obs, {"seed": seed}

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if not self._initialized:
            raise RuntimeError("Backend not initialized")
        self._step_count += 1
        obs = np.zeros((self.num_envs, self._obs_dim), dtype=np.float32)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=bool)
        trunc = np.zeros((self.num_envs,), dtype=bool)
        return obs, reward, done, trunc, {"step": self._step_count}

    def get_cpu_state(self, env_idx: int) -> CpuState:
        return {
            "pc": 0x100,
            "sp": 0xFFFE,
            "a": 0,
            "f": 0,
            "b": 0,
            "c": 0,
            "d": 0,
            "e": 0,
            "h": 0,
            "l": 0,
            "flags": {"z": 0, "n": 0, "h": 0, "c": 0},
        }

    def close(self) -> None:
        self._initialized = False


class TestGenerateActions:
    """Tests for generate_actions pure function."""

    def test_noop_returns_zeros(self) -> None:
        """noop generator returns all zeros."""
        actions = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=None,
            gen_name="noop",
        )
        assert actions.shape == (4,)
        assert actions.dtype == np.int32
        assert np.all(actions == 0)

    def test_noop_ignores_seed(self) -> None:
        """noop generator works the same regardless of seed."""
        actions1 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=42,
            gen_name="noop",
        )
        actions2 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=99,
            gen_name="noop",
        )
        np.testing.assert_array_equal(actions1, actions2)

    def test_seeded_random_deterministic(self) -> None:
        """seeded_random produces identical results for same inputs."""
        actions1 = generate_actions(
            step_idx=5,
            num_envs=8,
            seed=42,
            gen_name="seeded_random",
        )
        actions2 = generate_actions(
            step_idx=5,
            num_envs=8,
            seed=42,
            gen_name="seeded_random",
        )
        np.testing.assert_array_equal(actions1, actions2)

    def test_seeded_random_varies_by_step_idx(self) -> None:
        """seeded_random produces different actions for different step indices."""
        actions1 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=42,
            gen_name="seeded_random",
        )
        actions2 = generate_actions(
            step_idx=1,
            num_envs=4,
            seed=42,
            gen_name="seeded_random",
        )
        # Should be different (extremely high probability)
        assert not np.array_equal(actions1, actions2)

    def test_seeded_random_varies_by_seed(self) -> None:
        """seeded_random produces different actions for different seeds."""
        actions1 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=42,
            gen_name="seeded_random",
        )
        actions2 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=99,
            gen_name="seeded_random",
        )
        # Should be different (extremely high probability)
        assert not np.array_equal(actions1, actions2)

    def test_seeded_random_requires_seed(self) -> None:
        """seeded_random raises ValueError if seed is None."""
        with pytest.raises(ValueError, match="requires a seed"):
            generate_actions(
                step_idx=0,
                num_envs=4,
                seed=None,
                gen_name="seeded_random",
            )

    def test_seeded_random_action_range(self) -> None:
        """seeded_random produces actions in valid range [0, 8]."""
        actions = generate_actions(
            step_idx=0,
            num_envs=100,
            seed=42,
            gen_name="seeded_random",
        )
        assert np.all(actions >= 0)
        assert np.all(actions < 9)

    def test_unknown_generator_raises(self) -> None:
        """Unknown generator name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown action generator"):
            generate_actions(
                step_idx=0,
                num_envs=4,
                seed=42,
                gen_name="unknown_gen",
            )

    def test_single_env_batch_semantics(self) -> None:
        """Single env still returns array with correct shape."""
        actions = generate_actions(
            step_idx=0,
            num_envs=1,
            seed=42,
            gen_name="noop",
        )
        assert actions.shape == (1,)
        assert actions.dtype == np.int32


class TestGetActionGenMetadata:
    """Tests for get_action_gen_metadata function."""

    def test_metadata_structure(self) -> None:
        """Metadata contains name, version, seed."""
        meta = get_action_gen_metadata("noop", None)
        assert meta["name"] == "noop"
        assert meta["version"] == ACTION_GEN_VERSION
        assert meta["seed"] is None

    def test_metadata_with_seed(self) -> None:
        """Metadata records seed correctly."""
        meta = get_action_gen_metadata("seeded_random", 42)
        assert meta["name"] == "seeded_random"
        assert meta["version"] == ACTION_GEN_VERSION
        assert meta["seed"] == 42

    def test_version_is_stable(self) -> None:
        """Version string is stable for reproducibility."""
        assert ACTION_GEN_VERSION == "1.0"


# ---------------------------------------------------------------------------
# SPS Math Tests (with fake backend + mocked timer)
# ---------------------------------------------------------------------------


class TestSPSMath:
    """Tests for SPS formula correctness using fake backend."""

    def test_total_sps_formula(self) -> None:
        """total_sps = (steps * num_envs) / seconds."""
        backend = FakeVecBackend(num_envs=4)

        # Mock time.perf_counter to return fixed values
        # Start at 0.0, end at 1.0 (1 second elapsed)
        with mock.patch("time.perf_counter", side_effect=[0.0, 1.0]):
            results = run_benchmark(
                backend,
                steps=10,
                warmup_steps=0,
                action_gen="noop",
                actions_seed=None,
                frames_per_step=24,
            )

        # total_env_steps = 10 * 4 = 40
        # total_sps = 40 / 1.0 = 40.0
        assert results["total_env_steps"] == 40
        assert results["total_sps"] == 40.0

    def test_per_env_sps_formula(self) -> None:
        """per_env_sps = steps / seconds."""
        backend = FakeVecBackend(num_envs=4)

        with mock.patch("time.perf_counter", side_effect=[0.0, 2.0]):
            results = run_benchmark(
                backend,
                steps=10,
                warmup_steps=0,
                action_gen="noop",
                actions_seed=None,
                frames_per_step=24,
            )

        # per_env_sps = 10 / 2.0 = 5.0
        assert results["per_env_sps"] == 5.0

    def test_frames_per_sec_formula(self) -> None:
        """frames_per_sec = total_sps * frames_per_step."""
        backend = FakeVecBackend(num_envs=2)

        with mock.patch("time.perf_counter", side_effect=[0.0, 1.0]):
            results = run_benchmark(
                backend,
                steps=10,
                warmup_steps=0,
                action_gen="noop",
                actions_seed=None,
                frames_per_step=24,
            )

        # total_sps = 20 / 1.0 = 20.0
        # frames_per_sec = 20.0 * 24 = 480.0
        assert results["frames_per_sec"] == 480.0

    def test_single_env_sps(self) -> None:
        """SPS math correct for single-env backend."""
        backend = FakeVecBackend(num_envs=1)

        with mock.patch("time.perf_counter", side_effect=[0.0, 0.5]):
            results = run_benchmark(
                backend,
                steps=5,
                warmup_steps=0,
                action_gen="noop",
                actions_seed=None,
                frames_per_step=24,
            )

        # total_sps = per_env_sps = 5 / 0.5 = 10.0
        assert results["total_sps"] == 10.0
        assert results["per_env_sps"] == 10.0

    def test_warmup_excluded_from_timing(self) -> None:
        """Warmup steps not included in SPS calculation."""
        backend = FakeVecBackend(num_envs=1)

        # Timer only starts after warmup, so we only mock the measured portion
        with mock.patch("time.perf_counter", side_effect=[0.0, 1.0]):
            results = run_benchmark(
                backend,
                steps=10,
                warmup_steps=5,  # Should run but not be timed
                action_gen="noop",
                actions_seed=None,
                frames_per_step=24,
            )

        # Only 10 measured steps, not 15
        assert results["measured_steps"] == 10
        assert results["warmup_steps"] == 5


# ---------------------------------------------------------------------------
# Artifact Schema Tests
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """Tests for artifact JSON schema validation."""

    def test_artifact_is_valid_json(self) -> None:
        """write_artifact produces valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_artifact(
                Path(tmpdir),
                run_id="test_run",
                system_info={"platform": "test"},
                config={"backend": "fake"},
                results={"sps": 100.0},
            )

            # Should be valid JSON
            with open(path) as f:
                data = json.load(f)

            assert isinstance(data, dict)

    def test_artifact_required_keys(self) -> None:
        """write_artifact includes all required top-level keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_artifact(
                Path(tmpdir),
                run_id="test_run",
                system_info={"platform": "test"},
                config={"backend": "fake"},
                results={"sps": 100.0},
            )

            with open(path) as f:
                data = json.load(f)

            # Required keys per schema
            assert "schema_version" in data
            assert "run_id" in data
            assert "timestamp_utc" in data
            assert "system" in data
            assert "config" in data
            assert "results" in data

    def test_artifact_schema_version(self) -> None:
        """Artifact has correct schema_version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_artifact(
                Path(tmpdir),
                run_id="test_run",
                system_info={},
                config={},
                results={},
            )

            with open(path) as f:
                data = json.load(f)

            assert data["schema_version"] == 1

    def test_scaling_artifact_is_valid_json(self) -> None:
        """write_scaling_artifact produces valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_scaling_artifact(
                Path(tmpdir),
                run_id="test_scaling",
                system_info={"platform": "test"},
                sweep_config={"env_counts": [1, 2, 4]},
                results_list=[
                    {"num_envs": 1, "total_sps": 100.0},
                    {"num_envs": 2, "total_sps": 180.0},
                ],
            )

            with open(path) as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert data["run_id"] == "test_scaling"

    def test_scaling_artifact_required_keys(self) -> None:
        """write_scaling_artifact includes all required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_scaling_artifact(
                Path(tmpdir),
                run_id="test_scaling",
                system_info={},
                sweep_config={"env_counts": [1, 2]},
                results_list=[],
            )

            with open(path) as f:
                data = json.load(f)

            assert "schema_version" in data
            assert "run_id" in data
            assert "timestamp_utc" in data
            assert "system" in data
            assert "sweep_config" in data
            assert "results" in data
            assert isinstance(data["results"], list)

    def test_scaling_artifact_filename(self) -> None:
        """Scaling artifact has __scaling suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_scaling_artifact(
                Path(tmpdir),
                run_id="test_scaling",
                system_info={},
                sweep_config={},
                results_list=[],
            )

            assert path.name == "test_scaling__scaling.json"
