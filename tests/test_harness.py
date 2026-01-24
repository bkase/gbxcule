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
    get_action_codec_metadata,
    get_action_gen_metadata,
    run_benchmark,
    write_artifact,
    write_scaling_artifact,
)

from gbxcule.backends.common import ArraySpec, CpuState, Device
from gbxcule.core.action_codec import POKERED_PUFFER_V0_ID, get_action_codec

_ACTION_CODEC = get_action_codec(POKERED_PUFFER_V0_ID)
LEGACY_NUM_ACTIONS = _ACTION_CODEC.num_actions
LEGACY_ACTION_NAMES = list(_ACTION_CODEC.action_names)

# ---------------------------------------------------------------------------
# Fake Backend for testing
# ---------------------------------------------------------------------------


class FakeVecBackend:
    """Fake backend for testing harness without PyBoy dependency."""

    name: str = "fake_backend"
    device: Device = "cpu"

    def __init__(self, num_envs: int = 4, obs_dim: int = 32) -> None:
        self.num_envs = num_envs
        self.num_actions = 8  # Fake action count
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

    def read_memory(self, env_idx: int, lo: int, hi: int) -> bytes:
        """Return fake memory bytes."""
        return bytes(hi - lo)

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
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
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
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
        )
        actions2 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=99,
            gen_name="noop",
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
        )
        np.testing.assert_array_equal(actions1, actions2)

    def test_seeded_random_deterministic(self) -> None:
        """seeded_random produces identical results for same inputs."""
        actions1 = generate_actions(
            step_idx=5,
            num_envs=8,
            seed=42,
            gen_name="seeded_random",
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
        )
        actions2 = generate_actions(
            step_idx=5,
            num_envs=8,
            seed=42,
            gen_name="seeded_random",
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
        )
        np.testing.assert_array_equal(actions1, actions2)

    def test_seeded_random_varies_by_step_idx(self) -> None:
        """seeded_random produces different actions for different step indices."""
        actions1 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=42,
            gen_name="seeded_random",
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
        )
        actions2 = generate_actions(
            step_idx=1,
            num_envs=4,
            seed=42,
            gen_name="seeded_random",
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
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
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
        )
        actions2 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=99,
            gen_name="seeded_random",
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
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
                num_actions=LEGACY_NUM_ACTIONS,
                action_names=LEGACY_ACTION_NAMES,
            )

    def test_seeded_random_action_range(self) -> None:
        """seeded_random produces actions in valid range [0, 8]."""
        actions = generate_actions(
            step_idx=0,
            num_envs=100,
            seed=42,
            gen_name="seeded_random",
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
        )
        assert np.all(actions >= 0)
        assert np.all(actions < LEGACY_NUM_ACTIONS)

    def test_unknown_generator_raises(self) -> None:
        """Unknown generator name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown action generator"):
            generate_actions(
                step_idx=0,
                num_envs=4,
                seed=42,
                gen_name="unknown_gen",
                num_actions=LEGACY_NUM_ACTIONS,
                action_names=LEGACY_ACTION_NAMES,
            )

    def test_single_env_batch_semantics(self) -> None:
        """Single env still returns array with correct shape."""
        actions = generate_actions(
            step_idx=0,
            num_envs=1,
            seed=42,
            gen_name="noop",
            num_actions=LEGACY_NUM_ACTIONS,
            action_names=LEGACY_ACTION_NAMES,
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

    def test_striped_metadata_has_pattern(self) -> None:
        """striped metadata includes pattern and action names."""
        meta = get_action_gen_metadata(
            "striped",
            None,
            action_names=LEGACY_ACTION_NAMES,
            num_actions=LEGACY_NUM_ACTIONS,
        )
        assert meta["pattern_action_names"] == ["UP", "DOWN", "LEFT", "RIGHT"]


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
                sync_every=None,
                num_actions=LEGACY_NUM_ACTIONS,
                action_names=LEGACY_ACTION_NAMES,
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
                sync_every=None,
                num_actions=LEGACY_NUM_ACTIONS,
                action_names=LEGACY_ACTION_NAMES,
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
                sync_every=None,
                num_actions=LEGACY_NUM_ACTIONS,
                action_names=LEGACY_ACTION_NAMES,
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
                sync_every=None,
                num_actions=LEGACY_NUM_ACTIONS,
                action_names=LEGACY_ACTION_NAMES,
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
                sync_every=None,
                num_actions=LEGACY_NUM_ACTIONS,
                action_names=LEGACY_ACTION_NAMES,
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


# ---------------------------------------------------------------------------
# State Comparison Tests
# ---------------------------------------------------------------------------


class TestStateDiff:
    """Tests for CPU state normalization and comparison."""

    def test_normalize_cpu_state_all_keys(self) -> None:
        """normalize_cpu_state includes all canonical keys."""
        from harness import normalize_cpu_state

        raw_state = {
            "pc": 0x100,
            "sp": 0xFFFE,
            "a": 1,
            "f": 0xB0,
            "b": 2,
            "c": 3,
            "d": 4,
            "e": 5,
            "h": 6,
            "l": 7,
            "flags": {"z": 1, "n": 0, "h": 1, "c": 1},
        }

        normalized = normalize_cpu_state(raw_state)

        # Check all canonical keys present
        assert "pc" in normalized
        assert "sp" in normalized
        assert "flags" in normalized

    def test_diff_states_no_diff(self) -> None:
        """diff_states returns None when states match."""
        from harness import diff_states, normalize_cpu_state

        state = normalize_cpu_state(
            {
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
        )

        diff = diff_states(state, state)
        assert diff is None

    def test_diff_states_detects_diff(self) -> None:
        """diff_states detects register differences."""
        from harness import diff_states, normalize_cpu_state

        ref = normalize_cpu_state(
            {
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
        )
        dut = normalize_cpu_state(
            {
                "pc": 0x200,  # Different
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
        )

        diff = diff_states(ref, dut)
        assert diff is not None
        assert "pc" in diff
        assert diff["pc"]["ref"] == 0x100
        assert diff["pc"]["dut"] == 0x200


# ---------------------------------------------------------------------------
# Memory Region Parsing Tests
# ---------------------------------------------------------------------------


class TestMemRegionParsing:
    """Tests for memory region parsing and hashing helpers."""

    def test_parse_mem_region_valid(self) -> None:
        """parse_mem_region parses hex ranges with or without 0x."""
        from harness import parse_mem_region

        assert parse_mem_region("C000:C100") == (0xC000, 0xC100)
        assert parse_mem_region("0xC000:0xC100") == (0xC000, 0xC100)

    def test_parse_mem_region_invalid(self) -> None:
        """parse_mem_region rejects invalid ranges."""
        from harness import parse_mem_region

        with pytest.raises(ValueError):
            parse_mem_region("C000")
        with pytest.raises(ValueError):
            parse_mem_region("C100:C000")
        with pytest.raises(ValueError):
            parse_mem_region("0x0000:0x10001")

    def test_parse_mem_regions_empty(self) -> None:
        """parse_mem_regions returns empty list for None/empty."""
        from harness import parse_mem_regions

        assert parse_mem_regions(None) == []
        assert parse_mem_regions([]) == []

    def test_hash_memory_deterministic(self) -> None:
        """hash_memory is deterministic and stable."""
        from harness import hash_memory

        data = b"\x00\x01\x02\x03"
        assert hash_memory(data) == hash_memory(data)


# ---------------------------------------------------------------------------
# System Info Tests
# ---------------------------------------------------------------------------


class TestSystemInfo:
    """Tests for system info capture used in artifacts and mismatch bundles."""

    def test_get_system_info_includes_provenance_keys(self) -> None:
        """get_system_info always includes provenance keys (best-effort values)."""
        from harness import get_system_info

        info = get_system_info()

        assert "gpu" in info
        assert "gpu_names" in info
        assert "driver_version" in info
        assert "cuda_visible_devices" in info
        assert "warp" in info
        assert "warp_dist_name" in info
        assert "warp_dist_version" in info
        assert "warp_direct_url" in info
        assert "warp_wheel_source" in info

    def test_get_system_info_parses_nvidia_smi_output(self) -> None:
        """GPU name + driver version are parsed from nvidia-smi when present."""
        import subprocess

        from harness import get_system_info

        def fake_run(
            cmd: list[str], *args: Any, **kwargs: Any
        ) -> subprocess.CompletedProcess[str]:
            if cmd[:1] == ["nvidia-smi"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=(
                        "NVIDIA A100-SXM4-80GB, 535.104.05\n"
                        "NVIDIA A100-SXM4-80GB, 535.104.05\n"
                    ),
                    stderr="",
                )
            if cmd[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout="deadbeef\n", stderr=""
                )
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        with mock.patch("subprocess.run", side_effect=fake_run):
            info = get_system_info()

        assert info["gpu"] == "NVIDIA A100-SXM4-80GB"
        assert info["gpu_names"] == ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"]
        assert info["driver_version"] == "535.104.05"
        assert info["git_commit"] == "deadbeef"

    def test_get_system_info_reads_pep610_direct_url(self, tmp_path: Path) -> None:
        """Warp direct_url.json is recorded when present (PEP 610)."""
        from importlib import metadata as importlib_metadata

        from harness import get_system_info

        direct_url_path = tmp_path / "direct_url.json"
        direct_url_path.write_text(
            json.dumps({"url": "https://example.com/warp_lang-1.2.3-cp311.whl"})
        )

        class FakeDist:
            version = "1.2.3"
            metadata = {"Name": "warp-lang"}
            files = ["warp_lang-1.2.3.dist-info/direct_url.json"]

            def locate_file(self, file: object) -> Path:
                return direct_url_path

        def fake_distribution(name: str) -> FakeDist:
            if name == "warp-lang":
                return FakeDist()
            raise importlib_metadata.PackageNotFoundError(name)

        with mock.patch(
            "importlib.metadata.distribution", side_effect=fake_distribution
        ):
            info = get_system_info()

        assert info["warp_dist_version"] == "1.2.3"
        assert (
            info["warp_direct_url"] == "https://example.com/warp_lang-1.2.3-cp311.whl"
        )
        assert info["warp_wheel_source"] == "pep610"


# ---------------------------------------------------------------------------
# Mismatch Bundle Tests
# ---------------------------------------------------------------------------


class TestMismatchBundle:
    """Tests for mismatch bundle writing."""

    def test_write_mismatch_bundle_creates_files(self) -> None:
        """write_mismatch_bundle creates all required files."""
        from harness import write_mismatch_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            rom_path = Path(tmpdir) / "test.gb"
            rom_bytes = b"\x00\x01\x02\x03"
            rom_path.write_bytes(rom_bytes)
            bundle_path = write_mismatch_bundle(
                output_dir=Path(tmpdir),
                timestamp="20260101_120000",
                rom_path=rom_path,
                rom_sha256="abc123",
                ref_backend="pyboy_single",
                dut_backend="warp_vec",
                mismatch_step=5,
                env_idx=0,
                ref_state={"pc": 0x100},
                dut_state={"pc": 0x200},
                diff={"pc": {"ref": 0x100, "dut": 0x200}},
                action_codec=get_action_codec_metadata(POKERED_PUFFER_V0_ID),
                actions_trace=[[0], [0], [0], [0], [0], [0]],
                system_info={"platform": "test"},
                action_gen_name="noop",
                action_gen_seed=None,
                frames_per_step=24,
                release_after_frames=8,
                compare_every=1,
                verify_steps=10,
            )

            # Check all required files exist
            assert (bundle_path / "metadata.json").exists()
            assert (bundle_path / "ref_state.json").exists()
            assert (bundle_path / "dut_state.json").exists()
            assert (bundle_path / "diff.json").exists()
            assert (bundle_path / "actions.jsonl").exists()
            assert (bundle_path / "repro.sh").exists()
            assert (bundle_path / "rom.gb").exists()
            assert (bundle_path / "rom.gb").read_bytes() == rom_bytes

    def test_mismatch_bundle_metadata_schema(self) -> None:
        """Metadata has correct schema version."""
        from harness import MISMATCH_SCHEMA_VERSION, write_mismatch_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            rom_path = Path(tmpdir) / "test.gb"
            rom_bytes = b"\x10\x20"
            rom_path.write_bytes(rom_bytes)
            bundle_path = write_mismatch_bundle(
                output_dir=Path(tmpdir),
                timestamp="20260101_120000",
                rom_path=rom_path,
                rom_sha256="abc123",
                ref_backend="pyboy_single",
                dut_backend="warp_vec",
                mismatch_step=0,
                env_idx=0,
                ref_state={},
                dut_state={},
                diff={},
                action_codec=get_action_codec_metadata(POKERED_PUFFER_V0_ID),
                actions_trace=[],
                system_info={},
                action_gen_name="noop",
                action_gen_seed=None,
                frames_per_step=24,
                release_after_frames=8,
                compare_every=1,
                verify_steps=1,
            )

            with open(bundle_path / "metadata.json") as f:
                metadata = json.load(f)

            assert metadata["schema_version"] == MISMATCH_SCHEMA_VERSION
            assert metadata["mismatch_step"] == 0
            assert metadata["ref_backend"] == "pyboy_single"
            assert metadata["dut_backend"] == "warp_vec"
            assert metadata["rom_filename"] == "rom.gb"
            assert metadata["rom_size"] == len(rom_bytes)

    def test_repro_sh_uses_uv_and_actions_file(self) -> None:
        """repro.sh uses uv run and --actions-file."""
        from harness import write_mismatch_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            rom_path = Path(tmpdir) / "test.gb"
            rom_path.write_bytes(b"\x00")
            bundle_path = write_mismatch_bundle(
                output_dir=Path(tmpdir),
                timestamp="20260101_120000",
                rom_path=rom_path,
                rom_sha256="abc123",
                ref_backend="pyboy_single",
                dut_backend="warp_vec",
                mismatch_step=0,
                env_idx=0,
                ref_state={},
                dut_state={},
                diff={},
                action_codec=get_action_codec_metadata(POKERED_PUFFER_V0_ID),
                actions_trace=[],
                system_info={},
                action_gen_name="noop",
                action_gen_seed=None,
                frames_per_step=24,
                release_after_frames=8,
                compare_every=1,
                verify_steps=1,
            )

            with open(bundle_path / "repro.sh") as f:
                repro_content = f.read()

            assert "uv run" in repro_content
            assert "--actions-file" in repro_content
            assert "--verify" in repro_content
            assert "rom.gb" in repro_content
            assert "test.gb" not in repro_content

    def test_mismatch_bundle_mem_regions_and_dumps(self) -> None:
        """mismatch bundle records mem regions and writes dumps."""
        from harness import MEM_HASH_VERSION, write_mismatch_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            rom_path = Path(tmpdir) / "test.gb"
            rom_path.write_bytes(b"\x00\x01")
            mem_regions = [(0xC000, 0xC004)]
            mem_dumps = [(0xC000, 0xC004, b"\x01\x02\x03\x04", b"\x05\x06\x07\x08")]

            bundle_path = write_mismatch_bundle(
                output_dir=Path(tmpdir),
                timestamp="20260101_120000",
                rom_path=rom_path,
                rom_sha256="abc123",
                ref_backend="pyboy_single",
                dut_backend="warp_vec",
                mismatch_step=1,
                env_idx=0,
                ref_state={},
                dut_state={},
                diff={"memory": [{"lo": 0xC000, "hi": 0xC004}]},
                action_codec=get_action_codec_metadata(POKERED_PUFFER_V0_ID),
                mem_regions=mem_regions,
                mem_hash_version=MEM_HASH_VERSION,
                mem_dumps=mem_dumps,
                actions_trace=[],
                system_info={},
                action_gen_name="noop",
                action_gen_seed=None,
                frames_per_step=24,
                release_after_frames=8,
                compare_every=1,
                verify_steps=1,
            )

            with open(bundle_path / "metadata.json") as f:
                metadata = json.load(f)

            assert metadata["mem_regions"] == [{"lo": 0xC000, "hi": 0xC004}]
            assert metadata["mem_hash_version"] == MEM_HASH_VERSION

            ref_dump = bundle_path / "mem_ref_C000_C004.bin"
            dut_dump = bundle_path / "mem_dut_C000_C004.bin"
            assert ref_dump.exists()
            assert dut_dump.exists()
            assert ref_dump.read_bytes() == b"\x01\x02\x03\x04"
            assert dut_dump.read_bytes() == b"\x05\x06\x07\x08"

            with open(bundle_path / "repro.sh") as f:
                repro_content = f.read()

            assert "--mem-region C000:C004" in repro_content

    def test_mismatch_bundle_ppu_metadata_and_dumps(self) -> None:
        """mismatch bundle records PPU metadata and dump blobs when provided."""
        from harness import write_mismatch_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            rom_path = Path(tmpdir) / "test.gb"
            rom_path.write_bytes(b"\xaa\xbb")
            ppu_regs = {"ref": {"LCDC": 0x91}, "dut": {"LCDC": 0x90}}
            frame_hash_history = [
                {"step": 3, "ref_hash": "aaa", "dut_hash": "bbb"},
                {"step": 4, "ref_hash": "ccc", "dut_hash": "ddd"},
            ]
            ppu_mem_dump_meta = [
                {
                    "name": "oam",
                    "lo": 0xFE00,
                    "hi": 0xFEA0,
                    "dump_hi": 0xFE10,
                    "truncated": True,
                }
            ]
            ppu_mem_dump_blobs = [
                ("oam", 0xFE00, 0xFE10, b"\x01" * 0x10, b"\x02" * 0x10)
            ]

            bundle_path = write_mismatch_bundle(
                output_dir=Path(tmpdir),
                timestamp="20260101_120000",
                rom_path=rom_path,
                rom_sha256="abc123",
                ref_backend="pyboy_single",
                dut_backend="warp_vec",
                mismatch_step=2,
                env_idx=0,
                ref_state={},
                dut_state={},
                diff={},
                action_codec=get_action_codec_metadata(POKERED_PUFFER_V0_ID),
                actions_trace=[],
                system_info={},
                action_gen_name="noop",
                action_gen_seed=None,
                frames_per_step=24,
                release_after_frames=8,
                compare_every=1,
                verify_steps=5,
                ppu_regs=ppu_regs,
                frame_hash_history=frame_hash_history,
                ppu_mem_dump_meta=ppu_mem_dump_meta,
                ppu_mem_dump_blobs=ppu_mem_dump_blobs,
            )

            with open(bundle_path / "metadata.json") as f:
                metadata = json.load(f)

            assert metadata["ppu_regs"] == ppu_regs
            assert metadata["frame_hash_history"] == frame_hash_history
            assert metadata["ppu_mem_dumps"] == ppu_mem_dump_meta

            ref_dump = bundle_path / "oam_ref_FE00_FE10.bin"
            dut_dump = bundle_path / "oam_dut_FE00_FE10.bin"
            assert ref_dump.exists()
            assert dut_dump.exists()
            assert ref_dump.read_bytes() == b"\x01" * 0x10
            assert dut_dump.read_bytes() == b"\x02" * 0x10
