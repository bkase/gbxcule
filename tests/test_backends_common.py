"""Tests for backends/common.py types and validation helpers.

These tests are fast (no PyBoy import required) and verify:
- JSON serialization of dataclasses
- Action validation helpers
- CPU flag derivation
- Step output validation
- Run schema serialization
"""

import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from gbxcule.backends.common import (
    RESULT_SCHEMA_VERSION,
    ArraySpec,
    BackendSpec,
    RunConfig,
    RunResult,
    SystemInfo,
    as_i32_actions,
    empty_obs,
    flags_from_f,
    run_artifact_to_json_dict,
    validate_actions,
    validate_step_output,
)

# ---------------------------------------------------------------------------
# ArraySpec / BackendSpec JSON serialization
# ---------------------------------------------------------------------------


def test_array_spec_to_json_dict() -> None:
    """ArraySpec.to_json_dict produces JSON-serializable output."""
    spec = ArraySpec(shape=(4,), dtype="int32", meaning="action index")
    d = spec.to_json_dict()

    # Should serialize without error
    s = json.dumps(d)
    assert isinstance(s, str)

    # Shape converted to list
    assert d["shape"] == [4]
    assert d["dtype"] == "int32"
    assert d["meaning"] == "action index"


def test_backend_spec_to_json_dict() -> None:
    """BackendSpec.to_json_dict produces JSON-serializable output."""
    action_spec = ArraySpec(shape=(4,), dtype="int32", meaning="action")
    obs_spec = ArraySpec(shape=(4, 32), dtype="float32", meaning="observation")
    spec = BackendSpec(
        name="test_backend",
        device="cpu",
        num_envs=4,
        action=action_spec,
        obs=obs_spec,
    )
    d = spec.to_json_dict()

    # Should serialize without error
    s = json.dumps(d)
    assert isinstance(s, str)

    # Nested specs are also converted
    assert d["name"] == "test_backend"
    assert d["device"] == "cpu"
    assert d["num_envs"] == 4
    assert d["action"]["shape"] == [4]
    assert d["obs"]["shape"] == [4, 32]


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------


def test_validate_actions_correct() -> None:
    """Valid actions pass validation."""
    actions = np.array([0, 1, 2, 3], dtype=np.int32)
    validate_actions(actions, 4)  # Should not raise


def test_validate_actions_wrong_shape_raises() -> None:
    """2D actions raise ValueError."""
    actions = np.array([[0, 1], [2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="must be 1D"):
        validate_actions(actions, 4)


def test_validate_actions_wrong_length_raises() -> None:
    """Mismatched length raises ValueError."""
    actions = np.array([0, 1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="!= num_envs"):
        validate_actions(actions, 4)


def test_validate_actions_float_dtype_raises() -> None:
    """Float dtype raises ValueError."""
    actions = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="must be integer"):
        validate_actions(actions, 4)


def test_as_i32_actions_casts_int64() -> None:
    """int64 actions are cast to int32."""
    actions = np.array([0, 1, 2, 3], dtype=np.int64)
    result = as_i32_actions(actions, 4)
    assert result.dtype == np.int32


def test_as_i32_actions_preserves_int32() -> None:
    """int32 actions are returned as-is."""
    actions = np.array([0, 1, 2, 3], dtype=np.int32)
    result = as_i32_actions(actions, 4)
    assert result is actions  # Same object


# ---------------------------------------------------------------------------
# flags_from_f property test
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=255))
def test_flags_from_f_matches_f_register(f: int) -> None:
    """flags_from_f correctly extracts flag bits from F register.

    The F register stores flags in the upper nibble:
    - Bit 7: Z (zero)
    - Bit 6: N (subtract)
    - Bit 5: H (half-carry)
    - Bit 4: C (carry)
    """
    flags = flags_from_f(f)

    # Each flag is 0 or 1
    assert flags["z"] in (0, 1)
    assert flags["n"] in (0, 1)
    assert flags["h"] in (0, 1)
    assert flags["c"] in (0, 1)

    # Flags match the expected bits
    assert flags["z"] == ((f >> 7) & 1)
    assert flags["n"] == ((f >> 6) & 1)
    assert flags["h"] == ((f >> 5) & 1)
    assert flags["c"] == ((f >> 4) & 1)


def test_flags_from_f_specific_values() -> None:
    """Test specific flag values for clarity."""
    # Z=1, N=0, H=1, C=1 (0xB0 = 0b10110000)
    flags = flags_from_f(0xB0)
    assert flags["z"] == 1
    assert flags["n"] == 0
    assert flags["h"] == 1
    assert flags["c"] == 1

    # All flags set (0xF0 = 0b11110000)
    flags = flags_from_f(0xF0)
    assert flags["z"] == 1
    assert flags["n"] == 1
    assert flags["h"] == 1
    assert flags["c"] == 1

    # No flags set (lower nibble only)
    flags = flags_from_f(0x0F)
    assert flags["z"] == 0
    assert flags["n"] == 0
    assert flags["h"] == 0
    assert flags["c"] == 0


# ---------------------------------------------------------------------------
# validate_step_output
# ---------------------------------------------------------------------------


def test_validate_step_output_correct() -> None:
    """Valid step output passes validation."""
    num_envs, obs_dim = 4, 32
    obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
    reward = np.zeros((num_envs,), dtype=np.float32)
    done = np.zeros((num_envs,), dtype=np.bool_)
    trunc = np.zeros((num_envs,), dtype=np.bool_)

    validate_step_output(obs, reward, done, trunc, num_envs, obs_dim)


def test_validate_step_output_wrong_obs_shape_raises() -> None:
    """Wrong obs shape raises ValueError."""
    num_envs, obs_dim = 4, 32
    obs = np.zeros((num_envs, obs_dim + 1), dtype=np.float32)  # Wrong dim
    reward = np.zeros((num_envs,), dtype=np.float32)
    done = np.zeros((num_envs,), dtype=np.bool_)
    trunc = np.zeros((num_envs,), dtype=np.bool_)

    with pytest.raises(ValueError, match="obs shape"):
        validate_step_output(obs, reward, done, trunc, num_envs, obs_dim)


def test_validate_step_output_wrong_obs_dtype_raises() -> None:
    """Wrong obs dtype raises ValueError."""
    num_envs, obs_dim = 4, 32
    obs = np.zeros((num_envs, obs_dim), dtype=np.float64)  # Wrong dtype
    reward = np.zeros((num_envs,), dtype=np.float32)
    done = np.zeros((num_envs,), dtype=np.bool_)
    trunc = np.zeros((num_envs,), dtype=np.bool_)

    with pytest.raises(ValueError, match="obs dtype"):
        validate_step_output(obs, reward, done, trunc, num_envs, obs_dim)


def test_validate_step_output_wrong_done_dtype_raises() -> None:
    """Wrong done dtype raises ValueError."""
    num_envs, obs_dim = 4, 32
    obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
    reward = np.zeros((num_envs,), dtype=np.float32)
    done = np.zeros((num_envs,), dtype=np.int32)  # Wrong dtype
    trunc = np.zeros((num_envs,), dtype=np.bool_)

    with pytest.raises(ValueError, match="done dtype"):
        validate_step_output(obs, reward, done, trunc, num_envs, obs_dim)


# ---------------------------------------------------------------------------
# empty_obs helper
# ---------------------------------------------------------------------------


def test_empty_obs_shape_and_dtype() -> None:
    """empty_obs creates correct shape and dtype."""
    obs = empty_obs(4, 32)
    assert obs.shape == (4, 32)
    assert obs.dtype == np.float32
    assert np.all(obs == 0)


# ---------------------------------------------------------------------------
# Run schema JSON serialization
# ---------------------------------------------------------------------------


def test_system_info_to_json_dict() -> None:
    """SystemInfo.to_json_dict produces JSON-serializable output."""
    info = SystemInfo(
        platform="Linux",
        python="3.11.0",
        numpy="1.26.0",
        pyboy="2.6.1",
        warp=None,
        cpu="AMD Ryzen",
        gpu=None,
    )
    d = info.to_json_dict()
    s = json.dumps(d)
    assert isinstance(s, str)
    assert d["platform"] == "Linux"
    assert d["warp"] is None


def test_run_config_to_json_dict() -> None:
    """RunConfig.to_json_dict produces JSON-serializable output."""
    config = RunConfig(
        backend="pyboy_single",
        device="cpu",
        rom_path="bench/roms/out/ALU_LOOP.gb",
        rom_sha256="abc123",
        stage="full_step",
        num_envs=1,
        frames_per_step=24,
        release_after_frames=8,
        steps=1000,
        warmup_steps=100,
        actions_seed=42,
        sync_every=None,
    )
    d = config.to_json_dict()
    s = json.dumps(d)
    assert isinstance(s, str)
    assert d["backend"] == "pyboy_single"
    assert d["sync_every"] is None


def test_run_result_to_json_dict() -> None:
    """RunResult.to_json_dict produces JSON-serializable output."""
    result = RunResult(
        measured_steps=1000,
        seconds=1.5,
        total_sps=666.67,
        per_env_sps=666.67,
        frames_per_sec=16000.08,
    )
    d = result.to_json_dict()
    s = json.dumps(d)
    assert isinstance(s, str)
    assert d["measured_steps"] == 1000
    assert d["seconds"] == 1.5


def test_run_artifact_to_json_dict_roundtrip() -> None:
    """Full run artifact serializes and contains required keys."""
    system = SystemInfo(
        platform="Linux",
        python="3.11.0",
        numpy="1.26.0",
        pyboy="2.6.1",
        warp="1.11.0",
        cpu="AMD Ryzen",
        gpu="RTX 4090",
    )
    config = RunConfig(
        backend="pyboy_single",
        device="cpu",
        rom_path="test.gb",
        rom_sha256="abc",
        stage="full_step",
        num_envs=1,
        frames_per_step=24,
        release_after_frames=8,
        steps=100,
        warmup_steps=10,
        actions_seed=42,
        sync_every=10,
    )
    result = RunResult(
        measured_steps=100,
        seconds=0.5,
        total_sps=200.0,
        per_env_sps=200.0,
        frames_per_sec=4800.0,
    )

    artifact = run_artifact_to_json_dict(system, config, result)

    # Verify schema version
    assert artifact["schema_version"] == RESULT_SCHEMA_VERSION

    # Verify structure
    assert "system" in artifact
    assert "config" in artifact
    assert "result" in artifact

    # Verify roundtrip
    s = json.dumps(artifact)
    parsed = json.loads(s)
    assert parsed["schema_version"] == 1
    assert parsed["config"]["backend"] == "pyboy_single"


# ---------------------------------------------------------------------------
# CPU state hash helper
# ---------------------------------------------------------------------------


def test_hash_cpu_state_deterministic() -> None:
    """hash_cpu_state is deterministic for identical states."""
    from gbxcule.backends.common import CpuState
    from gbxcule.core.signatures import hash_cpu_state

    state: CpuState = {
        "pc": 0x0150,
        "sp": 0xFFFE,
        "a": 0x00,
        "f": 0xB0,
        "b": 0x01,
        "c": 0x13,
        "d": 0x00,
        "e": 0xD8,
        "h": 0x01,
        "l": 0x4D,
        "flags": {"z": 1, "n": 0, "h": 1, "c": 1},
        "instr_count": None,
        "cycle_count": None,
    }

    h1 = hash_cpu_state(state)
    h2 = hash_cpu_state(state)

    assert h1 == h2
    assert len(h1) == 64  # blake2b 32-byte digest = 64 hex chars


def test_hash_cpu_state_different_for_different_states() -> None:
    """hash_cpu_state produces different hashes for different states."""
    from gbxcule.backends.common import CpuState
    from gbxcule.core.signatures import hash_cpu_state

    state1: CpuState = {"pc": 0x0150, "sp": 0xFFFE, "a": 0}
    state2: CpuState = {"pc": 0x0151, "sp": 0xFFFE, "a": 0}

    h1 = hash_cpu_state(state1)
    h2 = hash_cpu_state(state2)

    assert h1 != h2


def test_hash_cpu_state_exclude_counters() -> None:
    """hash_cpu_state can exclude counters from hash."""
    from gbxcule.backends.common import CpuState
    from gbxcule.core.signatures import hash_cpu_state

    state_with: CpuState = {"pc": 0x0150, "instr_count": 100}
    state_without: CpuState = {"pc": 0x0150, "instr_count": 200}

    # With counters, different
    h1 = hash_cpu_state(state_with, include_counters=True)
    h2 = hash_cpu_state(state_without, include_counters=True)
    assert h1 != h2

    # Without counters, same (only pc matters)
    h3 = hash_cpu_state(state_with, include_counters=False)
    h4 = hash_cpu_state(state_without, include_counters=False)
    assert h3 == h4
