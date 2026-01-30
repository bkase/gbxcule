# pyright: reportTypedDictNotRequiredAccess=false
"""Consolidated backend compliance tests.

This module runs the standard compliance test suite against all backend
implementations using parametrization, reducing test duplication while
maintaining full coverage.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from gbxcule.backends.common import VecBackend

from .conftest import ROM_PATH, require_rom

# ---------------------------------------------------------------------------
# Backend configurations
# ---------------------------------------------------------------------------


@dataclass
class BackendConfig:
    """Configuration for a backend under test."""

    module: str
    class_name: str
    kwargs: dict[str, Any]
    expected_name: str
    expected_num_envs: int

    @property
    def id(self) -> str:
        return self.expected_name


BACKEND_CONFIGS = [
    BackendConfig(
        module="gbxcule.backends.pyboy_single",
        class_name="PyBoySingleBackend",
        kwargs={"obs_dim": 32},
        expected_name="pyboy_single",
        expected_num_envs=1,
    ),
    BackendConfig(
        module="gbxcule.backends.pyboy_vec_mp",
        class_name="PyBoyVecMpBackend",
        kwargs={"num_envs": 4, "num_workers": 2, "obs_dim": 32},
        expected_name="pyboy_vec_mp",
        expected_num_envs=4,
    ),
    BackendConfig(
        module="gbxcule.backends.warp_vec",
        class_name="WarpVecCpuBackend",
        kwargs={"num_envs": 4, "obs_dim": 32},
        expected_name="warp_vec_cpu",
        expected_num_envs=4,
    ),
]


def _create_backend(config: BackendConfig) -> VecBackend:
    """Create a backend instance from config."""
    require_rom(ROM_PATH)
    module = importlib.import_module(config.module)
    backend_class = getattr(module, config.class_name)
    return backend_class(str(ROM_PATH), **config.kwargs)


@pytest.fixture(params=BACKEND_CONFIGS, ids=lambda c: c.id)
def backend_config(request: pytest.FixtureRequest) -> BackendConfig:
    """Provide backend configuration."""
    return request.param


@pytest.fixture
def backend(backend_config: BackendConfig) -> Iterator[VecBackend]:
    """Create and cleanup a backend instance."""
    be = _create_backend(backend_config)
    yield be
    be.close()


def _valid_action(backend: VecBackend, idx: int) -> int:
    """Return a valid action index for a given backend."""
    if getattr(backend, "num_actions", 0) < 1:
        raise AssertionError("backend.num_actions must be >= 1")
    return min(idx, backend.num_actions - 1)


# ---------------------------------------------------------------------------
# Backend Init Tests
# ---------------------------------------------------------------------------


def test_backend_name(backend: VecBackend, backend_config: BackendConfig) -> None:
    """Backend has correct name attribute."""
    assert backend.name == backend_config.expected_name


def test_backend_device(backend: VecBackend) -> None:
    """Backend has device attribute."""
    assert backend.device == "cpu"


def test_backend_num_envs(backend: VecBackend, backend_config: BackendConfig) -> None:
    """Backend has correct num_envs attribute."""
    assert backend.num_envs == backend_config.expected_num_envs


def test_action_spec(backend: VecBackend) -> None:
    """Action spec has correct shape and dtype."""
    spec = backend.action_spec
    assert spec.shape == (backend.num_envs,)
    assert spec.dtype == "int32"


def test_obs_spec(backend: VecBackend) -> None:
    """Obs spec has correct shape and dtype."""
    spec = backend.obs_spec
    assert spec.shape == (backend.num_envs, 32)
    assert spec.dtype == "float32"


# ---------------------------------------------------------------------------
# Reset Tests
# ---------------------------------------------------------------------------


def test_reset_returns_obs_and_info(backend: VecBackend) -> None:
    """reset() returns tuple of (obs, info)."""
    obs, info = backend.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


def test_reset_obs_shape(backend: VecBackend) -> None:
    """reset() obs has correct shape."""
    obs, _ = backend.reset()
    assert obs.shape == (backend.num_envs, 32)


def test_reset_obs_dtype(backend: VecBackend) -> None:
    """reset() obs has dtype float32."""
    obs, _ = backend.reset()
    assert obs.dtype == np.float32


def test_reset_obs_normalized(backend: VecBackend) -> None:
    """reset() obs values are in [0, 1] range."""
    obs, _ = backend.reset()
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)


def test_reset_records_seed(backend: VecBackend) -> None:
    """reset() records seed in info dict."""
    _, info = backend.reset(seed=42)
    assert info["seed"] == 42


# ---------------------------------------------------------------------------
# Step Tests
# ---------------------------------------------------------------------------


def test_step_noop_returns_tuple(backend: VecBackend) -> None:
    """step() with noop returns 5-tuple."""
    backend.reset()
    action = _valid_action(backend, 0)
    actions = np.array([action] * backend.num_envs, dtype=np.int32)
    result = backend.step(actions)
    assert len(result) == 5


def test_step_obs_shape(backend: VecBackend) -> None:
    """step() obs has correct shape."""
    backend.reset()
    action = _valid_action(backend, 0)
    actions = np.array([action] * backend.num_envs, dtype=np.int32)
    obs, _, _, _, _ = backend.step(actions)
    assert obs.shape == (backend.num_envs, 32)


def test_step_obs_dtype(backend: VecBackend) -> None:
    """step() obs has dtype float32."""
    backend.reset()
    action = _valid_action(backend, 0)
    actions = np.array([action] * backend.num_envs, dtype=np.int32)
    obs, _, _, _, _ = backend.step(actions)
    assert obs.dtype == np.float32


def test_step_reward_shape_dtype(backend: VecBackend) -> None:
    """step() reward has shape (num_envs,) and dtype float32."""
    backend.reset()
    action = _valid_action(backend, 0)
    actions = np.array([action] * backend.num_envs, dtype=np.int32)
    _, reward, _, _, _ = backend.step(actions)
    assert reward.shape == (backend.num_envs,)
    assert reward.dtype == np.float32


def test_step_done_shape_dtype(backend: VecBackend) -> None:
    """step() done has shape (num_envs,) and dtype bool."""
    backend.reset()
    action = _valid_action(backend, 0)
    actions = np.array([action] * backend.num_envs, dtype=np.int32)
    _, _, done, _, _ = backend.step(actions)
    assert done.shape == (backend.num_envs,)
    assert done.dtype == np.bool_


def test_step_trunc_shape_dtype(backend: VecBackend) -> None:
    """step() trunc has shape (num_envs,) and dtype bool."""
    backend.reset()
    action = _valid_action(backend, 0)
    actions = np.array([action] * backend.num_envs, dtype=np.int32)
    _, _, _, trunc, _ = backend.step(actions)
    assert trunc.shape == (backend.num_envs,)
    assert trunc.dtype == np.bool_


def test_step_with_button_action(backend: VecBackend) -> None:
    """step() works with non-noop actions."""
    backend.reset()
    action = _valid_action(backend, 1)
    actions = np.array([action] * backend.num_envs, dtype=np.int32)
    obs, reward, done, trunc, info = backend.step(actions)
    assert obs.shape == (backend.num_envs, 32)
    assert reward.shape == (backend.num_envs,)


def test_step_multiple_actions(backend: VecBackend) -> None:
    """step() can be called multiple times with different actions."""
    backend.reset()
    for idx in [0, 1, 2]:
        action = _valid_action(backend, idx)
        actions = np.array([action] * backend.num_envs, dtype=np.int32)
        obs, _, _, _, _ = backend.step(actions)
        assert obs.shape == (backend.num_envs, 32)


# ---------------------------------------------------------------------------
# Get CPU State Tests
# ---------------------------------------------------------------------------


def test_get_cpu_state_returns_dict(backend: VecBackend) -> None:
    """get_cpu_state() returns a dict."""
    backend.reset()
    state = backend.get_cpu_state(0)
    assert isinstance(state, dict)


def test_get_cpu_state_has_registers(backend: VecBackend) -> None:
    """get_cpu_state() returns all canonical register keys."""
    backend.reset()
    state = backend.get_cpu_state(0)

    register_keys = ["pc", "sp", "a", "f", "b", "c", "d", "e", "h", "l"]
    for key in register_keys:
        assert key in state, f"Missing register: {key}"
        assert isinstance(state[key], int), f"Register {key} should be int"


def test_get_cpu_state_has_flags(backend: VecBackend) -> None:
    """get_cpu_state() includes flags dict."""
    backend.reset()
    state = backend.get_cpu_state(0)

    assert "flags" in state
    flags = state["flags"]
    assert isinstance(flags, dict)
    for key in ["z", "n", "h", "c"]:
        assert key in flags, f"Missing flag: {key}"
        assert flags[key] in (0, 1), f"Flag {key} should be 0 or 1"


def test_get_cpu_state_flags_derived_from_f(backend: VecBackend) -> None:
    """get_cpu_state() flags are correctly derived from F register."""
    backend.reset()
    state = backend.get_cpu_state(0)

    f = state["f"]
    flags = state["flags"]

    assert flags["z"] == ((f >> 7) & 1)
    assert flags["n"] == ((f >> 6) & 1)
    assert flags["h"] == ((f >> 5) & 1)
    assert flags["c"] == ((f >> 4) & 1)


def test_get_cpu_state_register_ranges(backend: VecBackend) -> None:
    """get_cpu_state() registers are in valid ranges."""
    backend.reset()
    state = backend.get_cpu_state(0)

    # 16-bit registers
    assert 0 <= state["pc"] <= 0xFFFF
    assert 0 <= state["sp"] <= 0xFFFF

    # 8-bit registers
    for reg in ["a", "f", "b", "c", "d", "e", "h", "l"]:
        assert 0 <= state[reg] <= 0xFF, f"Register {reg} out of range"


def test_get_cpu_state_has_counters(backend: VecBackend) -> None:
    """get_cpu_state() includes counter fields (may be None)."""
    backend.reset()
    state = backend.get_cpu_state(0)

    assert "instr_count" in state
    assert "cycle_count" in state


# ---------------------------------------------------------------------------
# Invalid Action Tests
# ---------------------------------------------------------------------------


def test_invalid_action_negative(backend: VecBackend) -> None:
    """Negative action index raises ValueError."""
    backend.reset()
    actions = np.array([-1] * backend.num_envs, dtype=np.int32)
    with pytest.raises((ValueError, RuntimeError), match="out of range"):
        backend.step(actions)


def test_invalid_action_too_large(backend: VecBackend) -> None:
    """Action index >= NUM_ACTIONS raises ValueError."""
    backend.reset()
    actions = np.array([backend.num_actions] * backend.num_envs, dtype=np.int32)
    with pytest.raises((ValueError, RuntimeError), match="out of range"):
        backend.step(actions)


def test_valid_actions_all_work(backend: VecBackend) -> None:
    """All valid action indices [0, NUM_ACTIONS) work."""
    backend.reset()
    # Batch actions across envs to reduce step count
    num_actions = backend.num_actions
    num_envs = backend.num_envs
    action_idx = 0
    while action_idx < num_actions:
        # Build batch with different actions per env (pad with action 0 if needed)
        batch = []
        for env in range(num_envs):
            if action_idx + env < num_actions:
                batch.append(action_idx + env)
            else:
                batch.append(0)  # Pad with valid action
        actions = np.array(batch, dtype=np.int32)
        obs, _, _, _, _ = backend.step(actions)
        assert obs.shape == (num_envs, 32)
        action_idx += num_envs


# ---------------------------------------------------------------------------
# Close Tests
# ---------------------------------------------------------------------------


def test_close_can_be_called_multiple_times(backend: VecBackend) -> None:
    """close() can be called multiple times without error."""
    backend.reset()
    backend.close()
    backend.close()  # Should not raise


def test_close_without_reset(backend: VecBackend) -> None:
    """close() without reset doesn't error."""
    backend.close()  # Should not raise
