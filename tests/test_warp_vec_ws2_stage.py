"""Stage kernel tests for warp_vec_cpu."""
# pyright: reportOptionalMemberAccess=false
# Tests access internal state that is Optional before reset()

from __future__ import annotations

import numpy as np

from gbxcule.backends.common import Stage
from gbxcule.backends.warp_vec import WarpVecCpuBackend

from .conftest import ROM_PATH, require_rom


def _make_backend(stage: Stage, frames_per_step: int = 0) -> WarpVecCpuBackend:
    require_rom(ROM_PATH)
    return WarpVecCpuBackend(
        str(ROM_PATH),
        num_envs=1,
        obs_dim=32,
        frames_per_step=frames_per_step,
        stage=stage,
    )


def _set_regs(backend: WarpVecCpuBackend, *, pc: int, sp: int, a: int) -> None:
    # These attributes are guaranteed non-None after reset()
    assert backend._pc is not None
    assert backend._sp is not None
    assert backend._a is not None
    assert backend._b is not None
    assert backend._c is not None
    assert backend._d is not None
    assert backend._e is not None
    assert backend._h is not None
    assert backend._l is not None
    assert backend._f is not None
    backend._pc.numpy()[0] = pc
    backend._sp.numpy()[0] = sp
    backend._a.numpy()[0] = a
    backend._b.numpy()[0] = 0x11
    backend._c.numpy()[0] = 0x22
    backend._d.numpy()[0] = 0x33
    backend._e.numpy()[0] = 0x44
    backend._h.numpy()[0] = 0x55
    backend._l.numpy()[0] = 0x66
    backend._f.numpy()[0] = 0xF0


def _write_wram(backend: WarpVecCpuBackend, data: bytes) -> None:
    backend.write_memory(0, 0xC000, data)


def _expected_reward(a: int, m0: int, m1: int) -> float:
    mix = (m0 + (m1 * 3)) & 0xFF
    val = (a ^ mix) & 0xFF
    return val / 255.0


def test_stage_emulate_only_does_not_write_outputs() -> None:
    backend = _make_backend("emulate_only", frames_per_step=0)
    try:
        backend.reset()
        assert backend._reward is not None
        assert backend._obs is not None
        backend._reward.numpy()[:] = 0.75
        backend._obs.numpy()[:] = 0.25
        actions = np.zeros((1,), dtype=np.int32)
        backend.step(actions)
        assert np.allclose(backend._reward.numpy(), 0.75)
        assert np.allclose(backend._obs.numpy(), 0.25)
    finally:
        backend.close()


def test_stage_reward_only_writes_reward() -> None:
    backend = _make_backend("reward_only", frames_per_step=0)
    try:
        backend.reset()
        backend._reward.numpy()[:] = -1.0
        backend._obs.numpy()[:] = 2.0
        _set_regs(backend, pc=0x1234, sp=0xBEEF, a=0x2A)
        _write_wram(backend, bytes([0x10, 0x05]))
        actions = np.zeros((1,), dtype=np.int32)
        backend.step(actions)
        expected = _expected_reward(0x2A, 0x10, 0x05)
        assert np.isclose(backend._reward.numpy()[0], expected)
        assert np.allclose(backend._obs.numpy(), 2.0)
    finally:
        backend.close()


def test_stage_obs_only_writes_obs() -> None:
    backend = _make_backend("obs_only", frames_per_step=0)
    try:
        backend.reset()
        backend._reward.numpy()[:] = -1.0
        backend._obs.numpy()[:] = 3.0
        _set_regs(backend, pc=0x0100, sp=0xFF00, a=0x2A)
        _write_wram(backend, bytes(range(16)))
        actions = np.zeros((1,), dtype=np.int32)
        backend.step(actions)
        obs = backend._obs.numpy()
        assert np.isclose(obs[0], 0x0100 / 65535.0)
        assert np.isclose(obs[2], 0x2A / 255.0)
        assert np.isclose(obs[10], 0x00 / 255.0)
        assert np.isclose(obs[11], 0x01 / 255.0)
        assert np.isclose(obs[26], (0x00 ^ 0x01) / 255.0)
        assert np.allclose(backend._reward.numpy(), -1.0)
    finally:
        backend.close()


def test_stage_full_step_matches_obs_and_reward() -> None:
    obs_backend = _make_backend("obs_only", frames_per_step=0)
    reward_backend = _make_backend("reward_only", frames_per_step=0)
    full_backend = _make_backend("full_step", frames_per_step=0)
    try:
        for backend in (obs_backend, reward_backend, full_backend):
            backend.reset()
            backend._reward.numpy()[:] = -1.0
            backend._obs.numpy()[:] = -1.0
            _set_regs(backend, pc=0x2000, sp=0x1234, a=0x2A)
            _write_wram(backend, bytes([0x10, 0x05]) + bytes(range(14)))

        actions = np.zeros((1,), dtype=np.int32)
        obs_backend.step(actions)
        reward_backend.step(actions)
        full_backend.step(actions)

        assert np.allclose(full_backend._obs.numpy(), obs_backend._obs.numpy())
        assert np.allclose(full_backend._reward.numpy(), reward_backend._reward.numpy())
    finally:
        obs_backend.close()
        reward_backend.close()
        full_backend.close()


def test_stage_does_not_affect_cpu_state() -> None:
    emulate_backend = _make_backend("emulate_only", frames_per_step=1)
    full_backend = _make_backend("full_step", frames_per_step=1)
    try:
        emulate_backend.reset(seed=123)
        full_backend.reset(seed=123)
        actions = np.zeros((1,), dtype=np.int32)
        emulate_backend.step(actions)
        full_backend.step(actions)
        assert emulate_backend.get_cpu_state(0) == full_backend.get_cpu_state(0)
        assert emulate_backend.read_memory(
            0, 0xC000, 0xC010
        ) == full_backend.read_memory(0, 0xC000, 0xC010)
    finally:
        emulate_backend.close()
        full_backend.close()
