"""Smoke tests for pyboy_puffer_vec backend (optional)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROM_PATH = Path(__file__).parent.parent / "bench" / "roms" / "out" / "ALU_LOOP.gb"


@pytest.mark.skipif(
    not ROM_PATH.exists(),
    reason="Test ROM not found; run `make roms` first.",
)
def test_puffer_vec_serial_smoke() -> None:
    pytest.importorskip("pufferlib")
    from gbxcule.backends.pyboy_puffer_vec import PyBoyPufferVecBackend

    backend = PyBoyPufferVecBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=1,
        vec_backend="puffer_serial",
    )
    try:
        obs, info = backend.reset(seed=123)
        assert obs.shape == backend.obs_spec.shape
        actions = np.zeros((backend.num_envs,), dtype=np.int32)
        obs, reward, done, trunc, info = backend.step(actions)
        assert obs.shape == backend.obs_spec.shape
        assert reward.shape == (backend.num_envs,)
        assert done.shape == (backend.num_envs,)
        assert trunc.shape == (backend.num_envs,)
    finally:
        backend.close()
