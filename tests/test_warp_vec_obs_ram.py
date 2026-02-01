"""Warp obs RAM readback tests vs PyBoy reference."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend

from .conftest import require_rom

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"
ROM_PATH = ROM_DIR / "MEM_RWB.gb"


def _expected_obs(ref: PyBoySingleBackend) -> np.ndarray:
    state = ref.get_cpu_state(0)
    mem = ref.read_memory(0, 0xC000, 0xC010)
    m = list(mem)
    obs = np.zeros((1, 32), dtype=np.float32)

    obs[0, 0] = np.float32(state["pc"] / 65535.0)
    obs[0, 1] = np.float32(state["sp"] / 65535.0)
    obs[0, 2] = np.float32(state["a"] / 255.0)
    obs[0, 3] = np.float32((state["f"] & 0xF0) / 255.0)
    obs[0, 4] = np.float32(state["b"] / 255.0)
    obs[0, 5] = np.float32(state["c"] / 255.0)
    obs[0, 6] = np.float32(state["d"] / 255.0)
    obs[0, 7] = np.float32(state["e"] / 255.0)
    obs[0, 8] = np.float32(state["h"] / 255.0)
    obs[0, 9] = np.float32(state["l"] / 255.0)

    for idx in range(16):
        obs[0, 10 + idx] = np.float32(m[idx] / 255.0)

    mix0 = (m[0] ^ m[1]) & 0xFF
    mix1 = (m[2] + (m[3] * 5)) & 0xFF
    mix2 = (m[4] ^ (state["a"] & 0xFF)) & 0xFF
    mix3 = (m[5] + (state["b"] & 0xFF)) & 0xFF
    mix4 = (m[6] ^ (state["c"] & 0xFF)) & 0xFF
    mix5 = (m[7] + (state["d"] & 0xFF)) & 0xFF

    obs[0, 26] = np.float32(mix0 / 255.0)
    obs[0, 27] = np.float32(mix1 / 255.0)
    obs[0, 28] = np.float32(mix2 / 255.0)
    obs[0, 29] = np.float32(mix3 / 255.0)
    obs[0, 30] = np.float32(mix4 / 255.0)
    obs[0, 31] = np.float32(mix5 / 255.0)

    return obs


def _run_obs_match(backend_cls, *, steps: int = 8) -> None:  # type: ignore[no-untyped-def]
    ref = PyBoySingleBackend(
        str(ROM_PATH),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
    )
    dut = backend_cls(
        str(ROM_PATH),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        stage="obs_only",
    )
    try:
        ref.reset()
        dut.reset()
        actions = np.zeros((1,), dtype=np.int32)
        for _ in range(steps):
            ref.step(actions)
            obs, _, _, _, _ = dut.step(actions)
            expected = _expected_obs(ref)
            np.testing.assert_allclose(obs, expected, atol=1e-6, rtol=0.0)
    finally:
        ref.close()
        dut.close()


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    wp = pytest.importorskip("warp")
    wp.init()
    return wp.is_cuda_available()


def test_obs_ram_matches_pyboy_cpu() -> None:
    require_rom(ROM_PATH)
    _run_obs_match(WarpVecCpuBackend, steps=8)


def test_obs_ram_matches_pyboy_cuda() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    require_rom(ROM_PATH)
    _run_obs_match(WarpVecCudaBackend, steps=8)
