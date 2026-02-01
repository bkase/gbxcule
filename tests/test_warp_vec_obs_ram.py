"""Warp obs RAM readback tests vs PyBoy reference."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend
from gbxcule.core.obs import build_obs_v3_from_state

from .conftest import require_rom

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"
ROM_PATH = ROM_DIR / "MEM_RWB.gb"


def _expected_obs(ref: PyBoySingleBackend) -> np.ndarray:
    state = ref.get_cpu_state(0)
    mem = ref.read_memory(0, 0xC000, 0xC010)
    obs_vec = build_obs_v3_from_state(
        pc=state["pc"],
        sp=state["sp"],
        a=state["a"],
        f=state["f"],
        b=state["b"],
        c=state["c"],
        d=state["d"],
        e=state["e"],
        h=state["h"],
        l_reg=state["l"],
        wram16=mem,
    )
    obs = np.zeros((1, obs_vec.shape[0]), dtype=np.float32)
    obs[0, : obs_vec.shape[0]] = obs_vec
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
