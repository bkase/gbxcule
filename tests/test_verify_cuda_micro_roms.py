"""CUDA verification tests against PyBoy for micro-ROMs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.harness import diff_states, normalize_cpu_state
from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCudaBackend

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"


def _cuda_available() -> bool:
    wp = pytest.importorskip("warp")
    wp.init()
    return wp.is_cuda_available()


def _verify_no_mismatch_cuda(rom_path: Path, steps: int = 32) -> None:
    ref = PyBoySingleBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
    )
    dut = WarpVecCudaBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
    )
    try:
        ref.reset()
        dut.reset()
        actions = np.zeros((1,), dtype=np.int32)
        for step_idx in range(steps):
            ref.step(actions)
            dut.step(actions)
            ref_state = normalize_cpu_state(ref.get_cpu_state(0))
            dut_state = normalize_cpu_state(dut.get_cpu_state(0))
            diff = diff_states(ref_state, dut_state)
            assert diff is None, f"Mismatch at step {step_idx}: {diff}"
    finally:
        ref.close()
        dut.close()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (ROM_DIR / "ALU_LOOP.gb").exists(),
    reason="Test ROM not found; run `make roms` first.",
)
def test_cuda_verify_alu_loop() -> None:
    _verify_no_mismatch_cuda(ROM_DIR / "ALU_LOOP.gb")
