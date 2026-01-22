"""Workstream 3 verification tests: warp_vec_cpu vs pyboy_single."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.harness import diff_states, normalize_cpu_state
from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"


def _verify_no_mismatch(rom_path: Path, steps: int = 8) -> None:
    ref = PyBoySingleBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
    )
    dut = WarpVecCpuBackend(
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


@pytest.mark.skipif(
    not (ROM_DIR / "ALU_LOOP.gb").exists(),
    reason="Test ROM not found; run `make roms` first.",
)
def test_verify_alu_loop() -> None:
    _verify_no_mismatch(ROM_DIR / "ALU_LOOP.gb")


@pytest.mark.skipif(
    not (ROM_DIR / "MEM_RWB.gb").exists(),
    reason="Test ROM not found; run `make roms` first.",
)
def test_verify_mem_rwb() -> None:
    _verify_no_mismatch(ROM_DIR / "MEM_RWB.gb")
