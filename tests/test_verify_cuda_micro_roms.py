"""CUDA verification tests against PyBoy for micro-ROMs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.harness import diff_states, normalize_cpu_state
from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCudaBackend

from .conftest import require_rom

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"


def _cuda_available() -> bool:
    wp = pytest.importorskip("warp")
    wp.init()
    return wp.is_cuda_available()


def _verify_no_mismatch_cuda(
    rom_path: Path,
    *,
    steps: int = 32,
    mem_region: tuple[int, int] | None = None,
) -> None:
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
            if mem_region is not None:
                lo, hi = mem_region
                ref_bytes = ref.read_memory(0, lo, hi)
                dut_bytes = dut.read_memory(0, lo, hi)
                assert ref_bytes == dut_bytes, (
                    f"Memory mismatch at step {step_idx} ({lo:04X}:{hi:04X})"
                )
    finally:
        ref.close()
        dut.close()


def test_cuda_verify_alu_loop() -> None:
    assert _cuda_available(), "CUDA not available"
    require_rom(ROM_DIR / "ALU_LOOP.gb")
    _verify_no_mismatch_cuda(ROM_DIR / "ALU_LOOP.gb")


def test_cuda_verify_mem_rwb() -> None:
    assert _cuda_available(), "CUDA not available"
    require_rom(ROM_DIR / "MEM_RWB.gb")
    _verify_no_mismatch_cuda(
        ROM_DIR / "MEM_RWB.gb",
        steps=64,
        mem_region=(0xC000, 0xC100),
    )
