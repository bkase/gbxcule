"""Workstream 3 verification tests: warp_vec_cpu vs pyboy_single."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bench.harness import diff_states, normalize_cpu_state
from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend

from .conftest import require_rom

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


def test_verify_alu_loop() -> None:
    require_rom(ROM_DIR / "ALU_LOOP.gb")
    _verify_no_mismatch(ROM_DIR / "ALU_LOOP.gb")


def test_verify_mem_rwb() -> None:
    require_rom(ROM_DIR / "MEM_RWB.gb")
    # MEM_RWB increments HL across the full address space; the first divergence
    # from incorrect ROM write handling shows up after several frames.
    _verify_no_mismatch(ROM_DIR / "MEM_RWB.gb", steps=64)


def test_verify_loads_basic() -> None:
    require_rom(ROM_DIR / "LOADS_BASIC.gb")
    _verify_no_mismatch(ROM_DIR / "LOADS_BASIC.gb", steps=64)


def test_verify_alu_flags() -> None:
    require_rom(ROM_DIR / "ALU_FLAGS.gb")
    _verify_no_mismatch(ROM_DIR / "ALU_FLAGS.gb", steps=64)


def test_verify_alu16_sp() -> None:
    require_rom(ROM_DIR / "ALU16_SP.gb")
    _verify_no_mismatch(ROM_DIR / "ALU16_SP.gb", steps=64)


def test_verify_flow_stack() -> None:
    require_rom(ROM_DIR / "FLOW_STACK.gb")
    _verify_no_mismatch(ROM_DIR / "FLOW_STACK.gb", steps=96)
