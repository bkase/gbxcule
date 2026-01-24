"""Workstream 3 verification tests: warp_vec_cpu vs pyboy_single."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.harness import diff_states, normalize_cpu_state
from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend

from .conftest import require_rom

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"


def _verify_no_mismatch(
    rom_path: Path,
    steps: int = 8,
    mem_region: tuple[int, int] | None = None,
) -> None:
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


def test_verify_alu_loop() -> None:
    require_rom(ROM_DIR / "ALU_LOOP.gb")
    _verify_no_mismatch(ROM_DIR / "ALU_LOOP.gb")


def test_verify_mem_rwb() -> None:
    require_rom(ROM_DIR / "MEM_RWB.gb")
    # MEM_RWB increments HL across the full address space; the first divergence
    # from incorrect ROM write handling shows up after several frames.
    _verify_no_mismatch(ROM_DIR / "MEM_RWB.gb", steps=64)


@pytest.mark.skipif(
    not (ROM_DIR / "MBC1_SWITCH.gb").exists(),
    reason="Test ROM not found; run `make roms` first.",
)
def test_verify_mbc1_switch() -> None:
    _verify_no_mismatch(
        ROM_DIR / "MBC1_SWITCH.gb", steps=64, mem_region=(0xC000, 0xC010)
    )


@pytest.mark.skipif(
    not (ROM_DIR / "MBC1_RAM.gb").exists(),
    reason="Test ROM not found; run `make roms` first.",
)
def test_verify_mbc1_ram() -> None:
    _verify_no_mismatch(ROM_DIR / "MBC1_RAM.gb", steps=64, mem_region=(0xC000, 0xC010))


@pytest.mark.skipif(
    not (ROM_DIR / "MBC3_SWITCH.gb").exists(),
    reason="Test ROM not found; run `make roms` first.",
)
def test_verify_mbc3_switch() -> None:
    _verify_no_mismatch(
        ROM_DIR / "MBC3_SWITCH.gb", steps=64, mem_region=(0xC000, 0xC010)
    )


@pytest.mark.skipif(
    not (ROM_DIR / "MBC3_RAM.gb").exists(),
    reason="Test ROM not found; run `make roms` first.",
)
def test_verify_mbc3_ram() -> None:
    _verify_no_mismatch(ROM_DIR / "MBC3_RAM.gb", steps=64, mem_region=(0xC000, 0xC010))


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


def test_verify_cb_bitops() -> None:
    require_rom(ROM_DIR / "CB_BITOPS.gb")
    _verify_no_mismatch(ROM_DIR / "CB_BITOPS.gb", steps=96)
