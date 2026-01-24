"""STAT interrupt verification vs PyBoy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend

from .conftest import require_rom

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"


def test_ppu_stat_irq_counter_matches_pyboy() -> None:
    rom_path = ROM_DIR / "PPU_STAT_IRQ.gb"
    require_rom(rom_path)

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
        for _ in range(12):
            ref.step(actions)
            dut.step(actions)

        ref_count = ref.read_memory(0, 0xC000, 0xC002)
        dut_count = dut.read_memory(0, 0xC000, 0xC002)
        assert ref_count == dut_count
    finally:
        ref.close()
        dut.close()
