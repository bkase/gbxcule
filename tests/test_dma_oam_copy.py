"""OAM DMA copy verification vs PyBoy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend

from .conftest import require_rom

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"


def test_dma_oam_copy_matches_pyboy() -> None:
    rom_path = ROM_DIR / "DMA_OAM_COPY.gb"
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
        for _ in range(4):
            ref.step(actions)
            dut.step(actions)

        ref_oam = ref.read_memory(0, 0xFE00, 0xFEA0)
        dut_oam = dut.read_memory(0, 0xFE00, 0xFEA0)
        assert ref_oam == dut_oam
    finally:
        ref.close()
        dut.close()

