"""CUDA read_memory tests for WarpVecCudaBackend."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from bench.roms.build_micro_rom import build_rom
from gbxcule.backends.warp_vec import BOOTROM_PATH, WarpVecCudaBackend

from .conftest import require_rom


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    wp = pytest.importorskip("warp")
    wp.init()
    return wp.is_cuda_available()


def test_cuda_read_memory_bootrom_and_rom() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    require_rom(BOOTROM_PATH)

    with tempfile.TemporaryDirectory() as tmpdir:
        rom_path = Path(tmpdir) / "test.gb"
        rom_bytes = build_rom("TEST", b"\x00")
        rom_path.write_bytes(rom_bytes)

        backend = WarpVecCudaBackend(
            str(rom_path),
            num_envs=2,
            frames_per_step=1,
            release_after_frames=0,
            obs_dim=32,
        )
        try:
            backend.reset()
            bootrom = BOOTROM_PATH.read_bytes()
            assert backend.read_memory(0, 0, 0x100) == bootrom
            assert backend.read_memory(1, 0, 0x100) == bootrom
            assert (
                backend.read_memory(0, 0x100, 0x100 + len(rom_bytes[0x100:]))
                == (rom_bytes[0x100:])
            )
            assert backend.read_memory(0, 0xC000, 0xC100) == b"\x00" * 0x100
        finally:
            backend.close()


def test_cuda_read_memory_invalid_ranges() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    with tempfile.TemporaryDirectory() as tmpdir:
        rom_path = Path(tmpdir) / "test.gb"
        rom_bytes = build_rom("TEST", b"\x00")
        rom_path.write_bytes(rom_bytes)

        backend = WarpVecCudaBackend(
            str(rom_path),
            num_envs=1,
            frames_per_step=1,
            release_after_frames=0,
            obs_dim=32,
        )
        try:
            backend.reset()
            with pytest.raises(ValueError, match="Invalid memory range"):
                backend.read_memory(0, -1, 1)
            with pytest.raises(ValueError, match="Invalid memory range"):
                backend.read_memory(0, 2, 1)
            with pytest.raises(ValueError, match="Invalid memory range"):
                backend.read_memory(0, 0, 0x20000)
            with pytest.raises(ValueError, match="env_idx"):
                backend.read_memory(2, 0, 1)
        finally:
            backend.close()
