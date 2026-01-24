"""Workstream 2 tests: ROM loading + ABI v0 memory model for warp_vec_cpu."""

from __future__ import annotations

import tempfile
from pathlib import Path

from bench.roms.build_micro_rom import build_rom
from gbxcule.backends.warp_vec import BOOTROM_PATH, WarpVecCpuBackend


def test_reset_loads_rom_prefix() -> None:
    """reset() loads ROM bytes and overlays boot ROM for env 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rom_path = Path(tmpdir) / "test.gb"
        rom_bytes = build_rom("TEST", b"\x00")
        rom_path.write_bytes(rom_bytes)

        backend = WarpVecCpuBackend(str(rom_path), num_envs=1, obs_dim=32)
        try:
            backend.reset()
            bootrom = BOOTROM_PATH.read_bytes()
            assert backend.read_memory(0, 0, 0x100) == bootrom
            assert (
                backend.read_memory(0, 0x100, 0x100 + len(rom_bytes[0x100:]))
                == rom_bytes[0x100:]
            )
        finally:
            backend.close()


def test_multi_env_isolation() -> None:
    """Each env has its own 64KB memory slice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rom_path = Path(tmpdir) / "test.gb"
        rom_bytes = build_rom("TEST", b"\x00")
        rom_path.write_bytes(rom_bytes)

        backend = WarpVecCpuBackend(str(rom_path), num_envs=2, obs_dim=32)
        try:
            backend.reset()
            backend.write_memory(0, 0xC000, b"\xab")
            assert backend.read_memory(0, 0xC000, 0xC001) == b"\xab"
            assert backend.read_memory(1, 0xC000, 0xC001) == b"\x00"
        finally:
            backend.close()
