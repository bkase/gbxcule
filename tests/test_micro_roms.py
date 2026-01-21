"""Tests for micro-ROM generation.

These tests verify:
- ROM files are generated correctly
- Header/global checksums are valid
- PyBoy can run the ROMs headless without crashing

Tests are fast and robust on macOS/cloud CPU-only environments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.roms.build_micro_rom import (
    build_all,
    build_alu_loop,
    build_mem_rwb,
    compute_global_checksum,
    compute_header_checksum,
    sha256_bytes,
)

# ---------------------------------------------------------------------------
# ROM generation tests
# ---------------------------------------------------------------------------


def test_build_all_creates_roms(tmp_path: Path) -> None:
    """build_all creates both ROM files."""
    results = build_all(tmp_path)

    assert len(results) == 2

    for name, path, sha in results:
        assert path.exists(), f"{name} was not created"
        assert path.stat().st_size == 32 * 1024, f"{name} is not 32KB"
        assert len(sha) == 64, f"{name} sha256 is wrong length"


def test_roms_are_deterministic() -> None:
    """Same generator produces identical bytes."""
    alu1 = build_alu_loop()
    alu2 = build_alu_loop()
    assert alu1 == alu2, "ALU_LOOP is not deterministic"

    mem1 = build_mem_rwb()
    mem2 = build_mem_rwb()
    assert mem1 == mem2, "MEM_RWB is not deterministic"


def test_sha256_is_deterministic() -> None:
    """sha256_bytes produces consistent hashes."""
    data = b"test data"
    h1 = sha256_bytes(data)
    h2 = sha256_bytes(data)
    assert h1 == h2
    assert len(h1) == 64


# ---------------------------------------------------------------------------
# Checksum validation tests
# ---------------------------------------------------------------------------


def test_alu_loop_header_checksum() -> None:
    """ALU_LOOP has valid header checksum."""
    rom = build_alu_loop()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_mem_rwb_header_checksum() -> None:
    """MEM_RWB has valid header checksum."""
    rom = build_mem_rwb()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_alu_loop_global_checksum() -> None:
    """ALU_LOOP has valid global checksum."""
    rom = build_alu_loop()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_mem_rwb_global_checksum() -> None:
    """MEM_RWB has valid global checksum."""
    rom = build_mem_rwb()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


# ---------------------------------------------------------------------------
# PyBoy headless smoke tests
# ---------------------------------------------------------------------------


@pytest.fixture
def rom_dir(tmp_path: Path) -> Path:
    """Build ROMs into a temp directory."""
    build_all(tmp_path)
    return tmp_path


def test_pyboy_headless_alu_loop(rom_dir: Path) -> None:
    """PyBoy can run ALU_LOOP headless without crashing."""
    from pyboy import PyBoy

    rom_path = rom_dir / "ALU_LOOP.gb"
    assert rom_path.exists()

    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)  # No speed limit

    # Tick 120 frames without rendering
    for _ in range(120):
        pyboy.tick(render=False)

    pyboy.stop(save=False)


def test_pyboy_headless_mem_rwb(rom_dir: Path) -> None:
    """PyBoy can run MEM_RWB headless without crashing."""
    from pyboy import PyBoy

    rom_path = rom_dir / "MEM_RWB.gb"
    assert rom_path.exists()

    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)  # No speed limit

    # Tick 120 frames without rendering
    for _ in range(120):
        pyboy.tick(render=False)

    pyboy.stop(save=False)


def test_pyboy_cpu_state_accessible(rom_dir: Path) -> None:
    """PyBoy CPU state can be read after running."""
    from pyboy import PyBoy

    rom_path = rom_dir / "ALU_LOOP.gb"
    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    # Run for some frames
    for _ in range(60):
        pyboy.tick(render=False)

    # Access CPU registers via register_file
    reg = pyboy.register_file
    assert hasattr(reg, "PC")
    assert hasattr(reg, "SP")
    assert hasattr(reg, "A")

    # Verify we can read values
    pc = reg.PC
    assert isinstance(pc, int)
    assert 0 <= pc <= 0xFFFF

    pyboy.stop(save=False)
