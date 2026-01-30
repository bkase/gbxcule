"""Tests for micro-ROM generation.

These tests verify:
- ROM files are generated correctly
- Header/global checksums are valid
- PyBoy can run the ROMs headless without crashing

Tests are fast and robust on macOS/cloud CPU-only environments.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from bench.roms.build_micro_rom import (
    build_all,
    build_alu16_sp,
    build_alu_flags,
    build_alu_loop,
    build_bg_scroll_anim,
    build_bg_scroll_signed,
    build_bg_static,
    build_cb_bitops,
    build_dma_oam_copy,
    build_ei_delay,
    build_flow_stack,
    build_joy_diverge_persist,
    build_loads_basic,
    build_mbc1_ram,
    build_mbc1_switch,
    build_mbc3_ram,
    build_mbc3_switch,
    build_mem_rwb,
    build_ppu_sprites,
    build_ppu_stat_irq,
    build_ppu_window,
    build_serial_hello,
    build_timer_div_basic,
    build_timer_irq_halt,
    compute_global_checksum,
    compute_header_checksum,
    sha256_bytes,
)

# ---------------------------------------------------------------------------
# ROM builder registry for parametrized tests
# ---------------------------------------------------------------------------

ROM_BUILDERS: list[tuple[Callable[[], bytes], str]] = [
    (build_alu_loop, "ALU_LOOP"),
    (build_mem_rwb, "MEM_RWB"),
    (build_serial_hello, "SERIAL_HELLO"),
    (build_dma_oam_copy, "DMA_OAM_COPY"),
    (build_timer_div_basic, "TIMER_DIV_BASIC"),
    (build_timer_irq_halt, "TIMER_IRQ_HALT"),
    (build_ei_delay, "EI_DELAY"),
    (build_joy_diverge_persist, "JOY_DIVERGE_PERSIST"),
    (build_loads_basic, "LOADS_BASIC"),
    (build_alu_flags, "ALU_FLAGS"),
    (build_alu16_sp, "ALU16_SP"),
    (build_flow_stack, "FLOW_STACK"),
    (build_cb_bitops, "CB_BITOPS"),
    (build_mbc1_switch, "MBC1_SWITCH"),
    (build_mbc1_ram, "MBC1_RAM"),
    (build_mbc3_switch, "MBC3_SWITCH"),
    (build_mbc3_ram, "MBC3_RAM"),
    (build_bg_static, "BG_STATIC"),
    (build_bg_scroll_anim, "BG_SCROLL_ANIM"),
    (build_ppu_window, "PPU_WINDOW"),
    (build_ppu_sprites, "PPU_SPRITES"),
    (build_ppu_stat_irq, "PPU_STAT_IRQ"),
    (build_bg_scroll_signed, "BG_SCROLL_SIGNED"),
]

# Representative ROMs for PyBoy smoke tests (covers different features)
PYBOY_SMOKE_ROMS = [
    "ALU_LOOP",
    "MEM_RWB",
    "SERIAL_HELLO",
    "JOY_DIVERGE_PERSIST",
    "LOADS_BASIC",
]

# ---------------------------------------------------------------------------
# ROM generation tests
# ---------------------------------------------------------------------------


def test_build_all_creates_roms(tmp_path: Path) -> None:
    """build_all creates all ROM files."""
    results = build_all(tmp_path)

    assert len(results) == 23

    for name, path, sha in results:
        assert path.exists(), f"{name} was not created"
        assert path.stat().st_size in {
            32 * 1024,
            64 * 1024,
        }, f"{name} has unexpected size"
        assert len(sha) == 64, f"{name} sha256 is wrong length"


@pytest.mark.parametrize("builder,name", ROM_BUILDERS)
def test_rom_is_deterministic(builder: Callable[[], bytes], name: str) -> None:
    """ROM builders produce identical bytes on repeated calls."""
    rom1 = builder()
    rom2 = builder()
    assert rom1 == rom2, f"{name} is not deterministic"


def test_sha256_is_deterministic() -> None:
    """sha256_bytes produces consistent hashes."""
    data = b"test data"
    h1 = sha256_bytes(data)
    h2 = sha256_bytes(data)
    assert h1 == h2
    assert len(h1) == 64


# ---------------------------------------------------------------------------
# Checksum validation tests (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder,name", ROM_BUILDERS)
def test_rom_header_checksum(builder: Callable[[], bytes], name: str) -> None:
    """ROM has valid header checksum at 0x014D."""
    rom = builder()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"{name} header checksum mismatch: {stored:02X} != {computed:02X}"
    )


@pytest.mark.parametrize("builder,name", ROM_BUILDERS)
def test_rom_global_checksum(builder: Callable[[], bytes], name: str) -> None:
    """ROM has valid global checksum at 0x014E-014F."""
    rom = builder()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"{name} global checksum mismatch: {stored:04X} != {computed:04X}"
    )


# ---------------------------------------------------------------------------
# PyBoy headless smoke tests (parametrized)
# ---------------------------------------------------------------------------


@pytest.fixture
def rom_dir(tmp_path: Path) -> Path:
    """Build ROMs into a temp directory."""
    build_all(tmp_path)
    return tmp_path


@pytest.mark.parametrize("rom_name", PYBOY_SMOKE_ROMS)
def test_pyboy_headless_smoke(rom_dir: Path, rom_name: str) -> None:
    """PyBoy can run ROM headless without crashing."""
    from pyboy import PyBoy

    rom_path = rom_dir / f"{rom_name}.gb"
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
