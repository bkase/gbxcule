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
    build_alu16_sp,
    build_alu_flags,
    build_alu_loop,
    build_bg_scroll_anim,
    build_bg_scroll_signed,
    build_bg_static,
    build_cb_bitops,
    build_ei_delay,
    build_flow_stack,
    build_joy_diverge_persist,
    build_loads_basic,
    build_mbc1_ram,
    build_mbc1_switch,
    build_mbc3_ram,
    build_mbc3_switch,
    build_mem_rwb,
    build_serial_hello,
    build_timer_div_basic,
    build_timer_irq_halt,
    compute_global_checksum,
    compute_header_checksum,
    sha256_bytes,
)

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


def test_roms_are_deterministic() -> None:
    """Same generator produces identical bytes."""
    alu1 = build_alu_loop()
    alu2 = build_alu_loop()
    assert alu1 == alu2, "ALU_LOOP is not deterministic"

    mem1 = build_mem_rwb()
    mem2 = build_mem_rwb()
    assert mem1 == mem2, "MEM_RWB is not deterministic"

    serial1 = build_serial_hello()
    serial2 = build_serial_hello()
    assert serial1 == serial2, "SERIAL_HELLO is not deterministic"

    timer_div1 = build_timer_div_basic()
    timer_div2 = build_timer_div_basic()
    assert timer_div1 == timer_div2, "TIMER_DIV_BASIC is not deterministic"

    timer_irq1 = build_timer_irq_halt()
    timer_irq2 = build_timer_irq_halt()
    assert timer_irq1 == timer_irq2, "TIMER_IRQ_HALT is not deterministic"

    ei1 = build_ei_delay()
    ei2 = build_ei_delay()
    assert ei1 == ei2, "EI_DELAY is not deterministic"

    joy1 = build_joy_diverge_persist()
    joy2 = build_joy_diverge_persist()
    assert joy1 == joy2, "JOY_DIVERGE_PERSIST is not deterministic"

    loads1 = build_loads_basic()
    loads2 = build_loads_basic()
    assert loads1 == loads2, "LOADS_BASIC is not deterministic"

    alu_flags1 = build_alu_flags()
    alu_flags2 = build_alu_flags()
    assert alu_flags1 == alu_flags2, "ALU_FLAGS is not deterministic"

    alu16_1 = build_alu16_sp()
    alu16_2 = build_alu16_sp()
    assert alu16_1 == alu16_2, "ALU16_SP is not deterministic"

    flow1 = build_flow_stack()
    flow2 = build_flow_stack()
    assert flow1 == flow2, "FLOW_STACK is not deterministic"

    cb1 = build_cb_bitops()
    cb2 = build_cb_bitops()
    assert cb1 == cb2, "CB_BITOPS is not deterministic"

    mbc1_sw1 = build_mbc1_switch()
    mbc1_sw2 = build_mbc1_switch()
    assert mbc1_sw1 == mbc1_sw2, "MBC1_SWITCH is not deterministic"

    mbc1_ram1 = build_mbc1_ram()
    mbc1_ram2 = build_mbc1_ram()
    assert mbc1_ram1 == mbc1_ram2, "MBC1_RAM is not deterministic"

    mbc3_sw1 = build_mbc3_switch()
    mbc3_sw2 = build_mbc3_switch()
    assert mbc3_sw1 == mbc3_sw2, "MBC3_SWITCH is not deterministic"

    mbc3_ram1 = build_mbc3_ram()
    mbc3_ram2 = build_mbc3_ram()
    assert mbc3_ram1 == mbc3_ram2, "MBC3_RAM is not deterministic"

    bg_static1 = build_bg_static()
    bg_static2 = build_bg_static()
    assert bg_static1 == bg_static2, "BG_STATIC is not deterministic"

    bg_scroll1 = build_bg_scroll_signed()
    bg_scroll2 = build_bg_scroll_signed()
    assert bg_scroll1 == bg_scroll2, "BG_SCROLL_SIGNED is not deterministic"

    bg_anim1 = build_bg_scroll_anim()
    bg_anim2 = build_bg_scroll_anim()
    assert bg_anim1 == bg_anim2, "BG_SCROLL_ANIM is not deterministic"


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


def test_joy_diverge_persist_header_checksum() -> None:
    """JOY_DIVERGE_PERSIST has valid header checksum."""
    rom = build_joy_diverge_persist()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_serial_hello_header_checksum() -> None:
    """SERIAL_HELLO has valid header checksum."""
    rom = build_serial_hello()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_timer_div_basic_header_checksum() -> None:
    """TIMER_DIV_BASIC has valid header checksum."""
    rom = build_timer_div_basic()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_timer_irq_halt_header_checksum() -> None:
    """TIMER_IRQ_HALT has valid header checksum."""
    rom = build_timer_irq_halt()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_ei_delay_header_checksum() -> None:
    """EI_DELAY has valid header checksum."""
    rom = build_ei_delay()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_loads_basic_header_checksum() -> None:
    """LOADS_BASIC has valid header checksum."""
    rom = build_loads_basic()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_alu_flags_header_checksum() -> None:
    """ALU_FLAGS has valid header checksum."""
    rom = build_alu_flags()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_alu16_sp_header_checksum() -> None:
    """ALU16_SP has valid header checksum."""
    rom = build_alu16_sp()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_flow_stack_header_checksum() -> None:
    """FLOW_STACK has valid header checksum."""
    rom = build_flow_stack()
    stored = rom[0x014D]
    computed = compute_header_checksum(rom)
    assert stored == computed, (
        f"Header checksum mismatch: {stored:02X} != {computed:02X}"
    )


def test_cb_bitops_header_checksum() -> None:
    """CB_BITOPS has valid header checksum."""
    rom = build_cb_bitops()
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


def test_joy_diverge_persist_global_checksum() -> None:
    """JOY_DIVERGE_PERSIST has valid global checksum."""
    rom = build_joy_diverge_persist()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_serial_hello_global_checksum() -> None:
    """SERIAL_HELLO has valid global checksum."""
    rom = build_serial_hello()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_timer_div_basic_global_checksum() -> None:
    """TIMER_DIV_BASIC has valid global checksum."""
    rom = build_timer_div_basic()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_timer_irq_halt_global_checksum() -> None:
    """TIMER_IRQ_HALT has valid global checksum."""
    rom = build_timer_irq_halt()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_ei_delay_global_checksum() -> None:
    """EI_DELAY has valid global checksum."""
    rom = build_ei_delay()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_flow_stack_global_checksum() -> None:
    """FLOW_STACK has valid global checksum."""
    rom = build_flow_stack()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_loads_basic_global_checksum() -> None:
    """LOADS_BASIC has valid global checksum."""
    rom = build_loads_basic()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_alu_flags_global_checksum() -> None:
    """ALU_FLAGS has valid global checksum."""
    rom = build_alu_flags()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_alu16_sp_global_checksum() -> None:
    """ALU16_SP has valid global checksum."""
    rom = build_alu16_sp()
    stored = (rom[0x014E] << 8) | rom[0x014F]
    computed = compute_global_checksum(rom)
    assert stored == computed, (
        f"Global checksum mismatch: {stored:04X} != {computed:04X}"
    )


def test_cb_bitops_global_checksum() -> None:
    """CB_BITOPS has valid global checksum."""
    rom = build_cb_bitops()
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


def test_pyboy_headless_serial_hello(rom_dir: Path) -> None:
    """PyBoy can run SERIAL_HELLO headless without crashing."""
    from pyboy import PyBoy

    rom_path = rom_dir / "SERIAL_HELLO.gb"
    assert rom_path.exists()

    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    for _ in range(120):
        pyboy.tick(render=False)

    pyboy.stop(save=False)


def test_pyboy_headless_joy_diverge_persist(rom_dir: Path) -> None:
    """PyBoy can run JOY_DIVERGE_PERSIST headless without crashing."""
    from pyboy import PyBoy

    rom_path = rom_dir / "JOY_DIVERGE_PERSIST.gb"
    assert rom_path.exists()

    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    for _ in range(120):
        pyboy.tick(render=False)

    pyboy.stop(save=False)


def test_pyboy_headless_loads_basic(rom_dir: Path) -> None:
    """PyBoy can run LOADS_BASIC headless without crashing."""
    from pyboy import PyBoy

    rom_path = rom_dir / "LOADS_BASIC.gb"
    assert rom_path.exists()

    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    for _ in range(120):
        pyboy.tick(render=False)

    pyboy.stop(save=False)


def test_pyboy_headless_alu_flags(rom_dir: Path) -> None:
    """PyBoy can run ALU_FLAGS headless without crashing."""
    from pyboy import PyBoy

    rom_path = rom_dir / "ALU_FLAGS.gb"
    assert rom_path.exists()

    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

    for _ in range(120):
        pyboy.tick(render=False)

    pyboy.stop(save=False)


def test_pyboy_headless_alu16_sp(rom_dir: Path) -> None:
    """PyBoy can run ALU16_SP headless without crashing."""
    from pyboy import PyBoy

    rom_path = rom_dir / "ALU16_SP.gb"
    assert rom_path.exists()

    pyboy = PyBoy(str(rom_path), window="null", sound_emulated=False)
    pyboy.set_emulation_speed(0)

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
