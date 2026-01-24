"""Tests for cartridge header parsing."""

from __future__ import annotations

import pytest

from gbxcule.core.cartridge import (
    CartType,
    clamp_bank,
    parse_cartridge_header,
    ram_size_from_code,
    rom_size_from_code,
)


def _make_rom(
    *,
    cart_type: int,
    rom_size_code: int,
    ram_size_code: int,
    rom_bytes: int = 32 * 1024,
) -> bytes:
    rom = bytearray(rom_bytes)
    rom[0x0147] = cart_type & 0xFF
    rom[0x0148] = rom_size_code & 0xFF
    rom[0x0149] = ram_size_code & 0xFF
    return bytes(rom)


def test_rom_size_map_basic() -> None:
    assert rom_size_from_code(0x00) == 32 * 1024
    assert rom_size_from_code(0x01) == 64 * 1024


def test_ram_size_map_basic() -> None:
    assert ram_size_from_code(0x00) == 0
    assert ram_size_from_code(0x02) == 8 * 1024


def test_parse_rom_only_header() -> None:
    rom = _make_rom(cart_type=0x00, rom_size_code=0x00, ram_size_code=0x00)
    spec = parse_cartridge_header(rom)
    assert spec.cart_type == CartType.ROM_ONLY
    assert spec.rom_bank_count == 2
    assert spec.ram_bank_count == 0
    assert not spec.has_ram
    assert not spec.has_battery
    assert not spec.has_rtc


def test_parse_mbc1_ram_header() -> None:
    rom = _make_rom(cart_type=0x02, rom_size_code=0x02, ram_size_code=0x03)
    spec = parse_cartridge_header(rom)
    assert spec.cart_type == CartType.MBC1_RAM
    assert spec.rom_bank_count == 8
    assert spec.ram_bank_count == 4
    assert spec.has_ram
    assert not spec.has_rtc


def test_parse_mbc3_timer_header() -> None:
    rom = _make_rom(cart_type=0x0F, rom_size_code=0x03, ram_size_code=0x00)
    spec = parse_cartridge_header(rom)
    assert spec.cart_type == CartType.MBC3_TIMER_BATTERY
    assert spec.has_rtc
    assert not spec.has_ram


def test_clamp_bank_power_of_two() -> None:
    assert clamp_bank(5, 8) == 5
    assert clamp_bank(9, 8) == 1


def test_clamp_bank_non_power_of_two() -> None:
    assert clamp_bank(7, 6) == 1


def test_invalid_rom_size_code() -> None:
    with pytest.raises(ValueError):
        rom_size_from_code(0xFF)


def test_invalid_ram_size_code() -> None:
    with pytest.raises(ValueError):
        ram_size_from_code(0xFF)


def test_invalid_cart_type() -> None:
    rom = _make_rom(cart_type=0x19, rom_size_code=0x00, ram_size_code=0x00)
    with pytest.raises(ValueError):
        parse_cartridge_header(rom)
