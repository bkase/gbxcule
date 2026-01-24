"""Cartridge header parsing and size helpers (DMG-era)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class CartType(IntEnum):
    """Supported cartridge types (subset)."""

    ROM_ONLY = 0x00
    MBC1 = 0x01
    MBC1_RAM = 0x02
    MBC1_RAM_BATTERY = 0x03
    MBC3_TIMER_BATTERY = 0x0F
    MBC3_TIMER_RAM_BATTERY = 0x10
    MBC3 = 0x11
    MBC3_RAM = 0x12
    MBC3_RAM_BATTERY = 0x13


ROM_SIZE_MAP: dict[int, int] = {
    0x00: 32 * 1024,
    0x01: 64 * 1024,
    0x02: 128 * 1024,
    0x03: 256 * 1024,
    0x04: 512 * 1024,
    0x05: 1024 * 1024,
    0x06: 2 * 1024 * 1024,
    0x07: 4 * 1024 * 1024,
    0x08: 8 * 1024 * 1024,
    0x52: int(1.1 * 1024 * 1024),
    0x53: int(1.2 * 1024 * 1024),
    0x54: int(1.5 * 1024 * 1024),
}

RAM_SIZE_MAP: dict[int, int] = {
    0x00: 0,
    0x01: 2 * 1024,
    0x02: 8 * 1024,
    0x03: 32 * 1024,
    0x04: 128 * 1024,
    0x05: 64 * 1024,
}

CART_ROM_BANK_SIZE = 0x4000
CART_RAM_BANK_SIZE = 0x2000
BOOTROM_SIZE = 0x100

MBC_KIND_ROM_ONLY = 0
MBC_KIND_MBC1 = 1
MBC_KIND_MBC3 = 2

# Per-env cart state layout (int32 array)
CART_STATE_STRIDE = 16
CART_STATE_MBC_KIND = 0
CART_STATE_RAM_ENABLE = 1
CART_STATE_ROM_BANK_LO = 2
CART_STATE_ROM_BANK_HI = 3
CART_STATE_RAM_BANK = 4
CART_STATE_BANK_MODE = 5
CART_STATE_BOOTROM_ENABLED = 6
CART_STATE_RTC_SELECT = 7
CART_STATE_RTC_LATCH = 8
CART_STATE_RTC_SECONDS = 9
CART_STATE_RTC_MINUTES = 10
CART_STATE_RTC_HOURS = 11
CART_STATE_RTC_DAYS_LOW = 12
CART_STATE_RTC_DAYS_HIGH = 13
CART_STATE_RTC_LAST_CYCLE = 14


@dataclass(frozen=True)
class CartridgeSpec:
    """Parsed cartridge metadata and derived sizes."""

    cart_type: CartType
    rom_size_code: int
    rom_byte_length: int
    rom_bank_count: int
    rom_bank_mask: int | None
    ram_size_code: int
    ram_byte_length: int
    ram_bank_count: int
    has_ram: bool
    has_battery: bool
    has_rtc: bool


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def rom_size_from_code(code: int) -> int:
    """Return ROM byte length from header code."""
    if code not in ROM_SIZE_MAP:
        raise ValueError(f"Unsupported ROM size code: 0x{code:02X}")
    return ROM_SIZE_MAP[code]


def ram_size_from_code(code: int) -> int:
    """Return RAM byte length from header code."""
    if code not in RAM_SIZE_MAP:
        raise ValueError(f"Unsupported RAM size code: 0x{code:02X}")
    return RAM_SIZE_MAP[code]


def rom_bank_count_from_bytes(rom_bytes: int) -> int:
    """Return ROM bank count (16KB banks) from ROM byte length."""
    if rom_bytes <= 0 or rom_bytes % 0x4000 != 0:
        raise ValueError(f"Invalid ROM byte length: {rom_bytes}")
    return rom_bytes // 0x4000


def ram_bank_count_from_bytes(ram_bytes: int) -> int:
    """Return RAM bank count (8KB banks) from RAM byte length."""
    if ram_bytes == 0:
        return 0
    if ram_bytes % 0x2000 != 0:
        raise ValueError(f"Invalid RAM byte length: {ram_bytes}")
    return ram_bytes // 0x2000


def clamp_bank(bank: int, bank_count: int) -> int:
    """Clamp or mask a bank number into a valid range."""
    if bank_count <= 0:
        return 0
    if _is_power_of_two(bank_count):
        return bank & (bank_count - 1)
    return bank % bank_count


def parse_cartridge_header(rom: bytes) -> CartridgeSpec:
    """Parse cartridge header fields into a canonical spec."""
    if len(rom) < 0x150:
        raise ValueError(f"ROM too small for header: {len(rom)} bytes")
    cart_type_raw = rom[0x0147]
    try:
        cart_type = CartType(cart_type_raw)
    except ValueError as exc:
        raise ValueError(f"Unsupported cartridge type: 0x{cart_type_raw:02X}") from exc

    rom_size_code = rom[0x0148]
    ram_size_code = rom[0x0149]
    rom_bytes = rom_size_from_code(rom_size_code)
    ram_bytes = ram_size_from_code(ram_size_code)
    rom_banks = rom_bank_count_from_bytes(rom_bytes)
    ram_banks = ram_bank_count_from_bytes(ram_bytes)
    rom_bank_mask = rom_banks - 1 if _is_power_of_two(rom_banks) else None

    has_ram = cart_type in {
        CartType.MBC1_RAM,
        CartType.MBC1_RAM_BATTERY,
        CartType.MBC3_RAM,
        CartType.MBC3_RAM_BATTERY,
        CartType.MBC3_TIMER_RAM_BATTERY,
    }
    has_battery = cart_type in {
        CartType.MBC1_RAM_BATTERY,
        CartType.MBC3_TIMER_BATTERY,
        CartType.MBC3_TIMER_RAM_BATTERY,
        CartType.MBC3_RAM_BATTERY,
    }
    has_rtc = cart_type in {
        CartType.MBC3_TIMER_BATTERY,
        CartType.MBC3_TIMER_RAM_BATTERY,
    }

    return CartridgeSpec(
        cart_type=cart_type,
        rom_size_code=rom_size_code,
        rom_byte_length=rom_bytes,
        rom_bank_count=rom_banks,
        rom_bank_mask=rom_bank_mask,
        ram_size_code=ram_size_code,
        ram_byte_length=ram_bytes,
        ram_bank_count=ram_banks,
        has_ram=has_ram,
        has_battery=has_battery,
        has_rtc=has_rtc,
    )
