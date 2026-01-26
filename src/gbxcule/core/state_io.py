# pyright: reportOptionalMemberAccess=false
"""PyBoy v9 state file I/O for gbxcule.

This module provides functions to load and save PyBoy v9 state files,
enabling state transfer between PyBoy and Warp backends.

See docs/pyboy_state_format.md for format specification.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

from gbxcule.core.abi import CYCLES_PER_SCANLINE, MEM_SIZE, SCREEN_H
from gbxcule.core.cartridge import (
    CART_STATE_BANK_MODE,
    CART_STATE_BOOTROM_ENABLED,
    CART_STATE_RAM_BANK,
    CART_STATE_RAM_ENABLE,
    CART_STATE_ROM_BANK_HI,
    CART_STATE_ROM_BANK_LO,
    CART_STATE_STRIDE,
)

if TYPE_CHECKING:
    from gbxcule.backends.warp_vec import WarpVecBaseBackend

# PyBoy state versions we support
PYBOY_STATE_VERSION_MIN = 9
PYBOY_STATE_VERSION_MAX = 15
PYBOY_STATE_VERSION_SAVE = 9  # Version we write

# Memory region sizes
VRAM_SIZE = 8192
OAM_SIZE = 160
WRAM_SIZE = 8192
HRAM_SIZE = 127
IO_PORTS_SIZE = 76
NON_IO_RAM0_SIZE = 96
NON_IO_RAM1_SIZE = 52
SCANLINE_PARAMS_SIZE = 720
SOUND_SIZE_V9 = 138
RENDERER_SIZE = 144 * 160 * 5  # 115200 bytes


@dataclass
class PyBoyState:
    """Parsed PyBoy state file (v9 DMG format)."""

    version: int
    bootrom_enabled: int

    # CPU registers
    a: int
    f: int
    b: int
    c: int
    d: int
    e: int
    h: int
    l_reg: int  # Named l_reg to avoid ambiguous variable name
    sp: int
    pc: int
    ime: int
    halted: int
    stopped: int
    ie: int  # Interrupt Enable (0xFFFF)
    interrupt_queued: int
    if_reg: int  # Interrupt Flag (0xFF0F)
    cpu_cycles: int

    # LCD state
    vram: bytes  # 8192 bytes
    oam: bytes  # 160 bytes
    lcdc: int
    stat: int
    ly: int
    lyc: int
    scy: int
    scx: int
    wy: int
    wx: int
    bgp: int
    obp0: int
    obp1: int
    lcd_clock: int
    lcd_clock_target: int
    next_stat_mode: int
    scanline_params: bytes  # 720 bytes

    # RAM
    wram: bytes  # 8192 bytes (0xC000-0xDFFF)
    hram: bytes  # 127 bytes (0xFF80-0xFFFE)
    io_ports: bytes  # 76 bytes (0xFF00-0xFF4B)
    non_io_ram0: bytes  # 96 bytes (0xFEA0-0xFEFF)
    non_io_ram1: bytes  # 52 bytes (0xFF4C-0xFF7F)

    # Timer
    div: int
    tima: int
    tma: int
    tac: int
    div_counter: int
    tima_counter: int

    # Cartridge
    rombank: int
    rambank: int
    rambank_enabled: int
    memorymodel: int
    cart_ram: bytes  # Concatenated cartridge RAM banks

    # Joypad
    joypad_directional: int  # D-pad bits (0=pressed, 1=released)
    joypad_standard: int  # Button bits (0=pressed, 1=released)

    # Serial (v15+)
    serial_sb: int
    serial_sc: int


def _read_u8(f: BinaryIO) -> int:
    data = f.read(1)
    if len(data) != 1:
        raise EOFError("Unexpected end of file")
    return data[0]


def _read_u16(f: BinaryIO) -> int:
    data = f.read(2)
    if len(data) != 2:
        raise EOFError("Unexpected end of file")
    return struct.unpack("<H", data)[0]


def _read_u64(f: BinaryIO) -> int:
    data = f.read(8)
    if len(data) != 8:
        raise EOFError("Unexpected end of file")
    return struct.unpack("<Q", data)[0]


def _write_u8(f: BinaryIO, value: int) -> None:
    f.write(struct.pack("B", value & 0xFF))


def _write_u16(f: BinaryIO, value: int) -> None:
    f.write(struct.pack("<H", value & 0xFFFF))


def _write_u64(f: BinaryIO, value: int) -> None:
    f.write(struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))


def _skip_sound_state(
    f: BinaryIO,
    version: int,
    *,
    max_sound_size: int | None = None,
) -> None:
    """Skip the sound state section.

    Sound state size varies by version but we can read field by field.
    """
    if version == 13:
        # Legacy layout: last_cycles, cycles, then channel states.
        f.read(8)  # last_cycles
        f.read(8)  # cycles
        _skip_sweep_channel_state(f)
        _skip_tone_channel_state(f)
        _skip_wave_channel_state(f)
        _skip_noise_channel_state(f)
        return

    if version >= 14:
        if max_sound_size is not None:
            if max_sound_size < 0:
                raise ValueError(
                    f"State file too small for sound section: {max_sound_size} bytes"
                )
            f.read(max_sound_size)
            return
        # Sound.save_state (v14+)
        f.read(8)  # audiobuffer_head
        samples_per_frame = _read_u64(f)
        f.read(8)  # cycles_per_sample (double)
        audiobuffer_length = (samples_per_frame + 1) * 2
        f.read(audiobuffer_length)
        f.read(1)  # speed_shift
        f.read(8)  # cycles_target (double)
        f.read(8)  # cycles_target_512Hz (double)
        f.read(8)  # cycles_to_interrupt
        f.read(8)  # cycles
        f.read(8)  # last_cycles
        f.read(8)  # div_apu_counter
        f.read(8)  # div_apu
        f.read(1)  # poweron
        f.read(1)  # disable_sampling
        f.read(8)  # channel enable bits

        _skip_sweep_channel_state(f)
        _skip_tone_channel_state(f)
        _skip_wave_channel_state(f)
        _skip_noise_channel_state(f)
        return

    # Fallback for older versions (v9-v12).
    f.read(SOUND_SIZE_V9)


def _skip_tone_channel_state(f: BinaryIO) -> None:
    f.read(2)  # wave_duty, init_length_timer
    f.read(3)  # envelope_volume, envelope_direction, envelope_pace
    f.read(2)  # sound_period
    f.read(1)  # length_enable
    f.read(1)  # enable
    f.read(8 * 6)  # lengthtimer, envelopetimer, periodtimer, period, waveframe, volume


def _skip_sweep_channel_state(f: BinaryIO) -> None:
    _skip_tone_channel_state(f)
    f.read(3)  # sweep_pace, sweep_direction, sweep_magnitude
    f.read(8)  # sweeptimer
    f.read(1)  # sweepenable
    f.read(8)  # shadow


def _skip_wave_channel_state(f: BinaryIO) -> None:
    f.read(16)  # wavetable
    f.read(1)  # dacpow
    f.read(1)  # init_length_timer
    f.read(1)  # volreg
    f.read(2)  # sound_period
    f.read(1)  # length_enable
    f.read(1)  # enable
    f.read(8 * 5)  # lengthtimer, periodtimer, period, waveframe, volumeshift


def _skip_noise_channel_state(f: BinaryIO) -> None:
    f.read(8)  # init_length_timer..length_enable
    f.read(1)  # enable
    # lengthtimer, periodtimer, envelopetimer, period, shiftregister, lfsrfeed, volume
    f.read(8 * 7)


def load_pyboy_state(
    path: str | Path, *, expected_cart_ram_size: int | None = None
) -> PyBoyState:
    """Load a PyBoy v9 state file.

    Args:
        path: Path to the .state file.

    Returns:
        Parsed PyBoyState object.

    Raises:
        ValueError: If state version is not 9 or CGB mode is detected.
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    with open(path, "rb") as f:
        return _load_pyboy_state_from_file(
            f, expected_cart_ram_size=expected_cart_ram_size
        )


def _load_pyboy_state_from_file(
    f: BinaryIO, *, expected_cart_ram_size: int | None = None
) -> PyBoyState:
    """Load PyBoy state from an open file handle.

    Supports versions 9-15.
    """
    # Header
    version = _read_u8(f)
    if version < PYBOY_STATE_VERSION_MIN or version > PYBOY_STATE_VERSION_MAX:
        raise ValueError(
            f"Unsupported state version {version}, "
            f"expected {PYBOY_STATE_VERSION_MIN}-{PYBOY_STATE_VERSION_MAX}"
        )

    bootrom_enabled = _read_u8(f)
    _key1 = _read_u8(f)
    _double_speed = _read_u8(f)
    cgb = _read_u8(f)
    if cgb != 0:
        raise ValueError("CGB mode not supported, only DMG")

    # CPU
    a = _read_u8(f)
    flags = _read_u8(f)
    b = _read_u8(f)
    c = _read_u8(f)
    d = _read_u8(f)
    e = _read_u8(f)
    hl = _read_u16(f)
    sp = _read_u16(f)
    pc = _read_u16(f)
    ime = _read_u8(f)
    halted = _read_u8(f)
    stopped = _read_u8(f)
    ie = _read_u8(f)
    interrupt_queued = _read_u8(f)
    if_reg = _read_u8(f)

    # v12+ has CPU cycles
    cpu_cycles = _read_u64(f) if version >= 12 else 0

    h = (hl >> 8) & 0xFF
    l_reg = hl & 0xFF

    # LCD
    vram = f.read(VRAM_SIZE)
    oam = f.read(OAM_SIZE)
    lcdc = _read_u8(f)
    bgp = _read_u8(f)
    obp0 = _read_u8(f)
    obp1 = _read_u8(f)
    stat = _read_u8(f)
    ly = _read_u8(f)
    lyc = _read_u8(f)
    scy = _read_u8(f)
    scx = _read_u8(f)
    wy = _read_u8(f)
    wx = _read_u8(f)

    # LCD timing/mode (v8+)
    _lcd_cgb = _read_u8(f)  # Should be 0 for DMG
    _lcd_double_speed = _read_u8(f)  # speed_shift
    if version >= 13:
        _lcd_last_cycles = _read_u64(f)
    lcd_clock = _read_u64(f)
    lcd_clock_target = _read_u64(f)
    next_stat_mode = _read_u8(f)

    # Scanline params (v9+)
    scanline_params = f.read(SCANLINE_PARAMS_SIZE)

    # Sound (v8+) - size varies; optionally derive from file size when provided.
    max_sound_size = None
    if expected_cart_ram_size is not None and version >= 14:
        sound_start = f.tell()
        file_end = f.seek(0, 2)
        f.seek(sound_start)
        timer_size = 8
        if version >= 12:
            timer_size += 8
        if version >= 13:
            timer_size += 8
        tail_size = 2 + (36 if version >= 15 else 0)
        min_after_sound = (
            RENDERER_SIZE
            + WRAM_SIZE
            + NON_IO_RAM0_SIZE
            + IO_PORTS_SIZE
            + HRAM_SIZE
            + NON_IO_RAM1_SIZE
            + timer_size
            + 5  # cart header fields
            + expected_cart_ram_size
            + tail_size
        )
        max_sound_size = file_end - sound_start - min_after_sound
    _skip_sound_state(f, version, max_sound_size=max_sound_size)

    # Renderer (skip - 115200 bytes)
    f.read(RENDERER_SIZE)

    # RAM
    wram = f.read(WRAM_SIZE)
    non_io_ram0 = f.read(NON_IO_RAM0_SIZE)
    io_ports = f.read(IO_PORTS_SIZE)
    hram = f.read(HRAM_SIZE)
    non_io_ram1 = f.read(NON_IO_RAM1_SIZE)

    # Timer (PyBoy order)
    div = _read_u8(f)
    tima = _read_u8(f)
    div_counter = _read_u16(f)
    tima_counter = _read_u16(f)
    tma = _read_u8(f)
    tac = _read_u8(f)

    # v12+ has timer last_cycles
    if version >= 12:
        _timer_last_cycles = _read_u64(f)  # Skip

    # v13+ has timer cycles_to_interrupt
    if version >= 13:
        _timer_cycles_to_interrupt = _read_u64(f)  # Skip

    # Cartridge
    rombank = _read_u16(f)
    rambank = _read_u8(f)
    rambank_enabled = _read_u8(f)
    memorymodel = _read_u8(f)
    # Cartridge RAM sits between cart header and the trailing joypad/serial bytes.
    serial_state_size = 36 if version >= 15 else 0
    tail_size = 2 + serial_state_size
    cart_ram_start = f.tell()
    file_end = f.seek(0, 2)
    cart_ram_size = file_end - tail_size - cart_ram_start
    if cart_ram_size < 0:
        raise ValueError(
            f"State file too small for cart RAM: {cart_ram_size} bytes remaining"
        )
    f.seek(cart_ram_start)
    cart_ram = b""
    if cart_ram_size:
        cart_ram = f.read(cart_ram_size)
        if len(cart_ram) != cart_ram_size:
            raise EOFError("Unexpected end of file while reading cart RAM")

    # Serial state is written last; joypad state is immediately before it.
    if version >= 15:
        f.seek(-(serial_state_size + 2), 2)
        joypad_directional = _read_u8(f)
        joypad_standard = _read_u8(f)
        serial_sb = _read_u8(f)
        serial_sc = _read_u8(f)
        _serial_transfer_enabled = _read_u8(f)
        _serial_internal_clock = _read_u8(f)
        f.read(8 * 4)  # last_cycles, cycles_to_interrupt, clock, clock_target
    else:
        f.seek(-2, 2)
        joypad_directional = _read_u8(f)
        joypad_standard = _read_u8(f)
        serial_sb = io_ports[0x01]
        serial_sc = io_ports[0x02]

    return PyBoyState(
        version=version,
        bootrom_enabled=bootrom_enabled,
        a=a,
        f=flags,
        b=b,
        c=c,
        d=d,
        e=e,
        h=h,
        l_reg=l_reg,
        sp=sp,
        pc=pc,
        ime=ime,
        halted=halted,
        stopped=stopped,
        ie=ie,
        interrupt_queued=interrupt_queued,
        if_reg=if_reg,
        cpu_cycles=cpu_cycles,
        vram=vram,
        oam=oam,
        lcdc=lcdc,
        stat=stat,
        ly=ly,
        lyc=lyc,
        scy=scy,
        scx=scx,
        wy=wy,
        wx=wx,
        bgp=bgp,
        obp0=obp0,
        obp1=obp1,
        lcd_clock=lcd_clock,
        lcd_clock_target=lcd_clock_target,
        next_stat_mode=next_stat_mode,
        scanline_params=scanline_params,
        wram=wram,
        hram=hram,
        io_ports=io_ports,
        non_io_ram0=non_io_ram0,
        non_io_ram1=non_io_ram1,
        div=div,
        tima=tima,
        tma=tma,
        tac=tac,
        div_counter=div_counter,
        tima_counter=tima_counter,
        rombank=rombank,
        rambank=rambank,
        rambank_enabled=rambank_enabled,
        memorymodel=memorymodel,
        cart_ram=cart_ram,
        joypad_directional=joypad_directional,
        joypad_standard=joypad_standard,
        serial_sb=serial_sb,
        serial_sc=serial_sc,
    )


def save_pyboy_state(state: PyBoyState, path: str | Path) -> None:
    """Save a PyBoyState to a v9 state file.

    Args:
        state: The state to save.
        path: Path to write the .state file.
    """
    path = Path(path)
    with open(path, "wb") as f:
        _save_pyboy_state_to_file(state, f)


def _save_pyboy_state_to_file(state: PyBoyState, f: BinaryIO) -> None:
    """Save PyBoy state to an open file handle."""
    # Header
    _write_u8(f, PYBOY_STATE_VERSION_SAVE)
    _write_u8(f, state.bootrom_enabled & 0x01)
    _write_u8(f, 0)  # key1 = 0 (DMG)
    _write_u8(f, 0)  # double_speed = false
    _write_u8(f, 0)  # cgb = false (DMG mode)

    # CPU
    _write_u8(f, state.a)
    _write_u8(f, state.f)
    _write_u8(f, state.b)
    _write_u8(f, state.c)
    _write_u8(f, state.d)
    _write_u8(f, state.e)
    _write_u16(f, (state.h << 8) | state.l_reg)  # HL
    _write_u16(f, state.sp)
    _write_u16(f, state.pc)
    _write_u8(f, state.ime)
    _write_u8(f, state.halted)
    _write_u8(f, state.stopped)
    _write_u8(f, state.ie)
    _write_u8(f, state.interrupt_queued)
    _write_u8(f, state.if_reg)

    # LCD - VRAM and OAM
    assert len(state.vram) == VRAM_SIZE
    f.write(state.vram)
    assert len(state.oam) == OAM_SIZE
    f.write(state.oam)

    # LCD registers
    _write_u8(f, state.lcdc)
    _write_u8(f, state.bgp)
    _write_u8(f, state.obp0)
    _write_u8(f, state.obp1)
    _write_u8(f, state.stat)
    _write_u8(f, state.ly)
    _write_u8(f, state.lyc)
    _write_u8(f, state.scy)
    _write_u8(f, state.scx)
    _write_u8(f, state.wy)
    _write_u8(f, state.wx)

    # LCD timing/mode (v8+)
    _write_u8(f, 0)  # lcd_cgb = 0 (DMG)
    _write_u8(f, 0)  # lcd_double_speed = 0 (DMG)
    _write_u64(f, state.lcd_clock)
    _write_u64(f, state.lcd_clock_target)
    _write_u8(f, state.next_stat_mode)

    # Scanline params (v9+)
    assert len(state.scanline_params) == SCANLINE_PARAMS_SIZE
    f.write(state.scanline_params)

    # Sound (write zeros - 138 bytes for v9)
    f.write(b"\x00" * SOUND_SIZE_V9)

    # Renderer (write zeros - 115200 bytes)
    # PyBoy will re-render on load anyway
    f.write(b"\x00" * RENDERER_SIZE)

    # RAM
    assert len(state.wram) == WRAM_SIZE
    f.write(state.wram)
    assert len(state.non_io_ram0) == NON_IO_RAM0_SIZE
    f.write(state.non_io_ram0)
    assert len(state.io_ports) == IO_PORTS_SIZE
    f.write(state.io_ports)
    assert len(state.hram) == HRAM_SIZE
    f.write(state.hram)
    assert len(state.non_io_ram1) == NON_IO_RAM1_SIZE
    f.write(state.non_io_ram1)

    # Timer
    _write_u8(f, state.div)
    _write_u8(f, state.tima)
    _write_u16(f, state.div_counter)
    _write_u16(f, state.tima_counter)
    _write_u8(f, state.tma)
    _write_u8(f, state.tac)

    # Cartridge (minimal - no RAM banks for Tetris)
    _write_u16(f, state.rombank)
    _write_u8(f, state.rambank)
    _write_u8(f, state.rambank_enabled)
    _write_u8(f, state.memorymodel)
    if state.cart_ram:
        f.write(state.cart_ram)

    # Joypad (at end)
    _write_u8(f, state.joypad_directional)
    _write_u8(f, state.joypad_standard)


def state_from_warp_backend(
    backend: WarpVecBaseBackend,
    env_idx: int = 0,
) -> PyBoyState:
    """Extract a PyBoyState from a Warp backend.

    Args:
        backend: The Warp backend to read from.
        env_idx: Environment index to extract.

    Returns:
        PyBoyState populated from the backend's current state.
    """
    if not backend._initialized:
        raise RuntimeError("Backend not initialized")

    # Synchronize if needed (for CUDA)
    if hasattr(backend, "_wp"):
        backend._wp.synchronize()

    mem = backend._mem.numpy()
    base = env_idx * MEM_SIZE

    # Extract memory regions
    vram = bytes(mem[base + 0x8000 : base + 0x8000 + VRAM_SIZE])
    oam = bytes(mem[base + 0xFE00 : base + 0xFE00 + OAM_SIZE])
    wram = bytes(mem[base + 0xC000 : base + 0xC000 + WRAM_SIZE])
    hram = bytes(mem[base + 0xFF80 : base + 0xFF80 + HRAM_SIZE])
    io_ports = bytes(mem[base + 0xFF00 : base + 0xFF00 + IO_PORTS_SIZE])
    non_io_ram0 = bytes(mem[base + 0xFEA0 : base + 0xFEA0 + NON_IO_RAM0_SIZE])
    non_io_ram1 = bytes(mem[base + 0xFF4C : base + 0xFF4C + NON_IO_RAM1_SIZE])

    cart_state = backend._cart_state.numpy()
    cart_base = env_idx * CART_STATE_STRIDE
    bootrom_enabled = int(cart_state[cart_base + CART_STATE_BOOTROM_ENABLED]) & 0x01
    rom_bank_lo = int(cart_state[cart_base + CART_STATE_ROM_BANK_LO]) & 0x7F
    rom_bank_hi = int(cart_state[cart_base + CART_STATE_ROM_BANK_HI]) & 0x03
    rombank = (rom_bank_hi << 5) | rom_bank_lo
    rambank = int(cart_state[cart_base + CART_STATE_RAM_BANK]) & 0xFF
    rambank_enabled = int(cart_state[cart_base + CART_STATE_RAM_ENABLE]) & 0x01
    memorymodel = int(cart_state[cart_base + CART_STATE_BANK_MODE]) & 0x01
    cart_ram = b""
    ram_byte_length = int(getattr(backend, "_ram_byte_length", 0))
    if ram_byte_length > 0 and backend._cart_ram is not None:
        cart_ram_np = backend._cart_ram.numpy()
        cart_ram_base = env_idx * ram_byte_length
        cart_ram = bytes(cart_ram_np[cart_ram_base : cart_ram_base + ram_byte_length])

    # Extract CPU registers
    a = int(backend._a.numpy()[env_idx]) & 0xFF
    f = int(backend._f.numpy()[env_idx]) & 0xF0  # Lower 4 bits always 0
    b = int(backend._b.numpy()[env_idx]) & 0xFF
    c = int(backend._c.numpy()[env_idx]) & 0xFF
    d = int(backend._d.numpy()[env_idx]) & 0xFF
    e = int(backend._e.numpy()[env_idx]) & 0xFF
    h = int(backend._h.numpy()[env_idx]) & 0xFF
    l_reg = int(backend._l.numpy()[env_idx]) & 0xFF
    sp = int(backend._sp.numpy()[env_idx]) & 0xFFFF
    pc = int(backend._pc.numpy()[env_idx]) & 0xFFFF
    ime = int(backend._ime.numpy()[env_idx]) & 0x01
    halted = int(backend._halted.numpy()[env_idx]) & 0x01
    cpu_cycles = int(backend._cycle_count.numpy()[env_idx])

    # Interrupts from memory
    ie = mem[base + 0xFFFF]
    if_reg = mem[base + 0xFF0F]

    # LCD registers from memory
    lcdc = mem[base + 0xFF40]
    stat = mem[base + 0xFF41]
    scy = mem[base + 0xFF42]
    scx = mem[base + 0xFF43]
    ly = int(backend._ppu_ly.numpy()[env_idx]) & 0xFF
    lyc = mem[base + 0xFF45]
    bgp = mem[base + 0xFF47]
    obp0 = mem[base + 0xFF48]
    obp1 = mem[base + 0xFF49]
    wy = mem[base + 0xFF4A]
    wx = mem[base + 0xFF4B]

    # Timer registers from memory
    div = mem[base + 0xFF04]
    tima = mem[base + 0xFF05]
    tma = mem[base + 0xFF06]
    tac = mem[base + 0xFF07]

    # Timer internal state (full 16-bit divider)
    div_counter_full = int(backend._div_counter.numpy()[env_idx]) & 0xFFFF
    div_counter = div_counter_full & 0xFF
    div = (div_counter_full >> 8) & 0xFF

    # PPU timing - compute lcd_clock from scanline state
    ppu_scanline_cycle = int(backend._ppu_scanline_cycle.numpy()[env_idx])
    lcd_clock = ly * CYCLES_PER_SCANLINE + ppu_scanline_cycle

    # Compute next_stat_mode from scanline cycle
    # Mode 2: OAM scan (0-79), Mode 3: Drawing (80-251), Mode 0: HBlank (252-455)
    # Mode 1: VBlank (LY 144-153)
    if ly >= 144:
        next_stat_mode = 1  # VBlank
    elif ppu_scanline_cycle < 80:
        next_stat_mode = 2  # OAM scan
    elif ppu_scanline_cycle < 252:
        next_stat_mode = 3  # Drawing
    else:
        next_stat_mode = 0  # HBlank

    # Generate scanline params (5 bytes per scanline)
    # Format: [SCX, SCY, WX, WY, palette?]
    scanline_params = bytearray(SCANLINE_PARAMS_SIZE)
    for i in range(144):
        scanline_params[i * 5 + 0] = scx
        scanline_params[i * 5 + 1] = scy
        scanline_params[i * 5 + 2] = wx
        scanline_params[i * 5 + 3] = wy
        scanline_params[i * 5 + 4] = 0

    return PyBoyState(
        version=PYBOY_STATE_VERSION_SAVE,
        bootrom_enabled=bootrom_enabled,
        a=a,
        f=f,
        b=b,
        c=c,
        d=d,
        e=e,
        h=h,
        l_reg=l_reg,
        sp=sp,
        pc=pc,
        ime=ime,
        halted=halted,
        stopped=0,
        ie=ie,
        interrupt_queued=0,
        if_reg=if_reg,
        cpu_cycles=cpu_cycles,
        vram=vram,
        oam=oam,
        lcdc=lcdc,
        stat=stat,
        ly=ly,
        lyc=lyc,
        scy=scy,
        scx=scx,
        wy=wy,
        wx=wx,
        bgp=bgp,
        obp0=obp0,
        obp1=obp1,
        lcd_clock=lcd_clock,
        lcd_clock_target=lcd_clock + CYCLES_PER_SCANLINE,
        next_stat_mode=next_stat_mode,
        scanline_params=bytes(scanline_params),
        wram=wram,
        hram=hram,
        io_ports=io_ports,
        non_io_ram0=non_io_ram0,
        non_io_ram1=non_io_ram1,
        div=div,
        tima=tima,
        tma=tma,
        tac=tac,
        div_counter=div_counter,
        tima_counter=0,  # Warp doesn't track TIMA counter separately
        rombank=rombank,
        rambank=rambank,
        rambank_enabled=rambank_enabled,
        memorymodel=memorymodel,
        cart_ram=cart_ram,
        joypad_directional=0x0F,  # All released
        joypad_standard=0x0F,  # All released
        serial_sb=mem[base + 0xFF01],
        serial_sc=mem[base + 0xFF02],
    )


def apply_state_to_warp_backend(
    state: PyBoyState,
    backend: WarpVecBaseBackend,
    env_idx: int = 0,
) -> None:
    """Apply a PyBoyState to a Warp backend.

    Args:
        state: The state to apply.
        backend: The Warp backend to modify.
        env_idx: Environment index to modify.
    """
    if not backend._initialized:
        raise RuntimeError("Backend not initialized")

    wp = backend._wp
    is_cuda = backend.device == "cuda"

    def _write_scalar(dest, value, np_dtype, wp_dtype) -> None:
        if dest is None:
            return
        if is_cuda:
            host_np = np.array([value], dtype=np_dtype)
            host = wp.array(host_np, dtype=wp_dtype, device="cpu")
            wp.copy(dest, host, dest_offset=int(env_idx), src_offset=0, count=1)
        else:
            dest.numpy()[env_idx] = value

    # CPU registers
    _write_scalar(backend._a, state.a, np.int32, wp.int32)
    _write_scalar(backend._f, state.f, np.int32, wp.int32)
    _write_scalar(backend._b, state.b, np.int32, wp.int32)
    _write_scalar(backend._c, state.c, np.int32, wp.int32)
    _write_scalar(backend._d, state.d, np.int32, wp.int32)
    _write_scalar(backend._e, state.e, np.int32, wp.int32)
    _write_scalar(backend._h, state.h, np.int32, wp.int32)
    _write_scalar(backend._l, state.l_reg, np.int32, wp.int32)
    _write_scalar(backend._sp, state.sp, np.int32, wp.int32)
    _write_scalar(backend._pc, state.pc, np.int32, wp.int32)
    _write_scalar(backend._ime, state.ime, np.int32, wp.int32)
    _write_scalar(backend._ime_delay, 0, np.int32, wp.int32)
    _write_scalar(backend._halted, 0, np.int32, wp.int32)

    # Memory
    mem_full = backend._mem.numpy()
    base = env_idx * MEM_SIZE
    mem_env = mem_full[base : base + MEM_SIZE].copy() if is_cuda else mem_full
    mem_offset = 0 if is_cuda else base
    cart_state_full = backend._cart_state.numpy()
    cart_base = env_idx * CART_STATE_STRIDE
    cart_state = (
        cart_state_full[cart_base : cart_base + CART_STATE_STRIDE].copy()
        if is_cuda
        else cart_state_full
    )

    cart_state_idx = cart_base if not is_cuda else 0
    cart_state[cart_state_idx + CART_STATE_BOOTROM_ENABLED] = (
        1 if state.bootrom_enabled != 0 else 0
    )
    # Preserve full 9-bit rombank (PyBoy uses 0-511 for 512KB ROMs).
    cart_state[cart_state_idx + CART_STATE_ROM_BANK_LO] = state.rombank & 0xFF
    cart_state[cart_state_idx + CART_STATE_ROM_BANK_HI] = (state.rombank >> 8) & 0xFF
    cart_state[cart_state_idx + CART_STATE_RAM_BANK] = state.rambank & 0xFF
    cart_state[cart_state_idx + CART_STATE_RAM_ENABLE] = state.rambank_enabled & 0x01
    cart_state[cart_state_idx + CART_STATE_BANK_MODE] = state.memorymodel & 0x01
    ram_byte_length = int(getattr(backend, "_ram_byte_length", 0))
    if ram_byte_length > 0 and backend._cart_ram is not None and state.cart_ram:
        cart_ram_bytes = np.frombuffer(state.cart_ram, dtype=np.uint8)
        cart_ram_env = np.zeros((ram_byte_length,), dtype=np.uint8)
        if cart_ram_bytes.size >= ram_byte_length:
            cart_ram_env[:] = cart_ram_bytes[:ram_byte_length]
        else:
            cart_ram_env[: cart_ram_bytes.size] = cart_ram_bytes
        if is_cuda:
            cart_ram_host = wp.array(cart_ram_env, dtype=wp.uint8, device="cpu")
            wp.copy(
                backend._cart_ram,
                cart_ram_host,
                dest_offset=int(env_idx) * ram_byte_length,
                src_offset=0,
                count=ram_byte_length,
            )
        else:
            cart_ram_np = backend._cart_ram.numpy()
            cart_ram_base = env_idx * ram_byte_length
            cart_ram_np[cart_ram_base : cart_ram_base + ram_byte_length] = cart_ram_env

    # VRAM (0x8000-0x9FFF)
    mem_env[mem_offset + 0x8000 : mem_offset + 0x8000 + VRAM_SIZE] = np.frombuffer(
        state.vram, dtype=np.uint8
    )

    # OAM (0xFE00-0xFE9F)
    mem_env[mem_offset + 0xFE00 : mem_offset + 0xFE00 + OAM_SIZE] = np.frombuffer(
        state.oam, dtype=np.uint8
    )

    # WRAM (0xC000-0xDFFF)
    mem_env[mem_offset + 0xC000 : mem_offset + 0xC000 + WRAM_SIZE] = np.frombuffer(
        state.wram, dtype=np.uint8
    )

    # HRAM (0xFF80-0xFFFE)
    mem_env[mem_offset + 0xFF80 : mem_offset + 0xFF80 + HRAM_SIZE] = np.frombuffer(
        state.hram, dtype=np.uint8
    )

    # IO ports (0xFF00-0xFF4B)
    mem_env[mem_offset + 0xFF00 : mem_offset + 0xFF00 + IO_PORTS_SIZE] = np.frombuffer(
        state.io_ports, dtype=np.uint8
    )
    # Joypad select bits are tracked separately in Warp.
    _write_scalar(
        backend._joyp_select,
        int(mem_env[mem_offset + 0xFF00] & 0x30),
        np.uint8,
        wp.uint8,
    )

    # Non-IO RAM regions
    mem_env[mem_offset + 0xFEA0 : mem_offset + 0xFEA0 + NON_IO_RAM0_SIZE] = (
        np.frombuffer(state.non_io_ram0, dtype=np.uint8)
    )
    mem_env[mem_offset + 0xFF4C : mem_offset + 0xFF4C + NON_IO_RAM1_SIZE] = (
        np.frombuffer(state.non_io_ram1, dtype=np.uint8)
    )

    # Key LCD registers (may override IO ports)
    mem_env[mem_offset + 0xFF40] = state.lcdc
    mem_env[mem_offset + 0xFF41] = state.stat
    mem_env[mem_offset + 0xFF42] = state.scy
    mem_env[mem_offset + 0xFF43] = state.scx
    mem_env[mem_offset + 0xFF44] = state.ly
    mem_env[mem_offset + 0xFF45] = state.lyc
    mem_env[mem_offset + 0xFF47] = state.bgp
    mem_env[mem_offset + 0xFF48] = state.obp0
    mem_env[mem_offset + 0xFF49] = state.obp1
    mem_env[mem_offset + 0xFF4A] = state.wy
    mem_env[mem_offset + 0xFF4B] = state.wx
    mem_env[mem_offset + 0xFF01] = state.serial_sb & 0xFF
    mem_env[mem_offset + 0xFF02] = state.serial_sc & 0xFF

    # Seed PPU latches from scanline params to render immediately after load.
    if (
        backend._bg_lcdc_latch_env0 is not None
        and backend._bg_scx_latch_env0 is not None
        and backend._bg_scy_latch_env0 is not None
        and backend._bg_bgp_latch_env0 is not None
        and backend._win_wx_latch_env0 is not None
        and backend._win_wy_latch_env0 is not None
        and backend._win_line_latch_env0 is not None
        and backend._obj_obp0_latch_env0 is not None
        and backend._obj_obp1_latch_env0 is not None
    ):
        scan = state.scanline_params
        if len(scan) >= SCREEN_H * 5:
            if is_cuda:
                lcdc_latch = np.zeros((SCREEN_H,), dtype=np.uint8)
                scx_latch = np.zeros((SCREEN_H,), dtype=np.uint8)
                scy_latch = np.zeros((SCREEN_H,), dtype=np.uint8)
                bgp_latch = np.zeros((SCREEN_H,), dtype=np.uint8)
                wx_latch = np.zeros((SCREEN_H,), dtype=np.uint8)
                wy_latch = np.zeros((SCREEN_H,), dtype=np.uint8)
                win_line_latch = np.zeros((SCREEN_H,), dtype=np.uint8)
                obp0_latch = np.zeros((SCREEN_H,), dtype=np.uint8)
                obp1_latch = np.zeros((SCREEN_H,), dtype=np.uint8)
            else:
                lcdc_latch = backend._bg_lcdc_latch_env0.numpy()
                scx_latch = backend._bg_scx_latch_env0.numpy()
                scy_latch = backend._bg_scy_latch_env0.numpy()
                bgp_latch = backend._bg_bgp_latch_env0.numpy()
                wx_latch = backend._win_wx_latch_env0.numpy()
                wy_latch = backend._win_wy_latch_env0.numpy()
                win_line_latch = backend._win_line_latch_env0.numpy()
                obp0_latch = backend._obj_obp0_latch_env0.numpy()
                obp1_latch = backend._obj_obp1_latch_env0.numpy()
            for y in range(SCREEN_H):
                idx = y * 5
                scx = scan[idx + 0]
                scy = scan[idx + 1]
                wx = scan[idx + 2]
                wy = scan[idx + 3]
                lcdc_latch[y] = state.lcdc
                scx_latch[y] = scx
                scy_latch[y] = scy
                bgp_latch[y] = state.bgp
                wx_latch[y] = wx
                wy_latch[y] = wy
                win_line_latch[y] = (y - wy) & 0xFF if y >= wy else 0
                obp0_latch[y] = state.obp0
                obp1_latch[y] = state.obp1
            if is_cuda:
                lcdc_host = wp.array(lcdc_latch, dtype=wp.uint8, device="cpu")
                scx_host = wp.array(scx_latch, dtype=wp.uint8, device="cpu")
                scy_host = wp.array(scy_latch, dtype=wp.uint8, device="cpu")
                bgp_host = wp.array(bgp_latch, dtype=wp.uint8, device="cpu")
                wx_host = wp.array(wx_latch, dtype=wp.uint8, device="cpu")
                wy_host = wp.array(wy_latch, dtype=wp.uint8, device="cpu")
                win_line_host = wp.array(win_line_latch, dtype=wp.uint8, device="cpu")
                obp0_host = wp.array(obp0_latch, dtype=wp.uint8, device="cpu")
                obp1_host = wp.array(obp1_latch, dtype=wp.uint8, device="cpu")
                wp.copy(backend._bg_lcdc_latch_env0, lcdc_host, count=SCREEN_H)
                wp.copy(backend._bg_scx_latch_env0, scx_host, count=SCREEN_H)
                wp.copy(backend._bg_scy_latch_env0, scy_host, count=SCREEN_H)
                wp.copy(backend._bg_bgp_latch_env0, bgp_host, count=SCREEN_H)
                wp.copy(backend._win_wx_latch_env0, wx_host, count=SCREEN_H)
                wp.copy(backend._win_wy_latch_env0, wy_host, count=SCREEN_H)
                wp.copy(backend._win_line_latch_env0, win_line_host, count=SCREEN_H)
                wp.copy(backend._obj_obp0_latch_env0, obp0_host, count=SCREEN_H)
                wp.copy(backend._obj_obp1_latch_env0, obp1_host, count=SCREEN_H)

    # Interrupts
    mem_env[mem_offset + 0xFF0F] = state.if_reg
    mem_env[mem_offset + 0xFFFF] = state.ie

    # Timer registers
    mem_env[mem_offset + 0xFF04] = state.div
    mem_env[mem_offset + 0xFF05] = state.tima
    mem_env[mem_offset + 0xFF06] = state.tma
    mem_env[mem_offset + 0xFF07] = state.tac

    # Timer internal state
    div_counter_full = ((int(state.div) & 0xFF) << 8) | (int(state.div_counter) & 0xFF)
    _write_scalar(backend._div_counter, div_counter_full, np.int32, wp.int32)
    # Align timer edge tracking to current DIV/TAC state.
    tac = int(state.tac)
    if (tac & 0x04) == 0:
        timer_prev_in = 0
    else:
        sel = tac & 0x03
        if sel == 0:
            bit = 9
        elif sel == 1:
            bit = 3
        elif sel == 2:
            bit = 5
        else:
            bit = 7
        timer_prev_in = (div_counter_full >> bit) & 0x1
    _write_scalar(backend._timer_prev_in, timer_prev_in, np.int32, wp.int32)
    _write_scalar(backend._tima_reload_pending, 0, np.int32, wp.int32)
    _write_scalar(backend._tima_reload_delay, 0, np.int32, wp.int32)

    # PPU state
    _write_scalar(backend._ppu_ly, state.ly, np.int32, wp.int32)
    scanline_cycle = int(state.lcd_clock % CYCLES_PER_SCANLINE)
    _write_scalar(backend._ppu_scanline_cycle, scanline_cycle, np.int32, wp.int32)

    cycles_per_frame = CYCLES_PER_SCANLINE * 154
    if state.cpu_cycles != 0:
        cycle_count = int(state.cpu_cycles)
    else:
        cycle_count = int(state.lcd_clock)
    cycle_in_frame = int(state.lcd_clock % cycles_per_frame)
    cycle_min = np.iinfo(np.int64).min
    cycle_max = np.iinfo(np.int64).max
    if cycle_count < cycle_min:
        cycle_count = cycle_min
    elif cycle_count > cycle_max:
        cycle_count = cycle_max
    _write_scalar(backend._cycle_count, cycle_count, np.int64, wp.int64)
    _write_scalar(backend._cycle_in_frame, cycle_in_frame, np.int32, wp.int32)

    # PPU window line counter (estimate from WY and LY)
    if state.ly >= state.wy:
        _write_scalar(backend._ppu_window_line, state.ly - state.wy, np.int32, wp.int32)
    else:
        _write_scalar(backend._ppu_window_line, 0, np.int32, wp.int32)

    # STAT previous state for edge triggering (mode0/mode1/mode2/lyc flags)
    mode = int(state.stat) & 0x03
    prev_mode0 = 1 if mode == 0 else 0
    prev_mode1 = 1 if mode == 1 else 0
    prev_mode2 = 1 if mode == 2 else 0
    prev_lyc = 1 if int(state.ly) == int(state.lyc) else 0
    _write_scalar(
        backend._ppu_stat_prev,
        prev_mode0 | (prev_mode1 << 1) | (prev_mode2 << 2) | (prev_lyc << 3),
        np.uint8,
        wp.uint8,
    )

    if is_cuda:
        mem_host = wp.array(mem_env, dtype=wp.uint8, device="cpu")
        wp.copy(backend._mem, mem_host, dest_offset=base, src_offset=0, count=MEM_SIZE)

        cart_host = wp.array(cart_state, dtype=wp.int32, device="cpu")
        wp.copy(
            backend._cart_state,
            cart_host,
            dest_offset=cart_base,
            src_offset=0,
            count=CART_STATE_STRIDE,
        )
        wp.synchronize()
