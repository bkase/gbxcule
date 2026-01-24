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

from gbxcule.core.abi import CYCLES_PER_SCANLINE, MEM_SIZE

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

    # Joypad
    joypad_directional: int  # D-pad bits (0=pressed, 1=released)
    joypad_standard: int  # Button bits (0=pressed, 1=released)


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


def _skip_sound_state(f: BinaryIO, version: int) -> None:
    """Skip the sound state section.

    Sound state size varies by version but we can read field by field.
    """
    # Main sound controller state
    _samples_in_buffer = _read_u16(f)
    _sample_rate = _read_u16(f)
    _buffer_max = _read_u16(f)
    _buffer_head_i = _read_u16(f)
    _samples_per_frame = _read_u16(f)
    _cycles_per_sample = _read_u64(f)
    _cycles_target = _read_u64(f)
    _cycles_512_target = _read_u64(f)
    _sound_cycles = _read_u64(f)
    _sound_last_cycles = _read_u64(f)
    _div_apu_counter = _read_u16(f)
    _div_apu_value = _read_u8(f)
    _power = _read_u8(f)
    _sampling_disabled = _read_u8(f)

    # 8 panning bits
    f.read(8)
    _nr50 = _read_u8(f)

    # Channel 1 (Sweep)
    _read_u8(f)  # wave_duty
    _read_u16(f)  # length_timer
    _read_u8(f)  # volume_start
    _read_u8(f)  # envelope_dir
    _read_u8(f)  # envelope_pace
    _read_u16(f)  # envelope_counter
    _read_u16(f)  # sound_period
    _read_u8(f)  # length_enable
    _read_u8(f)  # waveframe
    _read_u8(f)  # volume
    _read_u8(f)  # sweep_pace
    _read_u8(f)  # sweep_direction
    _read_u8(f)  # sweep_magnitude
    _read_u16(f)  # sweep_timer
    _read_u16(f)  # shadow_freq

    # Channel 2 (Tone)
    _read_u8(f)  # wave_duty
    _read_u16(f)  # length_timer
    _read_u8(f)  # volume_start
    _read_u8(f)  # envelope_dir
    _read_u8(f)  # envelope_pace
    _read_u16(f)  # envelope_counter
    _read_u16(f)  # sound_period
    _read_u8(f)  # length_enable
    _read_u8(f)  # waveframe
    _read_u8(f)  # volume

    # Channel 3 (Wave)
    f.read(16)  # wave_table
    _read_u8(f)  # dac_power
    _read_u16(f)  # length_timer
    _read_u8(f)  # volume_code
    _read_u16(f)  # sound_period
    _read_u8(f)  # length_enable
    _read_u8(f)  # waveframe
    _read_u8(f)  # volume_shift

    # Channel 4 (Noise)
    _read_u16(f)  # length_timer
    _read_u8(f)  # volume_start
    _read_u8(f)  # envelope_dir
    _read_u8(f)  # envelope_pace
    _read_u16(f)  # envelope_counter
    _read_u8(f)  # clock_power
    _read_u8(f)  # clock_width
    _read_u8(f)  # clock_divider
    _read_u8(f)  # length_enable
    _read_u16(f)  # shift_register
    _read_u16(f)  # lfsr_mask
    _read_u8(f)  # volume


def load_pyboy_state(path: str | Path) -> PyBoyState:
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
        return _load_pyboy_state_from_file(f)


def _load_pyboy_state_from_file(f: BinaryIO) -> PyBoyState:
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

    _bootrom = _read_u8(f)
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
    if version >= 12:
        _cpu_cycles = _read_u64(f)  # Skip - we don't use this

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
    _lcd_double_speed = _read_u8(f)  # Should be 0 for DMG

    # v11+ has last_cycles
    if version >= 11:
        _lcd_last_cycles = _read_u64(f)  # Skip

    lcd_clock = _read_u64(f)
    lcd_clock_target = _read_u64(f)
    next_stat_mode = _read_u8(f)

    # Scanline params
    scanline_params = f.read(SCANLINE_PARAMS_SIZE)

    # v11+ has additional LCD flags
    if version >= 11:
        _frame_done = _read_u8(f)
        _first_frame = _read_u8(f)
        _reset_flag = _read_u8(f)

    # Sound (v8+) - need to skip, but size varies
    # For now, we'll skip to the renderer section by reading field by field
    _skip_sound_state(f, version)

    # Renderer (skip - 115200 bytes)
    f.read(RENDERER_SIZE)

    # RAM
    wram = f.read(WRAM_SIZE)
    non_io_ram0 = f.read(NON_IO_RAM0_SIZE)
    io_ports = f.read(IO_PORTS_SIZE)
    hram = f.read(HRAM_SIZE)
    non_io_ram1 = f.read(NON_IO_RAM1_SIZE)

    # Timer
    div = _read_u8(f)
    tima = _read_u8(f)
    tma = _read_u8(f)
    tac = _read_u8(f)
    div_counter = _read_u16(f)
    tima_counter = _read_u16(f)

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
    # Skip cart RAM if any (we read joypad from end)

    # v15+ has serial state before joypad, but joypad is still at the end
    # Read joypad from end of file
    f.seek(-2, 2)  # Seek to 2 bytes before end
    joypad_directional = _read_u8(f)
    joypad_standard = _read_u8(f)

    return PyBoyState(
        version=version,
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
        joypad_directional=joypad_directional,
        joypad_standard=joypad_standard,
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
    _write_u8(f, 0)  # bootrom_enabled = false
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

    # LCD timing/mode
    _write_u8(f, 0)  # lcd_cgb = 0 (DMG)
    _write_u8(f, 0)  # lcd_double_speed = 0 (DMG)
    _write_u64(f, state.lcd_clock)
    _write_u64(f, state.lcd_clock_target)
    _write_u8(f, state.next_stat_mode)

    # Scanline params
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
    _write_u8(f, state.tma)
    _write_u8(f, state.tac)
    _write_u16(f, state.div_counter)
    _write_u16(f, state.tima_counter)

    # Cartridge (minimal - no RAM banks for Tetris)
    _write_u16(f, state.rombank)
    _write_u8(f, state.rambank)
    _write_u8(f, state.rambank_enabled)
    _write_u8(f, state.memorymodel)

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

    # Timer internal state
    div_counter = int(backend._div_counter.numpy()[env_idx]) & 0xFFFF

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
        rombank=0,
        rambank=0,
        rambank_enabled=0,
        memorymodel=0,
        joypad_directional=0x0F,  # All released
        joypad_standard=0x0F,  # All released
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

    # CPU registers
    backend._a.numpy()[env_idx] = state.a
    backend._f.numpy()[env_idx] = state.f
    backend._b.numpy()[env_idx] = state.b
    backend._c.numpy()[env_idx] = state.c
    backend._d.numpy()[env_idx] = state.d
    backend._e.numpy()[env_idx] = state.e
    backend._h.numpy()[env_idx] = state.h
    backend._l.numpy()[env_idx] = state.l_reg
    backend._sp.numpy()[env_idx] = state.sp
    backend._pc.numpy()[env_idx] = state.pc
    backend._ime.numpy()[env_idx] = state.ime
    backend._halted.numpy()[env_idx] = state.halted

    # Memory
    mem = backend._mem.numpy()
    base = env_idx * MEM_SIZE

    # VRAM (0x8000-0x9FFF)
    mem[base + 0x8000 : base + 0x8000 + VRAM_SIZE] = np.frombuffer(
        state.vram, dtype=np.uint8
    )

    # OAM (0xFE00-0xFE9F)
    mem[base + 0xFE00 : base + 0xFE00 + OAM_SIZE] = np.frombuffer(
        state.oam, dtype=np.uint8
    )

    # WRAM (0xC000-0xDFFF)
    mem[base + 0xC000 : base + 0xC000 + WRAM_SIZE] = np.frombuffer(
        state.wram, dtype=np.uint8
    )

    # HRAM (0xFF80-0xFFFE)
    mem[base + 0xFF80 : base + 0xFF80 + HRAM_SIZE] = np.frombuffer(
        state.hram, dtype=np.uint8
    )

    # IO ports (0xFF00-0xFF4B)
    mem[base + 0xFF00 : base + 0xFF00 + IO_PORTS_SIZE] = np.frombuffer(
        state.io_ports, dtype=np.uint8
    )

    # Non-IO RAM regions
    mem[base + 0xFEA0 : base + 0xFEA0 + NON_IO_RAM0_SIZE] = np.frombuffer(
        state.non_io_ram0, dtype=np.uint8
    )
    mem[base + 0xFF4C : base + 0xFF4C + NON_IO_RAM1_SIZE] = np.frombuffer(
        state.non_io_ram1, dtype=np.uint8
    )

    # Key LCD registers (may override IO ports)
    mem[base + 0xFF40] = state.lcdc
    mem[base + 0xFF41] = state.stat
    mem[base + 0xFF42] = state.scy
    mem[base + 0xFF43] = state.scx
    mem[base + 0xFF44] = state.ly
    mem[base + 0xFF45] = state.lyc
    mem[base + 0xFF47] = state.bgp
    mem[base + 0xFF48] = state.obp0
    mem[base + 0xFF49] = state.obp1
    mem[base + 0xFF4A] = state.wy
    mem[base + 0xFF4B] = state.wx

    # Interrupts
    mem[base + 0xFF0F] = state.if_reg
    mem[base + 0xFFFF] = state.ie

    # Timer registers
    mem[base + 0xFF04] = state.div
    mem[base + 0xFF05] = state.tima
    mem[base + 0xFF06] = state.tma
    mem[base + 0xFF07] = state.tac

    # Timer internal state
    backend._div_counter.numpy()[env_idx] = state.div_counter

    # PPU state
    backend._ppu_ly.numpy()[env_idx] = state.ly
    backend._ppu_scanline_cycle.numpy()[env_idx] = state.lcd_clock % CYCLES_PER_SCANLINE

    # PPU window line counter (estimate from WY and LY)
    if state.ly >= state.wy:
        backend._ppu_window_line.numpy()[env_idx] = state.ly - state.wy
    else:
        backend._ppu_window_line.numpy()[env_idx] = 0

    # STAT previous state for edge triggering
    backend._ppu_stat_prev.numpy()[env_idx] = state.stat & 0x07
