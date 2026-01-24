# PyBoy State File Format

Binary format documentation for `.state` files produced by PyBoy.
All multi-byte integers are **little-endian**.

**Current version: 15** (but many existing files are v9)

## Version Differences Summary

| Feature | v9 | v12+ | v13+ | v15+ |
|---------|-----|------|------|------|
| CPU cycles (64-bit) | No | Yes | Yes | Yes |
| Timer last_cycles | No | Yes | Yes | Yes |
| Timer cycles_to_interrupt | No | No | Yes | Yes |
| LCD last_cycles | No | No | Yes | Yes |
| LCD clock/clock_target | Yes | Yes | Yes | Yes |
| Serial state | No | No | No | Yes |

## Overview (v9 DMG format - validated)

```
┌─────────────────────────────────────────┐
│ Header (1 byte)                         │  offset 0
├─────────────────────────────────────────┤
│ Boot/CGB flags (4 bytes)                │  offset 1
├─────────────────────────────────────────┤
│ CPU state (18 bytes)                    │  offset 5
├─────────────────────────────────────────┤
│ LCD/PPU state (~9.1KB)                  │  offset 23
│   - VRAM: 8192 bytes                    │
│   - OAM: 160 bytes                      │
│   - Registers: 11 bytes                 │
│   - Mode flags: 2 bytes                 │
│   - Timing: 17 bytes                    │
│   - Scanline params: 720 bytes          │
├─────────────────────────────────────────┤
│ Sound/APU state (138 bytes)             │  offset 9125
├─────────────────────────────────────────┤
│ Renderer state (115,200 bytes)          │  offset 9263
├─────────────────────────────────────────┤
│ RAM state (8,543 bytes)                 │  offset 124463
├─────────────────────────────────────────┤
│ Timer state (8 bytes)                   │  offset 133006
├─────────────────────────────────────────┤
│ Cartridge/MBC state (5 + RAM banks)     │  offset 133014
├─────────────────────────────────────────┤
│ Interaction/Joypad state (2 bytes)      │  offset -2 (end)
└─────────────────────────────────────────┘
```

---

## 1. Header

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0x00 | 1 | `state_version` | 9 for most existing files |

---

## 2. Boot/CGB Flags (v2+/v8+)

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0x01 | 1 | `bootrom_enabled` | 0 or 1 |
| 0x02 | 1 | `key1` | CGB speed switch (v8+) |
| 0x03 | 1 | `double_speed` | 0 = normal, 1 = double (v8+) |
| 0x04 | 1 | `cgb` | 0 = DMG, 1 = CGB mode (v8+) |

---

## 3. HDMA State (CGB only, v8+)

Only present if `cgb == 1`. Skip for DMG games like Tetris.

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| +0 | 1 | `hdma1` | Source high |
| +1 | 1 | `hdma2` | Source low |
| +2 | 1 | `hdma3` | Dest high |
| +3 | 1 | `hdma4` | Dest low |
| +4 | 1 | `hdma5` | Length/mode |
| +5 | 1 | `transfer_active` | 0 or 1 |
| +6 | 2 | `curr_src` | 16-bit LE |
| +8 | 2 | `curr_dst` | 16-bit LE |

---

## 4. CPU State (v9)

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| +0 | 1 | `A` | Accumulator |
| +1 | 1 | `F` | Flags (upper 4 bits: ZNHC) |
| +2 | 1 | `B` | |
| +3 | 1 | `C` | |
| +4 | 1 | `D` | |
| +5 | 1 | `E` | |
| +6 | 2 | `HL` | 16-bit LE |
| +8 | 2 | `SP` | 16-bit LE |
| +10 | 2 | `PC` | 16-bit LE |
| +12 | 1 | `interrupt_master_enable` | IME flag |
| +13 | 1 | `halted` | HALT state |
| +14 | 1 | `stopped` | STOP state |
| +15 | 1 | `interrupts_enabled_register` | IE (0xFFFF), v5+ |
| +16 | 1 | `interrupt_queued` | Pending interrupt, v8+ |
| +17 | 1 | `interrupts_flag_register` | IF (0xFF0F), v8+ |
| +18 | 8 | `cycles` | 64-bit LE, **v12+ only** |

**v9 Total: 18 bytes** (no cycles field)

---

## 5. LCD/PPU State (v9 DMG format)

### 5.1 Video RAM

| Size | Field | Notes |
|------|-------|-------|
| 8192 | `VRAM0` | 0x8000-0x9FFF |

### 5.2 OAM

| Size | Field | Notes |
|------|-------|-------|
| 160 | `OAM` | 0xFE00-0xFE9F (40 sprites × 4 bytes) |

### 5.3 Registers (11 bytes)

| Size | Field | Address |
|------|-------|---------|
| 1 | `LCDC` | 0xFF40 |
| 1 | `BGP` | 0xFF47 |
| 1 | `OBP0` | 0xFF48 |
| 1 | `OBP1` | 0xFF49 |
| 1 | `STAT.value` | 0xFF41 |
| 1 | `LY` | 0xFF44 |
| 1 | `LYC` | 0xFF45 |
| 1 | `SCY` | 0xFF42 |
| 1 | `SCX` | 0xFF43 |
| 1 | `WY` | 0xFF4A |
| 1 | `WX` | 0xFF4B |

### 5.4 Mode Flags (v8+, 2 bytes)

| Size | Field | Notes |
|------|-------|-------|
| 1 | `cgb` | 0 for DMG |
| 1 | `double_speed` | 0 for DMG |

### 5.5 Timing (v8+, 17 bytes)

| Size | Field | Notes |
|------|-------|-------|
| 8 | `clock` | 64-bit LE, cycles in current frame |
| 8 | `clock_target` | 64-bit LE |
| 1 | `next_stat_mode` | PPU mode (0-3) |

Note: `last_cycles` was added in v11+, not present in v9.

### 5.6 Scanline Parameters (720 bytes)

```
for scanline in 0..144:
    write 5 bytes of scanline parameters
```

These capture per-scanline register state for mid-frame effects.

### 5.7 CGB-only LCD Data (skip for DMG)

Only if `cgb == 1`:

| Size | Field | Notes |
|------|-------|-------|
| 8192 | `VRAM1` | Second VRAM bank |
| 1 | `VBK` | VRAM bank select |
| ... | Color palette registers | BCPS, BCPD, OCPS, OCPD |

**v9 DMG LCD total: 8192 + 160 + 11 + 2 + 17 + 720 = 9102 bytes**

---

## 6. Sound/APU State (v8+)

We **skip this section** for Tetris (no sound emulation needed).

### v9 Sound Size: 138 bytes (validated)

Structure for reference:

| Size | Field | Notes |
|------|-------|-------|
| 2 | `samples_in_buffer` | |
| 2 | `sample_rate` | |
| 2 | `buffer_max` | |
| 2 | `buffer_head_i` | |
| 2 | `samples_per_frame` | |
| 8 | `cycles_per_sample` | 64-bit |
| 8 | `cycles_target` | 64-bit |
| 8 | `cycles_512_target` | 64-bit |
| 8 | `sound_cycles` | 64-bit |
| 8 | `sound_last_cycles` | 64-bit |
| 2 | `div_apu_counter` | |
| 1 | `div_apu_value` | |
| 1 | `power` | |
| 1 | `sampling_disabled` | |
| 8 | panning bits | Left/right for 4 channels |
| 1 | `NR50` | |
| ~23 | Channel 1 (Sweep) | |
| ~15 | Channel 2 (Tone) | |
| ~26 | Channel 3 (Wave) | Includes 16-byte wave table |
| ~19 | Channel 4 (Noise) | |

**Skip 138 bytes for v9 DMG.**

---

## 7. Renderer State

### 7.1 Screen Buffer

```
for y in 0..144:
    for x in 0..160:
        write_32bit(pixel_color)  # 4 bytes LE
        write(attribute)          # 1 byte
```

| Size | Field | Notes |
|------|-------|-------|
| 115,200 | Screen buffer | 144 × 160 × 5 bytes |

---

## 8. RAM State

### 8.1 Internal RAM

| Size | Field | Notes |
|------|-------|-------|
| 8192 (DMG) or 32768 (CGB) | `internal_ram0` | WRAM |
| 96 | `non_io_internal_ram0` | 0xFEA0-0xFEFF |
| 76 | `io_ports` | 0xFF00-0xFF4B |
| 127 | `internal_ram1` | HRAM 0xFF80-0xFFFE |
| 52 | `non_io_internal_ram1` | 0xFF4C-0xFF7F |

**DMG total: ~8,543 bytes**
**CGB total: ~33,119 bytes**

---

## 9. Timer State (v5+)

| Size | Field | Version | Notes |
|------|-------|---------|-------|
| 1 | `DIV` | v5+ | 0xFF04 |
| 1 | `TIMA` | v5+ | 0xFF05 |
| 1 | `TMA` | v5+ | 0xFF06 |
| 1 | `TAC` | v5+ | 0xFF07 |
| 2 | `DIV_counter` | v5+ | 16-bit LE internal |
| 2 | `TIMA_counter` | v5+ | 16-bit LE internal |
| 8 | `last_cycles` | v12+ | Not in v9 |
| 8 | `_cycles_to_interrupt` | v13+ | Not in v9 |

**v9 Total: 8 bytes**

---

## 10. Cartridge/MBC State

### 10.1 Bank Registers

| Size | Field | Notes |
|------|-------|-------|
| 2 | `rombank_selected` | 16-bit LE |
| 1 | `rambank_selected` | |
| 1 | `rambank_enabled` | |
| 1 | `memorymodel` | MBC mode |

### 10.2 Cartridge RAM

For each RAM bank (0 to num_ram_banks):
```
write 8192 bytes of RAM bank data
```

**Tetris: No cartridge RAM (ROM-only)**
**Pokemon Red: 4 banks × 8KB = 32KB**

### 10.3 RTC State (MBC3 with RTC only)

If cartridge has RTC:
| Size | Field | Notes |
|------|-------|-------|
| 1 | `S` | Seconds |
| 1 | `M` | Minutes |
| 1 | `H` | Hours |
| 1 | `DL` | Day low |
| 1 | `DH` | Day high + flags |
| ... | Latched values | |
| 8 | `last_cycles` | 64-bit LE |

---

## 11. Interaction/Joypad State (v7+)

| Size | Field | Notes |
|------|-------|-------|
| 1 | `directional` | D-pad state (bits: Down/Up/Left/Right) |
| 1 | `standard` | Buttons (bits: Start/Select/B/A) |

Bit = 0 means pressed, bit = 1 means released.
Initial value: 0x0F (all released).

---

## 12. Serial State (v15+)

| Size | Field | Notes |
|------|-------|-------|
| 1 | `SB` | 0xFF01 |
| 1 | `SC` | 0xFF02 |
| 1 | `transfer_enabled` | |
| 1 | `internal_clock` | |
| 8 | `last_cycles` | 64-bit LE |
| 8 | `_cycles_to_interrupt` | 64-bit LE |
| 8 | `clock` | 64-bit LE |
| 8 | `clock_target` | 64-bit LE |

**Total: ~36 bytes**

---

## Tetris-Specific Notes

For loading a Tetris state into gbxcule:

### What we need:
- CPU registers (A, F, B, C, D, E, HL, SP, PC, IME, halted)
- Memory (VRAM, OAM, WRAM, HRAM, IO ports)
- PPU state (LY, STAT, scanline cycle position via lcd_clock)
- Timer state (DIV, TIMA, TMA, TAC, div_counter)
- Joypad state

### What we can ignore:
- Sound/APU (not emulated) - skip 138 bytes
- Renderer screen buffer (we'll re-render) - skip 115200 bytes
- CGB-specific data (Tetris is DMG-only)
- Cartridge RAM (Tetris has none)
- Serial (v9 doesn't have it anyway)
- Scanline parameters (we re-render from scratch)

### Expected Tetris v9 state file size:
- Header + flags: 5 bytes
- CPU: 18 bytes
- LCD/PPU: 9,102 bytes
- Sound: 138 bytes (skip)
- Renderer: 115,200 bytes (skip)
- RAM: 8,543 bytes
- Timer: 8 bytes
- Cartridge: 5 bytes (no RAM for Tetris)
- Joypad: 2 bytes

**Total: ~133KB** for Tetris (ROM-only game)

---

## Python Implementation (v9 format)

```python
"""PyBoy state file loader for gbxcule Warp backends."""

import struct
from dataclasses import dataclass
from typing import BinaryIO


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
    l: int
    sp: int
    pc: int
    ime: int
    halted: int
    ie: int  # Interrupt Enable (0xFFFF)
    if_reg: int  # Interrupt Flag (0xFF0F)

    # LCD state
    vram: bytes  # 8192 bytes
    oam: bytes   # 160 bytes
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

    # RAM
    wram: bytes       # 8192 bytes (0xC000-0xDFFF)
    hram: bytes       # 127 bytes (0xFF80-0xFFFE)
    io_ports: bytes   # 76 bytes (0xFF00-0xFF4B)

    # Timer
    div: int
    tima: int
    tma: int
    tac: int
    div_counter: int

    # Joypad
    joypad_directional: int  # D-pad bits
    joypad_standard: int     # Button bits


def read_u8(f: BinaryIO) -> int:
    return struct.unpack('B', f.read(1))[0]


def read_u16(f: BinaryIO) -> int:
    return struct.unpack('<H', f.read(2))[0]


def read_u64(f: BinaryIO) -> int:
    return struct.unpack('<Q', f.read(8))[0]


def load_pyboy_state(path: str) -> PyBoyState:
    """Load a PyBoy v9 state file."""
    with open(path, 'rb') as f:
        # Header
        version = read_u8(f)
        if version != 9:
            raise ValueError(f"Unsupported state version {version}, expected 9")

        bootrom = read_u8(f)
        key1 = read_u8(f)
        double_speed = read_u8(f)
        cgb = read_u8(f)
        if cgb != 0:
            raise ValueError("CGB mode not supported")

        # CPU
        a = read_u8(f)
        flags = read_u8(f)
        b = read_u8(f)
        c = read_u8(f)
        d = read_u8(f)
        e = read_u8(f)
        hl = read_u16(f)
        sp = read_u16(f)
        pc = read_u16(f)
        ime = read_u8(f)
        halted = read_u8(f)
        stopped = read_u8(f)
        ie = read_u8(f)
        interrupt_queued = read_u8(f)
        if_reg = read_u8(f)

        h = (hl >> 8) & 0xFF
        l = hl & 0xFF

        # LCD
        vram = f.read(8192)
        oam = f.read(160)
        lcdc = read_u8(f)
        bgp = read_u8(f)
        obp0 = read_u8(f)
        obp1 = read_u8(f)
        stat = read_u8(f)
        ly = read_u8(f)
        lyc = read_u8(f)
        scy = read_u8(f)
        scx = read_u8(f)
        wy = read_u8(f)
        wx = read_u8(f)

        # LCD timing/mode
        lcd_cgb = read_u8(f)
        lcd_double_speed = read_u8(f)
        lcd_clock = read_u64(f)
        lcd_clock_target = read_u64(f)
        next_stat_mode = read_u8(f)

        # Scanline params (skip - we'll re-render)
        f.read(720)

        # Sound (skip - 138 bytes for v9)
        f.read(138)

        # Renderer (skip - 115200 bytes)
        f.read(115200)

        # RAM
        wram = f.read(8192)
        non_io_ram0 = f.read(96)  # 0xFEA0-0xFEFF
        io_ports = f.read(76)
        hram = f.read(127)
        non_io_ram1 = f.read(52)  # 0xFF4C-0xFF7F

        # Timer
        div = read_u8(f)
        tima = read_u8(f)
        tma = read_u8(f)
        tac = read_u8(f)
        div_counter = read_u16(f)
        tima_counter = read_u16(f)

        # Cartridge (skip - we have ROM loaded separately)
        rombank = read_u16(f)
        rambank = read_u8(f)
        rambank_enabled = read_u8(f)
        memorymodel = read_u8(f)
        # Skip cart RAM if any

        # Read joypad from end of file
        f.seek(-2, 2)  # Seek to 2 bytes before end
        joypad_directional = read_u8(f)
        joypad_standard = read_u8(f)

        return PyBoyState(
            version=version,
            a=a, f=flags, b=b, c=c, d=d, e=e, h=h, l=l,
            sp=sp, pc=pc, ime=ime, halted=halted,
            ie=ie, if_reg=if_reg,
            vram=vram, oam=oam,
            lcdc=lcdc, stat=stat, ly=ly, lyc=lyc,
            scy=scy, scx=scx, wy=wy, wx=wx,
            bgp=bgp, obp0=obp0, obp1=obp1,
            lcd_clock=lcd_clock, lcd_clock_target=lcd_clock_target,
            next_stat_mode=next_stat_mode,
            wram=wram, hram=hram, io_ports=io_ports,
            div=div, tima=tima, tma=tma, tac=tac,
            div_counter=div_counter,
            joypad_directional=joypad_directional,
            joypad_standard=joypad_standard,
        )


def apply_state_to_warp_backend(state: PyBoyState, backend, env_idx: int = 0):
    """Apply a PyBoy state to a Warp backend."""
    import numpy as np

    # CPU registers
    backend._a.numpy()[env_idx] = state.a
    backend._f.numpy()[env_idx] = state.f
    backend._b.numpy()[env_idx] = state.b
    backend._c.numpy()[env_idx] = state.c
    backend._d.numpy()[env_idx] = state.d
    backend._e.numpy()[env_idx] = state.e
    backend._h.numpy()[env_idx] = state.h
    backend._l.numpy()[env_idx] = state.l
    backend._sp.numpy()[env_idx] = state.sp
    backend._pc.numpy()[env_idx] = state.pc
    backend._ime.numpy()[env_idx] = state.ime
    backend._halted.numpy()[env_idx] = state.halted

    # Memory - VRAM at 0x8000
    mem = backend._mem.numpy()
    base = env_idx * 0x10000  # MEM_SIZE

    # VRAM (0x8000-0x9FFF)
    mem[base + 0x8000:base + 0x8000 + 8192] = np.frombuffer(state.vram, dtype=np.uint8)

    # OAM (0xFE00-0xFE9F)
    mem[base + 0xFE00:base + 0xFE00 + 160] = np.frombuffer(state.oam, dtype=np.uint8)

    # WRAM (0xC000-0xDFFF)
    mem[base + 0xC000:base + 0xC000 + 8192] = np.frombuffer(state.wram, dtype=np.uint8)

    # HRAM (0xFF80-0xFFFE)
    mem[base + 0xFF80:base + 0xFF80 + 127] = np.frombuffer(state.hram, dtype=np.uint8)

    # IO ports (0xFF00-0xFF4B)
    mem[base + 0xFF00:base + 0xFF00 + 76] = np.frombuffer(state.io_ports, dtype=np.uint8)

    # Key LCD registers
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

    # PPU state (approximate from LCD clock)
    # lcd_clock counts cycles since frame start
    # 456 cycles per scanline, 154 scanlines per frame
    cycles_per_scanline = 456
    backend._ppu_ly.numpy()[env_idx] = state.ly
    backend._ppu_scanline_cycle.numpy()[env_idx] = state.lcd_clock % cycles_per_scanline
```

---

## References

- PyBoy source: https://github.com/Baekalfen/PyBoy
- `pyboy/core/mb.py` - Motherboard save/load
- `pyboy/core/cpu.py` - CPU state
- `pyboy/core/lcd.py` - LCD/PPU state
- `pyboy/core/ram.py` - RAM regions
- `pyboy/core/timer.py` - Timer state
- `pyboy/utils.py` - IntIOWrapper, STATE_VERSION
