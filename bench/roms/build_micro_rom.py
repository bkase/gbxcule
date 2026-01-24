"""Generate valid micro-ROMs for testing.

These are minimal, deterministic ROMs for correctness and performance testing.
They are license-safe (generated, not copyrighted content).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

# Default output directory
DEFAULT_OUT_DIR = Path(__file__).parent / "out"

# Nintendo logo (required for boot ROM validation)
# This is the exact sequence the Game Boy checks at 0x0104-0x0133
NINTENDO_LOGO = bytes.fromhex(
    "CEED6666CC0D000B03730083000C000D0008111F8889000E"
    "DCCC6EE6DDDDD999BBBB67636E0EECCCDDDC999FBBB9333E"
)


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of bytes, returning hex string."""
    return hashlib.sha256(data).hexdigest()


def compute_header_checksum(rom: bytes) -> int:
    """Compute the header checksum (byte at 0x014D).

    Per Pan Docs: x = 0; for i in 0x0134..0x014C: x = x - rom[i] - 1
    """
    checksum = 0
    for i in range(0x0134, 0x014D):
        checksum = (checksum - rom[i] - 1) & 0xFF
    return checksum


def compute_global_checksum(rom: bytes) -> int:
    """Compute the global checksum (bytes at 0x014E-0x014F).

    Per Pan Docs: sum of all bytes except the checksum bytes themselves.
    Returns a 16-bit value (big-endian in ROM).
    """
    total = 0
    for i, byte in enumerate(rom):
        if i not in (0x014E, 0x014F):
            total = (total + byte) & 0xFFFF
    return total


def build_rom(title: str, program: bytes) -> bytes:
    """Build a valid Game Boy ROM with the given title and program code.

    Args:
        title: ROM title (max 11 characters, uppercase ASCII).
        program: Machine code to place at 0x0150.

    Returns:
        Complete 32KB ROM bytes with valid checksums.
    """
    # ROM is 32KB (rom_size_code = 0x00)
    rom = bytearray(32 * 1024)

    # Entry point at 0x0100: NOP; JP 0x0150
    rom[0x0100] = 0x00  # NOP
    rom[0x0101] = 0xC3  # JP
    rom[0x0102] = 0x50  # low byte of 0x0150
    rom[0x0103] = 0x01  # high byte of 0x0150

    # Nintendo logo at 0x0104-0x0133
    rom[0x0104 : 0x0104 + len(NINTENDO_LOGO)] = NINTENDO_LOGO

    # Title at 0x0134-0x0143 (padded with zeros)
    title_bytes = title.upper().encode("ascii")[:11]
    rom[0x0134 : 0x0134 + len(title_bytes)] = title_bytes

    # Cartridge type at 0x0147: ROM ONLY (0x00)
    rom[0x0147] = 0x00

    # ROM size at 0x0148: 32KB (code 0x00)
    rom[0x0148] = 0x00

    # RAM size at 0x0149: No RAM (0x00)
    rom[0x0149] = 0x00

    # Destination code at 0x014A: Non-Japanese (0x01)
    rom[0x014A] = 0x01

    # Old licensee code at 0x014B: 0x00
    rom[0x014B] = 0x00

    # ROM version at 0x014C: 0x00
    rom[0x014C] = 0x00

    # Program code at 0x0150
    rom[0x0150 : 0x0150 + len(program)] = program

    # Header checksum at 0x014D (must be computed after title/metadata)
    rom[0x014D] = compute_header_checksum(rom)

    # Global checksum at 0x014E-0x014F (big-endian)
    global_checksum = compute_global_checksum(rom)
    rom[0x014E] = (global_checksum >> 8) & 0xFF
    rom[0x014F] = global_checksum & 0xFF

    return bytes(rom)


def apply_rom_patches(rom: bytes, patches: dict[int, bytes]) -> bytes:
    """Apply ROM patches at absolute addresses and recompute checksums."""
    data = bytearray(rom)
    for addr, payload in patches.items():
        if addr < 0 or addr + len(payload) > len(data):
            raise ValueError(f"patch out of range: {addr:04X}")
        data[addr : addr + len(payload)] = payload
    data[0x014D] = compute_header_checksum(data)
    global_checksum = compute_global_checksum(data)
    data[0x014E] = (global_checksum >> 8) & 0xFF
    data[0x014F] = global_checksum & 0xFF
    return bytes(data)


def build_alu_loop() -> bytes:
    """Build ALU_LOOP.gb - a tight ALU-heavy loop.

    This ROM executes a deterministic loop of ALU operations.
    Good for testing CPU correctness and measuring ALU throughput.

    Assembly:
        LD A, 0       ; A = 0
        LD B, 0       ; B = 0
    loop:
        INC A         ; A++
        ADD A, B      ; A += B
        INC B         ; B++
        JR loop       ; infinite loop (-5)
    """
    program = bytes(
        [
            0x3E,
            0x00,  # LD A, 0
            0x06,
            0x00,  # LD B, 0
            # loop (offset 4):
            0x3C,  # INC A
            0x80,  # ADD A, B
            0x04,  # INC B
            0x18,
            0xFB,  # JR -5 (back to INC A)
        ]
    )
    return build_rom("ALU_LOOP", program)


def build_mem_rwb() -> bytes:
    """Build MEM_RWB.gb - WRAM read/write benchmark.

    This ROM performs memory loads and stores in a loop over 0xC000-0xC0FF.
    Good for testing memory correctness and measuring memory throughput.

    Assembly:
        LD HL, 0xC000  ; HL points to WRAM start
        LD A, 0        ; A = 0
    loop:
        LD (HL), A     ; Write A to (HL)
        INC A          ; A++
        LD B, (HL)     ; Read (HL) into B
        INC HL         ; HL++ (wraps within WRAM range)
        JR loop        ; infinite loop (-6)
    """
    program = bytes(
        [
            0x21,
            0x00,
            0xC0,  # LD HL, 0xC000
            0x3E,
            0x00,  # LD A, 0
            # loop (offset 5):
            0x77,  # LD (HL), A
            0x3C,  # INC A
            0x46,  # LD B, (HL)
            0x23,  # INC HL
            0x18,
            0xFA,  # JR -6 (back to LD (HL), A)
        ]
    )
    return build_rom("MEM_RWB", program)


def build_serial_hello() -> bytes:
    """Build SERIAL_HELLO.gb - serial output smoke test.

    This ROM writes "OK" to the serial port using SB/SC.

    Assembly:
        LD HL, 0xFF01
        LD A, 'O'
        LD (HL), A       ; SB
        INC HL           ; SC
        LD A, 0x81
        LD (HL), A       ; trigger transfer
        LD HL, 0xFF01
        LD A, 'K'
        LD (HL), A
        INC HL
        LD A, 0x81
        LD (HL), A
    loop:
        JR loop
    """
    program = bytes(
        [
            0x21,
            0x01,
            0xFF,  # LD HL, 0xFF01
            0x3E,
            0x4F,  # LD A, 'O'
            0x77,  # LD (HL), A
            0x23,  # INC HL
            0x3E,
            0x81,  # LD A, 0x81
            0x77,  # LD (HL), A
            0x21,
            0x01,
            0xFF,  # LD HL, 0xFF01
            0x3E,
            0x4B,  # LD A, 'K'
            0x77,  # LD (HL), A
            0x23,  # INC HL
            0x3E,
            0x81,  # LD A, 0x81
            0x77,  # LD (HL), A
            0x18,
            0xFE,  # JR -2
        ]
    )
    return build_rom("SERIAL_HELLO", program)


def build_dma_oam_copy() -> bytes:
    """Build DMA_OAM_COPY.gb - OAM DMA copy correctness ROM.

    This ROM fills 0xC000..0xC09F with incrementing bytes (0..159),
    triggers OAM DMA from 0xC000 via 0xFF46, then loops forever.

    Assembly:
        LD HL, 0xC000
        LD A, 0x00
        LD B, 0xA0
    loop:
        LD (HL), A
        INC A
        INC HL
        DEC B
        JR NZ, loop
        LD A, 0xC0
        LD (0xFF46), A
    halt:
        JR halt
    """
    program = bytes(
        [
            0x21,
            0x00,
            0xC0,  # LD HL, 0xC000
            0x3E,
            0x00,  # LD A, 0x00
            0x06,
            0xA0,  # LD B, 0xA0 (160 bytes)
            # loop:
            0x77,  # LD (HL), A
            0x3C,  # INC A
            0x23,  # INC HL
            0x05,  # DEC B
            0x20,
            0xFA,  # JR NZ, -6
            0x3E,
            0xC0,  # LD A, 0xC0
            0xEA,
            0x46,
            0xFF,  # LD (0xFF46), A
            # halt:
            0x18,
            0xFE,  # JR -2
        ]
    )
    return build_rom("DMA_OAM_COPY", program)


def build_timer_div_basic() -> bytes:
    """Build TIMER_DIV_BASIC.gb - DIV/TIMA sampling + DIV reset glitch.

    This ROM enables the timer (TAC=0x04), samples DIV/TIMA into WRAM,
    and uses the slow clock for stability.
    """
    program = bytes(
        [
            0x3E,
            0x00,  # LD A, 0x00
            0xEA,
            0x06,
            0xFF,  # LD (0xFF06), A ; TMA = 0
            0xEA,
            0x05,
            0xFF,  # LD (0xFF05), A ; TIMA = 0
            0xEA,
            0x04,
            0xFF,  # LD (0xFF04), A ; DIV reset
            0x3E,
            0x04,  # LD A, 0x04 ; TAC enable, freq=00
            0xEA,
            0x07,
            0xFF,  # LD (0xFF07), A
            # loop:
            0xFA,
            0x04,
            0xFF,  # LD A, (0xFF04)
            0xEA,
            0x00,
            0xC0,  # LD (0xC000), A
            0xFA,
            0x05,
            0xFF,  # LD A, (0xFF05)
            0xEA,
            0x01,
            0xC0,  # LD (0xC001), A
            0x18,
            0xF2,  # JR -14
        ]
    )
    return build_rom("TIMER_DIV", program)


def build_timer_irq_halt() -> bytes:
    """Build TIMER_IRQ_HALT.gb - timer interrupt + HALT wake check."""
    program = bytes(
        [
            0x31,
            0x00,
            0xC1,  # LD SP, 0xC100
            0x3E,
            0x00,  # LD A, 0x00
            0xEA,
            0x0F,
            0xFF,  # LD (0xFF0F), A ; IF = 0
            0x3E,
            0x04,  # LD A, 0x04
            0xEA,
            0xFF,
            0xFF,  # LD (0xFFFF), A ; IE = timer
            0x3E,
            0xAB,  # LD A, 0xAB
            0xEA,
            0x06,
            0xFF,  # LD (0xFF06), A ; TMA
            0x3E,
            0xFE,  # LD A, 0xFE
            0xEA,
            0x05,
            0xFF,  # LD (0xFF05), A ; TIMA
            0x3E,
            0x05,  # LD A, 0x05 ; TAC enable, freq=01
            0xEA,
            0x07,
            0xFF,  # LD (0xFF07), A
            0xFB,  # EI
            0x76,  # HALT
            0x18,
            0xFE,  # JR -2
        ]
    )
    rom = build_rom("TIMER_IRQ", program)
    isr = bytes(
        [
            0x3E,
            0x42,  # LD A, 0x42
            0xEA,
            0x10,
            0xC0,  # LD (0xC010), A
            0xFA,
            0x11,
            0xC0,  # LD A, (0xC011)
            0x3C,  # INC A
            0xEA,
            0x11,
            0xC0,  # LD (0xC011), A
            0xD9,  # RETI
        ]
    )
    return apply_rom_patches(rom, {0x0050: isr})


def build_ei_delay() -> bytes:
    """Build EI_DELAY.gb - EI delay returns correct PC on IRQ."""
    program = bytes(
        [
            0x31,
            0x00,
            0xC1,  # LD SP, 0xC100
            0x3E,
            0x01,  # LD A, 0x01
            0xEA,
            0xFF,
            0xFF,  # LD (0xFFFF), A ; IE = VBlank
            0xEA,
            0x0F,
            0xFF,  # LD (0xFF0F), A ; IF = VBlank
            0xFB,  # EI
            0x00,  # NOP (next instruction)
            0x3E,
            0x42,  # LD A, 0x42
            0xEA,
            0x00,
            0xC0,  # LD (0xC000), A
            0x18,
            0xFE,  # JR -2
        ]
    )
    rom = build_rom("EI_DELAY", program)
    isr = bytes(
        [
            0xFA,
            0xFE,
            0xC0,  # LD A, (0xC0FE)
            0xEA,
            0x10,
            0xC0,  # LD (0xC010), A
            0xFA,
            0xFF,
            0xC0,  # LD A, (0xC0FF)
            0xEA,
            0x11,
            0xC0,  # LD (0xC011), A
            0xD9,  # RETI
        ]
    )
    return apply_rom_patches(rom, {0x0040: isr})


def build_joy_diverge_persist() -> bytes:
    """Build JOY_DIVERGE_PERSIST.gb - JOYP-driven divergence benchmark.

    This ROM reads JOYP (0xFF00) each outer loop, updates a persistent
    mode byte in WRAM, and branches into one of four inner loops. Each
    outer iteration writes a deterministic signature to 0xC000:0xC010.
    """
    code = bytearray()
    labels: dict[str, int] = {}
    jr_fixups: list[tuple[int, str]] = []
    jp_fixups: list[tuple[int, str]] = []
    base_addr = 0x0150

    def label(name: str) -> None:
        labels[name] = len(code)

    def emit(*bytes_: int) -> None:
        code.extend(bytes_)

    def emit_jr(opcode: int, target: str) -> None:
        pc = len(code)
        emit(opcode, 0x00)
        jr_fixups.append((pc, target))

    def emit_jp(target: str) -> None:
        pc = len(code)
        emit(0xC3, 0x00, 0x00)  # JP a16
        jp_fixups.append((pc, target))

    def ld_a_d8(val: int) -> None:
        emit(0x3E, val & 0xFF)

    def ld_b_d8(val: int) -> None:
        emit(0x06, val & 0xFF)

    def ld_hl_d16(addr: int) -> None:
        emit(0x21, addr & 0xFF, (addr >> 8) & 0xFF)

    def ld_a_hl() -> None:
        emit(0x7E)

    def ld_b_hl() -> None:
        emit(0x46)

    def ld_hl_a() -> None:
        emit(0x77)

    def inc_a() -> None:
        emit(0x3C)

    def inc_hl() -> None:
        emit(0x23)

    def dec_b() -> None:
        emit(0x05)

    def add_a_b() -> None:
        emit(0x80)

    def and_d8(val: int) -> None:
        emit(0xE6, val & 0xFF)

    # --- Program ---
    label("outer")
    ld_hl_d16(0xFF00)
    ld_a_d8(0x20)
    ld_hl_a()
    ld_a_hl()
    and_d8(0x0F)
    ld_hl_d16(0xC100)
    ld_b_hl()
    add_a_b()
    and_d8(0x03)
    ld_hl_a()
    ld_b_hl()
    dec_b()
    emit_jr(0x28, "stub1")  # JR Z
    dec_b()
    emit_jr(0x28, "stub2")  # JR Z
    dec_b()
    emit_jr(0x28, "stub3")  # JR Z
    emit_jr(0x18, "stub0")  # JR

    label("stub0")
    emit_jp("loop0")
    label("stub1")
    emit_jp("loop1")
    label("stub2")
    emit_jp("loop2")
    label("stub3")
    emit_jp("loop3")

    label("loop0")
    ld_b_d8(0x10)
    label("loop0_inner")
    inc_a()
    add_a_b()
    dec_b()
    emit_jr(0x20, "loop0_inner")  # JR NZ
    emit_jp("write_sig")

    label("loop1")
    ld_hl_d16(0xC000)
    ld_b_d8(0x10)
    label("loop1_inner")
    ld_hl_a()
    inc_a()
    inc_hl()
    dec_b()
    emit_jr(0x20, "loop1_inner")  # JR NZ
    emit_jp("write_sig")

    label("loop2")
    ld_hl_d16(0xC000)
    ld_b_d8(0x10)
    label("loop2_inner")
    ld_hl_a()
    inc_a()
    for _ in range(5):
        inc_hl()
    dec_b()
    emit_jr(0x20, "loop2_inner")  # JR NZ
    emit_jp("write_sig")

    label("loop3")
    ld_b_d8(0x10)
    label("loop3_inner")
    inc_a()
    and_d8(0x01)
    emit_jr(0x28, "loop3_skip")  # JR Z
    inc_a()
    label("loop3_skip")
    dec_b()
    emit_jr(0x20, "loop3_inner")  # JR NZ
    emit_jp("write_sig")

    label("write_sig")
    ld_hl_d16(0xC000)
    ld_b_d8(0x10)
    label("sig_loop")
    ld_hl_a()
    inc_a()
    inc_hl()
    dec_b()
    emit_jr(0x20, "sig_loop")  # JR NZ
    emit_jp("outer")

    # Resolve JR fixups
    for pc, target in jr_fixups:
        if target not in labels:
            raise ValueError(f"Unknown JR label: {target}")
        offset = labels[target] - (pc + 2)
        if offset < -128 or offset > 127:
            raise ValueError(f"JR offset out of range for {target}: {offset}")
        code[pc + 1] = offset & 0xFF

    # Resolve JP fixups (absolute addresses)
    for pc, target in jp_fixups:
        if target not in labels:
            raise ValueError(f"Unknown JP label: {target}")
        addr = base_addr + labels[target]
        code[pc + 1] = addr & 0xFF
        code[pc + 2] = (addr >> 8) & 0xFF

    return build_rom("JOY_PERSIST", bytes(code))


def build_loads_basic() -> bytes:
    """Build LOADS_BASIC.gb - load/store instruction coverage loop.

    Exercises:
    - LD r,r (register copies)
    - LD r,d8 and LD rr,d16
    - LD (HL),r and LD r,(HL)
    - LDI/LDD variants
    - LD (BC)/(DE) and LD A,(BC)/(DE)
    - LD (a16),A and LD A,(a16)
    - LDH (a8),A and LDH A,(a8)
    - LD (C),A and LD A,(C)
    - LD (a16),SP
    """
    program = bytes(
        [
            0x3E,
            0x12,  # LD A,0x12
            0x47,  # LD B,A
            0x48,  # LD C,B
            0x51,  # LD D,C
            0x5A,  # LD E,D
            0x63,  # LD H,E
            0x6C,  # LD L,H
            0x7D,  # LD A,L
            0x21,
            0x00,
            0xC0,  # LD HL,0xC000
            0x01,
            0x10,
            0xC0,  # LD BC,0xC010
            0x11,
            0x20,
            0xC0,  # LD DE,0xC020
            0x31,
            0xF0,
            0xC0,  # LD SP,0xC0F0
            0x77,  # LD (HL),A
            0x22,  # LDI (HL),A
            0x3E,
            0x34,  # LD A,0x34
            0x77,  # LD (HL),A
            0x32,  # LDD (HL),A
            0x7E,  # LD A,(HL)
            0x02,  # LD (BC),A
            0x0A,  # LD A,(BC)
            0x12,  # LD (DE),A
            0x1A,  # LD A,(DE)
            0x0E,
            0x80,  # LD C,0x80
            0xE2,  # LD (C),A
            0xF2,  # LD A,(C)
            0xE0,
            0x81,  # LDH (0x81),A
            0xF0,
            0x81,  # LDH A,(0x81)
            0xEA,
            0x30,
            0xC0,  # LD (0xC030),A
            0xFA,
            0x30,
            0xC0,  # LD A,(0xC030)
            0x08,
            0x40,
            0xC0,  # LD (0xC040),SP
            0x21,
            0x40,
            0xC0,  # LD HL,0xC040
            0x2A,  # LDI A,(HL)
            0x7E,  # LD A,(HL)
            0xC3,
            0x50,
            0x01,  # JP 0x0150
        ]
    )
    return build_rom("LOADS_BASIC", program)


def build_alu_flags() -> bytes:
    """Build ALU_FLAGS.gb - ALU/flags coverage loop."""
    program = bytes(
        [
            0x21,
            0x00,
            0xC0,  # LD HL,0xC000
            0x3E,
            0x0F,  # LD A,0x0F
            0x77,  # LD (HL),A
            0x3E,
            0x01,  # LD A,0x01
            0x06,
            0x10,  # LD B,0x10
            0x80,  # ADD A,B
            0x88,  # ADC A,B
            0x86,  # ADD A,(HL)
            0x8E,  # ADC A,(HL)
            0xC6,
            0x01,  # ADD A,0x01
            0x37,  # SCF
            0xCE,
            0x01,  # ADC A,0x01
            0x90,  # SUB B
            0x37,  # SCF
            0x98,  # SBC A,B
            0x96,  # SUB (HL)
            0x37,  # SCF
            0x9E,  # SBC A,(HL)
            0xD6,
            0x01,  # SUB 0x01
            0x37,  # SCF
            0xDE,
            0x01,  # SBC A,0x01
            0xA0,  # AND B
            0xA6,  # AND (HL)
            0xE6,
            0x0F,  # AND 0x0F
            0xB0,  # OR B
            0xB6,  # OR (HL)
            0xF6,
            0x0F,  # OR 0x0F
            0xA8,  # XOR B
            0xAE,  # XOR (HL)
            0xEE,
            0x0F,  # XOR 0x0F
            0xB8,  # CP B
            0xBE,  # CP (HL)
            0xFE,
            0x0E,  # CP 0x0E
            0x3C,  # INC A
            0x3D,  # DEC A
            0x34,  # INC (HL)
            0x35,  # DEC (HL)
            0x3E,
            0x09,  # LD A,0x09
            0xC6,
            0x01,  # ADD A,0x01
            0x27,  # DAA
            0x2F,  # CPL
            0x37,  # SCF
            0x3F,  # CCF
            0xC3,
            0x50,
            0x01,  # JP 0x0150
        ]
    )
    return build_rom("ALU_FLAGS", program)


def build_alu16_sp() -> bytes:
    """Build ALU16_SP.gb - 16-bit ALU + SP instruction coverage loop."""
    program = bytes(
        [
            0x21,
            0x00,
            0x10,  # LD HL,0x1000
            0x01,
            0x34,
            0x12,  # LD BC,0x1234
            0x11,
            0x78,
            0x56,  # LD DE,0x5678
            0x31,
            0xF0,
            0xFF,  # LD SP,0xFFF0
            0x09,  # ADD HL,BC
            0x19,  # ADD HL,DE
            0x29,  # ADD HL,HL
            0x39,  # ADD HL,SP
            0x03,  # INC BC
            0x0B,  # DEC BC
            0x13,  # INC DE
            0x1B,  # DEC DE
            0x23,  # INC HL
            0x2B,  # DEC HL
            0x33,  # INC SP
            0x3B,  # DEC SP
            0xE8,
            0xFE,  # ADD SP,-2
            0xF8,
            0x05,  # LD HL,SP+5
            0xC3,
            0x50,
            0x01,  # JP 0x0150
        ]
    )
    return build_rom("ALU16_SP", program)


def build_flow_stack() -> bytes:
    """Build FLOW_STACK.gb - control flow + stack opcode coverage loop."""
    program = bytes(
        [
            0x31,
            0xF0,
            0xFF,  # LD SP,0xFFF0
            0x21,
            0x84,
            0x01,  # LD HL,0x0184 (JP target)
            0x01,
            0x34,
            0x12,  # LD BC,0x1234
            0xC5,  # PUSH BC
            0xD1,  # POP DE
            0xAF,  # XOR A (Z=1, C=0)
            0xC4,
            0x7A,
            0x01,  # CALL NZ,0x017A (not taken)
            0xCC,
            0x7C,
            0x01,  # CALL Z,0x017C (taken)
            0x20,
            0x02,  # JR NZ,+2 (not taken)
            0x28,
            0x01,  # JR Z,+1 (taken)
            0x00,  # NOP (skipped if JR Z taken)
            0x3E,
            0x01,  # LD A,0x01 (Z=0)
            0xB7,  # OR A
            0x30,
            0x03,  # JR NC,+3 (taken)
            0x38,
            0x04,  # JR C,+4 (not taken)
            0x00,  # NOP
            0xC4,
            0x7E,
            0x01,  # CALL NZ,0x017E (taken)
            0x37,  # SCF (C=1)
            0xD4,
            0x80,
            0x01,  # CALL NC,0x0180 (not taken)
            0xDC,
            0x82,
            0x01,  # CALL C,0x0182 (taken)
            0xE9,  # JP (HL)
            0xC0,  # RET NZ
            0xC9,  # RET
            0xC8,  # RET Z
            0xC9,  # RET
            0xC0,  # RET NZ
            0xC9,  # RET
            0xD0,  # RET NC
            0xC9,  # RET
            0xD8,  # RET C
            0xC9,  # RET
            0xAF,  # XOR A (Z=1)
            0xC2,
            0x98,
            0x01,  # JP NZ,0x0198 (not taken)
            0xCA,
            0x8E,
            0x01,  # JP Z,0x018E (taken)
            0x00,  # NOP
            0x00,  # NOP
            0x00,  # NOP
            0x37,  # SCF (C=1)
            0xD2,
            0x98,
            0x01,  # JP NC,0x0198 (not taken)
            0xDA,
            0x9B,
            0x01,  # JP C,0x019B (taken)
            0x00,  # NOP
            0x00,  # NOP
            0x00,  # NOP
            0xC3,
            0x50,
            0x01,  # JP 0x0150
            0xC3,
            0x50,
            0x01,  # JP 0x0150
        ]
    )
    return build_rom("FLOW_STACK", program)


def build_cb_bitops() -> bytes:
    """Build CB_BITOPS.gb - rotates/shifts + CB bit operations coverage loop."""
    program = bytes(
        [
            0x3E,
            0x81,  # LD A, 0x81
            0x06,
            0x01,  # LD B, 0x01
            0x0E,
            0x80,  # LD C, 0x80
            0x16,
            0x3C,  # LD D, 0x3C
            0x1E,
            0xF0,  # LD E, 0xF0
            0x26,
            0xC0,  # LD H, 0xC0
            0x2E,
            0x00,  # LD L, 0x00
            0x36,
            0x55,  # LD (HL), 0x55
            0x07,  # RLCA
            0x17,  # RLA
            0x0F,  # RRCA
            0x1F,  # RRA
            0xCB,
            0x00,  # RLC B
            0xCB,
            0x09,  # RRC C
            0xCB,
            0x12,  # RL D
            0xCB,
            0x1B,  # RR E
            0xCB,
            0x20,  # SLA B
            0xCB,
            0x29,  # SRA C
            0xCB,
            0x37,  # SWAP A
            0xCB,
            0x3A,  # SRL D
            0xCB,
            0x06,  # RLC (HL)
            0xCB,
            0x0E,  # RRC (HL)
            0xCB,
            0x16,  # RL (HL)
            0xCB,
            0x1E,  # RR (HL)
            0xCB,
            0x26,  # SLA (HL)
            0xCB,
            0x2E,  # SRA (HL)
            0xCB,
            0x36,  # SWAP (HL)
            0xCB,
            0x3E,  # SRL (HL)
            0xCB,
            0x40,  # BIT 0,B
            0xCB,
            0x49,  # BIT 1,C
            0xCB,
            0x7E,  # BIT 7,(HL)
            0xCB,
            0x82,  # RES 0,D
            0xCB,
            0x8B,  # RES 1,E
            0xCB,
            0xBE,  # RES 7,(HL)
            0xCB,
            0xC0,  # SET 0,B
            0xCB,
            0xC9,  # SET 1,C
            0xCB,
            0xFE,  # SET 7,(HL)
            0x18,
            0xFE,  # JR -2
        ]
    )
    return build_rom("CB_BITOPS", program)


def build_bg_static() -> bytes:
    """Build BG_STATIC.gb - static background with unsigned tiles."""
    program = bytes(
        [
            0xAF,  # XOR A
            0xE0,
            0x40,  # LDH (LCDC),A (LCD off)
            0x3E,
            0xE4,  # LD A,0xE4 (BGP)
            0xE0,
            0x47,  # LDH (BGP),A
            0xAF,  # XOR A
            0xE0,
            0x42,  # LDH (SCY),A
            0xE0,
            0x43,  # LDH (SCX),A
            0x21,
            0x00,
            0x80,  # LD HL,0x8000 (tile data)
            0x06,
            0x10,  # LD B,16
            0xAF,  # XOR A
            0x22,  # LD (HL+),A
            0x05,  # DEC B
            0x20,
            0xFB,  # JR NZ,-5
            0x06,
            0x08,  # LD B,8
            0x3E,
            0xFF,  # LD A,0xFF
            0x22,  # LD (HL+),A
            0xAF,  # XOR A
            0x22,  # LD (HL+),A
            0x05,  # DEC B
            0x20,
            0xF8,  # JR NZ,-8
            0x21,
            0x00,
            0x98,  # LD HL,0x9800 (BG map)
            0x06,
            0x20,  # LD B,32
            0xAF,  # XOR A
            0x0E,
            0x20,  # LD C,32
            0x22,  # LD (HL+),A
            0xEE,
            0x01,  # XOR 0x01
            0x0D,  # DEC C
            0x20,
            0xFA,  # JR NZ,-6
            0x05,  # DEC B
            0x20,
            0xF4,  # JR NZ,-12
            0x3E,
            0x91,  # LD A,0x91 (LCD on, unsigned tiles)
            0xE0,
            0x40,  # LDH (LCDC),A
            0x18,
            0xFE,  # JR -2
        ]
    )
    return build_rom("BG_STATIC", program)


def build_ppu_window() -> bytes:
    """Build PPU_WINDOW.gb - window overlay over BG (scanline-latch safe)."""
    code = bytearray()
    labels: dict[str, int] = {}
    jr_fixups: list[tuple[int, str]] = []

    def label(name: str) -> None:
        labels[name] = len(code)

    def emit(*bytes_: int) -> None:
        code.extend(bytes_)

    def emit_jr(opcode: int, target: str) -> None:
        pc = len(code)
        emit(opcode, 0x00)
        jr_fixups.append((pc, target))

    # LCD off
    emit(0xAF)  # XOR A
    emit(0xE0, 0x40)  # LDH (LCDC),A
    emit(0x3E, 0xE4)  # LD A,0xE4 (BGP)
    emit(0xE0, 0x47)  # LDH (BGP),A
    emit(0xAF)  # XOR A
    emit(0xE0, 0x42)  # LDH (SCY),A
    emit(0xE0, 0x43)  # LDH (SCX),A
    emit(0x3E, 0x20)  # LD A,0x20 (WX = 32)
    emit(0xE0, 0x4B)  # LDH (WX),A
    emit(0x3E, 0x08)  # LD A,0x08 (WY = 8)
    emit(0xE0, 0x4A)  # LDH (WY),A

    # Tile data: tile 0 = 0x00, tile 1 = 0xFF/0x00 per row
    emit(0x21, 0x00, 0x80)  # LD HL,0x8000
    emit(0x06, 0x10)  # LD B,16
    emit(0xAF)  # XOR A
    label("tile0")
    emit(0x22)  # LD (HL+),A
    emit(0x05)  # DEC B
    emit_jr(0x20, "tile0")

    # Tile 1: rows use colors 1,2,3,0 (repeat) for visible shading
    for val in (
        0xFF, 0x00,  # color 1
        0x00, 0xFF,  # color 2
        0xFF, 0xFF,  # color 3
        0x00, 0x00,  # color 0
        0xFF, 0x00,
        0x00, 0xFF,
        0xFF, 0xFF,
        0x00, 0x00,
    ):
        emit(0x3E, val)  # LD A,val
        emit(0x22)  # LD (HL+),A

    # BG map: fill with tile 0
    emit(0x21, 0x00, 0x98)  # LD HL,0x9800
    emit(0x06, 0x20)  # LD B,32
    emit(0xAF)  # XOR A
    label("bg_row")
    emit(0x0E, 0x20)  # LD C,32
    label("bg_col")
    emit(0x22)  # LD (HL+),A
    emit(0x0D)  # DEC C
    emit_jr(0x20, "bg_col")
    emit(0x05)  # DEC B
    emit_jr(0x20, "bg_row")

    # Window map: fill with tile 1
    emit(0x21, 0x00, 0x9C)  # LD HL,0x9C00
    emit(0x06, 0x20)  # LD B,32
    emit(0x3E, 0x01)  # LD A,0x01
    label("win_row")
    emit(0x0E, 0x20)  # LD C,32
    label("win_col")
    emit(0x22)  # LD (HL+),A
    emit(0x0D)  # DEC C
    emit_jr(0x20, "win_col")
    emit(0x05)  # DEC B
    emit_jr(0x20, "win_row")

    # LCD on: BG + window + unsigned tiles + window map @ 0x9C00
    emit(0x3E, 0xF1)  # LD A,0xF1
    emit(0xE0, 0x40)  # LDH (LCDC),A
    label("halt")
    emit_jr(0x18, "halt")

    # Resolve JR fixups
    for pc, target in jr_fixups:
        if target not in labels:
            raise ValueError(f"Unknown JR label: {target}")
        offset = labels[target] - (pc + 2)
        if offset < -128 or offset > 127:
            raise ValueError(f"JR offset out of range for {target}: {offset}")
        code[pc + 1] = offset & 0xFF

    return build_rom("PPU_WINDOW", bytes(code))


def build_ppu_sprites() -> bytes:
    """Build PPU_SPRITES.gb - sprite rendering coverage ROM."""
    code = bytearray()
    labels: dict[str, int] = {}
    jr_fixups: list[tuple[int, str]] = []

    def label(name: str) -> None:
        labels[name] = len(code)

    def emit(*bytes_: int) -> None:
        code.extend(bytes_)

    def emit_jr(opcode: int, target: str) -> None:
        pc = len(code)
        emit(opcode, 0x00)
        jr_fixups.append((pc, target))

    def ld_a_d8(val: int) -> None:
        emit(0x3E, val & 0xFF)

    def ld_hl_d16(addr: int) -> None:
        emit(0x21, addr & 0xFF, (addr >> 8) & 0xFF)

    def ld_hl_a() -> None:
        emit(0x22)

    # LCD off
    emit(0xAF)  # XOR A
    emit(0xE0, 0x40)  # LDH (LCDC),A
    ld_a_d8(0xE4)  # BGP
    emit(0xE0, 0x47)  # LDH (BGP),A
    ld_a_d8(0xE4)  # OBP0
    emit(0xE0, 0x48)  # LDH (OBP0),A
    ld_a_d8(0x1B)  # OBP1 (inverted)
    emit(0xE0, 0x49)  # LDH (OBP1),A
    emit(0xAF)  # XOR A
    emit(0xE0, 0x42)  # LDH (SCY),A
    emit(0xE0, 0x43)  # LDH (SCX),A

    # Tile data: tiles 0..3
    ld_hl_d16(0x8000)
    emit(0x06, 0x10)  # LD B,16
    emit(0xAF)  # XOR A
    label("tile0")
    ld_hl_a()
    emit(0x05)  # DEC B
    emit_jr(0x20, "tile0")

    # Tile 1: color 1 stripes (lo=FF, hi=00)
    emit(0x06, 0x08)  # LD B,8
    label("tile1")
    ld_a_d8(0xFF)
    ld_hl_a()
    emit(0xAF)
    ld_hl_a()
    emit(0x05)
    emit_jr(0x20, "tile1")

    # Tile 2: color 2 stripes (lo=00, hi=FF)
    emit(0x06, 0x08)  # LD B,8
    label("tile2")
    emit(0xAF)
    ld_hl_a()
    ld_a_d8(0xFF)
    ld_hl_a()
    emit(0x05)
    emit_jr(0x20, "tile2")

    # Tile 3: color 3 solid (lo=FF, hi=FF)
    emit(0x06, 0x10)  # LD B,16
    label("tile3")
    ld_a_d8(0xFF)
    ld_hl_a()
    emit(0x05)
    emit_jr(0x20, "tile3")

    # BG map: alternating 0/1
    ld_hl_d16(0x9800)
    emit(0x06, 0x20)  # LD B,32
    emit(0xAF)
    label("bg_row")
    emit(0x0E, 0x20)  # LD C,32
    label("bg_col")
    ld_hl_a()
    emit(0xEE, 0x01)  # XOR 0x01
    emit(0x0D)  # DEC C
    emit_jr(0x20, "bg_col")
    emit(0x05)  # DEC B
    emit_jr(0x20, "bg_row")

    # OAM data in WRAM (0xC000), then DMA copy
    ld_hl_d16(0xC000)
    for byte in (
        0x38,
        0x30,
        0x02,
        0x00,
        0x38,
        0x30,
        0x03,
        0x00,
        0x38,
        0x48,
        0x02,
        0x80,
        0x38,
        0x60,
        0x02,
        0x20,
        0x38,
        0x78,
        0x02,
        0x40,
        0x38,
        0x90,
        0x02,
        0x10,
    ):
        ld_a_d8(byte)
        ld_hl_a()

    ld_a_d8(0xC0)
    emit(0xEA, 0x46, 0xFF)  # LD (0xFF46),A

    # LCD on: BG + OBJ + 8x16
    ld_a_d8(0x97)
    emit(0xE0, 0x40)  # LDH (LCDC),A
    label("halt")
    emit_jr(0x18, "halt")

    for pc, target in jr_fixups:
        if target not in labels:
            raise ValueError(f"Unknown JR label: {target}")
        offset = labels[target] - (pc + 2)
        if offset < -128 or offset > 127:
            raise ValueError(f"JR offset out of range for {target}: {offset}")
        code[pc + 1] = offset & 0xFF

    return build_rom("PPU_SPRITES", bytes(code))


def build_bg_scroll_signed() -> bytes:
    """Build BG_SCROLL_SIGNED.gb - scrolled background with signed tiles."""
    program = bytes(
        [
            0xAF,  # XOR A
            0xE0,
            0x40,  # LDH (LCDC),A (LCD off)
            0x3E,
            0xE4,  # LD A,0xE4 (BGP)
            0xE0,
            0x47,  # LDH (BGP),A
            0x3E,
            0x08,  # LD A,0x08 (SCY)
            0xE0,
            0x42,  # LDH (SCY),A
            0x3E,
            0x04,  # LD A,0x04 (SCX)
            0xE0,
            0x43,  # LDH (SCX),A
            0x21,
            0x00,
            0x88,  # LD HL,0x8800 (signed tile data)
            0x06,
            0x08,  # LD B,8
            0x3E,
            0xAA,  # LD A,0xAA
            0x22,  # LD (HL+),A
            0x3E,
            0xCC,  # LD A,0xCC
            0x22,  # LD (HL+),A
            0x05,  # DEC B
            0x20,
            0xF7,  # JR NZ,-9
            0x21,
            0x00,
            0x98,  # LD HL,0x9800 (BG map)
            0x3E,
            0x80,  # LD A,0x80 (tile index -128)
            0x06,
            0x20,  # LD B,32
            0x0E,
            0x20,  # LD C,32
            0x22,  # LD (HL+),A
            0x0D,  # DEC C
            0x20,
            0xFC,  # JR NZ,-4
            0x05,  # DEC B
            0x20,
            0xF7,  # JR NZ,-9
            0x3E,
            0x81,  # LD A,0x81 (LCD on, signed tiles)
            0xE0,
            0x40,  # LDH (LCDC),A
            0x18,
            0xFE,  # JR -2
        ]
    )
    return build_rom("BG_SCROLL", program)


def atomic_write(path: Path, data: bytes) -> None:
    """Write data to path atomically using temp file + rename.

    This ensures partial writes never exist at the target path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (ensures same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, data)
        os.close(fd)
        os.rename(tmp_path, path)
    except Exception:
        os.close(fd)
        os.unlink(tmp_path)
        raise


def build_all(out_dir: Path | None = None) -> list[tuple[str, Path, str]]:
    """Build all micro-ROMs and return their info.

    Args:
        out_dir: Output directory. Defaults to bench/roms/out/.

    Returns:
        List of (name, path, sha256) tuples for each generated ROM.
    """
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR

    out_dir.mkdir(parents=True, exist_ok=True)

    roms = [
        ("ALU_LOOP.gb", build_alu_loop()),
        ("MEM_RWB.gb", build_mem_rwb()),
        ("SERIAL_HELLO.gb", build_serial_hello()),
        ("DMA_OAM_COPY.gb", build_dma_oam_copy()),
        ("TIMER_DIV_BASIC.gb", build_timer_div_basic()),
        ("TIMER_IRQ_HALT.gb", build_timer_irq_halt()),
        ("EI_DELAY.gb", build_ei_delay()),
        ("JOY_DIVERGE_PERSIST.gb", build_joy_diverge_persist()),
        ("LOADS_BASIC.gb", build_loads_basic()),
        ("ALU_FLAGS.gb", build_alu_flags()),
        ("ALU16_SP.gb", build_alu16_sp()),
        ("FLOW_STACK.gb", build_flow_stack()),
        ("CB_BITOPS.gb", build_cb_bitops()),
        ("BG_STATIC.gb", build_bg_static()),
        ("PPU_WINDOW.gb", build_ppu_window()),
        ("PPU_SPRITES.gb", build_ppu_sprites()),
        ("BG_SCROLL_SIGNED.gb", build_bg_scroll_signed()),
    ]

    results: list[tuple[str, Path, str]] = []
    for name, data in roms:
        path = out_dir / name
        atomic_write(path, data)
        sha = sha256_bytes(data)
        results.append((name, path, sha))

    return results


def main() -> None:
    """CLI entry point for micro-ROM generation."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic micro-ROMs for testing."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    results = build_all(args.out_dir)

    for _name, path, sha in results:
        size = path.stat().st_size
        print(f"{path}  {size:>6} bytes  sha256:{sha[:16]}...")


if __name__ == "__main__":
    main()
