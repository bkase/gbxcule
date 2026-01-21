"""Generate valid micro-ROMs for testing.

These are minimal, deterministic ROMs for correctness and performance testing.
They are license-safe (generated, not copyrighted content).
"""

from pathlib import Path

# Output directory
OUT_DIR = Path(__file__).parent / "out"

# Nintendo logo (required for boot ROM validation)
# This is the exact sequence the Game Boy checks
NINTENDO_LOGO = bytes(
    [
        0xCE,
        0xED,
        0x66,
        0x66,
        0xCC,
        0x0D,
        0x00,
        0x0B,
        0x03,
        0x73,
        0x00,
        0x83,
        0x00,
        0x0C,
        0x00,
        0x0D,
        0x00,
        0x08,
        0x11,
        0x1F,
        0x88,
        0x89,
        0x00,
        0x0E,
        0xDC,
        0xCC,
        0x6E,
        0xE6,
        0xDD,
        0xDD,
        0xD9,
        0x99,
        0xBB,
        0xBB,
        0x67,
        0x63,
        0x6E,
        0x0E,
        0xEC,
        0xCC,
        0xDD,
        0xDC,
        0x99,
        0x9F,
        0xBB,
        0xB9,
        0x33,
        0x3E,
    ]
)


def compute_header_checksum(header: bytes) -> int:
    """Compute the header checksum (byte at 0x14D)."""
    checksum = 0
    for byte in header[0x34:0x4D]:  # 0x134-0x14C relative to 0x100
        checksum = (checksum - byte - 1) & 0xFF
    return checksum


def build_rom(title: str, code: bytes) -> bytes:
    """Build a valid Game Boy ROM with the given title and code.

    Args:
        title: ROM title (max 11 characters for old format)
        code: Machine code to place at 0x150+

    Returns:
        Complete ROM bytes
    """
    # ROM is 32KB minimum
    rom = bytearray(32 * 1024)

    # Entry point at 0x100: JP 0x150 (jump to main code)
    rom[0x100] = 0xC3  # JP
    rom[0x101] = 0x50  # low byte of 0x150
    rom[0x102] = 0x01  # high byte of 0x150
    rom[0x103] = 0x00  # NOP padding

    # Nintendo logo at 0x104-0x133
    rom[0x104 : 0x104 + len(NINTENDO_LOGO)] = NINTENDO_LOGO

    # Title at 0x134-0x143 (padded with zeros)
    title_bytes = title.upper().encode("ascii")[:11]
    rom[0x134 : 0x134 + len(title_bytes)] = title_bytes

    # Cartridge type at 0x147: ROM only
    rom[0x147] = 0x00

    # ROM size at 0x148: 32KB (code 0)
    rom[0x148] = 0x00

    # RAM size at 0x149: No RAM
    rom[0x149] = 0x00

    # Destination code at 0x14A: Non-Japanese
    rom[0x14A] = 0x01

    # Old licensee code at 0x14B
    rom[0x14B] = 0x00

    # ROM version at 0x14C
    rom[0x14C] = 0x00

    # Header checksum at 0x14D
    rom[0x14D] = compute_header_checksum(rom[0x100:0x150])

    # Global checksum at 0x14E-0x14F (not validated by boot ROM, leave as 0)

    # Main code at 0x150
    rom[0x150 : 0x150 + len(code)] = code

    return bytes(rom)


def build_alu_loop() -> bytes:
    """Build ALU_LOOP.gb - a tight ALU-heavy loop.

    This ROM executes a deterministic loop of ALU operations.
    Good for testing CPU correctness and measuring ALU throughput.
    """
    # Assembly:
    #   LD A, 0       ; A = 0
    #   LD B, 0       ; B = 0
    # loop:
    #   INC A         ; A++
    #   ADD A, B      ; A += B
    #   INC B         ; B++
    #   JR loop       ; infinite loop
    code = bytes(
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
    return build_rom("ALU_LOOP", code)


def build_mem_rwb() -> bytes:
    """Build MEM_RWB.gb - memory read/write benchmark.

    This ROM performs memory loads and stores in a loop.
    Good for testing memory correctness and measuring memory throughput.
    """
    # Assembly:
    #   LD HL, 0xC000  ; HL points to WRAM
    #   LD A, 0        ; A = 0
    # loop:
    #   LD (HL), A     ; Write A to memory
    #   INC A          ; A++
    #   LD B, (HL)     ; Read memory into B
    #   INC HL         ; HL++
    #   JR loop        ; infinite loop
    code = bytes(
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
    return build_rom("MEM_RWB", code)


def main() -> None:
    """Generate all micro-ROMs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    roms = [
        ("ALU_LOOP.gb", build_alu_loop()),
        ("MEM_RWB.gb", build_mem_rwb()),
    ]

    for name, data in roms:
        path = OUT_DIR / name
        path.write_bytes(data)
        print(f"Generated {path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
