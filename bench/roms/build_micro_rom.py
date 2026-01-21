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
