"""Load/store instruction templates for Warp CPU stepping."""
# ruff: noqa: F841, F821

from __future__ import annotations

import warp as wp

# Placeholders for linting; the real values are injected into the kernel globals.
ROM_LIMIT = 0
CART_RAM_START = 0
CART_RAM_END = 0


def template_ld_r8_d8(pc_i: int, REG_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """LD r8, d8 template (REG_i placeholder)."""
    REG_i = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_ld_r16_d16(
    pc_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD r16, d16 template (HREG_i/LREG_i placeholders)."""
    lo = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    hi = read8(
        i,
        base,
        (pc_i + 2) & 0xFFFF,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    reg16 = ((hi << 8) | lo) & 0xFFFF
    HREG_i = (reg16 >> 8) & 0xFF
    LREG_i = reg16 & 0xFF
    pc_i = (pc_i + 3) & 0xFFFF
    cycles = 12


def template_ld_hl_r8(
    pc_i: int, HREG_i: int, LREG_i: int, SRC_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD (HL), r8 template (HREG_i/LREG_i/SRC_i placeholders)."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    write8(
        i,
        base,
        hl,
        SRC_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_r8_hl(
    pc_i: int, HREG_i: int, LREG_i: int, DST_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD r8, (HL) template (HREG_i/LREG_i/DST_i placeholders)."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    DST_i = read8(
        i,
        base,
        hl,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8
