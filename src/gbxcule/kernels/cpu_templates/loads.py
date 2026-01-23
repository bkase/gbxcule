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
    REG_i = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_ld_r16_d16(
    pc_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD r16, d16 template (HREG_i/LREG_i placeholders)."""
    lo = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
    hi = wp.int32(mem[base + ((pc_i + 2) & 0xFFFF)])
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
    if hl == 0xFF00:
        joyp_select[i] = wp.uint8(SRC_i) & wp.uint8(0x30)
    # Cartridge ROM (0x0000-0x7FFF) is read-only on real hardware.
    # MEM_RWB intentionally walks HL across the full address space,
    # so we must ignore writes into ROM to avoid self-modifying code
    # that diverges from PyBoy.
    elif hl >= ROM_LIMIT and not (hl >= CART_RAM_START and hl < CART_RAM_END):
        mem[base + hl] = wp.uint8(SRC_i)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_r8_hl(
    pc_i: int, HREG_i: int, LREG_i: int, DST_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD r8, (HL) template (HREG_i/LREG_i/DST_i placeholders)."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    if hl == 0xFF00:
        action_i = wp.int32(actions[i])
        joyp_sel = wp.int32(joyp_select[i])
        DST_i = joyp_read(
            action_i, frames_done, release_after_frames, joyp_sel, action_codec_id
        )
    else:
        # This repo's micro-ROMs are built with "no cart RAM".
        # Reads in 0xA000-0xBFFF return open-bus (0xFF);
        # writes are ignored.
        if hl >= CART_RAM_START and hl < CART_RAM_END:
            DST_i = 0xFF
        else:
            DST_i = wp.int32(mem[base + hl])
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8
