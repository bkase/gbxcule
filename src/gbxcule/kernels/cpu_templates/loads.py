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


def template_ld_r8_r8(pc_i: int, DST_i: int, SRC_i: int) -> None:
    """LD r8, r8 template (DST_i/SRC_i placeholders)."""
    DST_i = SRC_i & 0xFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_ld_hl_d8(
    pc_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD (HL), d8 template."""
    val = read8(
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
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    write8(
        i,
        base,
        hl,
        val,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 12


def template_ld_sp_d16(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """LD SP, d16 template."""
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
    sp_i = ((hi << 8) | lo) & 0xFFFF
    pc_i = (pc_i + 3) & 0xFFFF
    cycles = 12


def template_ld_sp_hl(pc_i: int, sp_i: int, HREG_i: int, LREG_i: int) -> None:
    """LD SP, HL template."""
    sp_i = ((HREG_i << 8) | LREG_i) & 0xFFFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_a16_a(pc_i: int, a_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """LD (a16), A template."""
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
    addr = ((hi << 8) | lo) & 0xFFFF
    write8(
        i,
        base,
        addr,
        a_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 3) & 0xFFFF
    cycles = 16


def template_ld_a_a16(pc_i: int, a_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """LD A, (a16) template."""
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
    addr = ((hi << 8) | lo) & 0xFFFF
    a_i = read8(
        i,
        base,
        addr,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    pc_i = (pc_i + 3) & 0xFFFF
    cycles = 16


def template_ldh_a8_a(pc_i: int, a_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """LDH (a8), A template."""
    off = read8(
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
    addr = (0xFF00 | off) & 0xFFFF
    write8(
        i,
        base,
        addr,
        a_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 12


def template_ldh_a_a8(pc_i: int, a_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """LDH A, (a8) template."""
    off = read8(
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
    addr = (0xFF00 | off) & 0xFFFF
    a_i = read8(
        i,
        base,
        addr,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 12


def template_ldh_c_a(
    pc_i: int, a_i: int, c_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD (C), A template (FF00+C)."""
    addr = (0xFF00 | c_i) & 0xFFFF
    write8(
        i,
        base,
        addr,
        a_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ldh_a_c(
    pc_i: int, a_i: int, c_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD A, (C) template (FF00+C)."""
    addr = (0xFF00 | c_i) & 0xFFFF
    a_i = read8(
        i,
        base,
        addr,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_a_bc(
    pc_i: int, a_i: int, b_i: int, c_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD A, (BC) template."""
    addr = ((b_i << 8) | c_i) & 0xFFFF
    a_i = read8(
        i,
        base,
        addr,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_a_de(
    pc_i: int, a_i: int, d_i: int, e_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD A, (DE) template."""
    addr = ((d_i << 8) | e_i) & 0xFFFF
    a_i = read8(
        i,
        base,
        addr,
        mem,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_bc_a(
    pc_i: int, a_i: int, b_i: int, c_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD (BC), A template."""
    addr = ((b_i << 8) | c_i) & 0xFFFF
    write8(
        i,
        base,
        addr,
        a_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_de_a(
    pc_i: int, a_i: int, d_i: int, e_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LD (DE), A template."""
    addr = ((d_i << 8) | e_i) & 0xFFFF
    write8(
        i,
        base,
        addr,
        a_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_hl_inc_a(
    pc_i: int, a_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LDI (HL), A template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    write8(
        i,
        base,
        hl,
        a_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    hl = (hl + 1) & 0xFFFF
    HREG_i = (hl >> 8) & 0xFF
    LREG_i = hl & 0xFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_hl_dec_a(
    pc_i: int, a_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LDD (HL), A template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    write8(
        i,
        base,
        hl,
        a_i,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    hl = (hl - 1) & 0xFFFF
    HREG_i = (hl >> 8) & 0xFF
    LREG_i = hl & 0xFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_a_hl_inc(
    pc_i: int, a_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LDI A, (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    a_i = read8(
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
    hl = (hl + 1) & 0xFFFF
    HREG_i = (hl >> 8) & 0xFF
    LREG_i = hl & 0xFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_a_hl_dec(
    pc_i: int, a_i: int, HREG_i: int, LREG_i: int, base: int, mem: wp.array
) -> None:  # type: ignore[name-defined]
    """LDD A, (HL) template."""
    hl = ((HREG_i << 8) | LREG_i) & 0xFFFF
    a_i = read8(
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
    hl = (hl - 1) & 0xFFFF
    HREG_i = (hl >> 8) & 0xFF
    LREG_i = hl & 0xFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_ld_a16_sp(pc_i: int, sp_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """LD (a16), SP template."""
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
    addr = ((hi << 8) | lo) & 0xFFFF
    lo_sp = sp_i & 0xFF
    hi_sp = (sp_i >> 8) & 0xFF
    write8(
        i,
        base,
        addr,
        lo_sp,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    write8(
        i,
        base,
        (addr + 1) & 0xFFFF,
        hi_sp,
        mem,
        joyp_select,
        serial_buf,
        serial_len,
        serial_overflow,
    )
    pc_i = (pc_i + 3) & 0xFFFF
    cycles = 20
