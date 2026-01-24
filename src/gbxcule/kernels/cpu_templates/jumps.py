"""Jump instruction templates for Warp CPU stepping."""
# ruff: noqa: F821, F841

from __future__ import annotations

import warp as wp


# Helper for type checking (injected as @wp.func in the kernel).
def sign8(x: int) -> int: ...


def template_jp_a16(pc_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JP a16 template."""
    lo = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
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
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    pc_i = ((hi << 8) | lo) & 0xFFFF
    cycles = 16


def template_jr_r8(pc_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JR r8 template."""
    off = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    off = sign8(off)
    pc_i = (pc_i + 2 + off) & 0xFFFF
    cycles = 12


def template_jr_nz_r8(pc_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JR NZ, r8 template."""
    off = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    off = sign8(off)
    z = (f_i >> 7) & 0x1
    take = wp.where(z == 0, 1, 0)
    pc_i = wp.where(take != 0, (pc_i + 2 + off) & 0xFFFF, (pc_i + 2) & 0xFFFF)
    cycles = wp.where(take != 0, 12, 8)


def template_jr_z_r8(pc_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JR Z, r8 template."""
    off = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    off = sign8(off)
    z = (f_i >> 7) & 0x1
    take = wp.where(z != 0, 1, 0)
    pc_i = wp.where(take != 0, (pc_i + 2 + off) & 0xFFFF, (pc_i + 2) & 0xFFFF)
    cycles = wp.where(take != 0, 12, 8)


def template_jr_nc_r8(pc_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JR NC, r8 template."""
    off = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    off = sign8(off)
    cflag = (f_i >> 4) & 0x1
    take = wp.where(cflag == 0, 1, 0)
    pc_i = wp.where(take != 0, (pc_i + 2 + off) & 0xFFFF, (pc_i + 2) & 0xFFFF)
    cycles = wp.where(take != 0, 12, 8)


def template_jr_c_r8(pc_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JR C, r8 template."""
    off = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    off = sign8(off)
    cflag = (f_i >> 4) & 0x1
    take = wp.where(cflag != 0, 1, 0)
    pc_i = wp.where(take != 0, (pc_i + 2 + off) & 0xFFFF, (pc_i + 2) & 0xFFFF)
    cycles = wp.where(take != 0, 12, 8)


def template_jp_nz_a16(pc_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JP NZ, a16 template."""
    lo = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
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
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    addr = ((hi << 8) | lo) & 0xFFFF
    z = (f_i >> 7) & 0x1
    take = wp.where(z == 0, 1, 0)
    pc_i = wp.where(take != 0, addr, (pc_i + 3) & 0xFFFF)
    cycles = wp.where(take != 0, 16, 12)


def template_jp_z_a16(pc_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JP Z, a16 template."""
    lo = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
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
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    addr = ((hi << 8) | lo) & 0xFFFF
    z = (f_i >> 7) & 0x1
    take = wp.where(z != 0, 1, 0)
    pc_i = wp.where(take != 0, addr, (pc_i + 3) & 0xFFFF)
    cycles = wp.where(take != 0, 16, 12)


def template_jp_nc_a16(pc_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JP NC, a16 template."""
    lo = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
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
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    addr = ((hi << 8) | lo) & 0xFFFF
    cflag = (f_i >> 4) & 0x1
    take = wp.where(cflag == 0, 1, 0)
    pc_i = wp.where(take != 0, addr, (pc_i + 3) & 0xFFFF)
    cycles = wp.where(take != 0, 16, 12)


def template_jp_c_a16(pc_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JP C, a16 template."""
    lo = read8(
        i,
        base,
        (pc_i + 1) & 0xFFFF,
        mem,
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
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
        rom,
        bootrom,
        cart_ram,
        cart_state,
        rom_bank_count,
        rom_bank_mask,
        ram_bank_count,
        actions,
        joyp_select,
        frames_done,
        release_after_frames,
        action_codec_id,
    )
    addr = ((hi << 8) | lo) & 0xFFFF
    cflag = (f_i >> 4) & 0x1
    take = wp.where(cflag != 0, 1, 0)
    pc_i = wp.where(take != 0, addr, (pc_i + 3) & 0xFFFF)
    cycles = wp.where(take != 0, 16, 12)


def template_jp_hl(pc_i: int, HREG_i: int, LREG_i: int) -> None:
    """JP (HL) template."""
    pc_i = ((HREG_i << 8) | LREG_i) & 0xFFFF
    cycles = 4
