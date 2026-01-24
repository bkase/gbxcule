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
    pc_i = ((hi << 8) | lo) & 0xFFFF
    cycles = 16


def template_jr_r8(pc_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JR r8 template."""
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
