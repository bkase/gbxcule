"""Jump instruction templates for Warp CPU stepping."""
# ruff: noqa: F841

from __future__ import annotations

import warp as wp


# Helper for type checking (injected as @wp.func in the kernel).
def sign8(x: int) -> int: ...


def template_jp_a16(pc_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JP a16 template."""
    lo = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
    hi = wp.int32(mem[base + ((pc_i + 2) & 0xFFFF)])
    pc_i = ((hi << 8) | lo) & 0xFFFF
    cycles = 16


def template_jr_r8(pc_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """JR r8 template."""
    off = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
    off = sign8(off)
    pc_i = (pc_i + 2 + off) & 0xFFFF
    cycles = 12
