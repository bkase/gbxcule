"""ALU instruction templates for Warp CPU stepping."""
# ruff: noqa: F841

from __future__ import annotations

import warp as wp


# Helper for type checking (injected as @wp.func in the kernel).
def make_flags(z: int, n: int, h: int, c: int) -> int: ...


def template_inc_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """8-bit INC template (REG_i placeholder)."""
    old = REG_i
    REG_i = (REG_i + 1) & 0xFF
    z = wp.where(REG_i == 0, 1, 0)
    hflag = wp.where((old & 0x0F) == 0x0F, 1, 0)
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_inc_r16(pc_i: int, HREG_i: int, LREG_i: int) -> None:
    """16-bit INC template (HREG_i/LREG_i placeholders)."""
    reg16 = ((HREG_i << 8) | LREG_i) & 0xFFFF
    reg16 = (reg16 + 1) & 0xFFFF
    HREG_i = (reg16 >> 8) & 0xFF
    LREG_i = reg16 & 0xFF
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 8


def template_add_a_r8(pc_i: int, a_i: int, f_i: int, REG_i: int) -> None:
    """ADD A, r8 template (REG_i placeholder)."""
    sum_ab = a_i + REG_i
    res = sum_ab & 0xFF
    z = wp.where(res == 0, 1, 0)
    hflag = wp.where(((a_i & 0x0F) + (REG_i & 0x0F)) > 0x0F, 1, 0)
    cflag = wp.where(sum_ab > 0xFF, 1, 0)
    a_i = res
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_and_a_d8(pc_i: int, a_i: int, f_i: int, base: int, mem: wp.array) -> None:  # type: ignore[name-defined]
    """AND A, d8 template."""
    val = wp.int32(mem[base + ((pc_i + 1) & 0xFFFF)])
    res = a_i & val
    a_i = res & 0xFF
    z = wp.where(a_i == 0, 1, 0)
    f_i = make_flags(z, 0, 1, 0)
    pc_i = (pc_i + 2) & 0xFFFF
    cycles = 8


def template_dec_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    """8-bit DEC template (REG_i placeholder)."""
    old = REG_i
    REG_i = (REG_i - 1) & 0xFF
    z = wp.where(REG_i == 0, 1, 0)
    hflag = wp.where((old & 0x0F) == 0x00, 1, 0)
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 1, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4
