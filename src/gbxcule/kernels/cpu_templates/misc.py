"""Misc instruction templates for Warp CPU stepping."""
# ruff: noqa: F841

from __future__ import annotations


def template_nop(pc_i: int) -> None:
    """NOP template."""
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_default(pc_i: int) -> None:
    """Default template for unknown opcodes."""
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4


def template_trap_unprefixed(
    pc_i: int,
    trap_i: int,
    trap_pc_i: int,
    trap_opcode_i: int,
    trap_kind_i: int,
    opcode: int,
) -> None:
    """Trap template for unknown unprefixed opcodes."""
    if trap_i == 0:
        trap_i = 1
        trap_pc_i = pc_i & 0xFFFF
        trap_opcode_i = opcode & 0xFF
        trap_kind_i = 1
    cycles = 0


def template_trap_cb(
    pc_i: int,
    trap_i: int,
    trap_pc_i: int,
    trap_opcode_i: int,
    trap_kind_i: int,
    cb_opcode: int,
) -> None:
    """Trap template for unknown CB-prefixed opcodes."""
    if trap_i == 0:
        trap_i = 1
        trap_pc_i = pc_i & 0xFFFF
        trap_opcode_i = cb_opcode & 0xFF
        trap_kind_i = 2
    cycles = 0
