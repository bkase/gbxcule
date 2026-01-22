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
