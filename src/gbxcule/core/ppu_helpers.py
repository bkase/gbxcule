"""Pure helpers for PPU decision logic (testable outside Warp)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatIrqConditions:
    """STAT interrupt source conditions (booleans)."""

    mode0: bool
    mode1: bool
    mode2: bool
    lyc: bool


def stat_irq_edge(
    prev: StatIrqConditions, curr: StatIrqConditions, stat_value: int
) -> bool:
    """Return True if any enabled STAT source transitions false -> true."""
    stat_bits = stat_value & 0x78
    enable_mode0 = (stat_bits >> 3) & 1
    enable_mode1 = (stat_bits >> 4) & 1
    enable_mode2 = (stat_bits >> 5) & 1
    enable_lyc = (stat_bits >> 6) & 1

    edge_mode0 = enable_mode0 and (not prev.mode0) and curr.mode0
    edge_mode1 = enable_mode1 and (not prev.mode1) and curr.mode1
    edge_mode2 = enable_mode2 and (not prev.mode2) and curr.mode2
    edge_lyc = enable_lyc and (not prev.lyc) and curr.lyc

    return bool(edge_mode0 or edge_mode1 or edge_mode2 or edge_lyc)


def sprite_pixel_visible(obj_color: int, bg_color: int, obj_priority: int) -> bool:
    """Return True if an OBJ pixel should be drawn over BG/Window."""
    if obj_color == 0:
        return False
    return not (obj_priority != 0 and bg_color != 0)
