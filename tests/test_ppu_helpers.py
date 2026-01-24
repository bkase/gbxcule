"""Unit tests for pure PPU helper functions."""

from __future__ import annotations

from gbxcule.core.ppu_helpers import (
    StatIrqConditions,
    sprite_pixel_visible,
    stat_irq_edge,
)


def test_stat_irq_edge_requires_enabled_source() -> None:
    prev = StatIrqConditions(mode0=False, mode1=False, mode2=False, lyc=False)
    curr = StatIrqConditions(mode0=True, mode1=False, mode2=False, lyc=False)
    assert stat_irq_edge(prev, curr, stat_value=0x00) is False


def test_stat_irq_edge_on_mode0_transition() -> None:
    prev = StatIrqConditions(mode0=False, mode1=False, mode2=False, lyc=False)
    curr = StatIrqConditions(mode0=True, mode1=False, mode2=False, lyc=False)
    assert stat_irq_edge(prev, curr, stat_value=0x08) is True
    assert stat_irq_edge(curr, curr, stat_value=0x08) is False


def test_stat_irq_edge_on_lyc_transition() -> None:
    prev = StatIrqConditions(mode0=False, mode1=False, mode2=False, lyc=False)
    curr = StatIrqConditions(mode0=False, mode1=False, mode2=False, lyc=True)
    assert stat_irq_edge(prev, curr, stat_value=0x40) is True
    assert stat_irq_edge(curr, curr, stat_value=0x40) is False


def test_stat_irq_edge_multiple_sources() -> None:
    prev = StatIrqConditions(mode0=False, mode1=True, mode2=False, lyc=False)
    curr = StatIrqConditions(mode0=True, mode1=True, mode2=False, lyc=False)
    assert stat_irq_edge(prev, curr, stat_value=0x18) is True


def test_sprite_pixel_visible_transparent() -> None:
    assert sprite_pixel_visible(obj_color=0, bg_color=0, obj_priority=0) is False
    assert sprite_pixel_visible(obj_color=0, bg_color=2, obj_priority=1) is False


def test_sprite_pixel_visible_priority() -> None:
    assert sprite_pixel_visible(obj_color=2, bg_color=0, obj_priority=1) is True
    assert sprite_pixel_visible(obj_color=2, bg_color=1, obj_priority=1) is False
    assert sprite_pixel_visible(obj_color=2, bg_color=3, obj_priority=0) is True
