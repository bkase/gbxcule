"""Scanline-accurate PPU timing tests for Warp CPU backend."""

from __future__ import annotations

import numpy as np

from gbxcule.backends.warp_vec import WarpVecCpuBackend
from gbxcule.core.abi import CYCLES_PER_SCANLINE

from .conftest import ROM_PATH, require_rom


def _make_backend(frames_per_step: int = 1) -> WarpVecCpuBackend:
    require_rom(ROM_PATH)
    return WarpVecCpuBackend(
        str(ROM_PATH),
        num_envs=1,
        obs_dim=32,
        frames_per_step=frames_per_step,
    )


def _neutralize_bootrom(backend: WarpVecCpuBackend) -> None:
    """Overwrite boot ROM area with NOPs to keep early execution deterministic."""
    backend.write_memory(0, 0x0000, bytes([0x00] * 0x100))


def test_scanline_state_invariants_and_vblank_request() -> None:
    backend = _make_backend(frames_per_step=1)
    try:
        backend.reset()
        _neutralize_bootrom(backend)
        # Disable interrupts so IF bit 0 remains set after VBlank request.
        backend.write_memory(0, 0xFFFF, bytes([0x00]))
        backend.write_memory(0, 0xFF0F, bytes([0x00]))
        if backend._ime is not None:
            backend._ime.numpy()[0] = 0

        # Ensure LCD is enabled so scanlines advance.
        backend.write_memory(0, 0xFF40, bytes([0x80]))

        actions = np.zeros((1,), dtype=np.int32)
        backend.step(actions)

        assert backend._ppu_ly is not None
        assert backend._ppu_scanline_cycle is not None

        ly = int(backend._ppu_ly.numpy()[0])
        scanline_cycle = int(backend._ppu_scanline_cycle.numpy()[0])

        assert 0 <= ly < 154
        assert 0 <= scanline_cycle < CYCLES_PER_SCANLINE

        if_val = backend.read_memory(0, 0xFF0F, 0xFF10)[0]
        assert (if_val & 0x01) == 0x01
    finally:
        backend.close()


def test_scanline_latch_captures_env0() -> None:
    backend = _make_backend(frames_per_step=1)
    try:
        backend.reset()
        _neutralize_bootrom(backend)
        # Force LCD on and known register values before stepping.
        backend.write_memory(0, 0xFF40, bytes([0x91]))
        backend.write_memory(0, 0xFF42, bytes([0x22]))
        backend.write_memory(0, 0xFF43, bytes([0x11]))
        backend.write_memory(0, 0xFF47, bytes([0xE4]))

        actions = np.zeros((1,), dtype=np.int32)
        backend.step(actions)

        assert backend._bg_lcdc_latch_env0 is not None
        assert backend._bg_scx_latch_env0 is not None
        assert backend._bg_scy_latch_env0 is not None
        assert backend._bg_bgp_latch_env0 is not None

        lcdc_latch = int(backend._bg_lcdc_latch_env0.numpy()[0])
        scx_latch = int(backend._bg_scx_latch_env0.numpy()[0])
        scy_latch = int(backend._bg_scy_latch_env0.numpy()[0])
        bgp_latch = int(backend._bg_bgp_latch_env0.numpy()[0])

        assert lcdc_latch == 0x91
        assert scx_latch == 0x11
        assert scy_latch == 0x22
        assert bgp_latch == 0xE4
    finally:
        backend.close()
