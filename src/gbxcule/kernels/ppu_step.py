"""Warp kernels for PPU rendering (scanline-accurate BG only)."""

from __future__ import annotations

import warp as wp

from gbxcule.core.abi import SCREEN_H, SCREEN_W
from gbxcule.kernels.cpu_step import get_warp


@wp.kernel
def ppu_render_bg_env0(
    mem: wp.array(dtype=wp.uint8),
    bg_lcdc_latch_env0: wp.array(dtype=wp.uint8),
    bg_scx_latch_env0: wp.array(dtype=wp.uint8),
    bg_scy_latch_env0: wp.array(dtype=wp.uint8),
    bg_bgp_latch_env0: wp.array(dtype=wp.uint8),
    frame_bg_shade_env0: wp.array(dtype=wp.uint8),
):
    idx = wp.tid()
    if idx >= SCREEN_W * SCREEN_H:
        return
    y = idx // SCREEN_W
    x = idx - y * SCREEN_W

    lcdc = wp.int32(bg_lcdc_latch_env0[y]) & 0xFF
    scx = wp.int32(bg_scx_latch_env0[y]) & 0xFF
    scy = wp.int32(bg_scy_latch_env0[y]) & 0xFF
    bgp = wp.int32(bg_bgp_latch_env0[y]) & 0xFF

    shade = wp.int32(0)
    if (lcdc & 0x01) != 0:
        sx = (x + scx) & 0xFF
        sy = (y + scy) & 0xFF

        tile_map = wp.int32(0x9800)
        if (lcdc & 0x08) != 0:
            tile_map = wp.int32(0x9C00)

        tile_x = sx >> 3
        tile_y = sy >> 3
        map_addr = tile_map + tile_y * 32 + tile_x
        tile_id = wp.int32(mem[map_addr]) & 0xFF

        tile_addr = wp.int32(0x8000)
        if (lcdc & 0x10) != 0:
            tile_addr = tile_addr + tile_id * 16
        else:
            signed_id = tile_id
            if signed_id >= 128:
                signed_id = signed_id - 256
            tile_addr = wp.int32(0x9000) + signed_id * 16

        row = sy & 7
        lo = wp.int32(mem[tile_addr + row * 2]) & 0xFF
        hi = wp.int32(mem[tile_addr + row * 2 + 1]) & 0xFF

        bit = 7 - (sx & 7)
        color = ((hi >> bit) & 1) << 1 | ((lo >> bit) & 1)
        shade = (bgp >> (color * 2)) & 0x03

    frame_bg_shade_env0[idx] = wp.uint8(shade)


def get_ppu_render_bg_kernel():  # type: ignore[no-untyped-def]
    """Return the BG render kernel (ensures Warp is initialized)."""
    get_warp()
    return ppu_render_bg_env0
