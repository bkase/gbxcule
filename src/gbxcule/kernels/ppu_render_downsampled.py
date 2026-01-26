"""Warp kernels for snapshot downsampled PPU rendering (all envs)."""

from __future__ import annotations

import warp as wp

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W, MEM_SIZE
from gbxcule.kernels.cpu_step import get_warp

IO_LCDC = 0xFF40
IO_SCY = 0xFF42
IO_SCX = 0xFF43
IO_BGP = 0xFF47
IO_OBP0 = 0xFF48
IO_OBP1 = 0xFF49
IO_WY = 0xFF4A
IO_WX = 0xFF4B

TILE_DATA_0 = 0x8000
TILE_DATA_1 = 0x9000
TILE_MAP_0 = 0x9800
TILE_MAP_1 = 0x9C00
OAM_BASE = 0xFE00


@wp.kernel
def ppu_render_shades_downsampled_all_envs(
    mem: wp.array(dtype=wp.uint8),
    out_pixels: wp.array(dtype=wp.uint8),
):
    idx = wp.tid()
    out_stride = DOWNSAMPLE_W * DOWNSAMPLE_H
    env_idx = idx // out_stride
    if env_idx < 0:
        return
    offset = idx - env_idx * out_stride
    if offset < 0 or offset >= out_stride:
        return
    oy = offset // DOWNSAMPLE_W
    ox = offset - oy * DOWNSAMPLE_W

    x = ox * 2
    y = oy * 2

    base = env_idx * MEM_SIZE

    lcdc = wp.int32(mem[base + IO_LCDC]) & 0xFF
    scy = wp.int32(mem[base + IO_SCY]) & 0xFF
    scx = wp.int32(mem[base + IO_SCX]) & 0xFF
    bgp = wp.int32(mem[base + IO_BGP]) & 0xFF
    obp0 = wp.int32(mem[base + IO_OBP0]) & 0xFF
    obp1 = wp.int32(mem[base + IO_OBP1]) & 0xFF
    wy = wp.int32(mem[base + IO_WY]) & 0xFF
    wx = wp.int32(mem[base + IO_WX]) & 0xFF

    shade = wp.int32(0)
    bg_color = wp.int32(0)

    if (lcdc & 0x01) != 0:
        use_window = wp.int32(0)
        if (lcdc & 0x20) != 0 and y >= wy:
            wx_start = wx - 7
            if x >= wx_start:
                use_window = 1

        if use_window != 0:
            wx_start = wx - 7
            win_x = x - wx_start
            win_y = y - wy
            tile_map = wp.int32(TILE_MAP_0)
            if (lcdc & 0x40) != 0:
                tile_map = wp.int32(TILE_MAP_1)
            tile_x = (win_x & 0xFF) >> 3
            tile_y = (win_y & 0xFF) >> 3
            map_addr = tile_map + tile_y * 32 + tile_x
            tile_id = wp.int32(mem[base + map_addr]) & 0xFF
            tile_addr = wp.int32(TILE_DATA_0)
            if (lcdc & 0x10) != 0:
                tile_addr = tile_addr + tile_id * 16
            else:
                signed_id = tile_id
                if signed_id >= 128:
                    signed_id = signed_id - 256
                tile_addr = wp.int32(TILE_DATA_1) + signed_id * 16
            row = win_y & 7
            lo = wp.int32(mem[base + tile_addr + row * 2]) & 0xFF
            hi = wp.int32(mem[base + tile_addr + row * 2 + 1]) & 0xFF
            bit = 7 - (win_x & 7)
            bg_color = ((hi >> bit) & 1) << 1 | ((lo >> bit) & 1)
        else:
            sx = (x + scx) & 0xFF
            sy = (y + scy) & 0xFF

            tile_map = wp.int32(TILE_MAP_0)
            if (lcdc & 0x08) != 0:
                tile_map = wp.int32(TILE_MAP_1)

            tile_x = sx >> 3
            tile_y = sy >> 3
            map_addr = tile_map + tile_y * 32 + tile_x
            tile_id = wp.int32(mem[base + map_addr]) & 0xFF

            tile_addr = wp.int32(TILE_DATA_0)
            if (lcdc & 0x10) != 0:
                tile_addr = tile_addr + tile_id * 16
            else:
                signed_id = tile_id
                if signed_id >= 128:
                    signed_id = signed_id - 256
                tile_addr = wp.int32(TILE_DATA_1) + signed_id * 16

            row = sy & 7
            lo = wp.int32(mem[base + tile_addr + row * 2]) & 0xFF
            hi = wp.int32(mem[base + tile_addr + row * 2 + 1]) & 0xFF

            bit = 7 - (sx & 7)
            bg_color = ((hi >> bit) & 1) << 1 | ((lo >> bit) & 1)

        shade = (bgp >> (bg_color * 2)) & 0x03

    if (lcdc & 0x02) != 0:
        sprite_height = wp.int32(8)
        if (lcdc & 0x04) != 0:
            sprite_height = 16
        sprites_on_line = wp.int32(0)
        best_found = wp.int32(0)
        best_x = wp.int32(0)
        best_oam = wp.int32(0)
        best_color = wp.int32(0)
        best_attr = wp.int32(0)
        for sprite_idx in range(40):
            oam_addr = wp.int32(OAM_BASE + sprite_idx * 4)
            sprite_y = wp.int32(mem[base + oam_addr]) & 0xFF
            sprite_x = wp.int32(mem[base + oam_addr + 1]) & 0xFF
            tile_id = wp.int32(mem[base + oam_addr + 2]) & 0xFF
            attr = wp.int32(mem[base + oam_addr + 3]) & 0xFF
            if sprite_x == 0 or sprite_y == 0 or sprite_x >= 168 or sprite_y >= 160:
                continue
            sprite_y_top = sprite_y - 16
            sprite_x_left = sprite_x - 8
            if y < sprite_y_top or y >= sprite_y_top + sprite_height:
                continue
            if sprites_on_line >= 10:
                continue
            sprites_on_line = sprites_on_line + 1
            if x < sprite_x_left or x >= sprite_x_left + 8:
                continue
            row = y - sprite_y_top
            yflip = (attr >> 6) & 1
            if yflip != 0:
                row = sprite_height - 1 - row
            if sprite_height == 16:
                tile_id = tile_id & 0xFE
                if row >= 8:
                    tile_id = tile_id + 1
                    row = row - 8
            tile_addr = wp.int32(TILE_DATA_0) + tile_id * 16
            lo = wp.int32(mem[base + tile_addr + row * 2]) & 0xFF
            hi = wp.int32(mem[base + tile_addr + row * 2 + 1]) & 0xFF
            px = x - sprite_x_left
            xflip = (attr >> 5) & 1
            bit = px
            if xflip == 0:
                bit = 7 - px
            color = ((hi >> bit) & 1) << 1 | ((lo >> bit) & 1)
            if color == 0:
                continue
            if best_found == 0:
                best_found = 1
                best_x = sprite_x_left
                best_oam = sprite_idx
                best_color = color
                best_attr = attr
            else:
                if sprite_x_left < best_x:
                    best_x = sprite_x_left
                    best_oam = sprite_idx
                    best_color = color
                    best_attr = attr
                elif sprite_x_left == best_x and sprite_idx < best_oam:
                    best_oam = sprite_idx
                    best_color = color
                    best_attr = attr

        if best_found != 0:
            priority = (best_attr >> 7) & 1
            if not (priority != 0 and bg_color != 0):
                palette = obp0
                if ((best_attr >> 4) & 1) != 0:
                    palette = obp1
                shade = (palette >> (best_color * 2)) & 0x03

    out_pixels[idx] = wp.uint8(shade)


def get_ppu_render_downsampled_kernel():  # type: ignore[no-untyped-def]
    """Return the snapshot downsampled renderer kernel."""
    get_warp()
    return ppu_render_shades_downsampled_all_envs
