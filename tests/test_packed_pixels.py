from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.core.abi import (  # noqa: E402
    DOWNSAMPLE_H,
    DOWNSAMPLE_W,
    DOWNSAMPLE_W_BYTES,
    PACK_PIXELS_PER_BYTE,
    PACKED_FRAME_BYTES,
)
from gbxcule.rl.packed_pixels import (  # noqa: E402
    get_diff_lut,
    pack_2bpp_u8,
    unpack_2bpp_u8,
)


def test_packed_constants_consistency() -> None:
    assert DOWNSAMPLE_W % PACK_PIXELS_PER_BYTE == 0
    assert DOWNSAMPLE_W_BYTES == DOWNSAMPLE_W // PACK_PIXELS_PER_BYTE
    assert PACKED_FRAME_BYTES == DOWNSAMPLE_H * DOWNSAMPLE_W_BYTES


def test_pack_unpack_roundtrip() -> None:
    torch.manual_seed(0)
    pixels = torch.randint(0, 4, (4, DOWNSAMPLE_H, DOWNSAMPLE_W), dtype=torch.uint8)
    packed = pack_2bpp_u8(pixels)
    unpacked = unpack_2bpp_u8(packed)
    assert torch.equal(unpacked, pixels)


def test_diff_lut_correctness() -> None:
    torch.manual_seed(1)
    pixels_a = torch.randint(0, 4, (2, DOWNSAMPLE_H, DOWNSAMPLE_W), dtype=torch.uint8)
    pixels_b = torch.randint(0, 4, (2, DOWNSAMPLE_H, DOWNSAMPLE_W), dtype=torch.uint8)
    packed_a = pack_2bpp_u8(pixels_a)
    packed_b = pack_2bpp_u8(pixels_b)
    diff_lut = get_diff_lut()
    diff_lut_dist = diff_lut[packed_a.to(torch.int64), packed_b.to(torch.int64)].sum(
        dim=(-2, -1)
    )
    ref_dist = (
        (pixels_a.to(torch.int16) - pixels_b.to(torch.int16)).abs().sum(dim=(-2, -1))
    )
    assert torch.equal(diff_lut_dist, ref_dist)
