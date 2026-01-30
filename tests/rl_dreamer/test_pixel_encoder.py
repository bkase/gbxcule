from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.encoders import Packed2PixelEncoder  # noqa: E402
from gbxcule.rl.packed_pixels import pack_2bpp_u8  # noqa: E402


def test_packed2_unpack_normalization() -> None:
    pixels = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.uint8)
    packed = pack_2bpp_u8(pixels)
    norm = Packed2PixelEncoder.unpack_and_normalize(packed)
    expected = torch.tensor(
        [[[[-0.5, -1.0 / 6.0, 1.0 / 6.0, 0.5]]]], dtype=torch.float32
    )
    assert torch.allclose(norm, expected, atol=1e-6)
