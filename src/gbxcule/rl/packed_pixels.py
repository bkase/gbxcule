"""Packed 2bpp framebuffer utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_UNPACK_LUT_CACHE: dict[tuple[str, str], torch.Tensor] = {}
_DIFF_LUT_CACHE: dict[str, torch.Tensor] = {}


def _require_torch():
    import importlib

    return importlib.import_module("torch")


def _device_key(device) -> str:  # type: ignore[no-untyped-def]
    if device is None:
        return "cpu"
    return str(device)


def get_unpack_lut(device=None, dtype=None):  # type: ignore[no-untyped-def]
    """Return a LUT of shape [256, 4] for unpacking bytes to 2bpp pixels."""
    torch = _require_torch()
    if dtype is None:
        dtype = torch.uint8
    device_key = _device_key(device)
    key = (device_key, str(dtype))
    if key in _UNPACK_LUT_CACHE:
        return _UNPACK_LUT_CACHE[key]
    vals = torch.arange(256, dtype=torch.uint8)
    p0 = vals & 0x03
    p1 = (vals >> 2) & 0x03
    p2 = (vals >> 4) & 0x03
    p3 = (vals >> 6) & 0x03
    lut = torch.stack((p0, p1, p2, p3), dim=1).to(dtype=dtype)
    if device is not None:
        lut = lut.to(device)
    _UNPACK_LUT_CACHE[key] = lut
    return lut


def pack_2bpp_u8(unpacked_u8):  # type: ignore[no-untyped-def]
    """Pack u8 pixels (0..3) into 2bpp bytes along the last dimension."""
    torch = _require_torch()
    if unpacked_u8.dtype is not torch.uint8:
        raise ValueError("unpacked_u8 must be uint8")
    if unpacked_u8.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be divisible by 4")
    w = int(unpacked_u8.shape[-1])
    packed = unpacked_u8.reshape(*unpacked_u8.shape[:-1], w // 4, 4)
    b0 = packed[..., 0]
    b1 = packed[..., 1]
    b2 = packed[..., 2]
    b3 = packed[..., 3]
    return b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)


def unpack_2bpp_u8(packed_u8, lut=None):  # type: ignore[no-untyped-def]
    """Unpack 2bpp bytes into u8 pixels (0..3) along the last dimension."""
    torch = _require_torch()
    if packed_u8.dtype is not torch.uint8:
        raise ValueError("packed_u8 must be uint8")
    if lut is None:
        lut = get_unpack_lut(device=packed_u8.device, dtype=torch.uint8)
    unpacked = lut[packed_u8.to(torch.int64)]
    return unpacked.reshape(*packed_u8.shape[:-1], packed_u8.shape[-1] * 4)


def get_diff_lut(device=None):  # type: ignore[no-untyped-def]
    """Return a LUT of shape [256, 256] with per-byte L1 distances."""
    torch = _require_torch()
    device_key = _device_key(device)
    if device_key in _DIFF_LUT_CACHE:
        return _DIFF_LUT_CACHE[device_key]
    lut = get_unpack_lut(device=None, dtype=torch.int16)
    diff = (lut[:, None, :] - lut[None, :, :]).abs().sum(dim=-1)
    diff_u8 = diff.to(dtype=torch.uint8)
    if device is not None:
        diff_u8 = diff_u8.to(device)
    _DIFF_LUT_CACHE[device_key] = diff_u8
    return diff_u8
