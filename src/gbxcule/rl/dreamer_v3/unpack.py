"""Packed2 unpack switch for Dreamer v3 (M6 optional)."""

from __future__ import annotations

from gbxcule.rl.packed_pixels import unpack_2bpp_u8


def unpack_packed2(  # type: ignore[no-untyped-def]
    packed_u8,
    *,
    impl: str = "lut",
):
    """Unpack packed2 observations using the selected implementation."""
    if impl == "lut":
        return unpack_2bpp_u8(packed_u8)
    if impl in ("warp", "triton"):
        raise NotImplementedError(f"unpack_impl={impl} not available yet")
    raise ValueError("unpack_impl must be one of {'lut', 'warp', 'triton'}")
