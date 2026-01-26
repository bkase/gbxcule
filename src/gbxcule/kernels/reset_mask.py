"""Warp kernels for masked reset copies."""

from __future__ import annotations

import warp as wp

from gbxcule.kernels.cpu_step import get_warp


@wp.kernel
def reset_copy_strided_u8(
    mask: wp.array(dtype=wp.uint8),
    src: wp.array(dtype=wp.uint8),
    dest: wp.array(dtype=wp.uint8),
    stride: wp.int32,
) -> None:
    idx = wp.tid()
    if stride <= 0:
        return
    env_idx = idx // stride
    if mask[env_idx] == 0:
        return
    offset = idx - env_idx * stride
    dest[idx] = src[offset]


@wp.kernel
def reset_copy_strided_i32(
    mask: wp.array(dtype=wp.uint8),
    src: wp.array(dtype=wp.int32),
    dest: wp.array(dtype=wp.int32),
    stride: wp.int32,
) -> None:
    idx = wp.tid()
    if stride <= 0:
        return
    env_idx = idx // stride
    if mask[env_idx] == 0:
        return
    offset = idx - env_idx * stride
    dest[idx] = src[offset]


@wp.kernel
def reset_copy_scalar_i32(
    mask: wp.array(dtype=wp.uint8),
    src: wp.array(dtype=wp.int32),
    dest: wp.array(dtype=wp.int32),
) -> None:
    idx = wp.tid()
    if mask[idx] == 0:
        return
    dest[idx] = src[0]


@wp.kernel
def reset_copy_scalar_i64(
    mask: wp.array(dtype=wp.uint8),
    src: wp.array(dtype=wp.int64),
    dest: wp.array(dtype=wp.int64),
) -> None:
    idx = wp.tid()
    if mask[idx] == 0:
        return
    dest[idx] = src[0]


@wp.kernel
def reset_copy_scalar_u8(
    mask: wp.array(dtype=wp.uint8),
    src: wp.array(dtype=wp.uint8),
    dest: wp.array(dtype=wp.uint8),
) -> None:
    idx = wp.tid()
    if mask[idx] == 0:
        return
    dest[idx] = src[0]


def ensure_reset_kernels_loaded() -> None:
    """Ensure Warp is initialized so kernels can be launched."""
    get_warp()
