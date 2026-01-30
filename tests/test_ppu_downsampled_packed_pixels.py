"""Packed downsampled pixel renderer parity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gbxcule.backends.warp_vec import WarpVecCpuBackend
from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W, DOWNSAMPLE_W_BYTES
from gbxcule.rl.packed_pixels import unpack_2bpp_u8

from .conftest import require_rom

torch = pytest.importorskip("torch")

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"
ROM_PATH = ROM_DIR / "BG_STATIC.gb"


def test_packed_render_matches_unpacked_cpu() -> None:
    require_rom(ROM_PATH)
    backend = WarpVecCpuBackend(
        str(ROM_PATH),
        num_envs=2,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels=True,
        render_pixels_packed=True,
    )
    try:
        backend.reset(seed=0)
        actions = np.zeros((backend.num_envs,), dtype=np.int32)
        for _ in range(3):
            backend.step(actions)
            unpacked_np = (
                backend.pixels_wp()
                .numpy()
                .reshape(backend.num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W)
            )
            packed_np = (
                backend.pixels_packed_wp()
                .numpy()
                .reshape(backend.num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
            )
            packed_t = torch.from_numpy(packed_np.astype(np.uint8))
            unpacked_t = torch.from_numpy(unpacked_np.astype(np.uint8))
            roundtrip = unpack_2bpp_u8(packed_t)
            assert torch.equal(roundtrip, unpacked_t)
    finally:
        backend.close()
