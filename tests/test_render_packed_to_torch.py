"""CUDA tests for rendering packed pixels into external torch buffers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import warp as wp

from gbxcule.backends.warp_vec import WarpVecCudaBackend
from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES

from .conftest import require_rom

torch = pytest.importorskip("torch")

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"
ROM_PATH = ROM_DIR / "BG_STATIC.gb"


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    return wp.is_cuda_available()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_render_packed_to_external_buffer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")
    require_rom(ROM_PATH)
    backend = WarpVecCudaBackend(
        str(ROM_PATH),
        num_envs=2,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels_packed=True,
    )
    try:
        backend.reset(seed=0)
        backend.render_pixels_snapshot()
        internal = backend.pixels_packed_torch().clone()

        out = torch.empty(
            (backend.num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
            device="cuda",
            dtype=torch.uint8,
        )
        backend.render_pixels_snapshot_packed_to_torch(out)
        torch.cuda.synchronize()
        assert torch.equal(out, internal)

        frame_bytes = backend.num_envs * DOWNSAMPLE_H * DOWNSAMPLE_W_BYTES
        out2 = torch.empty(
            (2, backend.num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
            device="cuda",
            dtype=torch.uint8,
        )
        backend.render_pixels_snapshot_packed_to_torch(
            out2, base_offset_bytes=frame_bytes
        )
        torch.cuda.synchronize()
        assert torch.equal(
            out2.view(-1)[frame_bytes : frame_bytes * 2].view_as(out), internal
        )
    finally:
        backend.close()
