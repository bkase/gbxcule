from __future__ import annotations

import os
from pathlib import Path

import pytest

from gbxcule.backends.warp_vec import WarpVecCudaBackend


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


ROM_PATH = Path("red.gb")


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_no_implicit_render_when_disabled() -> None:
    if not ROM_PATH.exists():
        pytest.skip("ROM missing")
    backend = WarpVecCudaBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=1,
        render_pixels=True,
        render_pixels_packed=False,
        render_on_step=False,
    )
    try:
        backend.reset(seed=0)
        backend.render_pixels_snapshot_torch()
        before = backend.pixels_torch().clone()
        import torch

        actions = torch.zeros((1,), dtype=torch.int32, device="cuda")
        backend.step_torch(actions)
        after = backend.pixels_torch()
        assert torch.equal(before, after)
    finally:
        backend.close()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_explicit_render_to_buffer() -> None:
    if not ROM_PATH.exists():
        pytest.skip("ROM missing")
    backend = WarpVecCudaBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=1,
        render_pixels=False,
        render_pixels_packed=False,
        render_on_step=False,
    )
    try:
        backend.reset(seed=0)
        import torch

        out = torch.zeros((1, 1, 72, 20), dtype=torch.uint8, device="cuda")
        backend.render_pixels_snapshot_packed_to_torch(out, 0)
        assert out.sum().item() > 0
    finally:
        backend.close()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_render_to_offset() -> None:
    if not ROM_PATH.exists():
        pytest.skip("ROM missing")
    backend = WarpVecCudaBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=1,
        render_pixels=False,
        render_pixels_packed=False,
        render_on_step=False,
    )
    try:
        backend.reset(seed=0)
        import torch

        buffer = torch.zeros((2, 1, 1, 72, 20), dtype=torch.uint8, device="cuda")
        frame_bytes = 1 * 72 * 20
        backend.render_pixels_snapshot_packed_to_torch(buffer, frame_bytes)
        assert buffer[0].sum().item() == 0
        assert buffer[1].sum().item() > 0
    finally:
        backend.close()
