"""Deterministic tests for downsampled pixel renderer (M1)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend
from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W, SCREEN_H, SCREEN_W
from gbxcule.kernels.ppu_render_downsampled import get_ppu_render_downsampled_kernel

from .conftest import require_rom

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"
ROM_PATH = ROM_DIR / "BG_STATIC.gb"


def _blake2b_hex(data: bytes) -> str:
    import hashlib

    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _downsample_frame(frame_bytes: bytes) -> np.ndarray:
    frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(SCREEN_H, SCREEN_W)
    return frame[::2, ::2].copy()


def _pix_env0(backend: WarpVecCpuBackend) -> np.ndarray:
    pix = backend.pixels_wp().numpy()
    pix = pix.reshape(backend.num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W)
    return np.array(pix[0], copy=True)


def _run_deterministic_trace(backend: WarpVecCpuBackend, steps: int) -> list[str]:
    hashes: list[str] = []
    actions = np.zeros((backend.num_envs,), dtype=np.int32)
    for _ in range(steps):
        backend.step(actions)
        hashes.append(_blake2b_hex(_pix_env0(backend).tobytes()))
    return hashes


def test_downsampled_pixels_cpu_deterministic() -> None:
    require_rom(ROM_PATH)
    backend = WarpVecCpuBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels=True,
    )
    try:
        backend.reset(seed=0)
        hashes_a = _run_deterministic_trace(backend, steps=4)
        backend.reset(seed=0)
        hashes_b = _run_deterministic_trace(backend, steps=4)
        assert hashes_a == hashes_b
    finally:
        backend.close()


def test_downsampled_pixels_multi_env_diverge() -> None:
    require_rom(ROM_PATH)
    backend = WarpVecCpuBackend(
        str(ROM_PATH),
        num_envs=2,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels=True,
    )
    try:
        backend.reset(seed=0)

        lcdc = bytes([0x91])
        bgp = bytes([0xE4])
        scx = bytes([0x00])
        scy = bytes([0x00])

        backend.write_memory(0, 0xFF40, lcdc)
        backend.write_memory(0, 0xFF47, bgp)
        backend.write_memory(0, 0xFF43, scx)
        backend.write_memory(0, 0xFF42, scy)

        backend.write_memory(1, 0xFF40, lcdc)
        backend.write_memory(1, 0xFF47, bgp)
        backend.write_memory(1, 0xFF43, scx)
        backend.write_memory(1, 0xFF42, scy)

        tile_zero = bytes([0x00, 0x00] * 8)
        tile_full = bytes([0xFF, 0xFF] * 8)
        backend.write_memory(0, 0x8000, tile_zero)
        backend.write_memory(1, 0x8000, tile_full)
        tile_map = bytes([0x00] * 0x400)
        backend.write_memory(0, 0x9800, tile_map)
        backend.write_memory(1, 0x9800, tile_map)

        kernel = get_ppu_render_downsampled_kernel()
        backend._wp.launch(
            kernel,
            dim=backend.num_envs * DOWNSAMPLE_W * DOWNSAMPLE_H,
            inputs=[backend._mem, backend._pix],
            device=backend._device,
        )
        backend._wp.synchronize()

        pix = backend.pixels_wp().numpy().reshape(
            backend.num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W
        )
        env0_unique = np.unique(pix[0])
        env1_unique = np.unique(pix[1])
        assert env0_unique.size == 1 and env0_unique[0] == 0
        assert env1_unique.size == 1 and env1_unique[0] == 3
    finally:
        backend.close()


def test_downsampled_pixels_env0_matches_downsampled_bg() -> None:
    require_rom(ROM_PATH)
    backend = WarpVecCpuBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_bg=True,
        render_pixels=True,
    )
    try:
        backend.reset(seed=0)
        actions = np.zeros((1,), dtype=np.int32)
        backend.step(actions)
        frame = backend.read_frame_bg_shade_env0()
        down = _downsample_frame(frame)
        pix = _pix_env0(backend)
        mismatch = float(np.mean(down != pix))
        mean_diff = float(abs(down.mean() - pix.mean()))
        assert mismatch < 0.6, f"Mismatch ratio too high: {mismatch:.3f}"
        assert mean_diff <= 0.5, f"Mean diff too high: {mean_diff:.3f}"
    finally:
        backend.close()


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def test_downsampled_pixels_cuda_parity() -> None:
    if os.environ.get("GBXCULE_M1_CUDA") != "1":
        pytest.skip("Set GBXCULE_M1_CUDA=1 to enable CUDA parity test")
    if not _cuda_available():
        pytest.skip("CUDA not available")
    require_rom(ROM_PATH)
    cpu = WarpVecCpuBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels=True,
    )
    cuda = WarpVecCudaBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels=True,
    )
    try:
        cpu.reset(seed=0)
        cuda.reset(seed=0)
        actions = np.zeros((1,), dtype=np.int32)
        for _ in range(3):
            cpu.step(actions)
            cuda.step(actions)
            cpu_pix = cpu.pixels_wp().numpy().tobytes()
            cuda_pix = cuda.pixels_wp().numpy().tobytes()
            assert cpu_pix == cuda_pix
    finally:
        cpu.close()
        cuda.close()
