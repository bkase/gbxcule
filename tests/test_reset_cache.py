from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend
from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W, MEM_SIZE
from gbxcule.core.cartridge import CART_STATE_ROM_BANK_LO, CART_STATE_STRIDE
from gbxcule.core.reset_cache import ResetCache

from .conftest import require_rom

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"
ROM_PATH = ROM_DIR / "BG_STATIC.gb"


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def test_reset_cache_cpu_masked_restore() -> None:
    require_rom(ROM_PATH)
    backend = WarpVecCpuBackend(
        str(ROM_PATH),
        num_envs=3,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels=True,
    )
    try:
        backend.reset(seed=0)
        backend.render_pixels_snapshot()
        cache = ResetCache.from_backend(backend, env_idx=0)
        assert backend._pc is not None
        assert backend._cart_state is not None
        assert backend._mem is not None
        assert cache.mem is not None
        assert cache.pc is not None
        assert cache.cart_state is not None
        assert cache.pix is not None

        backend.write_memory(1, 0xC000, b"\xaa\xbb")
        backend.write_memory(2, 0xC000, b"\xcc\xdd")
        backend._pc.numpy()[1] = 0x1234
        backend._pc.numpy()[2] = 0x5678
        cart_state = backend._cart_state.numpy()
        cart_state[1 * CART_STATE_STRIDE + CART_STATE_ROM_BANK_LO] = 5
        cart_state[2 * CART_STATE_STRIDE + CART_STATE_ROM_BANK_LO] = 7

        pix = (
            backend.pixels_wp()
            .numpy()
            .reshape(backend.num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W)
        )
        pix[1].fill(3)
        pix[2].fill(2)

        mask = np.array([0, 1, 0], dtype=np.uint8)
        cache.apply_mask_np(mask)

        mem = backend._mem.numpy()
        base1 = MEM_SIZE * 1
        base2 = MEM_SIZE * 2
        assert mem[base1 + 0xC000 : base1 + 0xC002].tolist() == list(
            cache.mem.numpy()[0xC000:0xC002]
        )
        assert mem[base2 + 0xC000 : base2 + 0xC002].tolist() == [0xCC, 0xDD]

        assert backend._pc.numpy()[1] == cache.pc.numpy()[0]
        assert backend._pc.numpy()[2] == 0x5678

        cart_state = backend._cart_state.numpy().reshape(
            backend.num_envs, CART_STATE_STRIDE
        )
        assert (
            cart_state[1, CART_STATE_ROM_BANK_LO]
            == cache.cart_state.numpy()[CART_STATE_ROM_BANK_LO]
        )
        assert cart_state[2, CART_STATE_ROM_BANK_LO] == 7

        pix = (
            backend.pixels_wp()
            .numpy()
            .reshape(backend.num_envs, DOWNSAMPLE_H, DOWNSAMPLE_W)
        )
        pix_snap = cache.pix.numpy().reshape(DOWNSAMPLE_H, DOWNSAMPLE_W)
        assert np.array_equal(pix[1], pix_snap)
        assert np.all(pix[2] == 2)
    finally:
        backend.close()


def test_reset_cache_cuda_masked_restore_optional() -> None:
    if not _cuda_available():
        pytest.skip("CUDA not available")
    torch = pytest.importorskip("torch")
    require_rom(ROM_PATH)
    backend = WarpVecCudaBackend(
        str(ROM_PATH),
        num_envs=2,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels=True,
    )
    try:
        backend.reset(seed=0)
        backend.render_pixels_snapshot_torch()
        cache = ResetCache.from_backend(backend, env_idx=0)

        backend.write_memory(1, 0xC000, b"\xaa\xbb\xcc\xdd")
        actions = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        for _ in range(4):
            backend.step_torch(actions)

        pre = backend.read_memory(1, 0xC000, 0xC010)
        snap = cache.mem.numpy()[0xC000:0xC010].tobytes()
        assert pre != snap, "env did not diverge from snapshot"

        mask = torch.tensor([0, 1], device="cuda", dtype=torch.uint8)
        cache.apply_mask_torch(mask)

        post = backend.read_memory(1, 0xC000, 0xC010)
        assert post == snap
    finally:
        backend.close()
