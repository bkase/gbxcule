"""Tests for RL M2 pixels wrapper (CUDA optional)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gbxcule.rl.pokered_pixels_env import PokeredPixelsEnv

from .conftest import require_rom

ROM_PATH = (
    Path(__file__).parent.parent / "bench" / "roms" / "out" / "BG_SCROLL_SIGNED.gb"
)


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def _require_torch():
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception:
        pytest.skip("torch not installed", allow_module_level=True)


def _hash_pixels_u64(pix, torch):  # type: ignore[no-untyped-def]
    flat = pix.reshape(pix.shape[0], -1).to(torch.int64)
    prime = torch.tensor(1315423911, device=flat.device, dtype=torch.int64)
    return (flat * prime).sum(dim=1)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_pixels_wrapper_determinism() -> None:
    torch = _require_torch()
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")
    require_rom(ROM_PATH)

    def run_trace(seed: int) -> list[list[int]]:
        env = PokeredPixelsEnv(str(ROM_PATH), num_envs=2, frames_per_step=1)
        try:
            env.reset(seed=seed)
            gen = torch.Generator(device="cuda")
            gen.manual_seed(seed)
            hashes: list[list[int]] = []
            for _ in range(4):
                actions = torch.randint(
                    0,
                    env.backend.num_actions,
                    (env.num_envs,),
                    device="cuda",
                    dtype=torch.int32,
                    generator=gen,
                )
                env.step(actions)
                h = _hash_pixels_u64(env.pixels, torch).tolist()
                hashes.append(h)
            return hashes
        finally:
            env.close()

    assert run_trace(1234) == run_trace(1234)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_pixels_pointer_stability() -> None:
    torch = _require_torch()
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")
    require_rom(ROM_PATH)

    env = PokeredPixelsEnv(str(ROM_PATH), num_envs=1, frames_per_step=1)
    try:
        env.reset(seed=0)
        ptr0 = env.pixels.data_ptr()
        actions = torch.zeros((1,), device="cuda", dtype=torch.int32)
        env.step(actions)
        ptr1 = env.pixels.data_ptr()
        assert ptr0 == ptr1
    finally:
        env.close()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_stream_ordering_updates_pixels() -> None:
    torch = _require_torch()
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")
    require_rom(ROM_PATH)

    env = PokeredPixelsEnv(str(ROM_PATH), num_envs=1, frames_per_step=1)
    try:
        env.reset(seed=0)
        h0 = _hash_pixels_u64(env.pixels, torch)[0].item()
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1234)
        changed = False
        for _ in range(8):
            actions = torch.randint(
                0,
                env.backend.num_actions,
                (1,),
                device="cuda",
                dtype=torch.int32,
                generator=gen,
            )
            env.step(actions)
            h1 = _hash_pixels_u64(env.pixels, torch)[0].item()
            if h1 != h0:
                changed = True
                break
        assert changed, "pixel hash did not change after steps"
    finally:
        env.close()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_stack_k1_matches_pixels() -> None:
    torch = _require_torch()
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")
    require_rom(ROM_PATH)

    env = PokeredPixelsEnv(str(ROM_PATH), num_envs=2, frames_per_step=1, stack_k=1)
    try:
        obs = env.reset(seed=0)
        assert obs.shape[1] == 1
        assert torch.equal(obs[:, 0], env.pixels)
        actions = torch.zeros((env.num_envs,), device="cuda", dtype=torch.int32)
        obs = env.step(actions)
        assert torch.equal(obs[:, 0], env.pixels)
    finally:
        env.close()
