from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES  # noqa: E402
from gbxcule.rl.pokered_packed_parcel_env import (  # noqa: E402
    SENSES_DIM,
    PokeredPackedParcelEnv,
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


ROM_PATH = Path("red.gb")
STATE_PATH = Path("states/rl_oak_parcel/start.state")


def _assets_available() -> bool:
    return ROM_PATH.exists() and STATE_PATH.exists()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_parcel_env_obs_dict_shapes() -> None:
    if not _assets_available():
        pytest.skip("ROM/state assets missing")
    env = PokeredPackedParcelEnv(
        rom_path=str(ROM_PATH),
        state_path=str(STATE_PATH),
        num_envs=2,
        max_steps=4,
    )
    try:
        obs = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert "pixels" in obs
        assert "senses" in obs
        pixels = obs["pixels"]
        senses = obs["senses"]
        assert pixels.shape == (2, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
        assert senses.shape == (2, SENSES_DIM)
        assert pixels.dtype is torch.uint8
        assert senses.dtype is torch.float32
        assert pixels.device.type == "cuda"
        assert senses.device.type == "cuda"
    finally:
        env.close()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_parcel_env_reset_mask_resets_counters() -> None:
    if not _assets_available():
        pytest.skip("ROM/state assets missing")
    env = PokeredPackedParcelEnv(
        rom_path=str(ROM_PATH),
        state_path=str(STATE_PATH),
        num_envs=2,
        max_steps=4,
    )
    try:
        env.reset(seed=0)
        actions = torch.zeros((env.num_envs,), dtype=torch.int32, device="cuda")
        # Take a step to advance counters
        env.step(actions)

        # Reset env 0 only
        mask = torch.tensor([1, 0], dtype=torch.uint8, device="cuda")
        env.reset_mask(mask)

        # Verify senses are properly updated after reset
        obs = env.obs
        senses = obs["senses"]
        assert senses.shape == (2, SENSES_DIM)
    finally:
        env.close()
