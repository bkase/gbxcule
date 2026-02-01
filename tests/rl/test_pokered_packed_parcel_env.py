from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES  # noqa: E402
from gbxcule.rl.pokered_packed_parcel_env import (  # noqa: E402
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
GOAL_DIR = Path("states/rl_stage1_exit_oak")


def _assets_available() -> bool:
    return (
        ROM_PATH.exists()
        and GOAL_DIR.exists()
        and (GOAL_DIR / "goal_template.npy").exists()
        and (GOAL_DIR / "goal_template.meta.json").exists()
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_parcel_env_obs_dict_shapes() -> None:
    if not _assets_available():
        pytest.skip("ROM/goal assets missing")
    env = PokeredPackedParcelEnv(
        rom_path=str(ROM_PATH),
        goal_dir=str(GOAL_DIR),
        num_envs=2,
        max_steps=4,
    )
    try:
        obs = env.reset(seed=0)
        assert isinstance(obs, dict)
        assert "pixels" in obs
        assert "snow" in obs
        pixels = obs["pixels"]
        snow = obs["snow"]
        assert pixels.shape == (2, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
        assert snow.shape == (2, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
        assert pixels.dtype is torch.uint8
        assert snow.dtype is torch.uint8
        assert pixels.device.type == "cuda"
        assert snow.device.type == "cuda"
    finally:
        env.close()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_parcel_env_epoch_snow_resets() -> None:
    if not _assets_available():
        pytest.skip("ROM/goal assets missing")
    env = PokeredPackedParcelEnv(
        rom_path=str(ROM_PATH),
        goal_dir=str(GOAL_DIR),
        num_envs=2,
        max_steps=4,
    )
    try:
        obs = env.reset(seed=0)
        snow_before = obs["snow"].clone()
        actions = torch.zeros((env.num_envs,), dtype=torch.int32, device="cuda")
        obs, _, _, _, _ = env.step(actions)
        assert torch.equal(obs["snow"], snow_before)

        mask = torch.tensor([1, 0], dtype=torch.uint8, device="cuda")
        env.reset_mask(mask)
        snow_after = obs["snow"]
        assert not torch.equal(snow_after[0], snow_before[0])
        assert torch.equal(snow_after[1], snow_before[1])
    finally:
        env.close()
