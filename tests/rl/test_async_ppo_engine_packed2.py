from __future__ import annotations

import os
from pathlib import Path

import pytest

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES
from gbxcule.rl.async_ppo_engine import AsyncPPOEngine, AsyncPPOEngineConfig


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
STATE_PATH = Path("states/rl_stage1_exit_oak/start.state")
GOAL_DIR = Path("states/rl_stage1_exit_oak")


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_engine_e2e_packed2() -> None:
    if not ROM_PATH.exists() or not STATE_PATH.exists() or not GOAL_DIR.exists():
        pytest.skip("ROM/state/goal assets missing")
    cfg = AsyncPPOEngineConfig(
        rom_path=str(ROM_PATH),
        state_path=str(STATE_PATH),
        goal_dir=str(GOAL_DIR),
        obs_format="packed2",
        num_envs=1,
        steps_per_rollout=1,
        updates=1,
        minibatch_size=1,
    )
    engine = AsyncPPOEngine(cfg)
    try:
        metrics = engine.run(updates=1)
        assert metrics["env_steps"] > 0
        rollout = engine.rollout_buffers[0]
        assert rollout.obs.shape[-2:] == (DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
        assert rollout.obs_slot(0).sum().item() > 0
    finally:
        engine.close()
