from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

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


def _assets_available() -> bool:
    return ROM_PATH.exists() and STATE_PATH.exists() and GOAL_DIR.exists()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_guardrails() -> None:
    if not _assets_available():
        pytest.skip("ROM/state/goal assets missing")
    cfg = AsyncPPOEngineConfig(
        rom_path=str(ROM_PATH),
        state_path=str(STATE_PATH),
        goal_dir=str(GOAL_DIR),
        device="cuda",
        num_envs=1,
        steps_per_rollout=1,
        updates=1,
        minibatch_size=1,
    )
    engine = AsyncPPOEngine(cfg)
    try:
        engine._assert_cuda_tensors()
        assert engine.rollout_buffers[0].obs.device.type == "cuda"
        for param in engine.actor_model.parameters():
            assert param.device.type == "cuda"
        for param in engine.learner_model.parameters():
            assert param.device.type == "cuda"
    finally:
        engine.close()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_failfast_on_cpu_tensor() -> None:
    if not _assets_available():
        pytest.skip("ROM/state/goal assets missing")
    cfg = AsyncPPOEngineConfig(
        rom_path=str(ROM_PATH),
        state_path=str(STATE_PATH),
        goal_dir=str(GOAL_DIR),
        device="cuda",
        num_envs=1,
        steps_per_rollout=1,
        updates=1,
        minibatch_size=1,
    )
    engine = AsyncPPOEngine(cfg)
    try:
        engine.obs = engine.obs.cpu()
        with pytest.raises(RuntimeError, match="CUDA tensor"):
            engine.run(updates=1)
    finally:
        engine.close()


def test_profile_tool_exists() -> None:
    result = subprocess.run(
        ["python", "tools/rl_profile_one_update.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
