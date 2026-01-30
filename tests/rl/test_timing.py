from __future__ import annotations

import os
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


def test_overlap_efficiency_calculation() -> None:
    t_actor, t_learner, t_total = 50.0, 50.0, 75.0
    efficiency = (t_actor + t_learner) / t_total
    assert efficiency > 1.0


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_timing_metrics_logged() -> None:
    if not ROM_PATH.exists() or not STATE_PATH.exists() or not GOAL_DIR.exists():
        pytest.skip("ROM/state/goal assets missing")
    cfg = AsyncPPOEngineConfig(
        rom_path=str(ROM_PATH),
        state_path=str(STATE_PATH),
        goal_dir=str(GOAL_DIR),
        num_envs=4,
        steps_per_rollout=2,
        updates=2,
        minibatch_size=8,
    )
    engine = AsyncPPOEngine(cfg)
    try:
        metrics = engine.run(updates=2)
        assert "t_actor_rollout_ms" in metrics
        assert "t_learner_update_ms" in metrics
        assert "t_total_update_ms" in metrics
        assert "overlap_efficiency" in metrics
    finally:
        engine.close()
