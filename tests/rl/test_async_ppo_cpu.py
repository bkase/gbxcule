from __future__ import annotations

from pathlib import Path

import pytest

from gbxcule.rl.async_ppo_engine import AsyncPPOEngine, AsyncPPOEngineConfig
from gbxcule.rl.experiment import Experiment

ROM_PATH = Path("red.gb")
STATE_PATH = Path("states/rl_stage1_exit_oak/start.state")
GOAL_DIR = Path("states/rl_stage1_exit_oak")


def _skip_if_assets_missing() -> None:
    if not ROM_PATH.exists() or not STATE_PATH.exists() or not GOAL_DIR.exists():
        pytest.skip("ROM/state/goal assets missing")


def _base_cpu_config() -> AsyncPPOEngineConfig:
    return AsyncPPOEngineConfig(
        rom_path=str(ROM_PATH),
        state_path=str(STATE_PATH),
        goal_dir=str(GOAL_DIR),
        device="cpu",
        num_envs=1,
        steps_per_rollout=1,
        updates=1,
        minibatch_size=1,
    )


def test_engine_runs_on_cpu() -> None:
    _skip_if_assets_missing()
    engine = AsyncPPOEngine(_base_cpu_config())
    try:
        metrics = engine.run(updates=1)
        assert metrics["env_steps"] > 0
    finally:
        engine.close()


def test_cpu_produces_standard_artifacts(tmp_path: Path) -> None:
    _skip_if_assets_missing()
    exp = Experiment(algo="ppo", rom_id="test", tag="cpu", run_root=tmp_path)
    engine = AsyncPPOEngine(_base_cpu_config(), experiment=exp)
    try:
        engine.run(updates=1)
    finally:
        engine.close()
    assert (exp.run_dir / "meta.json").exists()
    assert (exp.run_dir / "config.json").exists()
    metrics_path = exp.run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    assert metrics_path.read_text().strip() != ""


def test_cpu_deterministic() -> None:
    _skip_if_assets_missing()
    cfg = _base_cpu_config()
    engine1 = AsyncPPOEngine(cfg)
    engine2 = AsyncPPOEngine(cfg)
    try:
        metrics1 = engine1.run(updates=1)
        metrics2 = engine2.run(updates=1)
    finally:
        engine1.close()
        engine2.close()
    assert metrics1["env_steps"] == metrics2["env_steps"]


def test_cpu_sequential_execution() -> None:
    _skip_if_assets_missing()
    engine = AsyncPPOEngine(_base_cpu_config())
    try:
        assert engine.actor_stream is None
        assert engine.learner_stream is None
    finally:
        engine.close()
