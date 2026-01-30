from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def test_sweep_tool_exists() -> None:
    result = subprocess.run(
        ["python", "tools/rl_bench_sweep_async_ppo.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--num-envs" in result.stdout


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_sweep_outputs(tmp_path: Path) -> None:
    rom_path = Path("red.gb")
    state_path = Path("states/rl_stage1_exit_oak/start.state")
    goal_dir = Path("states/rl_stage1_exit_oak")
    if not rom_path.exists() or not state_path.exists() or not goal_dir.exists():
        pytest.skip("ROM/state/goal assets missing")

    result = subprocess.run(
        [
            "python",
            "tools/rl_bench_sweep_async_ppo.py",
            f"--output-dir={tmp_path}",
            "--num-envs=1,2",
            "--updates=1",
            "--steps-per-rollout=1",
            "--minibatch-size=1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0

    results_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.md"
    plots_dir = tmp_path / "plots"

    assert results_path.exists()
    assert summary_path.exists()
    assert plots_dir.exists()

    records = [json.loads(line) for line in results_path.read_text().splitlines()]
    assert len(records) >= 2
    record = records[0]
    for field in ("num_envs", "sps", "elapsed_s", "env_steps", "overlap_efficiency"):
        assert field in record

    plots = list(plots_dir.glob("*.png"))
    assert len(plots) >= 1
