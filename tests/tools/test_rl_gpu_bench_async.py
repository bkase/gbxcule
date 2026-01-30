from __future__ import annotations

import ast
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


def test_tool_imports_engine() -> None:
    source = Path("tools/rl_gpu_bench_async.py").read_text()
    tree = ast.parse(source)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert any("async_ppo_engine" in ast.unparse(node) for node in imports)


def test_tool_uses_experiment_harness() -> None:
    source = Path("tools/rl_gpu_bench_async.py").read_text()
    assert "Experiment" in source


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_tool_creates_standard_artifacts(tmp_path: Path) -> None:
    rom_path = Path("red.gb")
    state_path = Path("states/rl_stage1_exit_oak/start.state")
    goal_dir = Path("states/rl_stage1_exit_oak")
    if not rom_path.exists() or not state_path.exists() or not goal_dir.exists():
        pytest.skip("ROM/state/goal assets missing")
    result = subprocess.run(
        [
            "python",
            "tools/rl_gpu_bench_async.py",
            "--num-envs=1",
            "--steps-per-rollout=1",
            "--updates=1",
            "--minibatch-size=1",
            f"--output-dir={tmp_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    run_dirs = list(tmp_path.glob("*__ppo__*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
