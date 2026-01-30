from __future__ import annotations

from pathlib import Path

from gbxcule.rl.experiment import Experiment


def test_experiment_creates_run_dir(tmp_path: Path) -> None:
    exp = Experiment(algo="ppo", rom_id="test", tag="unit", run_root=tmp_path)
    assert exp.run_dir.exists()
    assert (exp.run_dir / "meta.json").exists()
    assert (exp.run_dir / "config.json").exists()


def test_atomic_json_write(tmp_path: Path) -> None:
    exp = Experiment(algo="ppo", rom_id="test", tag="metrics", run_root=tmp_path)
    exp.log_metrics({"sps": 1000})
    assert not list(exp.run_dir.rglob("*.tmp"))


def test_failure_bundle_on_exception(tmp_path: Path) -> None:
    exp = Experiment(algo="ppo", rom_id="test", tag="fail", run_root=tmp_path)
    failure_dir = exp.write_failure_bundle(
        kind="exception",
        error=ValueError("test error"),
        extra={"note": "unit-test"},
    )
    assert failure_dir.exists()
    assert (failure_dir / "failure.json").exists()
