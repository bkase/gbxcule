from __future__ import annotations

import pytest
import torch

from gbxcule.rl.experiment import Experiment
from gbxcule.rl.failfast import assert_device, assert_finite, assert_shape


def test_assert_finite_catches_nan() -> None:
    t = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(AssertionError):
        assert_finite(t, "test_tensor", None, "trace-1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_assert_device_catches_cpu_fallback() -> None:
    t_cpu = torch.zeros(10)
    with pytest.raises(AssertionError):
        assert_device(t_cpu, torch.device("cuda:0"), "test_tensor", None, "trace-1")


def test_assert_shape_catches_mismatch() -> None:
    t = torch.zeros(10, 20)
    with pytest.raises(AssertionError):
        assert_shape(t, (10, 30), "test_tensor", None, "trace-1")


def test_failfast_writes_failure_bundle(tmp_path) -> None:
    exp = Experiment(algo="ppo", rom_id="test", tag="failfast", run_root=tmp_path)
    bad = torch.tensor([1.0, float("nan")])
    snapshots = {
        "obs": torch.zeros((10, 4)),
        "actions": torch.arange(10),
        "logits": torch.randn(10, 3),
        "loss": torch.tensor(1.0),
    }
    with pytest.raises(AssertionError):
        assert_finite(bad, "bad", exp, "trace-1", snapshots=snapshots)
    bundles = list((exp.run_dir / "failures").iterdir())
    assert bundles
    bundle = bundles[0]
    assert (bundle / "failure.json").exists()
    assert (bundle / "tensors.pt").exists()
