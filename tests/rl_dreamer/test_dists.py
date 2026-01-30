from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.dists import SymlogTwoHot  # noqa: E402
from tests.rl_dreamer.fixtures import load_fixture  # noqa: E402


def _load_bins():  # type: ignore[no-untyped-def]
    data = load_fixture("bins.json")
    return torch.tensor(data["bins"], dtype=torch.float32)


def test_symlog_twohot_log_prob_parity() -> None:
    bins = _load_bins()
    payload = load_fixture("symlog_twohot_dist.json")
    logits = torch.tensor(payload["logits"], dtype=torch.float32)
    targets = torch.tensor(payload["targets"], dtype=torch.float32).unsqueeze(-1)
    expected = torch.tensor(payload["log_prob"], dtype=torch.float32)
    dist = SymlogTwoHot(logits, bins=bins, dims=1)
    actual = dist.log_prob(targets).squeeze(-1)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_symlog_twohot_mean_parity() -> None:
    bins = _load_bins()
    payload = load_fixture("symlog_twohot_dist.json")
    logits = torch.tensor(payload["logits"], dtype=torch.float32)
    expected = torch.tensor(payload["mean"], dtype=torch.float32)
    dist = SymlogTwoHot(logits, bins=bins, dims=1)
    actual = dist.mean.squeeze(-1)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_mode_matches_mean() -> None:
    bins = _load_bins()
    logits = torch.tensor([[0.0, 1.0, 2.0, 1.0, 0.0, -1.0, -2.0]], dtype=torch.float32)
    dist = SymlogTwoHot(logits, bins=bins, dims=1)
    assert torch.allclose(dist.mode, dist.mean, atol=1e-6, rtol=1e-6)
