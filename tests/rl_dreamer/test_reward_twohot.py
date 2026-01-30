from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.dists import TwoHotEncodingDistribution  # noqa: E402
from tests.rl_dreamer.fixtures import load_fixture  # noqa: E402


def test_twohot_bins_match_fixture() -> None:
    payload = load_fixture("bins.json")
    bins = torch.tensor(payload["bins"], dtype=torch.float32)
    logits = torch.zeros((1, bins.numel()), dtype=torch.float32)
    dist = TwoHotEncodingDistribution(
        logits,
        dims=1,
        low=float(payload["low"]),
        high=float(payload["high"]),
    )
    assert torch.allclose(dist.bins, bins, atol=1e-6, rtol=0.0)
