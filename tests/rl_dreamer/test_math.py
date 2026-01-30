from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3 import math as dv3_math  # noqa: E402
from tests.rl_dreamer.fixtures import load_fixture  # noqa: E402


def _load_bins():  # type: ignore[no-untyped-def]
    data = load_fixture("bins.json")
    return torch.tensor(data["bins"], dtype=torch.float32)


def test_symlog_symexp_roundtrip() -> None:
    values = torch.tensor(
        [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0],
        dtype=torch.float32,
    )
    roundtrip = dv3_math.symexp(dv3_math.symlog(values))
    assert torch.allclose(roundtrip, values, atol=1e-5, rtol=1e-5)


def test_symlog_cases_fixture() -> None:
    payload = load_fixture("symlog_cases.json")
    cases = payload["cases"]
    xs = torch.tensor([case["x"] for case in cases], dtype=torch.float32)
    expected_symlog = torch.tensor(
        [case["symlog"] for case in cases], dtype=torch.float32
    )
    expected_symexp = torch.tensor(
        [case["symexp"] for case in cases], dtype=torch.float32
    )
    actual_symlog = dv3_math.symlog(xs)
    actual_symexp = dv3_math.symexp(actual_symlog)
    assert torch.allclose(actual_symlog, expected_symlog, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual_symexp, expected_symexp, atol=1e-5, rtol=1e-5)


def test_twohot_mass_conservation() -> None:
    bins = _load_bins()
    values = torch.tensor([-1e9, -10.0, 0.0, 10.0, 1e9], dtype=torch.float32)
    weights = dv3_math.twohot(dv3_math.symlog(values), bins)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)
    assert torch.all(weights >= -1e-7)


def test_twohot_edge_clamp() -> None:
    bins = _load_bins()
    y_low = dv3_math.symlog(torch.tensor([-1e9], dtype=torch.float32))
    y_high = dv3_math.symlog(torch.tensor([1e9], dtype=torch.float32))
    w_low = dv3_math.twohot(y_low, bins).squeeze(0)
    w_high = dv3_math.twohot(y_high, bins).squeeze(0)
    assert w_low[0].item() == pytest.approx(1.0, abs=1e-5)
    assert w_high[-1].item() == pytest.approx(1.0, abs=1e-5)


def test_twohot_cases_fixture() -> None:
    bins = _load_bins()
    payload = load_fixture("twohot_cases.json")
    cases = payload["cases"]
    xs = torch.tensor([case["x"] for case in cases], dtype=torch.float32)
    expected = torch.tensor([case["weights"] for case in cases], dtype=torch.float32)
    actual = dv3_math.twohot(dv3_math.symlog(xs), bins)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
