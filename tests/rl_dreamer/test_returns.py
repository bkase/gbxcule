from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.imagination import compute_discounts  # noqa: E402
from gbxcule.rl.dreamer_v3.returns import lambda_returns  # noqa: E402
from tests.rl_dreamer.fixtures import load_fixture  # noqa: E402


def test_lambda_returns_cases() -> None:
    payload = load_fixture("lambda_returns_cases.json")
    for case in payload["cases"]:
        rewards = torch.tensor(case["rewards"], dtype=torch.float32).view(-1, 1, 1)
        values = torch.tensor(case["values"], dtype=torch.float32).view(-1, 1, 1)
        continues = torch.tensor(case["continues_gamma"], dtype=torch.float32).view(
            -1, 1, 1
        )
        out = lambda_returns(rewards, values, continues, lmbda=case["lambda"])
        expected = torch.tensor(case["expected"], dtype=torch.float32).view(-1, 1, 1)
        assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_discount_weights() -> None:
    continues = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32).view(-1, 1, 1)
    discount = compute_discounts(continues, gamma=0.9)
    expected = torch.tensor([1.0, 0.9, 0.0], dtype=torch.float32).view(-1, 1, 1)
    assert torch.allclose(discount, expected, atol=1e-6)
