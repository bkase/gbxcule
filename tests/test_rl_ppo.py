from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses  # noqa: E402


def _manual_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    steps = len(rewards)
    adv = [0.0 for _ in range(steps)]
    gae = 0.0
    for t in range(steps - 1, -1, -1):
        not_done = 0.0 if dones[t] else 1.0
        next_value = last_value if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        adv[t] = gae
    return adv


def test_compute_gae_matches_manual() -> None:
    rewards = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    values = torch.tensor([[0.5], [0.4], [0.3]], dtype=torch.float32)
    dones = torch.tensor([[False], [False], [True]])
    last_value = torch.tensor([0.0], dtype=torch.float32)
    adv, returns = compute_gae(
        rewards,
        values,
        dones,
        last_value,
        gamma=0.99,
        gae_lambda=0.95,
    )
    expected_adv = _manual_gae(
        [1.0, 1.0, 1.0],
        [0.5, 0.4, 0.3],
        [False, False, True],
        0.0,
        gamma=0.99,
        gae_lambda=0.95,
    )
    expected_adv_t = torch.tensor(expected_adv, dtype=torch.float32).unsqueeze(1)
    expected_returns = expected_adv_t + values
    assert torch.allclose(adv, expected_adv_t, atol=1e-5)
    assert torch.allclose(returns, expected_returns, atol=1e-5)


def test_ppo_losses_basic_sanity() -> None:
    logits = torch.zeros((4, 3), dtype=torch.float32)
    actions = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
    old_logprobs = logprob_from_logits(logits, actions)
    returns = torch.tensor([1.0, 0.5, -0.2, 0.1], dtype=torch.float32)
    advantages = torch.tensor([0.3, -0.1, 0.2, -0.4], dtype=torch.float32)
    values = torch.tensor([0.2, 0.2, 0.2, 0.2], dtype=torch.float32)
    losses = ppo_losses(
        logits,
        actions,
        old_logprobs,
        returns,
        advantages,
        values,
        clip=0.1,
        value_coef=0.5,
        entropy_coef=0.01,
    )
    assert math.isfinite(losses["loss_total"].item())
    assert math.isfinite(losses["approx_kl"].item())
    assert math.isfinite(losses["clipfrac"].item())


def test_ppo_gradients_update_params() -> None:
    torch.manual_seed(0)

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = torch.nn.Linear(4, 8)
            self.policy = torch.nn.Linear(8, 8)
            self.value = torch.nn.Linear(8, 1)

        def forward(self, x):  # type: ignore[no-untyped-def]
            h = torch.tanh(self.body(x))
            logits = self.policy(h)
            values = self.value(h).squeeze(-1)
            return logits, values

    model = TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    obs = torch.randn((6, 4), dtype=torch.float32)
    logits, values = model(obs)
    actions = torch.randint(0, 8, (6,), dtype=torch.int64)
    old_logprobs = logprob_from_logits(logits.detach(), actions)
    returns = torch.randn((6,), dtype=torch.float32)
    advantages = torch.randn((6,), dtype=torch.float32)

    losses = ppo_losses(
        logits,
        actions,
        old_logprobs,
        returns,
        advantages,
        values,
        clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    before = [p.detach().clone() for p in model.parameters()]
    opt.zero_grad()
    losses["loss_total"].backward()
    for param in model.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
    opt.step()
    after = list(model.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))
    assert changed
