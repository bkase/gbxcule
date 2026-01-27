from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W  # noqa: E402
from gbxcule.rl.a2c import a2c_td0_losses  # noqa: E402
from gbxcule.rl.models import PixelActorCriticCNN  # noqa: E402


class _ToyPixelsEnv:
    def __init__(self, num_envs: int, stack_k: int) -> None:
        self.num_envs = num_envs
        self.stack_k = stack_k
        self._step = torch.zeros((num_envs,), dtype=torch.int32)
        self._obs = torch.zeros(
            (num_envs, stack_k, DOWNSAMPLE_H, DOWNSAMPLE_W), dtype=torch.uint8
        )

    def reset(self):  # type: ignore[no-untyped-def]
        self._step.zero_()
        self._obs.zero_()
        return self._obs

    def step(self, actions):  # type: ignore[no-untyped-def]
        self._step.add_(1)
        self._obs[:, :-1].copy_(self._obs[:, 1:])
        fill = (actions % 4).to(torch.uint8).view(self.num_envs, 1, 1)
        self._obs[:, -1].copy_(fill.expand(-1, DOWNSAMPLE_H, DOWNSAMPLE_W))
        reward = self._obs[:, -1].to(torch.float32).mean(dim=(1, 2)) / 3.0
        done = self._step >= 2
        trunc = self._step >= 3
        return self._obs, reward, done, trunc, {}


def test_a2c_smoke_update() -> None:
    torch.manual_seed(0)
    env = _ToyPixelsEnv(num_envs=2, stack_k=2)
    model = PixelActorCriticCNN(num_actions=8, in_frames=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = [p.detach().clone() for p in model.parameters()]
    obs = env.reset()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(2):
        logits, values = model(obs)
        actions_i64 = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        next_obs, reward, done, trunc, _ = env.step(actions_i64.to(torch.int32))
        with torch.no_grad():
            _, v_next = model(next_obs)
        losses = a2c_td0_losses(
            logits,
            actions_i64,
            values,
            reward,
            done,
            trunc,
            v_next,
            gamma=0.99,
            value_coef=0.5,
            entropy_coef=0.01,
        )
        (losses["loss_total"] / 2.0).backward()
        obs = next_obs

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    assert torch.isfinite(losses["loss_total"]).item()

    deltas = [(b - a).abs().sum().item() for b, a in zip(before, model.parameters())]
    assert any(delta > 0.0 for delta in deltas)


def test_a2c_module_available() -> None:
    assert callable(a2c_td0_losses)
