from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W  # noqa: E402
from gbxcule.rl.models import PixelActorCriticCNN  # noqa: E402
from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses  # noqa: E402
from gbxcule.rl.rollout import RolloutBuffer  # noqa: E402


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


def test_m5_smoke_rollout_update() -> None:
    torch.manual_seed(0)
    env = _ToyPixelsEnv(num_envs=2, stack_k=4)
    model = PixelActorCriticCNN(num_actions=8, in_frames=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rollout = RolloutBuffer(steps=2, num_envs=2, stack_k=4, device="cpu")

    obs = env.reset()
    for _ in range(2):
        logits, values = model(obs)
        actions_i64 = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        logprobs = logprob_from_logits(logits, actions_i64)
        actions = actions_i64.to(torch.int32)
        next_obs, reward, done, trunc, _ = env.step(actions)
        rollout.add(obs, actions, reward, done | trunc, values, logprobs)
        obs = next_obs

    _, last_value = model(obs)
    advantages, returns = compute_gae(
        rollout.rewards,
        rollout.values,
        rollout.dones,
        last_value,
        gamma=0.99,
        gae_lambda=0.95,
    )

    batch = rollout.as_batch(flatten_obs=True)
    logits, values = model(batch["obs_u8"])
    losses = ppo_losses(
        logits,
        batch["actions"],
        batch["logprobs"],
        returns.reshape(-1),
        advantages.reshape(-1),
        values,
        clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    optimizer.zero_grad()
    losses["loss_total"].backward()
    optimizer.step()
    assert torch.isfinite(losses["loss_total"]).item()
