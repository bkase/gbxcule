from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W  # noqa: E402
from gbxcule.rl.models import PixelActorCriticCNN  # noqa: E402
from gbxcule.rl.rollout import RolloutBuffer  # noqa: E402


def test_pixel_actor_critic_shapes() -> None:
    model = PixelActorCriticCNN(num_actions=7, in_frames=4)
    obs = torch.zeros((2, 4, DOWNSAMPLE_H, DOWNSAMPLE_W), dtype=torch.uint8)
    logits, values = model(obs)
    assert logits.shape == (2, 7)
    assert values.shape == (2,)
    assert logits.dtype is torch.float32
    assert values.dtype is torch.float32


def test_rollout_buffer_shapes() -> None:
    buffer = RolloutBuffer(steps=3, num_envs=2, stack_k=4, device="cpu")
    obs = torch.zeros(
        (2, 4, DOWNSAMPLE_H, DOWNSAMPLE_W), dtype=torch.uint8, device="cpu"
    )
    actions = torch.zeros((2,), dtype=torch.int32, device="cpu")
    rewards = torch.zeros((2,), dtype=torch.float32, device="cpu")
    dones = torch.zeros((2,), dtype=torch.bool, device="cpu")
    values = torch.zeros((2,), dtype=torch.float32, device="cpu")
    logprobs = torch.zeros((2,), dtype=torch.float32, device="cpu")
    for _ in range(3):
        buffer.add(obs, actions, rewards, dones, values, logprobs)
    batch = buffer.as_batch()
    assert batch["obs_u8"].shape == (6, 4, DOWNSAMPLE_H, DOWNSAMPLE_W)
    assert batch["actions"].shape == (6,)
    assert batch["rewards"].shape == (6,)
