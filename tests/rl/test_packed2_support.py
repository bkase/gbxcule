from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES
from gbxcule.rl.goal_template import load_goal_template
from gbxcule.rl.models import PixelActorCriticCNN
from gbxcule.rl.packed_pixels import pack_2bpp_u8, unpack_2bpp_u8
from gbxcule.rl.rollout import RolloutBuffer


def test_obs_stored_as_packed2() -> None:
    rollout = RolloutBuffer(
        steps=64,
        num_envs=4,
        stack_k=1,
        obs_format="packed2",
        device="cpu",
    )
    assert rollout.obs.shape == (64, 4, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
    assert rollout.obs.dtype is torch.uint8


def test_goal_stored_as_packed2() -> None:
    goal_dir = Path("states/rl_stage1_exit_oak")
    if not (goal_dir / "goal_template.npy").exists():
        pytest.skip("goal template missing")
    goal, _ = load_goal_template(goal_dir, obs_format="packed2")
    assert goal.shape[-2:] == (DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
    assert goal.dtype == np.uint8


def test_packed2_roundtrip() -> None:
    img_u8 = torch.randint(0, 4, (4, 1, 72, 80), dtype=torch.uint8)
    packed = pack_2bpp_u8(img_u8)
    assert packed.shape == (4, 1, 72, 20)
    unpacked = unpack_2bpp_u8(packed)
    assert torch.equal(img_u8, unpacked)


def test_policy_accepts_packed2() -> None:
    model = PixelActorCriticCNN(num_actions=5, in_frames=1, input_format="packed2")
    obs_packed = torch.randint(0, 256, (4, 1, 72, 20), dtype=torch.uint8)
    logits, values = model(obs_packed)
    assert logits.shape == (4, 5)
    assert values.shape == (4,)
