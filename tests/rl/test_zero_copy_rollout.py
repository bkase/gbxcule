from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from gbxcule.backends.warp_vec import WarpVecCudaBackend
from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES
from gbxcule.rl.rollout import RolloutBuffer


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


ROM_PATH = Path("red.gb")


def test_obs_slot_returns_view() -> None:
    rollout = RolloutBuffer(
        steps=16, num_envs=4, stack_k=1, obs_format="packed2", device="cpu"
    )
    slot = rollout.obs_slot(10)
    slot.fill_(42)
    assert rollout.obs[10].sum().item() == 42 * slot.numel()


def test_set_step_fields_no_obs_copy() -> None:
    rollout = RolloutBuffer(
        steps=8, num_envs=4, stack_k=1, obs_format="packed2", device="cpu"
    )
    rollout.set_step_fields(
        5,
        actions=torch.zeros(4, dtype=torch.int32),
        rewards=torch.ones(4, dtype=torch.float32),
        dones=torch.zeros(4, dtype=torch.bool),
        values=torch.zeros(4, dtype=torch.float32),
        logprobs=torch.zeros(4, dtype=torch.float32),
    )
    assert rollout.actions[5].sum().item() == 0
    assert rollout.rewards[5].sum().item() == 4.0


def test_slots_are_contiguous() -> None:
    rollout = RolloutBuffer(
        steps=8, num_envs=4, stack_k=1, obs_format="packed2", device="cpu"
    )
    for i in range(8):
        slot = rollout.obs_slot(i)
        assert slot.is_contiguous()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_render_into_rollout_slot() -> None:
    if not ROM_PATH.exists():
        pytest.skip("ROM missing")
    rollout = RolloutBuffer(
        steps=4, num_envs=1, stack_k=1, obs_format="packed2", device="cuda"
    )
    backend = WarpVecCudaBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=1,
        render_pixels=False,
        render_pixels_packed=False,
        render_on_step=False,
    )
    try:
        backend.reset(seed=0)
        slot = rollout.obs_slot(2)
        assert slot.shape == (1, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
        backend.render_pixels_snapshot_packed_to_torch(slot.flatten(), 0)
        assert slot.sum().item() > 0
    finally:
        backend.close()
