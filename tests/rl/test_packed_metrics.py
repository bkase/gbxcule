from __future__ import annotations

import inspect

import torch

from gbxcule.rl.packed_metrics import packed_l1_distance
from gbxcule.rl.packed_pixels import pack_2bpp_u8, unpack_2bpp_u8


def test_packed_l1_distance_exists() -> None:
    assert callable(packed_l1_distance)


def test_packed_l1_distance_matches_unpack() -> None:
    obs_u8 = torch.randint(0, 4, (4, 1, 72, 80), dtype=torch.uint8)
    goal_u8 = torch.randint(0, 4, (1, 1, 72, 80), dtype=torch.uint8)
    obs_packed = pack_2bpp_u8(obs_u8)
    goal_packed = pack_2bpp_u8(goal_u8)

    dist_packed = packed_l1_distance(obs_packed, goal_packed)

    obs_unpacked = unpack_2bpp_u8(obs_packed)
    goal_unpacked = unpack_2bpp_u8(goal_packed)
    dist_ref = (
        torch.abs(obs_unpacked.float() - goal_unpacked.float()).mean(dim=(1, 2, 3))
        / 3.0
    )
    assert torch.allclose(dist_packed, dist_ref, atol=0.01)


def test_packed_l1_uses_diff_lut() -> None:
    source = inspect.getsource(packed_l1_distance)
    assert "diff_lut" in source or "get_diff_lut" in source


def test_packed_l1_distance_output_shape() -> None:
    obs = torch.randint(0, 256, (16, 1, 72, 20), dtype=torch.uint8)
    goal = torch.randint(0, 256, (1, 1, 72, 20), dtype=torch.uint8)
    dist = packed_l1_distance(obs, goal)
    assert dist.shape == (16,)
