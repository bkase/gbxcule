from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES  # noqa: E402
from gbxcule.rl.dreamer_v3.replay import ReplayRing  # noqa: E402


def _step_tensors(
    ring: ReplayRing, *, value: int, is_first: bool, episode_id: int, cont: float
):
    obs = torch.full(
        (ring.num_envs, *ring.obs_shape),
        value,
        dtype=torch.uint8,
        device=ring.device,
    )
    action = torch.full((ring.num_envs,), value, dtype=torch.int32, device=ring.device)
    reward = torch.full(
        (ring.num_envs,), float(value), dtype=torch.float32, device=ring.device
    )
    is_first_t = torch.full(
        (ring.num_envs,), is_first, dtype=torch.bool, device=ring.device
    )
    continue_t = torch.full(
        (ring.num_envs,), float(cont), dtype=torch.float32, device=ring.device
    )
    episode_t = torch.full(
        (ring.num_envs,), episode_id, dtype=torch.int32, device=ring.device
    )
    return obs, action, reward, is_first_t, continue_t, episode_t


def _step_dict_tensors(
    ring: ReplayRing,
    *,
    value: int,
    senses_dim: int,
    is_first: bool,
    episode_id: int,
    cont: float,
):
    pixels = torch.full(
        (ring.num_envs, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
        value,
        dtype=torch.uint8,
        device=ring.device,
    )
    senses = torch.full(
        (ring.num_envs, senses_dim),
        float(value),
        dtype=torch.float32,
        device=ring.device,
    )
    obs = {"pixels": pixels, "senses": senses}
    action = torch.full((ring.num_envs,), value, dtype=torch.int32, device=ring.device)
    reward = torch.full(
        (ring.num_envs,), float(value), dtype=torch.float32, device=ring.device
    )
    is_first_t = torch.full(
        (ring.num_envs,), is_first, dtype=torch.bool, device=ring.device
    )
    continue_t = torch.full(
        (ring.num_envs,), float(cont), dtype=torch.float32, device=ring.device
    )
    episode_t = torch.full(
        (ring.num_envs,), episode_id, dtype=torch.int32, device=ring.device
    )
    return obs, action, reward, is_first_t, continue_t, episode_t


def test_shapes_and_dtypes() -> None:
    ring = ReplayRing(capacity=8, num_envs=4, device="cpu")
    assert ring.obs.shape == (8, 4, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
    assert ring.obs.dtype is torch.uint8
    assert ring.action.shape == (8, 4)
    assert ring.action.dtype is torch.int32
    assert ring.reward.shape == (8, 4)
    assert ring.reward.dtype is torch.float32
    assert ring.is_first.shape == (8, 4)
    assert ring.is_first.dtype is torch.bool
    assert ring.continues.shape == (8, 4)
    assert ring.continues.dtype is torch.float32
    assert ring.episode_id.shape == (8, 4)
    assert ring.episode_id.dtype is torch.int32


def test_obs_slot_returns_view() -> None:
    ring = ReplayRing(capacity=4, num_envs=2, device="cpu")
    slot = ring.obs_slot(2)
    slot.fill_(7)
    assert ring.obs[2].sum().item() == 7 * slot.numel()


def test_wraparound_chronological_sampling() -> None:
    ring = ReplayRing(capacity=5, num_envs=1, device="cpu")
    for t in range(7):
        is_first = t in (0, 5)
        episode_id = 0 if t < 5 else 1
        obs, action, reward, is_first_t, continue_t, episode_t = _step_tensors(
            ring, value=t, is_first=is_first, episode_id=episode_id, cont=1.0
        )
        ring.push_step(
            obs=obs,
            action=action,
            reward=reward,
            is_first=is_first_t,
            continue_=continue_t,
            episode_id=episode_t,
        )
    assert ring.size == 5
    gen = torch.Generator().manual_seed(0)
    batch = ring.sample_sequences(batch=1, seq_len=5, gen=gen)
    values = batch["obs"][:, 0, 0, 0, 0].cpu().tolist()
    assert values == [2, 3, 4, 5, 6]

    gen = torch.Generator().manual_seed(123)
    batch = ring.sample_sequences(batch=1, seq_len=3, gen=gen)
    values = batch["obs"][:, 0, 0, 0, 0].cpu().tolist()
    assert values[1] == values[0] + 1
    assert values[2] == values[1] + 1


def test_episode_invariants_and_negative_case() -> None:
    ring = ReplayRing(capacity=6, num_envs=2, device="cpu")
    episode_ids = [0, 0, 1, 1]
    is_firsts = [True, False, True, False]
    for t, (ep, first) in enumerate(zip(episode_ids, is_firsts, strict=True)):
        obs, action, reward, is_first_t, continue_t, episode_t = _step_tensors(
            ring, value=t, is_first=first, episode_id=ep, cont=1.0
        )
        ring.push_step(
            obs=obs,
            action=action,
            reward=reward,
            is_first=is_first_t,
            continue_=continue_t,
            episode_id=episode_t,
        )
    ring.check_invariants()

    bad = ReplayRing(capacity=4, num_envs=1, device="cpu")
    for t in range(3):
        first = t == 0
        ep = 0 if t < 2 else 2  # skip without is_first
        obs, action, reward, is_first_t, continue_t, episode_t = _step_tensors(
            bad, value=t, is_first=first, episode_id=ep, cont=1.0
        )
        bad.push_step(
            obs=obs,
            action=action,
            reward=reward,
            is_first=is_first_t,
            continue_=continue_t,
            episode_id=episode_t,
        )
    with pytest.raises(ValueError, match="episode_id"):
        bad.check_invariants()


def test_continue_semantics() -> None:
    ring = ReplayRing(capacity=4, num_envs=1, device="cpu")
    obs, action, reward, is_first_t, continue_t, episode_t = _step_tensors(
        ring, value=0, is_first=True, episode_id=0, cont=1.0
    )
    ring.push_step(
        obs=obs,
        action=action,
        reward=reward,
        is_first=is_first_t,
        continue_=continue_t,
        episode_id=episode_t,
    )
    obs, action, reward, is_first_t, continue_t, episode_t = _step_tensors(
        ring, value=1, is_first=False, episode_id=0, cont=0.0
    )
    ring.push_step(
        obs=obs,
        action=action,
        reward=reward,
        is_first=is_first_t,
        continue_=continue_t,
        episode_id=episode_t,
    )
    ring.check_invariants()
    assert ring.continues[1, 0].item() == 0.0


def test_deterministic_sampling() -> None:
    ring = ReplayRing(capacity=6, num_envs=2, device="cpu")
    for t in range(6):
        obs, action, reward, is_first_t, continue_t, episode_t = _step_tensors(
            ring, value=t, is_first=t == 0, episode_id=0, cont=1.0
        )
        ring.push_step(
            obs=obs,
            action=action,
            reward=reward,
            is_first=is_first_t,
            continue_=continue_t,
            episode_id=episode_t,
        )
    gen1 = torch.Generator().manual_seed(999)
    gen2 = torch.Generator().manual_seed(999)
    out1 = ring.sample_sequences(batch=3, seq_len=4, gen=gen1, return_indices=True)
    out2 = ring.sample_sequences(batch=3, seq_len=4, gen=gen2, return_indices=True)
    assert torch.equal(out1["obs"], out2["obs"])
    assert torch.equal(out1["meta"]["env_idx"], out2["meta"]["env_idx"])
    assert torch.equal(out1["meta"]["start_offset"], out2["meta"]["start_offset"])


def test_non_strict_sampling_across_is_first() -> None:
    ring = ReplayRing(capacity=5, num_envs=1, device="cpu")
    for t in range(4):
        obs, action, reward, is_first_t, continue_t, episode_t = _step_tensors(
            ring, value=t, is_first=t == 2, episode_id=0 if t < 2 else 1, cont=1.0
        )
        ring.push_step(
            obs=obs,
            action=action,
            reward=reward,
            is_first=is_first_t,
            continue_=continue_t,
            episode_id=episode_t,
        )
    gen = torch.Generator().manual_seed(0)
    batch = ring.sample_sequences(batch=1, seq_len=4, gen=gen)
    assert batch["is_first"][2, 0].item() is True


def test_dict_obs_shapes_and_sampling() -> None:
    senses_dim = 5
    obs_spec = {
        "pixels": ((1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES), torch.uint8),
        "senses": ((senses_dim,), torch.float32),
    }
    ring = ReplayRing(capacity=6, num_envs=2, device="cpu", obs_spec=obs_spec)
    for t in range(4):
        obs, action, reward, is_first_t, continue_t, episode_t = _step_dict_tensors(
            ring,
            value=t,
            senses_dim=senses_dim,
            is_first=t == 0,
            episode_id=0,
            cont=1.0,
        )
        ring.push_step(
            obs=obs,
            action=action,
            reward=reward,
            is_first=is_first_t,
            continue_=continue_t,
            episode_id=episode_t,
        )
    gen = torch.Generator().manual_seed(0)
    batch = ring.sample_sequences(batch=2, seq_len=3, gen=gen)
    assert set(batch["obs"].keys()) == {"pixels", "senses"}
    assert batch["obs"]["pixels"].shape == (3, 2, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
    assert batch["obs"]["pixels"].dtype is torch.uint8
    assert batch["obs"]["senses"].shape == (3, 2, senses_dim)
    assert batch["obs"]["senses"].dtype is torch.float32


def test_dict_obs_rejects_tensor_obs() -> None:
    obs_spec = {
        "pixels": ((1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES), torch.uint8),
        "senses": ((3,), torch.float32),
    }
    ring = ReplayRing(capacity=4, num_envs=2, device="cpu", obs_spec=obs_spec)
    obs = torch.zeros(
        (ring.num_envs, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES),
        dtype=torch.uint8,
        device=ring.device,
    )
    action = torch.zeros((ring.num_envs,), dtype=torch.int32, device=ring.device)
    reward = torch.zeros((ring.num_envs,), dtype=torch.float32, device=ring.device)
    is_first_t = torch.ones((ring.num_envs,), dtype=torch.bool, device=ring.device)
    continue_t = torch.ones((ring.num_envs,), dtype=torch.float32, device=ring.device)
    episode_t = torch.zeros((ring.num_envs,), dtype=torch.int32, device=ring.device)
    with pytest.raises(ValueError, match="obs must be a dict"):
        ring.push_step(
            obs=obs,
            action=action,
            reward=reward,
            is_first=is_first_t,
            continue_=continue_t,
            episode_id=episode_t,
        )
