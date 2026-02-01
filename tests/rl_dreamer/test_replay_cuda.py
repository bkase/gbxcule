from __future__ import annotations

import os
from pathlib import Path

import pytest

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES
from gbxcule.rl.dreamer_v3.cuda_guard import profile_no_host_memcpy
from gbxcule.rl.dreamer_v3.ingest_cuda import ReplayIngestorCUDA
from gbxcule.rl.dreamer_v3.replay_commit import ReplayCommitManager
from gbxcule.rl.dreamer_v3.replay_cuda import ReplayRingCUDA

torch = pytest.importorskip("torch")


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


ROM_PATH = Path("bench/roms/out/BG_STATIC.gb")


def _step_tensors(ring: ReplayRingCUDA, value: int):
    assert ring.obs_shape is not None
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
    is_first = torch.zeros((ring.num_envs,), dtype=torch.bool, device=ring.device)
    continue_t = torch.ones((ring.num_envs,), dtype=torch.float32, device=ring.device)
    episode_id = torch.zeros((ring.num_envs,), dtype=torch.int32, device=ring.device)
    return obs, action, reward, is_first, continue_t, episode_id


def _step_dict_tensors(ring: ReplayRingCUDA, value: int, senses_dim: int):
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
    is_first = torch.zeros((ring.num_envs,), dtype=torch.bool, device=ring.device)
    continue_t = torch.ones((ring.num_envs,), dtype=torch.float32, device=ring.device)
    episode_id = torch.zeros((ring.num_envs,), dtype=torch.int32, device=ring.device)
    return obs, action, reward, is_first, continue_t, episode_id


def test_commit_stride_logic() -> None:
    commit = ReplayCommitManager(commit_stride=2, safety_margin=1, device="cpu")
    assert commit.committed_t == -1
    commit.mark_written(0)
    assert commit.committed_t == -1
    commit.mark_written(1)
    assert commit.committed_t == 1
    commit.mark_written(2)
    assert commit.committed_t == 1
    commit.mark_written(3)
    assert commit.committed_t == 3
    assert commit.safe_max_t() == 2


def test_exclude_write_index_when_full() -> None:
    ring = ReplayRingCUDA(capacity=4, num_envs=1, device="cpu")
    for t in range(4):
        obs, action, reward, is_first, continue_t, episode_id = _step_tensors(
            ring, value=t
        )
        ring.push_step(
            obs=obs,
            action=action,
            reward=reward,
            is_first=is_first,
            continue_=continue_t,
            episode_id=episode_id,
        )
    assert ring.size == 4
    assert ring.head == 0
    gen = torch.Generator().manual_seed(0)
    batch = ring.sample_sequences(
        batch=8,
        seq_len=1,
        gen=gen,
        committed_t=ring.total_steps - 1,
        exclude_head=True,
        return_indices=True,
    )
    time_idx = batch["meta"]["time_idx"].view(-1)
    assert torch.all(time_idx != ring.head)


def test_ingestor_alignment_initial_step() -> None:
    ring = ReplayRingCUDA(capacity=8, num_envs=2, device="cpu")
    commit = ReplayCommitManager(commit_stride=1, safety_margin=0, device="cpu")
    ingestor = ReplayIngestorCUDA(ring, commit)

    def _render(slot):
        slot.fill_(3)

    ingestor.start(_render)
    action = torch.zeros((2,), dtype=torch.int32)
    ingestor.commit_action(action)
    assert ring.reward[0, 0].item() == 0.0
    assert ring.is_first[0, 0].item() is True
    assert ring.action[0, 0].item() == 0


def test_dict_obs_cpu_sampling() -> None:
    senses_dim = 4
    obs_spec = {
        "pixels": ((1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES), torch.uint8),
        "senses": ((senses_dim,), torch.float32),
    }
    ring = ReplayRingCUDA(capacity=6, num_envs=2, device="cpu", obs_spec=obs_spec)
    for t in range(4):
        obs, action, reward, is_first, continue_t, episode_id = _step_dict_tensors(
            ring, value=t, senses_dim=senses_dim
        )
        ring.push_step(
            obs=obs,
            action=action,
            reward=reward,
            is_first=is_first,
            continue_=continue_t,
            episode_id=episode_id,
        )
    gen = torch.Generator().manual_seed(0)
    batch = ring.sample_sequences(batch=2, seq_len=3, gen=gen)
    assert set(batch["obs"].keys()) == {"pixels", "senses"}
    assert batch["obs"]["pixels"].shape == (3, 2, 1, DOWNSAMPLE_H, DOWNSAMPLE_W_BYTES)
    assert batch["obs"]["pixels"].dtype is torch.uint8
    assert batch["obs"]["senses"].shape == (3, 2, senses_dim)
    assert batch["obs"]["senses"].dtype is torch.float32


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_direct_write_and_alignment() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")
    if not ROM_PATH.exists():
        pytest.skip("ROM missing")
    from gbxcule.backends.warp_vec import WarpVecCudaBackend

    ring = ReplayRingCUDA(capacity=4, num_envs=1, device="cuda")
    ring.assert_alignment()
    slot = ring.obs_slot(0)
    assert not isinstance(slot, dict)
    assert slot.is_contiguous()
    backend = WarpVecCudaBackend(
        str(ROM_PATH),
        num_envs=1,
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_pixels_packed=True,
    )
    try:
        backend.reset(seed=0)
        backend.render_pixels_snapshot()
        internal = backend.pixels_packed_torch().clone()
        backend.render_pixels_snapshot_packed_to_torch(slot, 0)
        torch.cuda.synchronize()
        assert torch.equal(slot[:, 0], internal)
    finally:
        backend.close()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_commit_event_gating() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")
    ring = ReplayRingCUDA(capacity=4, num_envs=1, device="cuda")
    commit = ReplayCommitManager(commit_stride=1, safety_margin=0, device="cuda")
    actor_stream = torch.cuda.Stream()
    learner_stream = torch.cuda.Stream()
    slot = ring.obs_slot(0)
    assert not isinstance(slot, dict)
    with torch.cuda.stream(actor_stream):
        slot.fill_(9)
        dummy = torch.zeros((1,), dtype=torch.int32, device="cuda")
        ring.push_step(
            obs=None,
            action=dummy,
            reward=torch.zeros((1,), dtype=torch.float32, device="cuda"),
            is_first=torch.zeros((1,), dtype=torch.bool, device="cuda"),
            continue_=torch.ones((1,), dtype=torch.float32, device="cuda"),
            episode_id=torch.zeros((1,), dtype=torch.int32, device="cuda"),
        )
        commit.mark_written(ring.total_steps - 1, stream=actor_stream)

    with torch.cuda.stream(learner_stream):
        commit.wait_for_commit(0, stream=learner_stream)
        observed = slot.clone()

    torch.cuda.synchronize()
    assert observed.sum().item() == 9 * observed.numel()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_cuda_no_host_memcpy_guard() -> None:
    if not torch.cuda.is_available():
        pytest.skip("torch CUDA not available")
    torch.zeros((1,), device="cuda")
    torch.cuda.synchronize()

    def _work():
        x = torch.zeros((1024, 1024), device="cuda")
        _ = x + 1
        torch.cuda.synchronize()

    profile_no_host_memcpy(_work)
