from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.async_ppo import AsyncPPOBufferManager  # noqa: E402


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_async_buffer_handshake() -> None:
    manager = AsyncPPOBufferManager(num_buffers=2)
    actor_stream = torch.cuda.Stream()
    learner_stream = torch.cuda.Stream()

    with torch.cuda.stream(actor_stream):
        manager.wait_free(0, actor_stream)
        _ = torch.zeros((16, 16), device="cuda")
        manager.mark_ready(0, actor_stream, policy_version=1)

    with torch.cuda.stream(learner_stream):
        manager.wait_ready(0, learner_stream)
        _ = torch.ones((8, 8), device="cuda")
        manager.mark_free(0, learner_stream)

    torch.cuda.synchronize()


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_policy_versioning() -> None:
    manager = AsyncPPOBufferManager(num_buffers=1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        manager.mark_ready(0, stream, policy_version=7)
    torch.cuda.synchronize()
    assert manager.policy_version(0) == 7
