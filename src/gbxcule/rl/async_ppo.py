"""Async PPO buffer orchestration utilities (CUDA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl. Install with `uv sync --group rl`."
        ) from exc


@dataclass
class AsyncPPOBufferState:
    ready_event: Any
    free_event: Any
    policy_version: int = 0


class AsyncPPOBufferManager:
    """Double-buffer coordination with CUDA streams and events."""

    def __init__(self, num_buffers: int = 2) -> None:
        torch = _require_torch()
        if num_buffers < 1:
            raise ValueError("num_buffers must be >= 1")
        self._torch = torch
        self._buffers = [
            AsyncPPOBufferState(
                ready_event=torch.cuda.Event(),
                free_event=torch.cuda.Event(),
                policy_version=0,
            )
            for _ in range(num_buffers)
        ]
        stream = torch.cuda.current_stream()
        for buf in self._buffers:
            buf.free_event.record(stream)

    @property
    def num_buffers(self) -> int:
        return len(self._buffers)

    def wait_free(self, buf_idx: int, stream) -> None:  # type: ignore[no-untyped-def]
        stream.wait_event(self._buffers[buf_idx].free_event)

    def wait_ready(self, buf_idx: int, stream) -> None:  # type: ignore[no-untyped-def]
        stream.wait_event(self._buffers[buf_idx].ready_event)

    def mark_ready(self, buf_idx: int, stream, policy_version: int) -> None:  # type: ignore[no-untyped-def]
        buf = self._buffers[buf_idx]
        buf.policy_version = int(policy_version)
        buf.ready_event.record(stream)

    def mark_free(self, buf_idx: int, stream) -> None:  # type: ignore[no-untyped-def]
        self._buffers[buf_idx].free_event.record(stream)

    def policy_version(self, buf_idx: int) -> int:
        return int(self._buffers[buf_idx].policy_version)
