"""Commit gating for CUDA replay ingestion (Dreamer v3 M6)."""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


class ReplayCommitManager:  # type: ignore[no-any-unimported]
    """Track commit fences for replay ingestion and safe sampling."""

    def __init__(
        self,
        *,
        commit_stride: int,
        safety_margin: int,
        device: str = "cuda",
        event_slots: int = 4,
    ) -> None:
        torch = _require_torch()
        if commit_stride < 1:
            raise ValueError("commit_stride must be >= 1")
        if safety_margin < 0:
            raise ValueError("safety_margin must be >= 0")
        if event_slots < 1:
            raise ValueError("event_slots must be >= 1")

        self._torch = torch
        self.commit_stride = int(commit_stride)
        self.safety_margin = int(safety_margin)
        device_obj = torch.device(device)
        if (
            device_obj.type == "cuda"
            and device_obj.index is None
            and torch.cuda.is_available()
        ):
            device_obj = torch.device("cuda", torch.cuda.current_device())
        self.device = device_obj
        self._write_t = -1
        self._committed_t = -1
        self._use_events = self.device.type == "cuda" and torch.cuda.is_available()
        self._events = []
        if self._use_events:
            self._events = [
                torch.cuda.Event(enable_timing=False) for _ in range(event_slots)
            ]

    @property
    def write_t(self) -> int:
        return self._write_t

    @property
    def committed_t(self) -> int:
        return self._committed_t

    def safe_max_t(self) -> int:
        return self._committed_t - self.safety_margin

    def mark_written(self, write_t: int, stream=None) -> None:  # type: ignore[no-untyped-def]
        """Update write pointer and record commit events when stride hits."""
        if write_t < 0:
            raise ValueError("write_t must be >= 0")
        self._write_t = int(write_t)
        if (write_t + 1) % self.commit_stride == 0:
            self.record_commit(write_t=write_t, stream=stream)

    def record_commit(self, write_t: int | None = None, stream=None) -> None:  # type: ignore[no-untyped-def]
        """Record a commit event for the latest completed write."""
        if write_t is None:
            write_t = self._write_t
        if write_t < 0:
            return
        self._committed_t = int(write_t)
        if not self._use_events:
            return
        slot = (self._committed_t // self.commit_stride) % len(self._events)
        event = self._events[slot]
        if stream is None:
            event.record()
        else:
            event.record(stream)

    def wait_for_commit(self, target_t: int, stream=None) -> None:  # type: ignore[no-untyped-def]
        """Wait until the latest commit event covers the requested target."""
        if (
            target_t <= self._committed_t
            and self._use_events
            and self._committed_t >= 0
        ):
            slot = (self._committed_t // self.commit_stride) % len(self._events)
            event = self._events[slot]
            if stream is None:
                self._torch.cuda.current_stream().wait_event(event)
            else:
                stream.wait_event(event)
