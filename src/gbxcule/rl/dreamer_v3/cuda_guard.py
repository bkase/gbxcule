"""CUDA profiler guardrails for zero-copy ingestion (Dreamer v3 M6)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


def find_memcpy_events(profile: Any) -> list[str]:
    events = []
    for event in profile.key_averages():
        key = getattr(event, "key", None) or getattr(event, "name", "")
        lowered = str(key).lower()
        if "memcpy" in lowered:
            events.append(str(key))
    return events


def find_host_memcpy_events(memcpy_events: Iterable[str]) -> list[str]:
    host_events = []
    for key in memcpy_events:
        lowered = key.lower()
        if "htod" in lowered or "dtoh" in lowered:
            host_events.append(key)
    return host_events


def assert_no_host_memcpy(profile: Any) -> None:
    memcpy_events = find_memcpy_events(profile)
    host_events = find_host_memcpy_events(memcpy_events)
    if host_events:
        joined = "\n".join(host_events)
        raise RuntimeError(f"Host memcpy events detected:\n{joined}")


def profile_no_host_memcpy(  # type: ignore[no-untyped-def]
    fn,
    *,
    activities=None,
) -> Any:
    torch = _require_torch()
    if activities is None:
        activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        fn()
    assert_no_host_memcpy(prof)
    return prof
