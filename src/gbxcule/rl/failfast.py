"""Fail-fast helpers for detecting tensor issues early."""

from __future__ import annotations

import traceback
from typing import Any


def _require_torch():  # type: ignore[no-untyped-def]
    import importlib

    return importlib.import_module("torch")


def _snapshot_tensors(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    torch = _require_torch()
    snapshots: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            snap = value.detach()
            if snap.ndim > 0:
                snap = snap[:8]
            snapshots[key] = snap.cpu()
        else:
            snapshots[key] = value
    return snapshots


def _write_failure_bundle(
    exp: Any,
    *,
    kind: str,
    name: str,
    trace_id: str | None,
    message: str,
    snapshots: dict[str, Any] | None,
) -> None:
    if exp is None:
        return
    writer = getattr(exp, "write_failure_bundle", None)
    if not callable(writer):
        return
    extra = {
        "assertion": name,
        "message": message,
        "traceback": "".join(traceback.format_stack()),
    }
    try:
        writer(
            kind=kind,
            error=message,
            trace_id=trace_id,
            extra=extra,
            tensors=_snapshot_tensors(snapshots),
        )
    except Exception:
        return


def assert_finite(
    t,  # type: ignore[no-untyped-def]
    name: str,
    exp: Any,
    trace_id: str | None,
    *,
    snapshots: dict[str, Any] | None = None,
) -> None:
    torch = _require_torch()
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if not torch.isfinite(t).all():
        message = f"{name} has non-finite values"
        _write_failure_bundle(
            exp,
            kind="non_finite",
            name=name,
            trace_id=trace_id,
            message=message,
            snapshots=snapshots,
        )
        raise AssertionError(message)


def assert_device(
    t,  # type: ignore[no-untyped-def]
    device,
    name: str,
    exp: Any,
    trace_id: str | None,
    *,
    snapshots: dict[str, Any] | None = None,
) -> None:
    torch = _require_torch()
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    expected = torch.device(device)
    if t.device != expected:
        message = f"{name} expected device {expected}, got {t.device}"
        _write_failure_bundle(
            exp,
            kind="device_drift",
            name=name,
            trace_id=trace_id,
            message=message,
            snapshots=snapshots,
        )
        raise AssertionError(message)


def assert_shape(
    t,  # type: ignore[no-untyped-def]
    expected: tuple[int, ...],
    name: str,
    exp: Any,
    trace_id: str | None,
    *,
    snapshots: dict[str, Any] | None = None,
) -> None:
    torch = _require_torch()
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    actual = tuple(t.shape)
    if actual != tuple(expected):
        message = f"{name} expected shape {expected}, got {actual}"
        _write_failure_bundle(
            exp,
            kind="shape_mismatch",
            name=name,
            trace_id=trace_id,
            message=message,
            snapshots=snapshots,
        )
        raise AssertionError(message)
