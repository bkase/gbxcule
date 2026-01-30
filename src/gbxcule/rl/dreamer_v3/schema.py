"""Dreamer v3 tensor contracts (M0)."""

from __future__ import annotations

from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


OBS_PACKED2_DTYPE = "uint8"
OBS_PACKED2_SHAPE = (1, 72, 20)
ACTION_DTYPE = "int32"
REWARD_DTYPE = "float32"
CONTINUE_DTYPE = "float32"
EPISODE_ID_DTYPE = "int32"


def _dtype_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    torch = _require_torch()
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def assert_time_major(t, name: str) -> None:  # type: ignore[no-untyped-def]
    torch = _require_torch()
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if t.ndim < 2:
        raise ValueError(f"{name} must have shape [T, B, ...]")


def assert_float32(t, name: str) -> None:  # type: ignore[no-untyped-def]
    torch = _require_torch()
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if t.dtype is not torch.float32:
        raise ValueError(f"{name} must be float32")


def assert_int(t, name: str) -> None:  # type: ignore[no-untyped-def]
    torch = _require_torch()
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if t.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be int32 or int64")


def assert_packed2_obs(t, name: str) -> None:  # type: ignore[no-untyped-def]
    torch = _require_torch()
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if t.dtype is not torch.uint8:
        raise ValueError(f"{name} must be uint8")
    if t.ndim != 5:
        raise ValueError(f"{name} must have shape [T, B, 1, 72, 20]")
    if tuple(t.shape[-3:]) != OBS_PACKED2_SHAPE:
        raise ValueError(f"{name} must have spatial shape {OBS_PACKED2_SHAPE}")


def assert_time_major_packed2(t, name: str) -> None:  # type: ignore[no-untyped-def]
    assert_time_major(t, name)
    assert_packed2_obs(t, name)
