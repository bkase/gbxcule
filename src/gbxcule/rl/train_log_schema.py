"""Schema helpers for train_log.jsonl."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeGuard

SCHEMA_VERSION = 1

REQUIRED_META_KEYS = {
    "schema_version",
    "run_id",
    "rom_path",
    "rom_sha256",
    "state_path",
    "state_sha256",
    "goal_dir",
    "goal_sha256",
    "action_codec_id",
    "frames_per_step",
    "release_after_frames",
    "stack_k",
    "num_envs",
    "num_actions",
    "seed",
}

REQUIRED_RECORD_KEYS = {
    "run_id",
    "trace_id",
    "env_steps",
    "opt_steps",
    "wall_time_s",
    "sps",
    "reward_mean",
    "done_rate",
    "trunc_rate",
    "reset_rate",
    "dist_p10",
    "dist_p50",
    "dist_p90",
    "entropy_mean",
    "value_mean",
    "value_std",
    "action_hist",
}


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> TypeGuard[int | float]:
    if not _is_number(value):
        return False
    if value != value:
        return False
    return value not in (float("inf"), float("-inf"))


def validate_meta(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_META_KEYS - meta.keys()
    if missing:
        errors.append(f"meta missing keys: {sorted(missing)}")
    if meta.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            "meta.schema_version must be "
            f"{SCHEMA_VERSION}, got {meta.get('schema_version')}"
        )
    for key in (
        "frames_per_step",
        "release_after_frames",
        "stack_k",
        "num_envs",
        "num_actions",
        "seed",
    ):
        if key in meta and not isinstance(meta[key], int):
            errors.append(f"meta.{key} must be int")
    return errors


def validate_record(
    record: dict[str, Any], *, num_actions: int | None = None
) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_RECORD_KEYS - record.keys()
    if missing:
        errors.append(f"record missing keys: {sorted(missing)}")
        return errors
    for key in (
        "env_steps",
        "opt_steps",
        "wall_time_s",
        "sps",
        "reward_mean",
        "done_rate",
        "trunc_rate",
        "reset_rate",
        "dist_p10",
        "dist_p50",
        "dist_p90",
        "entropy_mean",
        "value_mean",
        "value_std",
    ):
        if not _is_finite_number(record.get(key)):
            errors.append(f"record.{key} must be finite number")
    action_hist = record.get("action_hist")
    if not isinstance(action_hist, list) or not all(
        isinstance(x, int) for x in action_hist
    ):
        errors.append("record.action_hist must be list[int]")
    elif num_actions is not None and len(action_hist) != num_actions:
        errors.append(
            f"record.action_hist length {len(action_hist)} != num_actions {num_actions}"
        )
    for rate_key in ("done_rate", "trunc_rate", "reset_rate"):
        rate = record.get(rate_key)
        if _is_finite_number(rate) and not (0.0 <= float(rate) <= 1.0):
            errors.append(f"record.{rate_key} must be in [0, 1]")
    return errors


def iter_jsonl(path) -> Iterable[dict[str, Any]]:
    for line_idx, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = __import__("json").loads(line)
        except Exception as exc:  # pragma: no cover - caller handles
            raise ValueError(f"Invalid JSON on line {line_idx + 1}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object on line {line_idx + 1}")
        yield payload
