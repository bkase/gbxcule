"""Schema helpers for RL experiment artifacts."""

from __future__ import annotations

from typing import Any, TypeGuard

RL_RUN_SCHEMA_VERSION = 1
RL_METRICS_SCHEMA_VERSION = 1
RL_FAILURE_SCHEMA_VERSION = 1

META_REQUIRED_FIELDS = {
    "run_id",
    "timestamp_utc",
    "schema_version",
    "rom",
    "env",
    "pipeline",
    "algo",
    "code",
    "system",
}

META_REQUIRED_ROM_FIELDS = {"rom_path", "rom_sha256"}
META_REQUIRED_STATE_FIELDS = {"state_path", "state_sha256"}
META_REQUIRED_ENV_FIELDS = {
    "num_envs",
    "frames_per_step",
    "release_after_frames",
    "stack_k",
}
META_REQUIRED_PIPELINE_FIELDS = {"obs_format", "action_codec_id"}
META_REQUIRED_ALGO_FIELDS = {"algo_name", "algo_version"}
META_REQUIRED_CODE_FIELDS = {"git_commit", "git_dirty"}
META_REQUIRED_SYSTEM_FIELDS = {
    "platform",
    "python",
    "torch_version",
    "warp_version",
    "cuda_available",
    "gpu_name",
}

METRICS_REQUIRED_FIELDS = {
    "run_id",
    "trace_id",
    "schema_version",
    "wall_time_s",
    "env_steps",
    "sps",
}

FAILURE_REQUIRED_FIELDS = {
    "run_id",
    "timestamp_utc",
    "schema_version",
    "kind",
    "trace_id",
}


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> TypeGuard[int | float]:
    if not _is_number(value):
        return False
    if value != value:
        return False
    return value not in (float("inf"), float("-inf"))


def _missing_fields(payload: dict[str, Any], required: set[str]) -> set[str]:
    return required - set(payload.keys())


def validate_meta(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = _missing_fields(meta, META_REQUIRED_FIELDS)
    if missing:
        errors.append(f"meta missing keys: {sorted(missing)}")
        return errors
    if meta.get("schema_version") != RL_RUN_SCHEMA_VERSION:
        errors.append(
            "meta.schema_version must be "
            f"{RL_RUN_SCHEMA_VERSION}, got {meta.get('schema_version')}"
        )
    rom = meta.get("rom")
    if not isinstance(rom, dict):
        errors.append("meta.rom must be dict")
    else:
        missing_rom = _missing_fields(rom, META_REQUIRED_ROM_FIELDS)
        if missing_rom:
            errors.append(f"meta.rom missing keys: {sorted(missing_rom)}")
    state = meta.get("state")
    if state is not None:
        if not isinstance(state, dict):
            errors.append("meta.state must be dict")
        else:
            missing_state = _missing_fields(state, META_REQUIRED_STATE_FIELDS)
            if missing_state:
                errors.append(f"meta.state missing keys: {sorted(missing_state)}")
    env = meta.get("env")
    if not isinstance(env, dict):
        errors.append("meta.env must be dict")
    else:
        missing_env = _missing_fields(env, META_REQUIRED_ENV_FIELDS)
        if missing_env:
            errors.append(f"meta.env missing keys: {sorted(missing_env)}")
    pipeline = meta.get("pipeline")
    if not isinstance(pipeline, dict):
        errors.append("meta.pipeline must be dict")
    else:
        missing_pipeline = _missing_fields(pipeline, META_REQUIRED_PIPELINE_FIELDS)
        if missing_pipeline:
            errors.append(f"meta.pipeline missing keys: {sorted(missing_pipeline)}")
    algo = meta.get("algo")
    if not isinstance(algo, dict):
        errors.append("meta.algo must be dict")
    else:
        missing_algo = _missing_fields(algo, META_REQUIRED_ALGO_FIELDS)
        if missing_algo:
            errors.append(f"meta.algo missing keys: {sorted(missing_algo)}")
    code = meta.get("code")
    if not isinstance(code, dict):
        errors.append("meta.code must be dict")
    else:
        missing_code = _missing_fields(code, META_REQUIRED_CODE_FIELDS)
        if missing_code:
            errors.append(f"meta.code missing keys: {sorted(missing_code)}")
    system = meta.get("system")
    if not isinstance(system, dict):
        errors.append("meta.system must be dict")
    else:
        missing_system = _missing_fields(system, META_REQUIRED_SYSTEM_FIELDS)
        if missing_system:
            errors.append(f"meta.system missing keys: {sorted(missing_system)}")
        cuda_available = system.get("cuda_available")
        if not isinstance(cuda_available, bool):
            errors.append("meta.system.cuda_available must be bool")
    return errors


def validate_metrics_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = _missing_fields(record, METRICS_REQUIRED_FIELDS)
    if missing:
        errors.append(f"record missing keys: {sorted(missing)}")
        return errors
    if record.get("schema_version") != RL_METRICS_SCHEMA_VERSION:
        errors.append(
            "record.schema_version must be "
            f"{RL_METRICS_SCHEMA_VERSION}, got {record.get('schema_version')}"
        )
    for key in ("wall_time_s", "env_steps", "sps"):
        if not _is_finite_number(record.get(key)):
            errors.append(f"record.{key} must be finite number")
    return errors


def validate_failure_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = _missing_fields(record, FAILURE_REQUIRED_FIELDS)
    if missing:
        errors.append(f"failure missing keys: {sorted(missing)}")
        return errors
    if record.get("schema_version") != RL_FAILURE_SCHEMA_VERSION:
        errors.append(
            "failure.schema_version must be "
            f"{RL_FAILURE_SCHEMA_VERSION}, got {record.get('schema_version')}"
        )
    return errors
