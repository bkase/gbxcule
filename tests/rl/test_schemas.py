from __future__ import annotations

from gbxcule.rl.schemas import (
    META_REQUIRED_FIELDS,
    RL_FAILURE_SCHEMA_VERSION,
    RL_METRICS_SCHEMA_VERSION,
    RL_RUN_SCHEMA_VERSION,
    validate_meta,
    validate_metrics_record,
)


def _create_test_meta() -> dict:
    return {
        "schema_version": RL_RUN_SCHEMA_VERSION,
        "run_id": "run-test",
        "timestamp_utc": "2026-01-30T00:00:00+00:00",
        "rom": {"rom_path": "rom.gb", "rom_sha256": "a" * 64},
        "state": {"state_path": "state.bin", "state_sha256": "b" * 64},
        "env": {
            "num_envs": 4,
            "frames_per_step": 24,
            "release_after_frames": 8,
            "stack_k": 1,
        },
        "pipeline": {"obs_format": "u8", "action_codec_id": "codec"},
        "algo": {"algo_name": "ppo", "algo_version": "0"},
        "code": {"git_commit": "deadbeef", "git_dirty": False},
        "system": {
            "platform": "linux",
            "python": "3.11",
            "torch_version": "2.9.1",
            "warp_version": "1.11.0",
            "cuda_available": False,
            "gpu_name": None,
        },
    }


def test_meta_schema_has_required_fields() -> None:
    meta = _create_test_meta()
    errors = validate_meta(meta)
    assert errors == []
    for field in META_REQUIRED_FIELDS:
        assert field in meta


def test_metrics_record_schema() -> None:
    record = {
        "run_id": "x",
        "trace_id": "y",
        "schema_version": RL_METRICS_SCHEMA_VERSION,
        "wall_time_s": 1.0,
        "env_steps": 100,
        "sps": 100.0,
    }
    errors = validate_metrics_record(record)
    assert errors == []


def test_schema_versions_defined() -> None:
    assert all(
        v >= 1
        for v in [
            RL_RUN_SCHEMA_VERSION,
            RL_METRICS_SCHEMA_VERSION,
            RL_FAILURE_SCHEMA_VERSION,
        ]
    )
