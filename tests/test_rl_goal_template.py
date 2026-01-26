from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gbxcule.rl.goal_template import (
    GOAL_TEMPLATE_SCHEMA_VERSION,
    GoalTemplateMeta,
    compute_sha256,
    load_actions_trace_jsonl,
    load_goal_template,
    save_goal_template,
    validate_meta,
)


def _base_meta(tmp_path: Path) -> GoalTemplateMeta:
    return GoalTemplateMeta(
        schema_version=GOAL_TEMPLATE_SCHEMA_VERSION,
        created_at="2026-01-01T00:00:00Z",
        rom_path="rom.gb",
        rom_sha256="rom",
        state_path="start.state",
        state_sha256="state",
        actions_path="actions.jsonl",
        actions_sha256="actions",
        action_codec_id="pokemonred_puffer_v0",
        frames_per_step=24,
        release_after_frames=8,
        downsample_h=72,
        downsample_w=80,
        stack_k=4,
        shade_levels=4,
        dist_metric="l1_mean_norm",
        tau=0.05,
        k_consecutive=2,
        pipeline_version=1,
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    meta = _base_meta(tmp_path)
    template = np.zeros((72, 80), dtype=np.uint8)
    save_goal_template(tmp_path, template, meta, force=True)
    loaded, loaded_meta = load_goal_template(
        tmp_path,
        action_codec_id="pokemonred_puffer_v0",
        frames_per_step=24,
        release_after_frames=8,
        stack_k=None,
        dist_metric="l1_mean_norm",
        pipeline_version=1,
    )
    assert loaded.shape == template.shape
    assert loaded.dtype == template.dtype
    assert loaded_meta == meta


def test_meta_validation_mismatch(tmp_path: Path) -> None:
    meta = _base_meta(tmp_path)
    with pytest.raises(ValueError):
        validate_meta(meta, frames_per_step=12)


def test_actions_trace_loader(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    path.write_text(json.dumps([1, 2]) + "\n" + json.dumps([3, 4]) + "\n")
    actions = load_actions_trace_jsonl(path)
    assert actions == [[1, 2], [3, 4]]
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps({"action": 1}) + "\n")
    with pytest.raises(ValueError):
        load_actions_trace_jsonl(bad_path)


def test_compute_sha256(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc")
    assert compute_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
