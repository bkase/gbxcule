"""Goal template artifacts and metadata validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import json
import os
import tempfile

import numpy as np

from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W

GOAL_TEMPLATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GoalTemplateMeta:
    schema_version: int
    created_at: str
    rom_path: str
    rom_sha256: str
    state_path: str
    state_sha256: str
    actions_path: str
    actions_sha256: str
    action_codec_id: str
    frames_per_step: int
    release_after_frames: int
    downsample_h: int
    downsample_w: int
    stack_k: int
    shade_levels: int
    dist_metric: str
    tau: float
    k_consecutive: int
    pipeline_version: int

    @staticmethod
    def now_iso() -> str:
        return datetime.now(tz=UTC).isoformat()


def compute_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_actions_trace_jsonl(path: Path) -> list[list[int]]:
    actions: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_num} in {path}: {exc}"
                ) from exc
            if not isinstance(data, list):
                raise ValueError(
                    f"Expected JSON list at line {line_num} in {path}, got {type(data)}"
                )
            try:
                actions.append([int(x) for x in data])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid action list at line {line_num} in {path}: {exc}"
                ) from exc
    if not actions:
        raise ValueError(f"Actions trace is empty: {path}")
    return actions


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, prefix=path.name, suffix=".tmp"
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, prefix=path.name, suffix=".tmp"
    ) as tmp:
        np.save(tmp, arr)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _meta_from_dict(data: dict[str, Any]) -> GoalTemplateMeta:
    required = {field.name for field in GoalTemplateMeta.__dataclass_fields__.values()}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing meta fields: {sorted(missing)}")
    return GoalTemplateMeta(
        schema_version=int(data["schema_version"]),
        created_at=str(data["created_at"]),
        rom_path=str(data["rom_path"]),
        rom_sha256=str(data["rom_sha256"]),
        state_path=str(data["state_path"]),
        state_sha256=str(data["state_sha256"]),
        actions_path=str(data["actions_path"]),
        actions_sha256=str(data["actions_sha256"]),
        action_codec_id=str(data["action_codec_id"]),
        frames_per_step=int(data["frames_per_step"]),
        release_after_frames=int(data["release_after_frames"]),
        downsample_h=int(data["downsample_h"]),
        downsample_w=int(data["downsample_w"]),
        stack_k=int(data["stack_k"]),
        shade_levels=int(data["shade_levels"]),
        dist_metric=str(data["dist_metric"]),
        tau=float(data["tau"]),
        k_consecutive=int(data["k_consecutive"]),
        pipeline_version=int(data["pipeline_version"]),
    )


def _validate_template_shape(template: np.ndarray, meta: GoalTemplateMeta) -> None:
    if template.dtype != np.uint8:
        raise ValueError(f"template dtype must be uint8, got {template.dtype}")
    if template.ndim not in (2, 3):
        raise ValueError(f"template must be 2D or 3D, got {template.ndim}D")
    if template.ndim == 2:
        if template.shape != (meta.downsample_h, meta.downsample_w):
            raise ValueError(
                "template shape mismatch: "
                f"{template.shape} vs {(meta.downsample_h, meta.downsample_w)}"
            )
    else:
        if template.shape[1:] != (meta.downsample_h, meta.downsample_w):
            raise ValueError(
                "template shape mismatch: "
                f"{template.shape[1:]} vs {(meta.downsample_h, meta.downsample_w)}"
            )
        if template.shape[0] != meta.stack_k:
            raise ValueError(
                f"template stack_k {template.shape[0]} != meta.stack_k {meta.stack_k}"
            )


def validate_meta(
    meta: GoalTemplateMeta,
    *,
    action_codec_id: str | None = None,
    frames_per_step: int | None = None,
    release_after_frames: int | None = None,
    downsample_h: int | None = None,
    downsample_w: int | None = None,
    stack_k: int | None = None,
    shade_levels: int | None = None,
    dist_metric: str | None = None,
    pipeline_version: int | None = None,
) -> None:
    if meta.schema_version != GOAL_TEMPLATE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {meta.schema_version} != {GOAL_TEMPLATE_SCHEMA_VERSION}"
        )
    if action_codec_id is not None and meta.action_codec_id != action_codec_id:
        raise ValueError(
            f"action_codec_id {meta.action_codec_id} != expected {action_codec_id}"
        )
    if frames_per_step is not None and meta.frames_per_step != frames_per_step:
        raise ValueError(
            f"frames_per_step {meta.frames_per_step} != expected {frames_per_step}"
        )
    if (
        release_after_frames is not None
        and meta.release_after_frames != release_after_frames
    ):
        raise ValueError(
            "release_after_frames "
            f"{meta.release_after_frames} != expected {release_after_frames}"
        )
    if downsample_h is not None and meta.downsample_h != downsample_h:
        raise ValueError(
            f"downsample_h {meta.downsample_h} != expected {downsample_h}"
        )
    if downsample_w is not None and meta.downsample_w != downsample_w:
        raise ValueError(
            f"downsample_w {meta.downsample_w} != expected {downsample_w}"
        )
    if stack_k is not None and meta.stack_k != stack_k:
        raise ValueError(f"stack_k {meta.stack_k} != expected {stack_k}")
    if shade_levels is not None and meta.shade_levels != shade_levels:
        raise ValueError(
            f"shade_levels {meta.shade_levels} != expected {shade_levels}"
        )
    if dist_metric is not None and meta.dist_metric != dist_metric:
        raise ValueError(f"dist_metric {meta.dist_metric} != expected {dist_metric}")
    if pipeline_version is not None and meta.pipeline_version != pipeline_version:
        raise ValueError(
            f"pipeline_version {meta.pipeline_version} != expected {pipeline_version}"
        )


def save_goal_template(
    output_dir: Path,
    template: np.ndarray,
    meta: GoalTemplateMeta,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "goal_template.npy"
    meta_path = output_dir / "goal_template.meta.json"
    if not force and (template_path.exists() or meta_path.exists()):
        raise FileExistsError(f"Goal template already exists in {output_dir}")
    _validate_template_shape(template, meta)
    _atomic_save_npy(template_path, template)
    meta_json = json.dumps(asdict(meta), indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(meta_path, meta_json)
    return template_path, meta_path


def load_goal_template(
    output_dir: Path,
    *,
    action_codec_id: str | None = None,
    frames_per_step: int | None = None,
    release_after_frames: int | None = None,
    downsample_h: int | None = DOWNSAMPLE_H,
    downsample_w: int | None = DOWNSAMPLE_W,
    stack_k: int | None = None,
    shade_levels: int | None = 4,
    dist_metric: str | None = None,
    pipeline_version: int | None = None,
) -> tuple[np.ndarray, GoalTemplateMeta]:
    template_path = output_dir / "goal_template.npy"
    meta_path = output_dir / "goal_template.meta.json"
    if not template_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing goal_template files in {output_dir}")
    template = np.load(template_path)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = _meta_from_dict(data)
    _validate_template_shape(template, meta)
    validate_meta(
        meta,
        action_codec_id=action_codec_id,
        frames_per_step=frames_per_step,
        release_after_frames=release_after_frames,
        downsample_h=downsample_h,
        downsample_w=downsample_w,
        stack_k=stack_k,
        shade_levels=shade_levels,
        dist_metric=dist_metric,
        pipeline_version=pipeline_version,
    )
    return template, meta

