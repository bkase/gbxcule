"""Unified experiment harness for RL runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from gbxcule.rl.schemas import (
    RL_FAILURE_SCHEMA_VERSION,
    RL_METRICS_SCHEMA_VERSION,
    RL_RUN_SCHEMA_VERSION,
)


def _slug(text: str) -> str:
    out = []
    for ch in text.strip():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("-")
    slug = "".join(out).strip("-_")
    return slug or "run"


def _normalize_payload(payload: Any) -> Any:
    if is_dataclass(payload) and not isinstance(payload, type):
        return asdict(payload)
    return payload


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _append_jsonl_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            if existing:
                handle.write(existing)
                if not existing.endswith("\n"):
                    handle.write("\n")
            handle.write(json.dumps(record))
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _atomic_torch_save(path: Path, payload: Any) -> None:
    import importlib

    torch = importlib.import_module("torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


class Experiment:
    """Experiment run directory + artifact writer."""

    def __init__(
        self,
        *,
        algo: str,
        rom_id: str,
        tag: str,
        run_root: Path | str = "bench/runs/rl",
        meta: dict[str, Any] | None = None,
        config: Any | None = None,
        timestamp: datetime | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.algo = algo
        self.rom_id = rom_id
        self.tag = tag
        self.run_root = Path(run_root)
        self.timestamp = timestamp or datetime.now(UTC)
        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}__{_slug(algo)}__{_slug(rom_id)}__{_slug(tag)}"
        self.run_id, self.run_dir = self._reserve_run_dir(run_id)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.failures_dir = self.run_dir / "failures"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.trace_id = trace_id or uuid4().hex[:8]

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir.mkdir(parents=True, exist_ok=True)

        meta_payload = {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp.isoformat(),
            "schema_version": RL_RUN_SCHEMA_VERSION,
            "algo": self.algo,
            "rom_id": self.rom_id,
            "tag": self.tag,
        }
        if meta:
            meta_payload.update(_normalize_payload(meta))
        _write_json_atomic(self.run_dir / "meta.json", meta_payload)

        config_payload = _normalize_payload(config) if config is not None else {}
        _write_json_atomic(self.run_dir / "config.json", config_payload)

        if not self.metrics_path.exists():
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.touch()

    def _reserve_run_dir(self, base_id: str) -> tuple[str, Path]:
        run_id = base_id
        run_dir = self.run_root / run_id
        if not run_dir.exists():
            return run_id, run_dir
        suffix = uuid4().hex[:8]
        run_id = f"{base_id}__{suffix}"
        return run_id, self.run_root / run_id

    def log_metrics(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("trace_id", self.trace_id)
        payload.setdefault("schema_version", RL_METRICS_SCHEMA_VERSION)
        _append_jsonl_atomic(self.metrics_path, payload)

    def save_checkpoint(self, name: str, payload: Any) -> Path:
        path = self.checkpoints_dir / name
        _atomic_torch_save(path, payload)
        return path

    def write_failure_bundle(
        self,
        *,
        kind: str,
        error: Exception | str | None = None,
        trace_id: str | None = None,
        extra: dict[str, Any] | None = None,
        tensors: dict[str, Any] | None = None,
        repro: str | None = None,
    ) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        trace = trace_id or self.trace_id
        bundle_name = f"{timestamp}__{trace}__{_slug(kind)}"
        final_dir = self.failures_dir / bundle_name
        tmp_dir = self.failures_dir / f".tmp_{uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            failure_payload: dict[str, Any] = {
                "run_id": self.run_id,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "schema_version": RL_FAILURE_SCHEMA_VERSION,
                "kind": kind,
                "trace_id": trace,
            }
            if error is not None:
                if isinstance(error, Exception):
                    failure_payload["error_type"] = type(error).__name__
                    failure_payload["error_message"] = str(error)
                else:
                    failure_payload["error_message"] = str(error)
            if extra:
                failure_payload["extra"] = extra
            _write_json_atomic(tmp_dir / "failure.json", failure_payload)
            if tensors:
                _atomic_torch_save(tmp_dir / "tensors.pt", tensors)
            if repro:
                repro_path = tmp_dir / "repro.sh"
                repro_path.write_text(repro, encoding="utf-8")
            tmp_dir.replace(final_dir)
        except Exception:
            if tmp_dir.exists():
                for item in tmp_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                tmp_dir.rmdir()
            raise
        return final_dir
