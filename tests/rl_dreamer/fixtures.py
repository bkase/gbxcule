"""Fixture loader helpers for Dreamer v3 tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FixtureEntry:
    name: str
    dtype: str
    shape: list[int]
    file: str
    notes: str | None = None


def load_manifest(path: Path) -> list[FixtureEntry]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("manifest must be a list")
    entries: list[FixtureEntry] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("manifest entries must be objects")
        entries.append(
            FixtureEntry(
                name=str(item.get("name")),
                dtype=str(item.get("dtype")),
                shape=list(item.get("shape", [])),
                file=str(item.get("file")),
                notes=item.get("notes"),
            )
        )
    return entries


def fixture_root() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "dreamer_v3"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixture(name: str) -> Any:
    root = fixture_root()
    return load_json(root / name)


def read_fixture_dir(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    entries = load_manifest(manifest_path)
    return {entry.name: entry for entry in entries}
