"""Fixture loader tests for Dreamer v3."""

from __future__ import annotations

from pathlib import Path

from tests.rl_dreamer.fixtures import load_manifest, read_fixture_dir


def test_manifest_loads_entries() -> None:
    manifest = Path(__file__).parents[1] / "fixtures" / "dreamer_v3" / "manifest.json"
    entries = load_manifest(manifest)
    assert entries


def test_read_fixture_dir_contains_entries() -> None:
    root = Path(__file__).parents[1] / "fixtures" / "dreamer_v3"
    fixtures = read_fixture_dir(root)
    assert fixtures
    for entry in fixtures.values():
        assert (root / entry.file).exists()
