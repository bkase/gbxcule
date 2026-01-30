"""Fixture loader tests for Dreamer v3."""

from __future__ import annotations

from tests.rl_dreamer.fixtures import fixture_root, load_manifest, read_fixture_dir


def test_manifest_loads_entries() -> None:
    manifest = fixture_root() / "manifest.json"
    entries = load_manifest(manifest)
    assert entries


def test_read_fixture_dir_contains_entries() -> None:
    root = fixture_root()
    fixtures = read_fixture_dir(root)
    assert "bins" in fixtures
    assert "rssm_meta" in fixtures
    for entry in fixtures.values():
        assert (root / entry.file).exists()
