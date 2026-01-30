"""Fixture loader tests for Dreamer v3."""

from __future__ import annotations

from tests.rl_dreamer.fixtures import fixture_root, load_manifest, read_fixture_dir


def test_empty_manifest_loads() -> None:
    manifest = fixture_root() / "manifest.json"
    entries = load_manifest(manifest)
    assert entries


def test_read_fixture_dir_has_entries() -> None:
    root = fixture_root()
    fixtures = read_fixture_dir(root)
    assert "bins" in fixtures
    assert "symlog_cases" in fixtures
    assert "twohot_cases" in fixtures
    assert "symlog_twohot_dist" in fixtures
    assert "return_ema_cases" in fixtures
