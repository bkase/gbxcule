"""Tests for documentation consistency (docs-as-code lite)."""

import re
from pathlib import Path


def _get_makefile_targets() -> set[str]:
    """Extract all target names from Makefile."""
    makefile_path = Path(__file__).parent.parent / "Makefile"
    content = makefile_path.read_text()
    # Match target names (lines like "target:" or "target: deps")
    targets = set()
    for line in content.splitlines():
        match = re.match(r"^([a-zA-Z0-9_-]+):", line)
        if match:
            targets.add(match.group(1))
    return targets


def _get_readme_make_commands() -> set[str]:
    """Extract make command targets referenced in README."""
    readme_path = Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    # Match "make <target>" patterns (with word boundary after target)
    targets = set()
    for match in re.finditer(r"\bmake\s+([a-zA-Z0-9_-]+)\b", content):
        targets.add(match.group(1))
    return targets


def test_readme_make_commands_exist_in_makefile() -> None:
    """Ensure all make commands referenced in README exist as Makefile targets."""
    makefile_targets = _get_makefile_targets()
    readme_commands = _get_readme_make_commands()

    missing = readme_commands - makefile_targets
    assert not missing, (
        f"README references make targets that don't exist in Makefile: {missing}"
    )


def test_readme_references_key_commands() -> None:
    """Ensure README documents the essential commands."""
    readme_commands = _get_readme_make_commands()
    essential = {"setup", "bench", "verify", "check"}

    missing = essential - readme_commands
    assert not missing, f"README should document these essential commands: {missing}"
