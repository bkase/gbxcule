"""Golden Bridge fixture generator scaffold for Dreamer v3.

This tool should only be used from CLI. It may import sheeprl or other
reference implementations inside `main()` to avoid polluting runtime deps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Dreamer v3 parity fixtures (skeleton)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/dreamer_v3"),
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Fixture subset selector (placeholder)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest",
    )
    return parser.parse_args(argv)


def _ensure_reference_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sheeprl_path = repo_root / "third_party" / "sheeprl"
    sys.path.insert(0, str(sheeprl_path))


def _write_manifest(
    out_dir: Path, entries: list[dict[str, Any]], *, force: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    _ensure_reference_on_path()
    # Placeholder: implement fixture generation in M1+.
    entries: list[dict[str, Any]] = []
    _write_manifest(args.out, entries, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
