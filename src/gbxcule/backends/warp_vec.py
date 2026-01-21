"""Warp-based backend (CPU-debug + GPU)."""

# Lazy import to avoid heavy deps at module import time


def _import_warp():  # type: ignore[no-untyped-def]
    """Lazily import Warp."""
    import warp as wp

    return wp


# Implementation will be added in later stories
