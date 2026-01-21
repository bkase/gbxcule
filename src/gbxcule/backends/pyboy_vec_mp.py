"""Multiprocessing PyBoy backend (CPU baseline)."""

# Lazy import to avoid heavy deps at module import time


def _import_pyboy():  # type: ignore[no-untyped-def]
    """Lazily import PyBoy."""
    from pyboy import PyBoy

    return PyBoy


# Implementation will be added in later stories
