"""Warp kernels for CPU stepping."""

# Lazy import to avoid compilation at import time


def _import_warp():  # type: ignore[no-untyped-def]
    """Lazily import Warp."""
    import warp as wp

    return wp


# Kernel implementations will be added in later stories
