"""Warp kernels for CPU stepping."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_wp: Any | None = None
_warp_initialized = False
_inc_counter_kernel: Callable[..., Any] | None = None
_warp_warmed = False


def get_warp() -> Any:  # type: ignore[no-untyped-def]
    """Import Warp and initialize once."""
    global _wp, _warp_initialized
    if _wp is None:
        import warp as wp

        _wp = wp
    if not _warp_initialized:
        _wp.init()
        _warp_initialized = True
    return _wp


def get_inc_counter_kernel():  # type: ignore[no-untyped-def]
    """Return the counter increment kernel (cached)."""
    global _inc_counter_kernel
    if _inc_counter_kernel is None:
        wp = get_warp()

        @wp.kernel
        def inc_counter(counters: wp.array(dtype=wp.int32)):  # type: ignore[name-defined]
            i = wp.tid()
            counters[i] = counters[i] + 1

        _inc_counter_kernel = inc_counter
    return _inc_counter_kernel


def warmup_warp_cpu() -> None:
    """Warm up Warp on CPU by compiling and launching the trivial kernel once."""
    global _warp_warmed
    if _warp_warmed:
        return
    wp = get_warp()
    device = wp.get_device("cpu")
    counters = wp.zeros(1, dtype=wp.int32, device=device)
    kernel = get_inc_counter_kernel()
    wp.launch(kernel, dim=1, inputs=[counters], device=device)
    wp.synchronize()
    _warp_warmed = True
