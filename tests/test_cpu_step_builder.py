"""Tests for Warp cpu_step kernel generation pipeline."""

from __future__ import annotations

from pathlib import Path

from gbxcule.kernels import cpu_step


def test_cpu_step_kernel_written_to_disk(tmp_path: Path, monkeypatch) -> None:
    """Kernel source is written to disk (Warp requires file-backed modules)."""
    monkeypatch.setenv("GBXCULE_WARP_CACHE_DIR", str(tmp_path))
    cpu_step._cpu_step_kernels = {}
    cpu_step.get_cpu_step_kernel()
    generated = list(tmp_path.glob("cpu_step_*.py"))
    assert generated, "expected cpu_step module written to cache directory"
