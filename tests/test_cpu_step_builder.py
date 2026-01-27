"""Tests for Warp cpu_step kernel generation pipeline."""

from __future__ import annotations

from pathlib import Path

from gbxcule.kernels import cpu_step


def test_cpu_step_kernel_written_to_disk(tmp_path: Path, monkeypatch) -> None:
    """Kernel source is written to disk (Warp requires file-backed modules)."""
    monkeypatch.setenv("GBXCULE_WARP_CACHE_DIR", str(tmp_path))
    # Use monkeypatch to swap the cache dict so it's restored after test
    # (avoids clearing the session-warmed kernels for other tests)
    monkeypatch.setattr(cpu_step, "_cpu_step_kernels", {})
    cpu_step.get_cpu_step_kernel()
    generated = list(tmp_path.glob("cpu_step_*.py"))
    assert generated, "expected cpu_step module written to cache directory"


def test_cpu_step_action_mapping_uses_v1_indices(tmp_path: Path, monkeypatch) -> None:
    """Generated kernel should reflect v1 action indices (NOOP at 0)."""
    monkeypatch.setenv("GBXCULE_WARP_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cpu_step, "_cpu_step_kernels", {})
    cpu_step.get_cpu_step_kernel()
    generated = list(tmp_path.glob("cpu_step_*.py"))
    assert generated, "expected cpu_step module written to cache directory"
    source = generated[0].read_text(encoding="utf-8")
    assert "if action == 1" in source and "BUTTON_A" in source
    assert "elif action == 2" in source and "BUTTON_B" in source
    assert "elif action == 3" in source and "BUTTON_START" in source
    assert "if action == 4" in source and "DPAD_UP" in source
    assert "elif action == 5" in source and "DPAD_DOWN" in source
    assert "elif action == 6" in source and "DPAD_LEFT" in source
    assert "elif action == 7" in source and "DPAD_RIGHT" in source
