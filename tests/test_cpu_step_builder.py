"""Tests for Warp cpu_step kernel generation pipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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
    source = generated[0].read_text(encoding="utf-8")
    assert "@wp.kernel(enable_backward=False)" in source


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


def test_warp_compile_settings_are_explicit(monkeypatch) -> None:
    config = SimpleNamespace(
        mode="release",
        optimization_level=None,
        verify_cuda=False,
    )
    monkeypatch.setenv("GBXCULE_WARP_MODE", "release")
    monkeypatch.setenv("GBXCULE_WARP_OPTIMIZATION_LEVEL", "0")
    monkeypatch.setenv("GBXCULE_WARP_VERIFY_CUDA", "1")

    cpu_step._configure_warp(SimpleNamespace(config=config))

    assert config.mode == "release"
    assert config.optimization_level == 0
    assert config.verify_cuda is True


@pytest.mark.parametrize("value", ["-1", "4", "fast"])
def test_warp_optimization_level_rejects_invalid_values(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("GBXCULE_WARP_OPTIMIZATION_LEVEL", value)
    config = SimpleNamespace(
        mode="release",
        optimization_level=None,
        verify_cuda=False,
    )

    with pytest.raises(ValueError, match="integer from 0 to 3"):
        cpu_step._configure_warp(SimpleNamespace(config=config))
