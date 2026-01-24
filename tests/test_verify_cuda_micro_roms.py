"""CUDA verification tests against PyBoy for micro-ROMs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bench.harness import diff_states, normalize_cpu_state
from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend

from .conftest import require_rom

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    wp = pytest.importorskip("warp")
    wp.init()
    return wp.is_cuda_available()


def _verify_no_mismatch_cuda(
    rom_path: Path,
    *,
    steps: int = 32,
    mem_region: tuple[int, int] | None = None,
) -> None:
    ref = PyBoySingleBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
    )
    dut = WarpVecCudaBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
    )
    try:
        ref.reset()
        dut.reset()
        actions = np.zeros((1,), dtype=np.int32)
        for step_idx in range(steps):
            ref.step(actions)
            dut.step(actions)
            ref_state = normalize_cpu_state(ref.get_cpu_state(0))
            dut_state = normalize_cpu_state(dut.get_cpu_state(0))
            diff = diff_states(ref_state, dut_state)
            assert diff is None, f"Mismatch at step {step_idx}: {diff}"
            if mem_region is not None:
                lo, hi = mem_region
                ref_bytes = ref.read_memory(0, lo, hi)
                dut_bytes = dut.read_memory(0, lo, hi)
                assert ref_bytes == dut_bytes, (
                    f"Memory mismatch at step {step_idx} ({lo:04X}:{hi:04X})"
                )
    finally:
        ref.close()
        dut.close()


def _verify_no_mismatch_cpu_vs_cuda(
    rom_path: Path,
    *,
    steps: int = 8,
    mem_region: tuple[int, int] | None = None,
    compare_frame: bool = False,
    frame_warmup: int = 0,
) -> None:
    cpu = WarpVecCpuBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_bg=compare_frame,
    )
    cuda = WarpVecCudaBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_bg=compare_frame,
    )
    try:
        cpu.reset()
        cuda.reset()
        actions = np.zeros((1,), dtype=np.int32)
        for step_idx in range(steps):
            cpu.step(actions)
            cuda.step(actions)
            cpu_state = normalize_cpu_state(cpu.get_cpu_state(0))
            cuda_state = normalize_cpu_state(cuda.get_cpu_state(0))
            diff = diff_states(cpu_state, cuda_state)
            assert diff is None, f"Mismatch at step {step_idx}: {diff}"
            if mem_region is not None:
                lo, hi = mem_region
                cpu_bytes = cpu.read_memory(0, lo, hi)
                cuda_bytes = cuda.read_memory(0, lo, hi)
                assert cpu_bytes == cuda_bytes, (
                    f"Memory mismatch at step {step_idx} ({lo:04X}:{hi:04X})"
                )
            if compare_frame and step_idx >= frame_warmup:
                cpu_frame = cpu.read_frame_bg_shade_env0()
                cuda_frame = cuda.read_frame_bg_shade_env0()
                assert cpu_frame == cuda_frame, f"Frame mismatch at step {step_idx}"
    finally:
        cpu.close()
        cuda.close()


def test_cuda_verify_alu_loop() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    require_rom(ROM_DIR / "ALU_LOOP.gb")
    _verify_no_mismatch_cuda(ROM_DIR / "ALU_LOOP.gb")


def test_cuda_verify_mem_rwb() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    require_rom(ROM_DIR / "MEM_RWB.gb")
    _verify_no_mismatch_cuda(
        ROM_DIR / "MEM_RWB.gb",
        steps=64,
        mem_region=(0xC000, 0xC100),
    )


def test_cuda_parity_dma_oam_copy() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    require_rom(ROM_DIR / "DMA_OAM_COPY.gb")
    _verify_no_mismatch_cpu_vs_cuda(
        ROM_DIR / "DMA_OAM_COPY.gb",
        steps=8,
        mem_region=(0xFE00, 0xFEA0),
    )


def test_cuda_parity_stat_irq_counter() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    require_rom(ROM_DIR / "PPU_STAT_IRQ.gb")
    _verify_no_mismatch_cpu_vs_cuda(
        ROM_DIR / "PPU_STAT_IRQ.gb",
        steps=12,
        mem_region=(0xC000, 0xC002),
    )


def test_cuda_parity_window_frame() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    require_rom(ROM_DIR / "PPU_WINDOW.gb")
    _verify_no_mismatch_cpu_vs_cuda(
        ROM_DIR / "PPU_WINDOW.gb",
        steps=4,
        compare_frame=True,
        frame_warmup=2,
    )


def test_cuda_parity_sprites_frame() -> None:
    if not _cuda_available():
        pytest.skip("CUDA disabled for dev runs")
    require_rom(ROM_DIR / "PPU_SPRITES.gb")
    _verify_no_mismatch_cpu_vs_cuda(
        ROM_DIR / "PPU_SPRITES.gb",
        steps=4,
        compare_frame=True,
        frame_warmup=2,
    )
