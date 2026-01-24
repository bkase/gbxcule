"""Frame-hash comparisons for scanline-accurate BG renderer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gbxcule.backends.pyboy_single import PyBoySingleBackend
from gbxcule.backends.warp_vec import WarpVecCpuBackend

from .conftest import require_rom

# Add bench directory to path for harness helpers.
sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))

from harness import _frame_shades_from_backend, hash_memory  # noqa: E402

ROM_DIR = Path(__file__).parent.parent / "bench" / "roms" / "out"


def _verify_frame_hash_matches(
    rom_path: Path,
    *,
    warmup_frames: int = 2,
    compare_frames: int = 1,
) -> None:
    ref = PyBoySingleBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_frames=True,
    )
    dut = WarpVecCpuBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
        render_bg=True,
    )
    try:
        ref.reset()
        dut.reset()
        actions = np.zeros((1,), dtype=np.int32)

        for _ in range(warmup_frames):
            ref.step(actions)
            dut.step(actions)

        for frame_idx in range(compare_frames):
            ref.step(actions)
            dut.step(actions)
            ref_hash = hash_memory(_frame_shades_from_backend(ref).tobytes())
            dut_hash = hash_memory(_frame_shades_from_backend(dut).tobytes())
            assert ref_hash == dut_hash, (
                f"Frame hash mismatch at frame {frame_idx}: {ref_hash} vs {dut_hash}"
            )
    finally:
        ref.close()
        dut.close()


def test_bg_static_frame_hash_matches_pyboy() -> None:
    require_rom(ROM_DIR / "BG_STATIC.gb")
    _verify_frame_hash_matches(ROM_DIR / "BG_STATIC.gb")


def test_bg_scroll_signed_frame_hash_matches_pyboy() -> None:
    require_rom(ROM_DIR / "BG_SCROLL_SIGNED.gb")
    _verify_frame_hash_matches(ROM_DIR / "BG_SCROLL_SIGNED.gb")


def test_window_frame_hash_matches_pyboy() -> None:
    require_rom(ROM_DIR / "PPU_WINDOW.gb")
    _verify_frame_hash_matches(ROM_DIR / "PPU_WINDOW.gb")


def test_sprites_frame_hash_matches_pyboy() -> None:
    require_rom(ROM_DIR / "PPU_SPRITES.gb")
    _verify_frame_hash_matches(ROM_DIR / "PPU_SPRITES.gb")
