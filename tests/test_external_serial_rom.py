"""External serial ROM smoke test (skip if ROM missing)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gbxcule.backends.warp_vec import WarpVecCpuBackend


def test_external_serial_rom_pass_token() -> None:
    rom_path = Path(
        os.environ.get("GBXCULE_EXTERNAL_SERIAL_ROM", "bench/roms/external/serial.gb")
    )
    if not rom_path.exists():
        pytest.skip(f"External serial ROM not found: {rom_path}")
    pass_token = os.environ.get("GBXCULE_EXTERNAL_SERIAL_TOKEN", "PASS")
    steps = int(os.environ.get("GBXCULE_EXTERNAL_SERIAL_STEPS", "8"))

    backend = WarpVecCpuBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
    )
    try:
        backend.reset()
        actions = np.zeros((1,), dtype=np.int32)
        for _ in range(steps):
            backend.step(actions)
        data = backend.read_serial(0)
        assert pass_token.encode() in data
    finally:
        backend.close()
