"""Serial output capture tests for Warp backend."""

from __future__ import annotations

import numpy as np

from bench.roms.build_micro_rom import build_serial_hello
from gbxcule.backends.warp_vec import WarpVecCpuBackend


def test_serial_hello_outputs_ok(tmp_path) -> None:
    rom_path = tmp_path / "SERIAL_HELLO.gb"
    rom_path.write_bytes(build_serial_hello())

    backend = WarpVecCpuBackend(
        str(rom_path),
        frames_per_step=1,
        release_after_frames=0,
        obs_dim=32,
    )
    try:
        backend.reset()
        actions = np.zeros((1,), dtype=np.int32)
        for _ in range(2):
            backend.step(actions)
        data = backend.read_serial(0)
        assert b"OK" in data
    finally:
        backend.close()
