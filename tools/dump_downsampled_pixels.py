"""Dump downsampled pixel buffer to PNG (debug utility)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from gbxcule.backends.warp_vec import WarpVecCpuBackend, WarpVecCudaBackend
from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", default=None, help="Optional .state file path")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", default="downsampled_pixels.png")
    return parser.parse_args()


def _build_backend(rom: str, backend: str):
    backend_cls = WarpVecCpuBackend if backend == "cpu" else WarpVecCudaBackend
    return backend_cls(
        rom_path=rom,
        num_envs=1,
        frames_per_step=24,
        release_after_frames=8,
        obs_dim=32,
        render_pixels=True,
    )


def main() -> None:
    args = _parse_args()
    rom_path = Path(args.rom)
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")

    backend = _build_backend(str(rom_path), args.backend)
    try:
        backend.reset(seed=0)
        if args.state:
            backend.load_state_file(str(args.state), env_idx=0)
        actions = np.zeros((1,), dtype=np.int32)
        for _ in range(max(1, args.steps)):
            backend.step(actions)
        pix = backend.pixels_wp().numpy().reshape(1, DOWNSAMPLE_H, DOWNSAMPLE_W)[0]
        palette = np.array([255, 170, 85, 0], dtype=np.uint8)
        img = Image.fromarray(palette[pix], mode="L")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        img.save(output)
        print(f"Wrote {output}")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
