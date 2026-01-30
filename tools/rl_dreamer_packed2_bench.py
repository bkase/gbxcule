"""Packed2 unpack micro-benchmark for Dreamer v3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Packed2 unpack micro-benchmark")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _require_torch():  # type: ignore[no-untyped-def]
    import importlib

    return importlib.import_module("torch")


def main() -> int:
    args = _parse_args()
    torch = _require_torch()
    from gbxcule.rl.packed_pixels import unpack_2bpp_u8

    device = torch.device(args.device)
    width_bytes = args.width // 4
    packed = torch.randint(
        0,
        256,
        (args.seq, args.batch, 1, args.height, width_bytes),
        dtype=torch.uint8,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for _ in range(args.warmup):
        _ = unpack_2bpp_u8(packed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(args.iters):
        _ = unpack_2bpp_u8(packed)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_frames = args.seq * args.batch * args.iters
    total_pixels = total_frames * args.height * args.width
    pixels_per_sec = total_pixels / elapsed
    bytes_per_sec = pixels_per_sec  # 1 byte per pixel for uint8
    gb_per_sec = bytes_per_sec / 1e9
    ms_per_step = (elapsed / args.iters) * 1e3

    result = {
        "device": str(device),
        "seq": args.seq,
        "batch": args.batch,
        "height": args.height,
        "width": args.width,
        "iters": args.iters,
        "elapsed_sec": elapsed,
        "ms_per_step": ms_per_step,
        "pixels_per_sec": pixels_per_sec,
        "gb_per_sec": gb_per_sec,
    }
    print(json.dumps(result, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
