#!/usr/bin/env python3
"""Generate MP4 video from an action trace JSON file."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_file", help="Path to trace JSON file")
    parser.add_argument("--rom", default="red.gb", help="Path to ROM")
    parser.add_argument(
        "--output", "-o", help="Output MP4 path (default: same as trace with .mp4)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument(
        "--scale", type=int, default=4, help="Scale factor for output video"
    )
    args = parser.parse_args()

    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"Error: trace file not found: {trace_path}")
        return 1

    # Load trace
    with open(trace_path) as f:
        trace = json.load(f)

    start_state = trace["start_state"]
    actions = trace["actions"]
    steps = trace["steps"]

    print(f"Trace: {steps} steps from {start_state}")
    print(f"Actions: {len(actions)}")

    # Output path
    output_path = Path(args.output) if args.output else trace_path.with_suffix(".mp4")

    # Create backend
    from gbxcule.backends.warp_vec import WarpVecCpuBackend
    from gbxcule.core.state_io import apply_state_to_warp_backend, load_pyboy_state

    backend = WarpVecCpuBackend(
        args.rom,
        num_envs=1,
        frames_per_step=24,
        release_after_frames=8,
        render_pixels=True,
    )

    # Load initial state
    initial_state = load_pyboy_state(start_state, expected_cart_ram_size=32768)
    backend.reset()
    apply_state_to_warp_backend(initial_state, backend, env_idx=0)

    # Collect frames
    print("Collecting frames...")
    frames = []

    # Get initial frame
    backend.render_pixels_snapshot()
    pixels = backend.pixels_wp().numpy().reshape(72, 80)
    frames.append(pixels.copy())

    # Execute actions and collect frames
    for i, action in enumerate(actions):
        action_arr = np.array([action], dtype=np.int32)
        backend.step(action_arr)
        backend.render_pixels_snapshot()
        pixels = backend.pixels_wp().numpy().reshape(72, 80)
        frames.append(pixels.copy())

        if (i + 1) % 100 == 0:
            print(f"  Step {i + 1}/{len(actions)}")

    backend.close()
    print(f"Collected {len(frames)} frames")

    # Convert shade values (0-3) to grayscale (0-255)
    # GB palette: 0=white (255), 1=light gray, 2=dark gray, 3=black (0)
    palette = np.array([255, 170, 85, 0], dtype=np.uint8)

    # Create video
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python required. Install with: pip install opencv-python")
        return 1

    h, w = 72, 80
    out_h, out_w = h * args.scale, w * args.scale

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(  # type: ignore[attr-defined]
        str(output_path), fourcc, args.fps, (out_w, out_h), isColor=False
    )

    print(f"Writing video to {output_path}...")
    for frame in frames:
        # Convert to grayscale
        gray = palette[frame]
        # Scale up
        interp = cv2.INTER_NEAREST  # type: ignore[attr-defined]
        scaled = cv2.resize(gray, (out_w, out_h), interpolation=interp)  # type: ignore[attr-defined]
        writer.write(scaled)

    writer.release()
    print(f"Done! Video saved to: {output_path}")
    print(f"  Resolution: {out_w}x{out_h}")
    print(f"  Frames: {len(frames)}")
    print(f"  Duration: {len(frames) / args.fps:.1f}s at {args.fps} FPS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
