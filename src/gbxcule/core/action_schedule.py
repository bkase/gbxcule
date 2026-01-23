"""Shared action press/release timing helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def validate_schedule(frames_per_step: int, release_after_frames: int) -> None:
    """Validate press/release schedule parameters."""
    if frames_per_step < 0:
        raise ValueError("frames_per_step must be >= 0")
    if release_after_frames < 0:
        raise ValueError("release_after_frames must be >= 0")
    if release_after_frames > frames_per_step:
        raise ValueError(
            "release_after_frames cannot exceed frames_per_step "
            f"({release_after_frames} > {frames_per_step})"
        )


def pressed_for_frame(frame_idx: int, release_after_frames: int) -> bool:
    """Return True if input should be considered pressed at frame_idx."""
    if frame_idx < 0:
        raise ValueError("frame_idx must be >= 0")
    if release_after_frames < 0:
        raise ValueError("release_after_frames must be >= 0")
    return frame_idx < release_after_frames


def split_press_release_ticks(
    frames_per_step: int, release_after_frames: int
) -> tuple[int, int]:
    """Return (pressed_ticks, remaining_ticks) for a step."""
    validate_schedule(frames_per_step, release_after_frames)
    pressed_ticks = release_after_frames
    remaining_ticks = frames_per_step - pressed_ticks
    return pressed_ticks, remaining_ticks


def run_puffer_press_release_schedule(
    *,
    send_input: Callable[..., Any],
    tick: Callable[..., Any],
    press_event: Any,
    release_event: Any,
    frames_per_step: int,
    release_after_frames: int,
) -> None:
    """Run puffer-style press/release schedule against a PyBoy-like API."""
    validate_schedule(frames_per_step, release_after_frames)
    if frames_per_step < 1:
        raise ValueError("frames_per_step must be >= 1")

    send_input(press_event)
    send_input(release_event, delay=release_after_frames)

    if frames_per_step > 1:
        tick(frames_per_step - 1, render=False)
    tick(1, render=False)
