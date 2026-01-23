"""Shared action press/release timing helpers."""

from __future__ import annotations


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
