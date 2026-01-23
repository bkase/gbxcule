"""Tests for action schedule helpers."""

from __future__ import annotations

import pytest

from gbxcule.core.action_schedule import (
    pressed_for_frame,
    run_puffer_press_release_schedule,
    split_press_release_ticks,
    validate_schedule,
)


def test_validate_schedule_ok() -> None:
    """Valid schedules pass."""
    validate_schedule(24, 8)
    validate_schedule(1, 0)
    validate_schedule(0, 0)


def test_validate_schedule_negative_raises() -> None:
    """Negative values raise."""
    with pytest.raises(ValueError, match="frames_per_step"):
        validate_schedule(-1, 0)
    with pytest.raises(ValueError, match="release_after_frames"):
        validate_schedule(1, -1)


def test_validate_schedule_release_exceeds_frames() -> None:
    """release_after_frames > frames_per_step raises."""
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_schedule(4, 5)


def test_pressed_for_frame_basic() -> None:
    """pressed_for_frame matches expected boundaries."""
    assert pressed_for_frame(0, 1) is True
    assert pressed_for_frame(1, 1) is False
    assert pressed_for_frame(0, 0) is False


def test_pressed_for_frame_negative_index_raises() -> None:
    """Negative frame index raises."""
    with pytest.raises(ValueError, match="frame_idx"):
        pressed_for_frame(-1, 1)


def test_split_press_release_ticks() -> None:
    """split_press_release_ticks returns correct counts."""
    assert split_press_release_ticks(24, 8) == (8, 16)
    assert split_press_release_ticks(1, 1) == (1, 0)
    assert split_press_release_ticks(0, 0) == (0, 0)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def send_input(self, event: object, delay: int | None = None) -> None:
        self.calls.append(("send_input", (event,), {"delay": delay}))

    def tick(self, count: int, *, render: bool = False) -> None:
        self.calls.append(("tick", (count,), {"render": render}))


def test_run_puffer_press_release_schedule_frames_gt_one() -> None:
    recorder = _Recorder()
    run_puffer_press_release_schedule(
        send_input=recorder.send_input,
        tick=recorder.tick,
        press_event="PRESS",
        release_event="RELEASE",
        frames_per_step=4,
        release_after_frames=2,
    )
    assert recorder.calls == [
        ("send_input", ("PRESS",), {"delay": None}),
        ("send_input", ("RELEASE",), {"delay": 2}),
        ("tick", (3,), {"render": False}),
        ("tick", (1,), {"render": False}),
    ]


def test_run_puffer_press_release_schedule_frames_eq_one() -> None:
    recorder = _Recorder()
    run_puffer_press_release_schedule(
        send_input=recorder.send_input,
        tick=recorder.tick,
        press_event="PRESS",
        release_event="RELEASE",
        frames_per_step=1,
        release_after_frames=0,
    )
    assert recorder.calls == [
        ("send_input", ("PRESS",), {"delay": None}),
        ("send_input", ("RELEASE",), {"delay": 0}),
        ("tick", (1,), {"render": False}),
    ]


def test_run_puffer_press_release_schedule_invalid_frames() -> None:
    recorder = _Recorder()
    with pytest.raises(ValueError, match="frames_per_step"):
        run_puffer_press_release_schedule(
            send_input=recorder.send_input,
            tick=recorder.tick,
            press_event="PRESS",
            release_event="RELEASE",
            frames_per_step=0,
            release_after_frames=0,
        )
